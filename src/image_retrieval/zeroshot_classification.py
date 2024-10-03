from loguru import logger
from tqdm import tqdm
import clip
import json
import numpy as np
import os
import sys
import torch

# Standard CLIP ImageNet classification templates
IMAGENET_TEMPLATES = [
    'a photo of a {}.',
    'a rendering of a {}.',
    'a cropped photo of a {}.',
    'the photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of a {}.',
    'a close-up photo of a {}.',
    'a bright photo of a {}.',
    'a cropped photo of the {}.',
    'a photo of the {}.',
    'a good photo of a {}.',
    'a photo of one {}.',
    'a bad photo of the {}.',
    'a photo of the clean {}.',
    'a photo of the large {}.',
    'a photo of a small {}.'
]

def load_embeddings(embeddings_folder):
    """Load precomputed embeddings from folder."""
    embeddings = []
    filenames = []

    for file in os.listdir(embeddings_folder):
        if file.endswith('.npy'):
            data = np.load(os.path.join(embeddings_folder, file), allow_pickle=True).item()
            embeddings.append(torch.tensor(data['embeddings']))
            filenames.extend(data['filenames'])

    logger.info(f"Loaded {len(filenames)} images from {embeddings_folder}")
    return torch.cat(embeddings), filenames

def create_text_embeddings(model, categories, device):
    """Create text embeddings using CLIP templates and categories."""
    text_embeddings = []
    for category in categories:
        texts = [template.format(category) for template in IMAGENET_TEMPLATES]
        text_tokens = clip.tokenize(texts).to(device)
        with torch.no_grad():
            text_embedding = model.encode_text(text_tokens).mean(dim=0)
            text_embedding /= text_embedding.norm()
        text_embeddings.append(text_embedding)
    return torch.stack(text_embeddings)

def zero_shot_classification_batch(image_embeddings_batch, text_embeddings):
    """Perform zero-shot classification for a batch using image and text embeddings."""
    with torch.no_grad():
        logits = image_embeddings_batch @ text_embeddings.T
        probs = logits.softmax(dim=-1)
    return probs

def process_images(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    # Load embeddings
    image_embeddings, filenames = load_embeddings(args.embeddings_folder)

    # Load keywords from JSON file
    with open(args.json_file, 'r') as f:
        data = json.load(f)
        categories = data['keywords_to_partition']

    logger.info(f"Loaded {len(categories)} categories from {args.json_file}")

    # Create text embeddings
    logger.info("Creating text embeddings using CLIP templates...")
    text_embeddings = create_text_embeddings(model, categories, device).to(device)

    # Initialize classification results dictionary
    classification_results = {category: [] for category in categories}

    # Process in batches
    logger.info(f"Performing zero-shot classification in batches of size {args.batch_size}")
    for i in tqdm(range(0, len(image_embeddings), args.batch_size), desc="Classifying images"):
        batch_embeddings = image_embeddings[i:i + args.batch_size].to(device)
        batch_filenames = filenames[i:i + args.batch_size]

        # Perform zero-shot classification
        probs = zero_shot_classification_batch(batch_embeddings, text_embeddings)

        # Assign each image to the category with the highest probability
        predicted_indices = probs.argmax(dim=-1)
        for idx, pred_idx in enumerate(predicted_indices):
            category = categories[pred_idx.item()]
            classification_results[category].append(batch_filenames[idx])

    # Save results to the specified output file
    with open(args.output_file, 'w') as f:
        json.dump(classification_results, f, indent=4)

    logger.info(f"Classification results saved to {args.output_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Perform zero-shot classification on precomputed CLIP embeddings.")
    parser.add_argument('--embeddings_folder', type=str, required=True, help="Folder containing precomputed image embeddings.")
    parser.add_argument('--json_file', type=str, required=True, help="JSON file containing the 'keywords_to_partition' key.")
    parser.add_argument('--output_file', type=str, required=True, help="Name of the output JSON file.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for classification.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to use for computation.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode with detailed logging.")

    args = parser.parse_args()

    # Set logging level
    logger.remove()
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")


    logger.info("Starting zero-shot classification on precomputed embeddings")

    try:
        process_images(args)
    except Exception as e:
        logger.exception(f"Error occurred during processing: {e}")
    finally:
        logger.info("Script finished.")

if __name__ == "__main__":
    main()