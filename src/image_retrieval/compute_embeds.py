from PIL import Image
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import clip
import numpy as np
import os
import sys
import torch

class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform):
        self.folder = folder
        self.transform = transform
        self.image_filenames = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            return self.transform(img), os.path.basename(img_path)
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            return None, None  # Skip this image

def save_embeddings(output_folder, batch_embeddings, batch_filenames, batch_index):
    save_path = os.path.join(output_folder, f'embeddings_batch_{batch_index}.npy')
    np.save(save_path, {"filenames": batch_filenames, "embeddings": batch_embeddings.cpu().numpy()})
    logger.debug(f"Saved embeddings for {len(batch_filenames)} images to {save_path}")

def process_images(args):
    logger.info(f"Using device: {'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu'}")
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logger.info(f"Created output directory: {args.output_folder}")

    dataset = ImageFolderDataset(args.image_folder, preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    batch_index = 0
    logger.info(f"Starting embedding computation in batches of size {args.batch_size}")

    for batch_images, batch_fnames in tqdm(dataloader, desc="Computing embeddings"):
        valid_batch_images = []
        valid_filenames = []
        for img, fname in zip(batch_images, batch_fnames):
            if img is not None:
                valid_batch_images.append(img)
                valid_filenames.append(fname)
        
        if not valid_batch_images:
            continue  # Skip if no valid images in the batch

        batch_images_tensor = torch.stack(valid_batch_images).to(device)

        with torch.no_grad():
            batch_embeddings = model.encode_image(batch_images_tensor)
        
        save_embeddings(args.output_folder, batch_embeddings, valid_filenames, batch_index)
        batch_index += 1

    logger.info("Embedding computation completed.")

def main():
    parser = argparse.ArgumentParser(description="Compute and save CLIP embeddings for a folder of images.")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument('--output_folder', type=str, required=True, help="Output Folderto save the embeddings.")
    parser.add_argument('--batch_size', type=int, default=4096, help="Batch size for computing embeddings and saving them.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to use for computation.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode with detailed logging.")

    args = parser.parse_args()

    # Set logging level
    logger.remove()
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")


    logger.info("Starting the CLIP image embedding script")
    
    try:
        process_images(args)
    except Exception as e:
        logger.exception(f"Error occurred during processing: {e}")
    finally:
        logger.info("Script finished.")

if __name__ == "__main__":
    main()
