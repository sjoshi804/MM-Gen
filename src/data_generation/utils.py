from PIL import Image
from torchvision import transforms 
from transformers import CLIPProcessor, CLIPModel
import torch

class CLIPZeroShotClassifier:
    def __init__(self, model_name, batch_size, device, logger=None):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.batch_size = batch_size
        self.logger = logger
        if self.logger is None:
            from loguru import logger
            self.logger = logger
        
    def classify(self, texts, image_paths):
        self.logger.info("Performing zero-shot classification.")
        text_embeddings = self.compute_text_embeddings(texts)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        classification_map = {text: [] for text in texts}
        batch_size = self.batch_size

        for i in range(0, len(image_paths), batch_size):
            batch_image_paths = image_paths[i:i + batch_size]
            image_embeddings = self.compute_image_embeddings(batch_image_paths)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

            similarities = torch.matmul(image_embeddings, text_embeddings.T)
            image_to_text_indices = torch.argmax(similarities, dim=1)

            for img_idx, text_idx in enumerate(image_to_text_indices):
                classification_map[texts[text_idx.item()]].append(i + img_idx)

        self.logger.debug("Zero-shot classification completed.")
        return classification_map
    
    def compute_text_embeddings(self, texts):
        self.logger.debug("Computing text embeddings.")
        
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**inputs)
        
        self.logger.debug("Text embeddings computed.")
        return text_embeddings
    
    def compute_image_embeddings(self, image_paths):
        self.logger.debug("Computing image embeddings.")
        images = [Image.open(image_path) for image_path in image_paths]
        inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_embeddings = self.clip_model.get_image_features(**inputs)
        
        self.logger.debug("Image embeddings computed.")
        return image_embeddings
    
def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif'])

# give me a pytorch augmentation pipeline for images to augment "map" data s.t. I can create meaningful new images from it - gives similar
random_image_augmentation_pipeline = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with a 50% probability
        transforms.RandomVerticalFlip(p=0.1),    # Randomly flip the image vertically with a 10% probability
        transforms.RandomRotation(degrees=10),   # Randomly rotate the image by up to 10 degrees
        transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),  # Randomly crop and resize the image
    ]
)