from utils import CLIPZeroShotClassifier, is_image_file
import argparse
import json
import json
import os
import torch
from loguru import logger

class ImagePartitioner:
    def __init__(self, candidate_image_folder, task_desc, model_name, gpu, batch_size):
        self.candidate_image_folder = candidate_image_folder
        self.task_desc_path = task_desc
        with open(task_desc, 'r') as file:
            self.task_desc = json.load(file)
        self.model_name = model_name
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.keywords = self.load_keywords()

    def load_keywords(self):
        keywords = []
        for subgroup in self.task_desc.get('subgroups', []):
            keywords.append(subgroup.get('keyword', []))
        return keywords

    def partition_images(self):
        classifier = CLIPZeroShotClassifier(model_name=self.model_name, batch_size=self.batch_size, device=self.device)
        candidate_image_paths = [os.path.join(self.candidate_image_folder, f) for f in os.listdir(self.candidate_image_folder) if is_image_file(f)]
        keyword_image_partition = classifier.classify(
            self.keywords, 
            candidate_image_paths
        )
        
        for subgroup in self.task_desc["subgroups"]:
            logger.info(f"Processing subgroup: {subgroup['keyword']}")
            subgroup["candidate_image_paths"] = [candidate_image_paths[i] for i in keyword_image_partition[subgroup["keyword"]]]

        with open(self.task_desc_path, 'w') as file:
            json.dump(self.task_desc, file, indent=3)
            
def main():
    parser = argparse.ArgumentParser(description="Partition candidate images based on task description.")
    parser.add_argument('--candidate_image_folder', type=str, help='Path to the image folder')
    parser.add_argument('--task_desc', type=str, help='Path to the task description JSON file')
    parser.add_argument('--model_name', type=str, help='Path to the CLIP model', default="openai/clip-vit-large-patch14")
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for processing')

    args = parser.parse_args()

    partitioner = ImagePartitioner(
        candidate_image_folder=args.candidate_image_folder,
        task_desc=args.task_desc,
        model_name=args.model_name,
        gpu=args.gpu,
        batch_size=args.batch_size
    )

    partitioner.partition_images()

if __name__ == "__main__":
    main()