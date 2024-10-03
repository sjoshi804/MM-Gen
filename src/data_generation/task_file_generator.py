from datetime import datetime
from loguru import logger
from utils import CLIPZeroShotClassifier
import argparse
import json 
import os
import sys
import torch 

class TaskFileGenerator():
    def __init__(self, args):
        logger.info("Initializing TaskFileGenerator with arguments: {}", args)
        self.data_file = args.data_file
        with open(self.data_file, 'r') as file:
            self.data = json.load(file)
        self.keywords = args.keywords
        logger.debug(f"Keywords: {self.keywords}")
        self.output_prefix = args.output_prefix
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.input_folder = args.input_folder
        
        self.task_file = self.initialize_task_file()
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        self.zeroshot_classifier = CLIPZeroShotClassifier(self.model_name, self.batch_size, args.gpu, logger)
        self.DT_STR = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        self.output_path = os.path.join(args.output_folder, f"{self.output_prefix}_{self.DT_STR}.json")
        logger.info("TaskFileGenerator initialized successfully.")
    
    def initialize_task_file(self):
        logger.debug("Initializing task file.")
        task_file = {
            "image_folder": self.data["image_folder"],
            "dataset_description": self.data["skill"],
            "subgroups": [],
            "samples": self.data["samples"]
        } 
        return task_file
    
    def generate_task_file(self):
        logger.info("Generating task file.")
        image_keyword_partition = self.zeroshot_classifier.classify(
            self.keywords, 
            [os.path.join(self.input_folder, self.data["image_folder"], sample["image_1"]) for sample in self.data["samples"]]
        )
        for keyword, image_indices in image_keyword_partition.items():
            subgroup = {
                "keyword": keyword,
                "reference_sample_idx": image_indices
            }
            logger.debug(f"Keyword: {keyword}, Num Samples: {len(image_indices)}")
            self.task_file["subgroups"].append(subgroup)
        with open(self.output_path, 'w') as file:
            json.dump(self.task_file, file, indent=3)
        logger.info(f"Task file generated successfully and saved to {self.output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate skills from a data file and keywords.')
    parser.add_argument('--data_file', type=str, help='Path to the data file')
    parser.add_argument('--input_folder', type=str, default="", help='Input Folder (only for AML)')
    parser.add_argument('--output_folder', type=str, default="", help='Output Folder (only for AML)')
    parser.add_argument('--keywords', nargs='+', help='List of keywords')
    parser.add_argument('--output_prefix', type=str, help='Prefix for output file')
    parser.add_argument('--model_name', type=str, help='Path to the CLIP model', default="openai/clip-vit-large-patch14")
    parser.add_argument('--batch_size', type=int, help='Batch size', default=512)
    parser.add_argument('--gpu', type=int, help='GPU device to use', default=0)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    logger.remove()
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    logger.info("Starting the task file generation process.")
    task_file_generator = TaskFileGenerator(args)
    task_file_generator.generate_task_file()
    logger.info("Task file generation process completed.")

if __name__ == '__main__':
    main()
