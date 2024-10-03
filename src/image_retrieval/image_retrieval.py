from asyncio.subprocess import PIPE
from datetime import datetime
from loguru import logger
import argparse
import asyncio 
import json
import os
import sys

class ImageRetrieval:
    def __init__(self, args):
        self.skill_desc = args.skill_desc
        self.num = args.num
        self.candidate_images = args.candidate_images
        self.image_pool_index = args.image_pool_index
        self.logger = logger
        self.debug = args.debug
        self.dt_str = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        
        self.logger.info("Initialized ImageRetrieval with parameters:")
        self.logger.info(f"skill_desc: {self.skill_desc}")
        self.logger.info(f"candidate_images: {self.candidate_images}")
        self.logger.info(f"image_pool_index: {self.image_pool_index}")
        
        self.output_folder = os.path.join(self.candidate_images, self.dt_str)
        os.makedirs(self.output_folder, exist_ok=True)
        self.logger.info(f"Saving images to {self.output_folder}")
        
    def get_commands(self):
        with open(self.skill_desc, 'r') as file:
            self.skill_desc = json.load(file)
        keywords = self.skill_desc["keywords"]
        
        base_command = ["clip-retrieval", "filter",  "--indice_folder", self.image_pool_index, "--num_results", str(self.num), "--output_folder", str(self.output_folder)]
        
        # Start all retrieval
        for keyword in keywords:
            command = base_command + ["--query", f"\"{keyword}\""]
            command[-3] = os.path.join(self.output_folder, f"{keyword.strip().replace(' ', '_')}")
            os.makedirs(command[-3], exist_ok=True)
            command = " ".join(command)
            with open(os.path.join(self.output_folder, "_commands.sh"), "a") as f:
                f.write(f"# Keyword: {keyword}\n")
                f.write(command)
                f.write("\n\n") 
                
        os.chmod(os.path.join(self.output_folder, "_commands.sh"), 0o755)
        
        self.logger.info(f"Commands saved to {self.output_folder}/_commands.sh")
        return
    
    async def retrieve(self):
        with open(self.skill_desc, 'r') as file:
            self.skill_desc = json.load(file)
        keywords = self.skill_desc["keywords"]
        
        retrievers = []
        base_command = ["clip-retrieval", "filter",  "--indice_folder", self.image_pool_index, "--num_results", str(self.num), "--output_folder", str(self.output_folder)]
        
        # Start all retrieval
        for keyword in keywords:
            command = base_command + ["--query", f"\"{keyword}\""]
            command = " ".join(command)
            retrievers.append((keyword, await asyncio.subprocess.create_subprocess_shell(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)))
            
            with open(os.path.join(self.output_folder, "_commands.txt"), "a") as f:
                f.write(f"Keyword: {keyword}\n")
                f.write(command)
                f.write("\n\n") 
                
        # Wait for completion
        for keyword, retriever in retrievers:
            await retriever.wait()
            output = await retriever.stdout.read()
            output = output.decode('utf-8')
            with open(os.path.join(self.output_folder, "_logs.txt"), "a") as f:
                f.write(f"Keyword: {keyword}\n")
                f.write(output)
                f.write("\n\n")

        return                

def main():
    parser = argparse.ArgumentParser(description="Retrieve images based on skill descriptions.")
    parser.add_argument('--skill_desc', type=str, required=True, help="Path to the JSON file containing skill descriptions.")
    parser.add_argument('--num', type=int, required=True, help="Number of images to retrieve per keyword.")
    parser.add_argument('--candidate_images', type=str, required=True, help="Path to save the retrieved images.")
    parser.add_argument('--image_pool_index', type=str, default="/scratch/datacomp/image_pool_index", help="Path to the image pool index.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode with detailed logging.")

    args = parser.parse_args()

    # Set logging level
    logger.remove()
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")


    image_retrieval = ImageRetrieval(args)
    image_retrieval.get_commands()
    #asyncio.run(image_retrieval.retrieve())

if __name__ == "__main__":
    main()
