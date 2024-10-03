from datetime import datetime
from enum import Enum
from loguru import logger
from tqdm import tqdm 
import argparse
import json
import math
import os
import random
import re
import sys

class GenerationMode(Enum):
    VQA = "vqa"
    VQA_NR = "vqa_nr"
    TQA = "tqa"
    DESCRIPT = "descript"
    GENERIC = "generic"
    
SAVE_PATH = "generated_prompts"

class SubsetBatchSampler:
    def __init__(self, samples, batch_size, subset_idx=None, include_indices=False):
        if subset_idx is None:
            subset_idx = list(range(len(samples)))
        self.samples = [samples[i] for i in subset_idx]
        self.batch_size = batch_size
        self.original_indices = subset_idx
        self.indices = range(len(self.samples))
        self.include_indices = include_indices
        self.reset()

    def reset(self):
        # Shuffle the indices at the start of each epoch
        self.shuffled_indices = random.sample(self.indices, len(self.indices))
        self.index = 0

    def get_batch(self):
        # Check if we need to reset for a new epoch
        if self.index >= len(self.shuffled_indices):
            self.reset()
        
        # Determine batch end index
        end_index = min(self.index + self.batch_size, len(self.shuffled_indices))
        # Get the current batch indices
        batch_indices = self.shuffled_indices[self.index:end_index]
        # Move the index forward
        self.index = end_index

        # Retrieve the corresponding samples
        batch_samples = [self.samples[i] for i in batch_indices]
        
        if self.include_indices: 
            return self.convert_to_original_indices(batch_indices), batch_samples
        else:
            return batch_samples
    
    def convert_to_original_indices(self, batch_indices):
        return [self.original_indices[i] for i in batch_indices]

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

class PromptGenerator:
    def __init__(self, args, logger):
        """
        Initialize the Prompt Generator class.
        """
        self.min_gen_per_candidate = args.min_gen_per_candidate
        self.task_desc_path = os.path.join(args.input_folder, args.task_desc)
        with open(self.task_desc_path, 'r') as file:
            self.task_desc = json.load(file)
        self.total_gen = args.total_gen
        self.num_icl_samples = args.num_icl_samples
        self.mode = args.mode
        self.logger = logger
        self.logger.info(f"Generation Mode: {self.mode}")
                    
        # DT string to associate examples with time of run
        self.dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(args.output_folder, SAVE_PATH, f"{args.file_prefix}_prompts_{self.dt_str}.json")
        os.makedirs(os.path.join(args.output_folder, SAVE_PATH), exist_ok=True)

    def number_to_ordinal(self, n):
        ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
        if 1 <= n <= 10:
            return ordinals[n - 1]
        else:
            raise ValueError("Number out of range. Only numbers from 1 to 10 are supported.")
        
    def generate_prompts(self):
        """
        This method processes candidate images, constructs prompts.
        """
        self.logger.info("Generating prompts.")
        self.prompt_objects = []
        
        # Compute number of samples to generate per subgroup
        ref_subgroup_sizes = [len(subgroup["reference_sample_idx"]) for subgroup in self.task_desc["subgroups"]]
        total_samples = sum(ref_subgroup_sizes)
        subgroup_proportions = [size / total_samples for size in ref_subgroup_sizes]
        gen_subgroup_sizes = [math.ceil(self.total_gen * prop / self.min_gen_per_candidate) for prop in subgroup_proportions]
        
        pbar = tqdm(total=sum(gen_subgroup_sizes), desc="Generating Prompts")
        for subgroup, subgroup_size in zip(self.task_desc["subgroups"], gen_subgroup_sizes):
            # Create a sampler for the subgroup candidates
            candidate_sampler = SubsetBatchSampler(subgroup["candidate_image_paths"], 1)
            
            # Create a batch sampler for the reference samples
            ref_sampler = SubsetBatchSampler(
                self.task_desc["samples"], 
                self.num_icl_samples, 
                subset_idx=subgroup["reference_sample_idx"], 
                include_indices=True
            )
            
            for _ in range(subgroup_size):
                
                ##################################################################
                #                    Generate Prompts                            #
                ##################################################################
                candidate_image_path = candidate_sampler.get_batch()[0]
                
                # Load candidate image
                self.logger.debug(f"Processing candidate image: {candidate_image_path}")

                # Construct ICL Prompt
                ref_indices, ref_batch = ref_sampler.get_batch()
                prompt = []
                for i, sample in enumerate(ref_batch):
                    example_num = i + 1
                    image = os.path.join(self.task_desc["image_folder"], sample["image_1"])
                    query = sample["conversations"][0]["value"]
                    answer = sample["conversations"][1]["value"]
                    query = re.sub("<image.?.?.?>", f"the {self.number_to_ordinal(example_num)} image", query)
                    query = re.sub("<.*>", "", query)
                    prompt.extend([f"Example {example_num}", image, f"Q: {query}\n A: {answer}"])
                
                if self.mode != GenerationMode.TQA:
                    prompt.append(candidate_image_path)

                self.prompt_objects.append({
                    "keyword": subgroup["keyword"],
                    "prompt": prompt, 
                    "icl_indices": ref_indices
                })
                pbar.update(1)
        pbar.close()
        
        with open(SAVE_PATH, 'w') as file:
            prompt_file = {
                "task_desc_path": self.task_desc_path,
                "dataset_description": self.task_desc["dataset_description"],
                "num_icl_samples": self.num_icl_samples,
                "mode": self.mode.value,
                "min_gen_per_candidate": self.min_gen_per_candidate,
                "total_gen": sum(gen_subgroup_sizes),
                "prompts": self.prompt_objects
            }
            json.dump(prompt_file, file, indent=3)
            
def load_json_file(skill_desc):
    """
    Load the JSON file containing the skill description, image folder, and sample questions.

    :param skill_desc: Path to the JSON file.
    :return: Dictionary containing skill description, image folder, and sample questions.
    """
    logger.debug(f"Loading JSON file from {skill_desc}.")
    with open(skill_desc, 'r') as file:
        data = json.load(file)

    logger.debug("Loaded JSON data successfully.")
    return data

def main(args):
    # Set logging level
    logger.remove()
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    # Load data from JSON file
    args.mode = GenerationMode(args.mode)
    generator = PromptGenerator(args, logger)
    
    logger.info("Generating prompts")
    generator.generate_prompts()
    logger.info(f"Finished. See generated data in {generator.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Generator for MultiModal Data Generation")
    parser.add_argument("--task_desc", type=str, required=True, help="Path to the JSON file containing task description")
    parser.add_argument('--input_folder', type=str, default="", help='Input Folder')
    parser.add_argument('--output_folder', type=str, default="", help='Output Folder')
    parser.add_argument("--file_prefix", type=str, required=True, help="Prefix for file with generated questions")
    parser.add_argument("--total_gen", type=int, help="Total data to generate")
    parser.add_argument("--min_gen_per_candidate", type=int, default=1, help="Number of generated data points per candidate.")
    parser.add_argument("--num_icl_samples", type=int, default=1, help="Number of in context samples provided per generation.")
    parser.add_argument("--mode", choices=[mode.value for mode in GenerationMode], default=GenerationMode.VQA_NR, help="Mode to generate data")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging")

    args = parser.parse_args()
    main(args)