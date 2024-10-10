from PIL import Image
from datetime import datetime
from loguru import logger
from src.data_generation.gpt4 import GPTEndPoint
from src.data_generation.minimal_dep_utils import is_image_file, GenerationMode
from tqdm import tqdm 
import argparse
import json
import os
import random
import re
import signal
import sys
    
VQA_SYS_PROMPT = """You are an expert in <DATASET_DESC>. Your task is to generate high-quality question-answer pairs relevant to this skill similar to the following examples. 

Step-by-Step Process:
1. Analyze the Example: Review the provided example question-answer pair to understand the structure, focus, and context.
2. Understand the New Image: Infer relevant details, objects, and themes in the new image, considering how they relate to the skill.
3. Generate Questions: Create questions that reflect the context and content of the new image, ensuring they align with the skill and follow the example’s style.
4. If the question is a multiple choice question, make sure to include the options in the question.
5. Breakdown Reasoning: For each question, provide a detailed step-by-step breakdown of the reasoning process required to determine the correct answer. Highlight the key elements of the image and the logical steps leading to the answer.
6. Formulate Answers: After providing the reasoning, generate accurate and concise answers to the questions. Ensure that each answer is consistent with the reasoning provided.

Output Format:
Return the results as a JSON list of objects. Each object should include:
- "Q": The generated question (include options if it's multiple-choice).
- "R": The step-by-step reasoning that leads to the answer.
- "A": The generated answer.

Example Output:
[
  {"Q": "Generated question 1", "R": "Step-by-step reasoning for question 1", "A": "Generated answer 1"},
  {"Q": "Generated question 2", "R": "Step-by-step reasoning for question 2", "A": "Generated answer 2"}
]
"""

VQA_GEN_PROMPT = """Generate exactly <NUM> question-answer pairs for the following image. Each question-answer pair should adhere to the following guidelines:

1. **Question Format:** Follow the style and format of the provided example question. Ensure each question is clear, specific, and directly relevant to the image. Present the options with corresponding letters (e.g., A: <option_a>, B: <option_b>), placing each option on a new line.
   
2. **Reasoning Breakdown:** For each question, provide a detailed step-by-step breakdown of the reasoning process required to determine the correct answer. Explain the key elements in the image and the logical steps that lead to the answer.
   
3. **Answer Format:** After providing the reasoning, generate the correct answer. Ensure that each answer is consistent with the reasoning provided and directly supported by the image.

4. **JSON Format:** Structure each question-answer pair in JSON format as shown in the example. Ensure that all JSON keys ("Q", "R", "A") and values are correctly formatted and consistently applied.

Make sure the questions are varied in type and cover different aspects of the image to ensure a comprehensive and diverse set of questions and answers."""

VQA_NR_SYS_PROMPT = """You are an expert in <DATASET_DESC>. Given example image-question-answer tuples, your task is to generate *diverse* high-quality question-answer pairs relevant to this skill similar to the provided examples.

Step-by-Step Process:
1. Analyze the Example: Review the provided example question-answer pair to understand the structure, focus, and context.
2. Understand the New Image: Infer relevant details, objects, and themes in the new image, considering how they relate to the skill.
3. Generate Questions: Create questions that reflect the context and content of the new image, ensuring they align with the skill and follow the example’s style.
4. If the question is a multiple-choice question, make sure to include the options in the question.
5. Formulate Answers: Generate accurate and concise answers to the questions. Ensure each answer directly corresponds to the content of the new image.

Output Format:
Return the results as a JSON list of objects. Each object should include:
- "Q": The generated question (include options if it's multiple-choice).
- "A": The generated answer.

Example Output:
[
  {"Q": "Generated question 1", "A": "Generated answer 1"},
  {"Q": "Generated question 2", "A": "Generated answer 2"}
]
"""

VQA_NR_GEN_PROMPT = """Generate exactly <NUM> *diverse* question-answer pairs for the following image. Each question-answer pair should adhere to the following guidelines:

1. **Question Format:** Follow the style and format of the provided example questions. Ensure each question is clear, specific, and directly relevant to the following image. Present the options with corresponding letters (e.g., A: <option_a>, B: <option_b>), placing each option on a new line.

2. **Answer Format:** Generate the correct answer for each question. Ensure that each answer is directly supported by the image.

3. **JSON Format:** Structure each question-answer pair in JSON format as shown in the example. Ensure that all JSON keys ("Q", "A") and values are correctly formatted and consistently applied.

Make sure the questions are varied in type and cover different aspects of the image to ensure a comprehensive and diverse set of questions and answers.
"""

TQA_SYS_PROMPT = """You are an expert in <DATASET_DESC>. Your task is to generate high-quality question-answer pairs relevant to this skill similar to the following examples. 

1. **Hypothetical Image Description:** Start by providing a brief and clear description of the hypothetical image that could be related to the skill.
   
2. **Question Generation:** Based on the image description, generate relevant multiple-choice questions. Each question should be clear, specific, and directly related to the described image. Present the options with corresponding letters (e.g., A: <option_a>, B: <option_b>), placing each option on a new line.

3. **Answer Generation:** Provide the correct answer formatted as the letter of the option followed by the option text (e.g., "A. Option Text"). Each answer should be accurate, directly supported by the hypothetical image description, and consistent with the reasoning.

4. **Explanation:** For each question-answer pair, provide a concise explanation of your step-by-step reasoning, highlighting how the hypothetical image description informed your question and answer choices. Ensure that the answer aligns with the reasoning provided.

**Output Format:**
Return only the output as a JSON list of objects. Each object should include:
- `"I"`: The description of the hypothetical image.
- `"Q"`: The generated question (include options if it's a multiple-choice question).
- `"R"`: A step-by-step explanation of the reasoning behind the question-answer pair.
- `"A"`: The generated answer.

Example Output:
[
  {"I": "Hypothetical image description 1", "Q": "Generated question 1", "R": "Step-by-step reasoning for question 1", "A": "Generated answer 1"},
  {"I": "Hypothetical image description 2", "Q": "Generated question 2", "R": "Step-by-step reasoning for question 2", "A": "Generated answer 2"}
]
"""

TQA_GEN_PROMPT = """Generate exactly <NUM> multiple-choice question-answer pairs based on a hypothetical image that you will describe. Each question-answer pair should adhere to the following guidelines:

1. **Hypothetical Image Description:** Begin by providing a brief and clear description of the hypothetical image that will serve as the basis for the question-answer pair.

2. **Question Format:** Follow the style and format of the provided example question. Ensure that each question is clear, specific, and directly relevant to the hypothetical image. Present the options with corresponding letters (e.g., A: <option_a>, B: <option_b>), placing each option on a new line.

3. **Answer Format:** Provide the correct answer formatted as the letter of the option followed by the option text (e.g., "A. Option Text"). Each answer should be accurate, directly related to the question, and consistent with the reasoning provided.

4. **Explanation:** Include a brief explanation of how you determined the correct answer, ensuring that the answer is consistent with the step-by-step reasoning.

Ensure that the questions are diverse in type and cover different aspects of the hypothetical image to provide a comprehensive and varied set of questions and answers."""


VQA_TASK_DESC_SYS_PROMPT = """You are an expert in <DATASET_DESC>. Your task is to generate high-quality question-answer pairs relevant to this skill.

Step-by-Step Process:
1. Understand the New Image: Infer relevant details, objects, and themes in the new image, considering how they relate to the skill.
2. Generate Questions: Create questions that reflect the context and content of the new image, ensuring they align with the skill.
3. If the question is a multiple choice question, make sure to include the options in the question.
4. Breakdown Reasoning: For each question, provide a detailed step-by-step breakdown of the reasoning process required to determine the correct answer. Highlight the key elements of the image and the logical steps leading to the answer.
5. Formulate Answers: After providing the reasoning, generate accurate and concise answers to the questions. Ensure that each answer is consistent with the reasoning provided.

Output Format:
Return the results as a JSON list of objects. Each object should include:
- "Q": The generated question (include options if it's multiple-choice).
- "R": The step-by-step reasoning that leads to the answer.
- "A": The generated answer.

Example Output:
[
  {"Q": "Generated question 1", "R": "Step-by-step reasoning for question 1", "A": "Generated answer 1"},
  {"Q": "Generated question 2", "R": "Step-by-step reasoning for question 2", "A": "Generated answer 2"}
]
"""

VQA_TASK_DESC_GEN_PROMPT = """Generate exactly <NUM> question-answer pairs for the following image. Each question-answer pair should adhere to the following guidelines:

1. **Question Format:** Ensure each question is clear, specific, and directly relevant to the image. Present the options with corresponding letters (e.g., A: <option_a>, B: <option_b>), placing each option on a new line.
   
2. **Reasoning Breakdown:** For each question, provide a detailed step-by-step breakdown of the reasoning process required to determine the correct answer. Explain the key elements in the image and the logical steps that lead to the answer.
   
3. **Answer Format:** After providing the reasoning, generate the correct answer. Ensure that each answer is consistent with the reasoning provided and directly supported by the image.

4. **JSON Format:** Structure each question-answer pair in JSON format as shown in the example. Ensure that all JSON keys ("Q", "R", "A") and values are correctly formatted and consistently applied.

Make sure the questions are varied in type and cover different aspects of the image to ensure a comprehensive and diverse set of questions and answers."""


DESCRIPTION_SYS_PROMPT = """
You are an expert in <DATASET_DESC>. Your task is to generate high-quality data relevant to this skill. Given an example image and its question-answer pair, create descriptions using the aforementioned skill for a new image.

Return only the output as a JSON list of objects, where each object has a key "A" for the description.

Output Format:
[
  {"A": "Generated description 1"},
  {"A": "Generated description 2"},
  ...
]
"""

DESCRIPTION_GEN_PROMPT = "Generate exactly <NUM> highly detailed descriptions for the following image in valid aforementioned json format. Each description should be unique and contain multiple sentences."

GENERIC_DESCRIPTION_SYS_PROMPT = """
Your task is to generate high-quality descriptions.

Return only the output as a JSON list of objects, where each object has a key "A" for the description.

Output Format:
[
  {"A": "Generated description 1"},
  {"A": "Generated description 2"},
  ...
]
"""

GENERIC_DESCRIPTION_GEN_PROMPT = "Generate exactly <NUM> highly detailed descriptions for the following image in valid aforementioned JSON format. Each description should be unique and contain multiple sentences."

DESCRIPTION_TASK_DESC_SYS_PROMPT = """
You are an expert in <DATASET_DESC>. Your task is to generate high-quality data relevant to this skill. Create detailed descriptions using the aforementioned skill for a new image.

Return only the output as a JSON list of objects, where each object has a key "A" for the description.

Output Format:
[
  {"A": "Generated description 1"},
  {"A": "Generated description 2"},
  ...
]
"""

DESCRIPTION_TASK_DESC_GEN_PROMPT = "Generate exactly <NUM> highly detailed descriptions for the following image in valid aforementioned JSON format. Each description should be unique and contain multiple sentences."

LENGTH_CONSTRAINT = "Each description should be approximately <LEN> words long."

QA_FORMAT_FIX_SYS_PROMPT = """
You are a formatting agent that fixes improperly formatted JSON strings and ensures they conform to a specific format. Your task is to take any input string that is meant to represent JSON and correct it so that it becomes valid JSON according to the following structure:

Output Format:
[
  {"Q": "Generated question 1", "R": "Step-by-step reasoning for question 1", "A": "Generated answer 1"},
  {"Q": "Generated question 2", "R": "Step-by-step reasoning for question 2", "A": "Generated answer 2"},
  ...
]

Instructions:

1. Identify and correct common JSON errors, such as:
   - Missing or incorrect commas, colons, or brackets.
   - Misplaced or missing quotation marks.
   - Unescaped special characters.

2. Ensure that all keys are "Q" for questions, "R" for reasoning, and "A" for answers.

3. Make sure that the JSON array structure is followed, with each object in the array containing exactly three key-value pairs: one for the question, one for the reasoning, and one for the answer. Ensure all values are strings.

4. Ensure that each answer is consistent with the reasoning provided. If necessary, reformat or rewrite the input to match the required structure while preserving the original content's meaning as closely as possible.

Return ONLY the JSON.
"""

QA_NR_FORMAT_FIX_SYS_PROMPT = """You are a formatting agent that fixes improperly formatted JSON strings and ensures they conform to a specific format. Your task is to take any input string that is meant to represent JSON and correct it so that it becomes valid JSON according to the following structure:

Output Format:
[
  {"Q": "Generated question 1", "A": "Generated answer 1"},
  {"Q": "Generated question 2", "A": "Generated answer 2"},
  ...
]

Instructions:

1. Identify and correct common JSON errors, such as:
   - Missing or incorrect commas, colons, or brackets.
   - Misplaced or missing quotation marks.
   - Unescaped special characters.

2. Ensure that all keys are "Q" for questions and "A" for answers.

3. Make sure that the JSON array structure is followed, with each object in the array containing exactly two key-value pairs: one for the question and one for the answer. Ensure all values are strings.

4. Ensure that each answer is appropriate for the given question. If necessary, reformat or rewrite the input to match the required structure while preserving the original content's meaning as closely as possible.

Return ONLY the JSON.
"""


DESCRIPT_FORMAT_FIX_SYS_PROMPT = """
You are a formatting agent fixes improperly formatted JSON strings and ensures they conform to a specific format. Your task is to take any input string that is meant to represent JSON and correct it so that it becomes valid JSON according to the following structure:

[
  {"A": "Generated description 1"},
  {"A": "Generated description 2"},
  ...
]

Instructions:

1. Identify and correct common JSON errors, such as:
   - Missing or incorrect commas, colons, or brackets.
   - Misplaced or missing quotation marks.
   - Unescaped special characters.

2. Ensure that all keys are "A" for descriptions.

3. Make sure that the JSON array structure is followed, with each object in the array containing exactly one key-value pair for the description.

4. If necessary, reformat or rewrite the input to match the required structure while preserving the original content's meaning as closely as possible.

Return ONLY the json
"""

FIX_FORMAT_QA_GEN_PROMPT = """
Output Format:
[
  {"Q": "Generated question 1", "R": "Step-by-step reasoning for question 1", "A": "Generated answer 1"},
  {"Q": "Generated question 2", "R": "Step-by-step reasoning for question 2", "A": "Generated answer 2"},
  ...
]

Please fix the following string so that it becomes valid JSON according to the aforementioned format.
"""

FIX_FORMAT_DESCRIPT_GEN_PROMPT = """
[
  {"A": "Generated description 1"},
  {"A": "Generated answer 2"},
  ...
]

Please fix the following string so that it becomes valid JSON according to the aforementioned format.
"""

REASONING_PROMPT = "Return a detailed step-by-step reasoning first and the return the answer."

ONLY_ANS_PROMPT = "Provide a concise answer."

SAVE_PATH = "generated_data"

class BatchSampler:
    def __init__(self, samples, batch_size):
        self.samples = samples
        self.batch_size = batch_size
        self.indices = list(range(len(self.samples)))  # Store original indices
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
        return batch_indices, batch_samples

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

class MultimodalDataGenerator:
    def __init__(self, args, logger):
        """
        Initialize the MultimodalDataGenerator class.

        :param model_id: The ID of the pre-trained language model to use.
        """
        self.prompt_file = args.prompt_file
        self.model_name = args.model_name
        self.start_idx = args.start_idx 
        self.num_prompts = args.num_prompts
        self.logger = logger
        self.min_gen_per_candidate = self.prompt_file["min_gen_per_candidate"]
        self.input_folder = args.input_folder
        self.mode = GenerationMode(self.prompt_file["mode"])
        self.logger.info(f"Generation Mode: {self.mode}")
        
        # Create agent to fix formats
        if self.mode == GenerationMode.VQA or self.mode == GenerationMode.TQA:
            self.format_fix_agent = GPTEndPoint(self.model_name, self.logger, sys_prompt=QA_FORMAT_FIX_SYS_PROMPT)
        elif self.mode == GenerationMode.VQA_NR:
            self.format_fix_agent = GPTEndPoint(self.model_name, self.logger, sys_prompt=QA_NR_FORMAT_FIX_SYS_PROMPT)
        elif self.mode == GenerationMode.DESCRIPT or self.mode == GenerationMode.GENERIC:
            self.format_fix_agent = GPTEndPoint(self.model_name, self.logger, sys_prompt=DESCRIPT_FORMAT_FIX_SYS_PROMPT) 
        else:
            raise ValueError()
        
        # Create the corresponding GPT4 Endpoint 
        if self.mode == GenerationMode.DESCRIPT: 
            self.generation_prompt = DESCRIPTION_GEN_PROMPT
            self.sys_prompt = DESCRIPTION_SYS_PROMPT
        elif self.mode == GenerationMode.VQA:
            self.generation_prompt = VQA_GEN_PROMPT
            self.sys_prompt = VQA_SYS_PROMPT
        elif self.mode == GenerationMode.DESCRIPT_TASK_DESC: 
            self.generation_prompt = DESCRIPTION_TASK_DESC_GEN_PROMPT
            self.sys_prompt = DESCRIPTION_TASK_DESC_SYS_PROMPT
        elif self.mode == GenerationMode.VQA_TASK_DESC:
            self.generation_prompt = VQA_TASK_DESC_GEN_PROMPT
            self.sys_prompt = VQA_TASK_DESC_SYS_PROMPT
        elif self.mode == GenerationMode.VQA_NR:
            self.generation_prompt = VQA_NR_GEN_PROMPT
            self.sys_prompt = VQA_NR_SYS_PROMPT          
        elif self.mode == GenerationMode.TQA:
            self.generation_prompt = TQA_GEN_PROMPT
            self.sys_prompt = TQA_SYS_PROMPT
        elif self.mode == GenerationMode.GENERIC:
            self.generation_prompt = GENERIC_DESCRIPTION_GEN_PROMPT
            self.sys_prompt = GENERIC_DESCRIPTION_SYS_PROMPT
        else:
            raise ValueError(f"Invalid value for mode")
        self.generation_prompt = self.generation_prompt.replace("<NUM>", str(self.min_gen_per_candidate))        
        self.model = GPTEndPoint(self.model_name, sys_prompt=self.sys_prompt.replace("<DATASET_DESC>", self.prompt_file["dataset_description"]), logger=logger)
        
        # DT string to associate examples with time of run
        self.dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.gen_data_path = os.path.join(args.output_folder, SAVE_PATH, f"{args.file_prefix}_{self.dt_str}.json")
        os.makedirs(os.path.join(args.output_folder, SAVE_PATH), exist_ok=True)

    def generate_questions(self):
        """
        This method processes candidate images, constructs prompts, and generates 
        questions using a model. It handles errors gracefully and logs the progress 
        and any issues encountered.
        :return: None
        """
        self.logger.info("Generating questions based on skill description and sample questions.")
        self.gen_image_paths = []
        self.gen_text = []
        self.gen_icl_indices = []
        self.gen_keywords = []
        
        # Compute per candidate generation count 
        self.total_gen = self.min_gen_per_candidate * self.num_prompts
        self.logger.info(f"Will be generating {self.total_gen} examples.")
        
        try:
            ##################################################################
            #                     Try Generate & Parse                       #
            ##################################################################
            for prompt_object in tqdm(self.prompt_file["prompts"][self.start_idx:self.start_idx + self.num_prompts]):
                # Parse prompt object
                prompt = prompt_object["prompt"]
                keyword = prompt_object["keyword"]
                icl_indices = prompt_object["icl_indices"]

                # Extract candidate image path 
                candidate_image_path = os.path.join(self.input_folder, prompt[-1])
                self.logger.debug(f"Processing candidate image: {candidate_image_path}")
                
                # Load all images in prompts
                for i in range(len(prompt)):
                    if self.model_name == "mock":
                        prompt[i] = os.path.join(self.input_folder, prompt[i])
                        continue
                    if is_image_file(prompt[i]):
                        prompt[i] = Image.open(os.path.join(self.input_folder, prompt[i]))
                        
                # Get response from model
                response = self.model.generate(prompt + [self.generation_prompt])
                    
                # Remove all special token tags (these are specific to LLava style models)
                response = re.sub("<.*>", "", response)
                
                if self.model_name == "mock":
                    self.logger.debug("Mock model: tests complete.")
                    exit(0)
                    
                # Try parse, else retry
                try:
                        json_response = self.parse_response(response)
                except:
                    self.logger.error("Failing due to invalid format. Skipping ahead.")
                    continue
                        
                self.logger.debug(f"Generated questions for prompt and image: {response}")
                
                if len(json_response) != self.min_gen_per_candidate:
                    self.logger.error(f"Got {len(json_response)} examples, expected {self.min_gen_per_candidate}.")
                    
                self.gen_text.extend(json_response)
                for _ in range(len(json_response)):
                    self.gen_image_paths.append(candidate_image_path)
                    self.gen_icl_indices.append(icl_indices)
                    self.gen_keywords.append(keyword)
                self.end_idx = self.start_idx + i
                
        # Catch all to gracefully fail and save what is done so far
        except Exception as e:
            self.logger.error(f"Gracefully handling exception: {e}")
            min_len = min(len(self.gen_text), len(self.gen_image_paths), len(self.gen_icl_indices))
            self.gen_text = self.gen_text[:min_len]
            self.gen_image_paths = self.gen_image_paths[:min_len]
            self.gen_icl_indices = self.gen_icl_indices[:min_len]
            self.gen_keywords = self.gen_keywords[:min_len]
            self.end_idx = self.start_idx + i
        self.logger.info(f"Completed generation")
    
    def extract_json_part(self, input_string):
        # Use regex to find the JSON part within the input string
        json_match = re.search(r'\[.*\]', input_string, re.DOTALL)
        if json_match:
            json_part = json_match.group(0)
            json_part = re.sub(r'\\', r'\\\\', json_part)
            try:
                # Parse the JSON to ensure it's valid
                json_data = json.loads(json_part)
                return json.dumps(json_data, indent=3)  # Return the formatted JSON string
            except json.JSONDecodeError:
                return "Invalid JSON found."
        else:
            return "No JSON part found."
    
    def parse_response(self, response):
        num_tries = 0
        while num_tries < 3:
            try:
                json_response = json.loads(self.extract_json_part(response))
                self.logger.debug(json_response)
                for datum in json_response:
                    assert "A" in datum 
                    if self.mode == GenerationMode.VQA:
                        assert "Q" in datum
                        assert "R" in datum
                    elif self.mode == GenerationMode.VQA_NR:
                        assert "Q" in datum
                    elif self.mode == GenerationMode.TQA:
                        assert "Q" in datum
                        assert "I" in datum
                        assert "R" in datum
                    for key in datum:
                        datum[key] = str(datum[key])
                return json_response
            except:
                num_tries += 1
                self.logger.warning("Incorrect formatting: using format fixing agent.")
                self.logger.warning(response)
                response = self.format_fix_agent.generate([FIX_FORMAT_QA_GEN_PROMPT, response])
                self.logger.warning(response)
        raise ValueError("Unable to parse")
            
    def save_gen_text(self):
        # Formatting gen qa into desired json format
        formatted_gen_text = [] 
        
        for i, (gen_obj, image_path, indices, keyword) in tqdm(enumerate(zip(self.gen_text, self.gen_image_paths, self.gen_icl_indices, self.gen_keywords))):
            self.logger.debug(f"Question and Answer: {gen_obj}")
            choices = ""
            for key in gen_obj:
                if str(key).lower() == "choices":
                    choices = str(gen_obj[key])
                    choices = " " + choices + " "
                    
            if self.mode == GenerationMode.VQA:
                formatted_gen_text.append(
                    {
                        "id": i,
                        "image_1": image_path, 
                        "conversations":
                            [
                                {
                                    "from": "human",
                                    "value": "<image> " + gen_obj["Q"] + choices + "\n" + REASONING_PROMPT
                                }, 
                                {
                                    "from": "gpt",
                                    "value": gen_obj["R"] + "\n" + gen_obj["A"]
                                }
                            ],
                        "icl_indices": indices,
                        "keyword": keyword
                    }
                )
                
                formatted_gen_text.append(
                    {
                        "id": i, 
                        "image_1": image_path, 
                        "conversations":
                            [
                                {
                                    "from": "human",
                                    "value": "<image> " + gen_obj["Q"] + choices + "\n" + ONLY_ANS_PROMPT
                                }, 
                                {
                                    "from": "gpt",
                                    "value": gen_obj["A"]
                                }
                            ],
                        "icl_indices": indices,
                        "keyword": keyword
                    }
                )
            elif self.mode == GenerationMode.VQA_NR:
                formatted_gen_text.append(
                    {
                        "id": i, 
                        "image_1": image_path, 
                        "conversations":
                            [
                                {
                                    "from": "human",
                                    "value": "<image> " + gen_obj["Q"] + choices
                                }, 
                                {
                                    "from": "gpt",
                                    "value": gen_obj["A"]
                                }
                            ],
                        "icl_indices": indices,
                        "keyword": keyword
                    }
                )
            elif self.mode == GenerationMode.TQA:
                formatted_gen_text.append(
                    {
                        "id": i, 
                        "conversations":
                            [
                                {
                                    "from": "human",
                                    "value":  gen_obj["I"] + "\n" + gen_obj["Q"] + choices + "\n" + REASONING_PROMPT
                                }, 
                                {
                                    "from": "gpt",
                                    "value": gen_obj["R"] + "\n" + gen_obj["A"]
                                }
                            ],
                        "icl_indices": indices,
                        "keyword": keyword
                    }
                )
                
                formatted_gen_text.append(
                    {
                        "id": i, 
                        "image_1": image_path, 
                        "conversations":
                            [
                                {
                                    "from": "human",
                                    "value":  gen_obj["I"] + "\n"  + gen_obj["Q"] + choices + "\n" + ONLY_ANS_PROMPT
                                }, 
                                {
                                    "from": "gpt",
                                    "value": gen_obj["A"]
                                }
                            ],
                        "icl_indices": indices,
                        "keyword": keyword
                    }
                )
            else:
                EXP_SYS_PROMPT = f"Describe the image as an expert in " + self.prompt_file["dataset_description"]
                if GenerationMode.GENERIC:
                    EXP_SYS_PROMPT = "Describe the image."
                
                formatted_gen_text.append(
                    {
                        "id": i, 
                        "image_1": image_path, 
                        "conversations":
                            [
                                {
                                    "from": "human",
                                    "value": "<image> " + EXP_SYS_PROMPT
                                }, 
                                {
                                    "from": "gpt",
                                    "value": gen_obj["A"]
                                }
                            ],
                        "icl_indices": indices,
                        "keyword": keyword
                    }
                )
            
        # Saving formatted qa
        self.logger.info(f"Saving {len(formatted_gen_text)} generated examples.")
        with open(self.gen_data_path, 'w') as file:
            json.dump({
                "image_folder": "", # for backward compatibility
                "len_samples": len(formatted_gen_text),
                "start_idx": self.start_idx, 
                "end_idx": self.end_idx,
                "num_prompts": self.num_prompts,
                "samples": formatted_gen_text,
            }, file, indent=3)
            
        self.logger.info(f"Saved generated questions to {self.gen_data_path}")

def ctrl_c_handler(signum, frame):
    raise Exception("Gracefully handling exception: Ctrl-C Pressed. Terminating the process.")

# Registering the signal handler
signal.signal(signal.SIGINT, ctrl_c_handler)

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

def is_image_file(filename):
    image_extensions = ('.png', '.jpg', '.jpeg')
    return filename.lower().endswith(image_extensions)

def main(args):
    # Set logging level
    logger.remove()
    if args.debug or args.model_name == "mock":
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    # Load data from JSON file
    args.prompt_file = os.path.join(args.input_folder, args.prompt_file)
    args.prompt_file = load_json_file(args.prompt_file)
    if args.num_prompts == -1:
        args.num_prompts = len(args.prompt_file["prompts"])
    
    # Initialize generator
    generator = MultimodalDataGenerator(args, logger)
    
    logger.info("Generating questions and answers.")
    generator.generate_questions()
    
    logger.info("Saving generated questions and answers.")
    generator.save_gen_text()
    
    logger.info(f"Finished. See generated data in {generator.gen_data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Data Generator")
    parser.add_argument("--model_name", type=str, default="gpt-4o-450K", help="Name of model to use with GPT4 endpoint")
    parser.add_argument('--input_folder', type=str, default="", help='Input Folder')
    parser.add_argument('--output_folder', type=str, default="", help='Output Folder')
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the JSON file containing the prompts")
    parser.add_argument("--file_prefix", type=str, required=True, help="Prefix for file with generated questions")
    parser.add_argument("--start_idx", type=int, default=0, help="Index to start at in the list of the prompts.")
    parser.add_argument("--num_prompts", type=int, default=-1, help="Number of prompts to process.")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging")

    args = parser.parse_args()
    main(args)