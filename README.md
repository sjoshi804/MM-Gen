# MM-Instruct: A Blueprint for Data Generation for Efficiently Fine-tuning VLMs

## Step 1: Create Task File (from Data File)

Here, given a data file, in the llava data format, we will create a task file.

```bash
python src/data_generation/task_file_generator.py 
    --data_file {path_to_data_file} // Relative path inside input folder
    --input_folder {input_folder_path} // Path to parent input folder, this is also the parent folder for the image folder from data file
    --output_folder {output_folder_path} // Path to parent output folder
    --keywords {keyword_1} {keyword_2} ... {keyword_N} // List of keywords to partition data file for task file
    --output_prefix {output_file_prefix} // Prefix for generated task file 
    --model_name {clip_model_path} // Path to clip model (OPTIONAL)
    --batch_size {batch_size} // Batch size for embedding calculation for partitioning data (OPTIONAL)
    --gpu {gpu_device_number} // GPU Device numer (OPTIONAL)
    [--debug]
```

## Step 2: Generate Prompts (from Task File)

Here, given a task file, generate the prompts to supply to the stronger model. 

```bash
    python src/data_generation/prompt_generator.py //
        --task_desc {task_desc} // Path to task file 
        --input_folder {input_folder_path} // Path to parent input folder, this is also the parent folder for the image folder from data file
        --output_folder {output_folder_path} // Path to parent output folder
        --file_prefix {file_prefix} // Prefix for generated_prompts file
        --total_gen {total_gen} // Total number to generate
        --min_gen_per_candidate {min_gen_per_candidate} // Minimum number to generate per candidate (OPTIONAL)
        --num_icl_samples {num_icl_samples} // Number of ICL samples to provide (OPTIONAL)
        --mode {mode} // Generation Mode (OPTIONAL)
        {debug_flag}
```

## Step 3: Generate Data (from Generated Prompts)

Here, given a generated prompts file, generate the accompanying text (q/a) data for the candidate images. 

```bash
    python src/data_generation/data_generator.py
        --input_folder {input_folder_path} // Path to parent input folder, this is also the parent folder for the image folder from data file
        --output_folder {output_folder_path} // Path to parent output folder
        --prompt_file {prompt_file} // Path to prompt file, relative path from input folder
        --file_prefix {file_prefix} // Prefix for generated_data file
        --start_idx {start_idx} // Index of prompts we want to start from 
        --num_prompts {num_prompts} // Numer of prompts, starting at start_idx, for which we want to generate
        --model_name {model_name} // Model name for OpenAI API to use (OPTIONAL)
        {debug_flag}
```

To automatically split generated prompts to paralleize generation, use the following command. 

```bash
    python src/data_generation/batch_data_generator.py
        --input_folder {input_folder_path} // Path to parent input folder, this is also the parent folder for the image folder from data file
        --output_folder {output_folder_path} // Path to parent output folder
        --prompt_file  {prompt_file}  // Path to prompt file, relative path from input folder
        --file_prefix  {file_prefix}  // Prefix for generated_data file
        --num_parallel {num_parallel} // Number of parallel generations to run
        --model_name  {model_name}  // Model name for OpenAI API to use (OPTIONAL)
        {debug_flag}
```

Example command

```bash
python src/data_generation/batch_data_generator.py \
    --input_folder {input_folder_path} \
    --output_folder {output_folder_path} \
    --prompt_file generated_prompts/ai2d_test_prompts_20241002_203030.json \
    --file_prefix  ai2d_test  \
    --num_parallel 2 \
```