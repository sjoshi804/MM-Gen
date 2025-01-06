# MM-GEN: Enhancing Task Performance Through Targeted Multimodal Data Curation

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
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py \
        --input_folder {input_folder_path} \
        --output_folder {output_folder_path} \
        --prompt_file generated_prompts/ai2d_test_prompts_20241002_203030.json \
        --file_prefix  ai2d_test  \
        --num_parallel 2 \
```

## TODO

- [x] Mock out gpt4 to check what is being sent to gpt 4

# Commands 

## MM-Instruct 

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/spatial_map_gen_task_desc_2024_09_19_02:09:58.json \
        --input_folder . \
        --output_folder . \
        --file_prefix spatial_map_mminstruct_15k \
        --total_gen 15000 \
        --mode vqa_nr
```

Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/spatial_map_mminstruct_15k_prompts_20241010_115659.json --file_prefix spatial_map_mminstruct_15k --num_parallel 12
```


## Generic Caption Baseline 

### SpatialMap 

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/spatial_map_gen_task_desc_2024_09_19_02:09:58.json \
        --input_folder . \
        --output_folder . \
        --file_prefix spatial_map_generic_15k \
        --total_gen 15000 \
        --mode generic
```

Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/spatial_map_generic_15k_prompts_20241010_113453.json --file_prefix spatial_map_generic_15k --num_parallel 12
```

## [RUN DATA GENERATION COMMANDS HERE]: Task Description Baseline 

### SpatialMap 

**Q/A**

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/spatial_map_gen_task_desc_2024_09_19_02:09:58.json \
        --input_folder . \
        --output_folder . \
        --file_prefix spatial_map_vqa_task_desc_50k \
        --total_gen 50000 \
        --mode vqa_task_desc
```

Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/spatial_map_vqa_task_desc_50k_prompts_20241010_120435.json --file_prefix spatial_map_vqa_task_desc_50k --num_parallel 12
```

**DESCRIPT** 

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/spatial_map_gen_task_desc_2024_09_19_02:09:58.json \
        --input_folder . \
        --output_folder . \
        --file_prefix spatial_map_descript_task_desc_50k  \
        --total_gen 50000 \
        --mode descript_task_desc
```


Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/spatial_map_descript_task_desc_50k_prompts_20241010_120440.json --file_prefix spatial_map_descript_task_desc_50k --num_parallel 12
```


### AI2D 

**Q/A**

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/ai2d_gen_single_subgroup_2024_09_18_13:58:08.json \
        --input_folder . \
        --output_folder . \
        --file_prefix ai2d_vqa_task_desc_100k \
        --total_gen 100000 \
        --mode vqa_task_desc
```

Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/ai2d_vqa_task_desc_100k_prompts_20241010_120932.json --file_prefix ai2d_vqa_task_desc_100k --num_parallel 12
```

**DESCRIPT** 

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/ai2d_gen_single_subgroup_2024_09_18_13:58:08.json \
        --input_folder . \
        --output_folder . \
        --file_prefix ai2d_descript_task_desc_100k  \
        --total_gen 100000 \
        --mode descript_task_desc
```


Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/ai2d_descript_task_desc_100k_prompts_20241010_120938.json --file_prefix ai2d_descript_task_desc_100k --num_parallel 12
```


### ChartQA 

**Q/A**

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/chartqa_gen_single_subgroup_2024_09_18_13:58:08.json \
        --input_folder . \
        --output_folder . \
        --file_prefix chartqa_vqa_task_desc_150k \
        --total_gen 150000 \
        --mode vqa_task_desc
```

Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/chartqa_vqa_task_desc_150k_prompts_20241010_121210.json --file_prefix chartqa_vqa_task_desc_150k --num_parallel 12
```

**DESCRIPT** 

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/chartqa_gen_single_subgroup_2024_09_18_13:58:08.json \
        --input_folder . \
        --output_folder . \
        --file_prefix chartqa_descript_task_desc_150k  \
        --total_gen 150000 \
        --mode descript_task_desc
```


Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/chartqa_descript_task_desc_150k_prompts_20241010_121215.json --file_prefix chartqa_descript_task_desc_150k --num_parallel 12
```


## [RUN DATA GENERATION COMMANDS HERE] Ablations (ChartQA)

### Without Partition

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/chartqa_gen_single_subgroup_2024_09_18_13:58:08.json \
        --input_folder . \
        --output_folder . \
        --file_prefix chartqa_mminstruct_no_partition_150k \
        --total_gen 150000 \
        --mode vqa_nr
```

Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/chartqa_mminstruct_no_partition_150k_prompts_20241010_122100.json --file_prefix chartqa_mminstruct_no_partition_150k --num_parallel 12
```

### Imbalanced Ref Set

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/chartqa_gen_task_desc_imbalanced.json \
        --input_folder . \
        --output_folder . \
        --file_prefix chartqa_mminstruct_imbalanced_refset_150k \
        --total_gen 150000 \
        --mode vqa_nr
```

Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/chartqa_mminstruct_imbalanced_refset_150k_prompts_20241010_131412.json --file_prefix chartqa_mminstruct_imbalanced_refset_150k --num_parallel 12
```


### More In-Context Samples (3 and 5)

**3 ICL** 

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/chartqa_gen_task_desc_2024_09_18_07:50:34.json \
        --input_folder . \
        --output_folder . \
        --file_prefix chartqa_mminstruct_3icl_150k \
        --num_icl 3 \
        --total_gen 150000 \
        --mode vqa_nr
```

Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/chartqa_mminstruct_3icl_150k_prompts_20241010_123855.json --file_prefix chartqa_mminstruct_3icl_150k --num_parallel 12
```

**5 ICL** 

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/chartqa_gen_task_desc_2024_09_18_07:50:34.json \
        --input_folder . \
        --output_folder . \
        --file_prefix chartqa_mminstruct_5icl_150k \
        --num_icl 5 \
        --total_gen 150000 \
        --mode vqa_nr
```

Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/chartqa_mminstruct_5icl_150k_prompts_20241010_123910.json --file_prefix chartqa_mminstruct_5icl_150k --num_parallel 12
```


### ~30K (28299) examples, 1-3-5 captions per images

**1 caption per image** 

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/chartqa_gen_task_desc_2024_09_18_07:50:34.json \
        --input_folder . \
        --output_folder . \
        --file_prefix chartqa_mminstruct_1cap_30k \
        --total_gen 28299 \
        --mode vqa_nr
```

Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/chartqa_mminstruct_1cap_30k_prompts_20241010_123920.json --file_prefix chartqa_mminstruct_1cap_30k --num_parallel 12
```

**3 caption per image** 

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/spatial_map_gen_task_desc_3cap_per_img.json \
        --input_folder . \
        --output_folder . \
        --file_prefix chartqa_mminstruct_3cap_30k \
        --total_gen 28299 \
        --mode vqa_nr
```

Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/chartqa_mminstruct_3cap_30k_prompts_20241010_123954.json --file_prefix chartqa_mminstruct_3cap_30k --num_parallel 12
```

**5 caption per image** 

Prompt Generation Command:
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/prompt_generator.py \
        --task_desc task_files/spatial_map_gen_task_desc_5cap_per_img.json \
        --input_folder . \
        --output_folder . \
        --file_prefix chartqa_mminstruct_5cap_30k \
        --total_gen 28299 \
        --mode vqa_nr
```

Data Generation Command: 
```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/data_generation/batch_data_generator.py --input_folder <INPUT_FOLDER> --output_folder . --prompt_file generated_prompts/chartqa_mminstruct_5cap_30k_prompts_20241010_124001.json --file_prefix chartqa_mminstruct_5cap_30k --num_parallel 12
```