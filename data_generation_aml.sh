#!/bin/bash

# SpatialMap VQA Task Desc Data Generation Command
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/data_generation/batch_data_generator.py --input_folder /home/vivineet/projects/siddharth/data --output_folder /home/vivineet/projects/siddharth/data --prompt_file generated_prompts/spatial_map_vqa_task_desc_50k_prompts_20241010_120435.json --file_prefix spatial_map_vqa_task_desc_50k --num_parallel 12 --dry_run

# SpatialMap Descript Task Desc Data Generation Command
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/data_generation/batch_data_generator.py --input_folder /home/vivineet/projects/siddharth/data --output_folder /home/vivineet/projects/siddharth/data --prompt_file generated_prompts/spatial_map_descript_task_desc_50k_prompts_20241010_120440.json --file_prefix spatial_map_descript_task_desc_50k --num_parallel 12 --dry_run

# AI2D VQA Task Desc Data Generation Command
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/data_generation/batch_data_generator.py --input_folder /home/vivineet/projects/siddharth/data --output_folder /home/vivineet/projects/siddharth/data --prompt_file generated_prompts/ai2d_vqa_task_desc_100k_prompts_20241010_120932.json --file_prefix ai2d_vqa_task_desc_100k --num_parallel 12 --dry_run

# AI2D Descript Task Desc Data Generation Command
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/data_generation/batch_data_generator.py --input_folder /home/vivineet/projects/siddharth/data --output_folder /home/vivineet/projects/siddharth/data --prompt_file generated_prompts/ai2d_descript_task_desc_100k_prompts_20241010_120938.json --file_prefix ai2d_descript_task_desc_100k --num_parallel 12 --dry_run

# ChartQA VQA Task Desc Data Generation Command
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/data_generation/batch_data_generator.py --input_folder /home/vivineet/projects/siddharth/data --output_folder /home/vivineet/projects/siddharth/data --prompt_file generated_prompts/chartqa_vqa_task_desc_150k_prompts_20241010_121210.json --file_prefix chartqa_vqa_task_desc_150k --num_parallel 12 --dry_run

# ChartQA Descript Task Desc Data Generation Command
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/data_generation/batch_data_generator.py --input_folder /home/vivineet/projects/siddharth/data --output_folder /home/vivineet/projects/siddharth/data --prompt_file generated_prompts/chartqa_descript_task_desc_150k_prompts_20241010_121215.json --file_prefix chartqa_descript_task_desc_150k --num_parallel 12 --dry_run

# ChartQA Without Partition Data Generation Command
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/data_generation/batch_data_generator.py --input_folder /home/vivineet/projects/siddharth/data --output_folder /home/vivineet/projects/siddharth/data --prompt_file generated_prompts/chartqa_mminstruct_no_partition_150k_prompts_20241010_122100.json --file_prefix chartqa_mminstruct_no_partition_150k --num_parallel 12 --dry_run

# ChartQA Imbalanced Ref Set Data Generation Command
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/data_generation/batch_data_generator.py --input_folder /home/vivineet/projects/siddharth/data --output_folder /home/vivineet/projects/siddharth/data --prompt_file generated_prompts/chartqa_mminstruct_imbalanced_refset_150k_prompts_20241010_131412.json --file_prefix chartqa_mminstruct_imbalanced_refset_150k --num_parallel 12 --dry_run

# ChartQA More In-Context Samples (3 ICL) Data Generation Command
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/data_generation/batch_data_generator.py --input_folder /home/vivineet/projects/siddharth/data --output_folder /home/vivineet/projects/siddharth/data --prompt_file generated_prompts/chartqa_mminstruct_3icl_150k_prompts_20241010_123855.json --file_prefix chartqa_mminstruct_3icl_150k --num_parallel 12 --dry_run

# ChartQA More In-Context Samples (5 ICL) Data Generation Command
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/data_generation/batch_data_generator.py --input_folder /home/vivineet/projects/siddharth/data --output_folder /home/vivineet/projects/siddharth/data --prompt_file generated_prompts/chartqa_mminstruct_5icl_150k_prompts_20241010_123910.json --file_prefix chartqa_mminstruct_5icl_150k --num_parallel 12 --dry_run

# ChartQA 1 Caption per Image Data Generation Command
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/data_generation/batch_data_generator.py --input_folder /home/vivineet/projects/siddharth/data --output_folder /home/vivineet/projects/siddharth/data --prompt_file generated_prompts/chartqa_mminstruct_1cap_30k_prompts_20241010_123920.json --file_prefix chartqa_mminstruct_1cap_30k --num_parallel 12 --dry_run

# ChartQA 3 Captions per Image Data Generation Command
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/data_generation/batch_data_generator.py --input_folder /home/vivineet/projects/siddharth/data --output_folder /home/vivineet/projects/siddharth/data --prompt_file generated_prompts/chartqa_mminstruct_3cap_30k_prompts_20241010_123954.json --file_prefix chartqa_mminstruct_3cap_30k --num_parallel 12 --dry_run

# ChartQA 5 Captions per Image Data Generation Command
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/data_generation/batch_data_generator.py --input_folder /home/vivineet/projects/siddharth/data --output_folder /home/vivineet/projects/siddharth/data --prompt_file generated_prompts/chartqa_mminstruct_5cap_30k_prompts_20241010_124001.json --file_prefix chartqa_mminstruct_5cap_30k --num_parallel 12 --dry_run