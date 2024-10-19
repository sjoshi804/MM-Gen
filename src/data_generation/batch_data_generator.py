import argparse 
import json 
import os
import subprocess
import time

MAX_CONCURRENT_JOBS = 12

def main(args):
    prompt_file = os.path.join(args.input_folder, args.prompt_file)
    with open(prompt_file, 'r') as file:
        prompts = json.load(file)
    total_prompts = len(prompts["prompts"])
    
    chunk_size = total_prompts // args.num_parallel
    chunks_start_idx = [(i, chunk_size) for i in range(0, total_prompts, chunk_size)]
    chunks_start_idx[-1] = (chunks_start_idx[-1][0], -1)
    
    if args.num_parallel > MAX_CONCURRENT_JOBS:
        print(f"Number of parallel jobs {args.num_parallel} exceeds the maximum number of concurrent jobs {MAX_CONCURRENT_JOBS}.")
        print(f"Jobs batched into batches of size {MAX_CONCURRENT_JOBS}.")
        print(f"Will run batch {args.batch_num} now.")
    
    command_file = os.path.join(args.output_folder, f"commands_{args.file_prefix}.sh")
    print(f"Writing all commands to {command_file}")
    
    command_input_folder = args.input_folder if not args.aml else "$INPUT_FOLDER"
    command_output_folder = args.output_folder if not args.aml else "$OUTPUT_FOLDER"
    
    for run_id, (start_idx, num_prompts) in enumerate(chunks_start_idx):
        if int(run_id / MAX_CONCURRENT_JOBS) != args.batch_num:
            continue
        time.sleep(1)
        command = (
            f"python src/data_generation/data_generator.py "
            f"--model_name {args.model_name} "
            f"--input_folder {command_input_folder} "
            f"--output_folder {command_output_folder} "
            f"--prompt_file {args.prompt_file} "
            f"--file_prefix {args.file_prefix}_{run_id} "
            f"--start_idx {start_idx} "
            f"--num_prompts {num_prompts} "
            f"{'--debug' if args.debug else ''}"
        )
        
        print(f"Running {command}")
        
        output_file = None
        if not args.aml:
            log_folder = os.path.join(args.output_folder, "logs")
            os.makedirs(log_folder, exist_ok=True)
            output_file = os.path.join(log_folder, f"run_{run_id}_{args.file_prefix}.log")
        
            with open(output_file, 'w') as f:
                f.write(f"Command: {command}\n")
            
        print(f"Logging to file {output_file}")
        
        # Start the command as a new process and redirect stdout and stderr to the output file
        if not args.dry_run:
            with open(output_file, 'a') as f:
                process = subprocess.Popen(command, shell=True, stdout=f, stderr=subprocess.STDOUT)
        
        with open(command_file, 'a') as f:
            f.write(command + "\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Data Generator")
    parser.add_argument("--model_name", type=str, default="gpt-4o-450K", help="Name of model to use with GPT4 endpoint")
    parser.add_argument('--input_folder', type=str, default="", help='Input Folder')
    parser.add_argument('--output_folder', type=str, default="", help='Output Folder')
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the JSON file containing the prompts")
    parser.add_argument("--save_path", type=str, default="", help="Path to save the generated questions and answers")
    parser.add_argument("--file_prefix", type=str, required=True, help="Prefix for file with generated questions")
    parser.add_argument("--num_parallel", type=int, default=1, help="Index to start at in the list of the prompts.")
    parser.add_argument("--batch_num", type=int, default=0, help="Which batch of concurrent runs to start if running > max conccurent jobs")
    parser.add_argument("--dry_run", action='store_true', help="Dry run the script without executing the commands.")
    parser.add_argument("--aml", action='store_true', help="Run on aml")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging")

    args = parser.parse_args()
    main(args)
