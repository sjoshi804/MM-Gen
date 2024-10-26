import argparse
import os
import json
from glob import glob

def merge_json_files(folder_path, run_id, output_folder):
    # Find all JSON files in the directory with the run_id in their name
    json_files = sorted(glob(os.path.join(folder_path, f"*{run_id}*.json")), key=lambda x: int(x.split(f"{run_id}_")[-1].split("_")[0]))
    if not json_files:
        print(f"No JSON files found with run_id {run_id} in {folder_path}")
        return

    # Load the first JSON file
    with open(json_files[0], 'r') as f:
        merged_data = json.load(f)

    # Merge the 'samples' list from each subsequent JSON file into the first one
    start_idx = merged_data["num_prompts"]
    for json_file in json_files[1:]:
        with open(json_file, 'r') as f:
            data = json.load(f)
            print(data["start_idx"], start_idx)
            #assert data["start_idx"] == start_idx
            start_idx += data["num_prompts"]
            # assert len(data["samples"]) > int(0.9 * data["num_prompts"])
            merged_data['samples'].extend(data.get('samples', []))

    # Process samples
    merged_data["image_folder"] = os.path.dirname(merged_data["samples"][0]["image_1"])
    for sample in merged_data["samples"]:
        sample["image_1"] = os.path.basename(sample["image_1"])
        
    # Save the merged data to a new JSON file
    merged_data["len_samples"] = len(merged_data["samples"])
    output_file = os.path.join(output_folder, f"{run_id}.json")
    merged_data["image_folder"] = merged_data["image_folder"].replace("/home/vivineet/projects/siddharth/data/","")
        
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=3)
        
    print("CHECK IMAGE FOLDER PATH", merged_data["image_folder"])

    print(f"Merged JSON file saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge JSON files based on run_id")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing JSON files")
    parser.add_argument("--output_folder", type=str, help="Path to the folder to save the merged JSON file")
    parser.add_argument("--run_id", type=str, help="Run ID to filter JSON files")

    args = parser.parse_args()
    merge_json_files(args.folder_path, args.run_id, args.output_folder)