from tqdm import tqdm
import argparse
import math
import os
import random
import shutil

def copy_random_images(src_dir, dest_dir, num_images, num_skills):
    # Create the destination directory and subfolders if they don't exist
    os.makedirs(dest_dir, exist_ok=True)
    subfolders = [os.path.join(dest_dir, str(i)) for i in range(num_skills)]
    for folder in subfolders:
        os.makedirs(folder, exist_ok=True)

    # Get a list of all image files in the source directory
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(image_extensions)]

    # Select num_images random images (or fewer if less than num_images images available)
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    # Partition the images into roughly equal chunks for each subfolder
    chunk_size = math.ceil(len(selected_images) / num_skills)

    for i, folder in enumerate(subfolders):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk_images = selected_images[start_idx:end_idx]
        
        # Copy each image in the chunk to the corresponding subfolder with progress tracking
        for image in tqdm(chunk_images, desc=f"Copying to folder {i}", unit="image"):
            shutil.copy(os.path.join(src_dir, image), folder)

    print(f"Copied {len(selected_images)} images into {num_skills} subfolders in {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy random images from source to destination with partitioned subfolders.")
    parser.add_argument("--src_dir", type=str, required=True, help="Source directory containing images")
    parser.add_argument("--dest_dir", type=str, required=True, help="Destination directory to copy images to")
    parser.add_argument("--num_images", type=int, required=True, help="Number of random images to copy")
    parser.add_argument("--num_skills", type=int, required=True, help="Number of subfolders to create and partition images into")

    args = parser.parse_args()
    
    copy_random_images(args.src_dir, args.dest_dir, args.num_images, args.num_skills)
