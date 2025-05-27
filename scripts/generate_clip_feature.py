import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T # Kept for potential direct use, though CLIPProcessor handles most
from transformers import AutoImageProcessor, CLIPVisionModelWithProjection
from tqdm import tqdm
import os
import argparse

def get_clip_visual_features(pil_image, image_processor, vision_model, device):
    """
    Extracts CLIP visual features from a PIL image.
    Relies on the image_processor to handle necessary transformations (resize, crop, normalize).
    """
    inputs = image_processor(images=pil_image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    with torch.no_grad():
        # .image_embeds directly gives the [batch_size, projection_dim] tensor
        features = vision_model(pixel_values=pixel_values).image_embeds
    return features # Shape (1, feature_dim), already on target_device

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Extract CLIP visual features from images in a dataset.")
    parser.add_argument("--dataset_path", type=str, help="Path to the input dataset directory (containing .npz files).")
    parser.add_argument("--output_dir_path", type=str, help="Path to the directory where extracted .npy features will be saved.")
    args = parser.parse_args()

    clip_model_name = "openai/clip-vit-large-patch14"

    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using CLIP model: {clip_model_name}")

    # --- Load CLIP Model and Image Processor ---
    try:
        # The AutoImageProcessor handles resizing to model's expected input size (e.g., 224x224)
        # and normalization.
        image_processor = AutoImageProcessor.from_pretrained(clip_model_name)
        vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(device).eval()
    except Exception as e:
        print(f"Error loading CLIP model or image processor ('{clip_model_name}'): {e}")
        print("Please ensure the model name is correct and you have an internet connection if downloading.")
        exit(1)

    # --- Main Processing Logic ---
    dataset_path = args.dataset_path
    output_dir = args.output_dir_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Reading files from: {dataset_path}")
    try:
        file_list = os.listdir(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset path '{dataset_path}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error listing files in dataset path: {e}")
        exit(1)

    # Filter for relevant files (e.g., episode_*.npz)
    episode_files = sorted([f for f in file_list if f.startswith("episode_") and f.endswith(".npz")])
    print(f"Found {len(episode_files)} 'episode_*.npz' files to process.")

    if not episode_files:
        print("No episode files found to process. Exiting.")
        exit(0)

    for filename in tqdm(episode_files, desc="Extracting CLIP features"):
        if not filename.startswith("episode_") or not filename.endswith(".npz"):
            # This secondary check is redundant due to pre-filtering but kept for safety.
            continue
        
        file_path = os.path.join(dataset_path, filename)
        try:
            data = np.load(file_path, allow_pickle=True)
            if "rgb_static" not in data:
                print(f"Warning: 'rgb_static' key not found in {filename}. Skipping.")
                continue
            
            rgb_static_numpy = data["rgb_static"]
            if not isinstance(rgb_static_numpy, np.ndarray):
                print(f"Warning: 'rgb_static' in {filename} is not a numpy array. Skipping.")
                continue

            pil_image = Image.fromarray(rgb_static_numpy.astype(np.uint8))
            
            # Get CLIP features (tensor on device)
            feature_tensor = get_clip_visual_features(pil_image, image_processor, vision_model, device)
            
            # Convert to NumPy array on CPU
            feature_numpy = feature_tensor.cpu().numpy() # Shape will be (1, feature_dim) e.g. (1, 768)

            output_filename = os.path.join(output_dir, filename.replace(".npz", "_clip_feature.npy"))
            np.save(output_filename, feature_numpy)

        except Exception as e:
            print(f"Error processing file {filename}: {e}. Skipping.")

    print("CLIP feature extraction complete.")

if __name__ == "__main__":
    main()