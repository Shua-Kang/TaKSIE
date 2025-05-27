import torch
import torchvision
import numpy as np # It's good practice to import numpy as np
from PIL import Image

from torchvision import transforms
import torch.nn as nn # Not used in the provided snippet
import torchvision.transforms as T
from tqdm import tqdm
import os
import argparse # Import argparse for command-line arguments

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Extract R3M features from a dataset.")
parser.add_argument("--dataset_path", type=str, help="Path to the input dataset directory.")
parser.add_argument("--output_dir_path", type=str, help="Path to the directory where features will be saved.")
args = parser.parse_args()

# --- Configuration ---
device = "cuda:0" if torch.cuda.is_available() else "cpu" # Use CUDA if available, otherwise CPU
print(f"Using device: {device}")

# --- Load R3M Model ---
try:
    from r3m import load_r3m
    r3m = load_r3m("resnet50") # resnet18, resnet34
    r3m.eval()
    r3m.to(device)
    transforms_r3m = T.Compose([T.Resize(224), T.ToTensor()]) # ToTensor() divides by 255
except ImportError:
    print("Error: r3m library not found. Please ensure it is installed.")
    exit()
except Exception as e:
    print(f"Error loading R3M model: {e}")
    exit()

# --- Helper Function ---
def get_r3m_feature(rgb_static_img):
    """
    Extracts R3M features from a single RGB image.
    """
    # Ensure input is a NumPy array
    if not isinstance(rgb_static_img, np.ndarray):
        raise TypeError(f"Input image must be a NumPy array, but got {type(rgb_static_img)}")

    # Ensure the image is in HWC format (Height, Width, Channels)
    if rgb_static_img.ndim != 3 or rgb_static_img.shape[2] != 3:
        raise ValueError(f"Input image must be in HWC format with 3 channels, but got shape {rgb_static_img.shape}")

    # Convert to PIL Image, apply transforms, and get features
    img_pil = Image.fromarray(rgb_static_img.astype(np.uint8)) # Ensure correct data type for PIL
    processed_img = transforms_r3m(img_pil).unsqueeze(0).to(device) # Add batch dimension and move to device
    
    with torch.no_grad(): # Disable gradient calculations for inference
        # R3M expects input range [0, 255]
        input_features = r3m(processed_img * 255.0) # R3M's ToTensor already divides by 255, so we multiply back. Or, if r3m expects [0,1] then remove * 255.0
                                                     # According to common practice with R3M, it expects [0, 255] float tensors.
                                                     # The transforms_r3m ToTensor() converts images to [0,1].
                                                     # So, multiplying by 255.0 is correct here.

    return input_features.unsqueeze(0).cpu().numpy() # Remove batch dim and move to CPU, then to numpy

dataset_path = args.dataset_path
output_dir = args.output_dir_path

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

print(f"Reading files from: {dataset_path}")

try:
    file_list = os.listdir(dataset_path)
except FileNotFoundError:
    print(f"Error: Dataset path '{dataset_path}' not found.")
    exit()
except Exception as e:
    print(f"Error listing files in dataset path: {e}")
    exit()

files_to_process = [f for f in file_list if f.startswith("episode_") and f.endswith(".npz")]
print(f"Found {len(files_to_process)} episode files to process.")

print(f"Processing {len(files_to_process)} files...") # Update count after potential slicing

for filename in tqdm(files_to_process):
    try:
        file_path = os.path.join(dataset_path, filename)
        data = np.load(file_path, allow_pickle=True)

        if "rgb_static" not in data:
            print(f"Warning: 'rgb_static' not found in {filename}. Skipping.")
            continue

        rgb_static_image = data["rgb_static"]
        feature = get_r3m_feature(rgb_static_image)

        output_filename = os.path.join(output_dir, filename.replace(".npz", "_r3m_feature.npy")) # Save as .npy
        np.save(output_filename, feature)

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

print("Feature extraction complete.")