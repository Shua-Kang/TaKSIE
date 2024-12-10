from pipeline_controlnet import StableDiffusionControlNetPipelineLstm
from controlnet import ControlNetModelLstm
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
from PIL import Image, ImageDraw
import transformers

# Set the device to GPU for faster computation
device = "cuda:0"

# Paths to the base model and ControlNet
base_model_path = "ShuaKang/TaKSIE_unet"
controlnet_path = "ShuaKang/TaKSIE_controlnet"

# Load the ControlNet model
controlnet = ControlNetModelLstm.from_pretrained(
    controlnet_path, torch_dtype=torch.float32
).to(device)

# Load the pipeline with the ControlNet
pipe = StableDiffusionControlNetPipelineLstm.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float32
).to(device)
controlnet.eval()

# Speed up the diffusion process with a faster scheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Load CLIP image processor and vision model
imageProcessor = transformers.AutoImageProcessor.from_pretrained(
    "openai/clip-vit-large-patch14", use_auth_token=False
)
visionModel = (
    transformers.CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    .requires_grad_(False)
    .eval()
    .to(device)
)

# Function to extract CLIP features from an image
def get_clip_feature(rgb_static):
    input = imageProcessor(images=rgb_static, return_tensors="pt")
    input_features = (
        visionModel(input["pixel_values"].to(device))[0].unsqueeze(0).to(device)
    )
    return input_features

# Load and preprocess the control image
control_image = load_image("./example/example_0.jpg")
control_image = control_image.resize((256, 256))

# Generate the CLIP feature for the control image
progress_feature = get_clip_feature(control_image)
progress_feature = torch.cat(
    [progress_feature, progress_feature, progress_feature]
).transpose(0, 1)

# Define multiple prompts for image generation
prompts = [
    "slide down the switch",
    "turn off the LED",
    "push the red block to the right",
    "pick the red block up",
    "move slider left"
]

# Generate images for each prompt and store them
generated_images = []
for prompt in prompts:
    generated_image = pipe(
        prompt,
        num_inference_steps=50,
        negative_prompt="",
        image=control_image,
        class_label=progress_feature,
        guidance_scale=2.5,
        image_guidance_scale=2.5,
    ).images[0]
    generated_images.append((generated_image, prompt))

# Create a combined image with labels
font_size = 20
label_height = 2 * (font_size + 5)  # Space for two-line labels

# Calculate the dimensions for the combined image
combined_width = sum(img.width for img, _ in generated_images) + control_image.width
combined_height = max(img.height for img, _ in generated_images) + label_height

# Create a blank canvas for the combined image
combined_image = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
draw = ImageDraw.Draw(combined_image)

# Add the condition image with label
combined_image.paste(control_image, (0, 0))
draw.text(
    (10, control_image.height + 5),
    "Condition Image",
    fill="blue"
)

# Paste each generated image and its two-line prompt
x_offset = control_image.width
for img, prompt in generated_images:
    # Split the prompt into two lines if it's too long
    words = prompt.split()
    line1 = ""
    line2 = ""
    for word in words:
        if len(line1 + word) <= 20:
            line1 += f"{word} "
        else:
            line2 += f"{word} "
    prompt_text = f"{line1.strip()}\n{line2.strip()}"

    # Paste the image
    combined_image.paste(img, (x_offset, 0))
    # Add the two-line label
    draw.multiline_text(
        (x_offset + 10, img.height + 5),
        prompt_text,
        fill="black",
        align="center"
    )
    x_offset += img.width

# Save the combined image
combined_image.save("result_combined_with_labels.png")
