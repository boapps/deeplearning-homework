from datasets import Dataset, Image
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
import os
import torch
import numpy as np
from PIL import Image as PILImage
from data_split import get_split_indices

# Load the trained model
model = AutoModelForSemanticSegmentation.from_pretrained("../data/vit")

# Load the image processor
image_processor = AutoImageProcessor.from_pretrained("../data/vit")

# Set model to evaluation mode
model.eval()

# Set image paths
image_dir = "./img"

images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
image_paths = ["img/" + f for f in images]

# Get test set indices
split_indices = get_split_indices(image_dir)

test_image_paths = [image_paths[i] for i in split_indices["test"]]

# Create an output directory
output_dir = "../data/test_results"
os.makedirs(output_dir, exist_ok=True)

# Define a color palette for the segmentation map
num_classes = model.config.num_labels
palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

# Process each test image
for index, image_path in enumerate(test_image_paths):
    # Load image
    image = PILImage.open(image_path).convert("RGB")
    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the outputs to get the segmentation map
    predicted_seg = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]

    # Map the predicted segmentation to colors
    segmentation_map = palette[predicted_seg.numpy()]

    # Convert segmentation map to PIL Image
    seg_image = PILImage.fromarray(segmentation_map)

    # Overlay segmentation map on original image
    blended = PILImage.blend(image, seg_image, alpha=0.5)

    # Save the result
    output_path = os.path.join(output_dir, f"result_{index}.png")
    blended.save(output_path)