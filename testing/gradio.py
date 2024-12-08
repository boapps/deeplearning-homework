import gradio as gr
from PIL import Image
import torch
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
import numpy as np
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
from torchvision import transforms

checkpoint = "../data/vit.best"
id2label = {i: str(i) for i in range(20)}
id2label[255] = "255"
label2id = {str(i): i for i in range(20)}
label2id["255"] = 255

fixed_color_map = {
    0: (0, 0, 0),         # Empty
    1: (220, 20, 60),     # Aeroplane
    2: (255, 182, 193),   # Bicycle
    3: (255, 105, 180),   # Bird
    4: (255, 20, 147),    # Boat
    5: (255, 0, 255),     # Bottle
    6: (199, 21, 133),    # Bus
    7: (219, 112, 147),   # Car
    8: (255, 160, 122),   # Cat
    9: (255, 69, 0),      # Chair
    10: (255, 140, 0),    # Cow
    11: (255, 215, 0),    # Diningtable
    12: (255, 255, 0),    # Dog
    13: (173, 255, 47),   # Horse
    14: (0, 255, 0),      # Motorbike
    15: (0, 128, 0),      # Person
    16: (0, 255, 255),    # Potted Plant
    17: (0, 191, 255),    # Sheep
    18: (30, 144, 255),   # Sofa
    19: (0, 0, 255),      # Train
    20: (0, 0, 139)       # TV/Monitor
}

model = AutoModelForSemanticSegmentation.from_pretrained(
    checkpoint, id2label=id2label, label2id=label2id
)

image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)

def model_vit(image):
    inputs = image_processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        predicted_segmentation = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    segmented_image = np.zeros((*predicted_segmentation.shape, 3), dtype=np.uint8)
    for label, color in fixed_color_map.items():
        segmented_image[predicted_segmentation == label] = color

    segmented_image = Image.fromarray(segmented_image)
    return segmented_image

def model_cnn1(image):
    model = fcn_resnet50(weights=None, num_classes=21)
    
    checkpoint = torch.load('../data/cnn.pth', map_location=torch.device("cpu"))
    state_dict = {key.replace("model.", ""): value for key, value in checkpoint.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)['out']
        output = torch.nn.functional.interpolate(
            output,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        predicted_segmentation = output.argmax(dim=1).squeeze().cpu().numpy()

    segmented_image = np.zeros((*predicted_segmentation.shape, 3), dtype=np.uint8)
    for label, color in fixed_color_map.items():
        segmented_image[predicted_segmentation == label] = color

    segmented_image = Image.fromarray(segmented_image)
    return segmented_image

def model_cnn2(image):
    model = fcn_resnet101(weights=None, num_classes=21)
    checkpoint = torch.load('../data/cnn_v2.pth', map_location=torch.device("cpu"))
    state_dict = {key.replace("model.", ""): value for key, value in checkpoint.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)['out']
        output = torch.nn.functional.interpolate(
            output,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        predicted_segmentation = output.argmax(dim=1).squeeze().cpu().numpy()

    segmented_image = np.zeros((*predicted_segmentation.shape, 3), dtype=np.uint8)
    for label, color in fixed_color_map.items():
        segmented_image[predicted_segmentation == label] = color

    segmented_image = Image.fromarray(segmented_image)
    return segmented_image

def segment_image(image, model_choice):
    if model_choice == "MiT-B0":
        return model_vit(image)
    elif model_choice == "ResNet-50":
        return model_cnn1(image)
    elif model_choice == "ResNet-101":
        return model_cnn2(image)

def generate_color_legend():
    label_names = {
        0: "Empty",
        1: "Aeroplane",
        2: "Bicycle",
        3: "Bird",
        4: "Boat",
        5: "Bottle",
        6: "Bus",
        7: "Car",
        8: "Cat",
        9: "Chair",
        10: "Cow",
        11: "Diningtable",
        12: "Dog",
        13: "Horse",
        14: "Motorbike",
        15: "Person",
        16: "Potted Plant",
        17: "Sheep",
        18: "Sofa",
        19: "Train",
        20: "TV/Monitor"
    }
    html_content = "<table>"
    for label, color in fixed_color_map.items():
        html_content += f"<tr><td style='background-color: rgb{color}; width: 50px; height: 20px;'></td><td>{label_names[label]}</td></tr>"
    html_content += "</table>"
    return html_content

color_legend = generate_color_legend()

demo = gr.Blocks()

with demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            model_choice = gr.Dropdown(choices=["MiT-B0", "ResNet-50", "ResNet-101"], label="Select Model")
            submit_button = gr.Button("Segment Image")
        with gr.Column():
            segmented_image_output = gr.Image(type="pil", label="Segmented Image")
            gr.HTML(color_legend)
    
    submit_button.click(segment_image, inputs=[image_input, model_choice], outputs=segmented_image_output)

demo.launch()