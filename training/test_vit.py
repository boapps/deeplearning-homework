import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset, Image
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor, TrainingArguments, Trainer
import evaluate
from data_split import get_split_indices  # Import split function
import os

# Set paths and model checkpoint
checkpoint = "nvidia/mit-b0"

image_dir = "./img"
mask_dir = "./msk"

images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
image_paths = ["img/" + f for f in images]
label_paths = [f.replace("jpg", "png").replace("img", "msk") for f in image_paths]

split_indices = get_split_indices(image_dir)

test_dataset = Dataset.from_dict({
    "image": [image_paths[i] for i in split_indices["test"]],
    "label": [label_paths[i] for i in split_indices["test"]]
})

# Convert columns to `Image` format
test_dataset = test_dataset.cast_column("image", Image())
test_dataset = test_dataset.cast_column("label", Image())

id2label = {i: str(i) for i in range(20)}
id2label[255] = "255"
label2id = {str(i): i for i in range(20)}
label2id["255"] = 255

image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)

def test_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["label"]]
    inputs = image_processor(images, labels)
    return inputs

test_dataset.set_transform(test_transforms)

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=20,
            ignore_index=255,
            reduce_labels=False,
        )
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
        return metrics

model = AutoModelForSemanticSegmentation.from_pretrained(
    checkpoint, id2label=id2label, label2id=label2id
)

# Define new evaluation arguments
eval_args = TrainingArguments(
    output_dir="mit-b0-evaluation",
    learning_rate=6e-5,
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    save_total_limit=3,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=200,
    eval_steps=200,
    logging_steps=10,
    eval_accumulation_steps=16,
    remove_unused_columns=False,
)

# Set up trainer for evaluation
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.evaluate()

model.save_pretrained("../data/vit")
