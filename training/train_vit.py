from datasets import Dataset, DatasetDict, Image
from torch.utils.data import random_split
import os
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from transformers import AutoImageProcessor
import numpy as np
import torch
from torch import nn
import evaluate

checkpoint = "nvidia/mit-b0"

image_paths_train = [
    "img/" + f for f in sorted(os.listdir("img")) if f.endswith((".jpg"))
]
label_paths_train = [
    f.replace("jpg", "png").replace("img", "msk") for f in image_paths_train
]


def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict(
        {"image": sorted(image_paths), "label": sorted(label_paths)}
    )
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())
    return dataset


train_dataset = create_dataset(image_paths_train, label_paths_train)
dataset = train_dataset.train_test_split(seed=42)

id2label = {i: str(i) for i in range(20)}
id2label[255] = "255"

label2id = {str(i): i for i in range(20)}
label2id["255"] = 255


def train_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["label"]]
    inputs = image_processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["label"]]
    inputs = image_processor(images, labels)
    return inputs


dataset["train"].set_transform(train_transforms)
dataset["test"].set_transform(val_transforms)


image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)

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

training_args = TrainingArguments(
    output_dir="mit-b0-pascal-voc",
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
    # push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("vit")
