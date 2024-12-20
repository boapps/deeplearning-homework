from datasets import Dataset, Image
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from transformers import AutoImageProcessor
import os
import numpy as np
import torch
from torch import nn
import evaluate
from data_split import get_split_indices  # Import split function

checkpoint = "nvidia/mit-b0"

# Set image and mask paths because we are using the same dataset all the time
image_dir = "./img"
mask_dir = "./msk"

images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
image_paths = ["img/" + f for f in images]
label_paths = [f.replace("jpg", "png").replace("img", "msk") for f in image_paths]

# Get consistent split indices
split_indices = get_split_indices(image_dir)

train_dataset = Dataset.from_dict({
    "image": [image_paths[i] for i in split_indices["train"]],
    "label": [label_paths[i] for i in split_indices["train"]]
})
print(len(train_dataset))
val_dataset = Dataset.from_dict({
    "image": [image_paths[i] for i in split_indices["val"]],
    "label": [label_paths[i] for i in split_indices["val"]]
})
test_dataset = Dataset.from_dict({
    "image": [image_paths[i] for i in split_indices["test"]],
    "label": [label_paths[i] for i in split_indices["test"]]
})

# Convert the image and label columns to `Image` format
train_dataset = train_dataset.cast_column("image", Image())
train_dataset = train_dataset.cast_column("label", Image())
val_dataset = val_dataset.cast_column("image", Image())
val_dataset = val_dataset.cast_column("label", Image())
test_dataset = test_dataset.cast_column("image", Image())
test_dataset = test_dataset.cast_column("label", Image())

id2label = {i: str(i) for i in range(20)}
id2label[255] = "255"
label2id = {str(i): i for i in range(20)}
label2id["255"] = 255

image_processor = AutoImageProcessor.from_pretrained(
    'nvidia/mit-b0',
    do_resize=True,
    size={'height': 512, 'width': 512},
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
    do_reduce_labels=False
)
image_processor.save_pretrained("../data/vit")

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

train_dataset.set_transform(train_transforms)
val_dataset.set_transform(val_transforms)
test_dataset.set_transform(val_transforms)

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
    learning_rate=3e-4,
    warmup_ratio=0.05,
    weight_decay=0.05,
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=24,
    gradient_accumulation_steps=2,
    fp16=True,
    remove_unused_columns=False,
    report_to=None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

model.save_pretrained("../data/vit")

