import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torchmetrics import JaccardIndex
from torchmetrics.segmentation import MeanIoU
import numpy as np
import torch.nn.functional as F


class VOCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = [
            f
            for f in sorted(os.listdir(image_dir))
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(
            self.mask_dir, img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        )

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = np.array(mask, dtype=np.int64)
        return image, torch.from_numpy(mask)


def create_data_loaders(
    image_dir, mask_dir, batch_size=8, num_workers=4
):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    mask_transform = transforms.Compose(
        [transforms.Resize((256, 256), interpolation=Image.NEAREST)]
    )

    full_dataset = VOCDataset(
        image_dir, mask_dir, transform=transform, mask_transform=mask_transform
    )

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [0.6, 0.2, 0.2],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# Run evaluation on the validation set
def evaluate_model(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    jaccard_meter = JaccardIndex(task="multiclass", num_classes=21, ignore_index=255)
    mean_iou_meter = MeanIoU(num_classes=21)
    val_loss_meter = 0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch
            outputs = model(images)
            loss = F.cross_entropy(outputs, masks, ignore_index=255)
            preds = outputs.argmax(dim=1)

            val_loss_meter += loss.item()
            jaccard_meter(preds, masks)
            mean_iou_meter(preds, masks)
            count += 1

    avg_val_loss = val_loss_meter / count
    avg_jaccard = jaccard_meter.compute()
    avg_mean_iou = mean_iou_meter.compute()

    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Jaccard Index: {avg_jaccard:.4f}")
    print(f"Validation Mean IoU: {avg_mean_iou:.4f}")
