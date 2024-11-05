import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import pytorch_lightning as pl
from torch import nn
from torchmetrics import JaccardIndex
from torchmetrics.segmentation import MeanIoU
import numpy as np
import torchvision.models.segmentation as seg_models
import torch.nn.functional as F
import wandb

wandb.init(project="segmentation_project", entity="boapps")

class VOCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = [f for f in sorted(os.listdir(image_dir)) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png').replace('.jpeg', '.png'))
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
         
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        mask = np.array(mask, dtype=np.int64)
        return image, torch.from_numpy(mask)

def create_data_loaders(image_dir, mask_dir, batch_size=8, val_split=0.2, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST)
    ])

    full_dataset = VOCDataset(image_dir, mask_dir, transform=transform, mask_transform=mask_transform)
    
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Total dataset size: {total_size}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

class UNet(pl.LightningModule):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.model = seg_models.fcn_resnet50(pretrained=False, num_classes=num_classes)
        self.jaccard = JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=255)
        self.mean_iou = MeanIoU(num_classes=num_classes) # no ignore_index :(

    def forward(self, x):
        return self.model(x)['out']
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, masks, ignore_index=255)
        self.log("train_loss", loss)
        
        # Log to WandB
        wandb.log({"train_loss": loss})
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, masks, ignore_index=255)
        preds = outputs.argmax(dim=1)
        
        jaccard = self.jaccard(preds, masks)
        mean_iou = self.mean_iou(preds, masks)
        
        self.log("val_loss", loss)
        self.log("val_jaccard", jaccard)
        self.log("val_mean_iou", mean_iou)
        
        # Log to WandB
        wandb.log({"val_loss": loss, "val_jaccard": jaccard, "val_mean_iou": mean_iou})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# Initialize model and trainer
model = UNet(num_classes=21)
trainer = pl.Trainer(max_epochs=3, accelerator='auto', logger=pl.loggers.WandbLogger())
train_loader, val_loader = create_data_loaders('./img', './msk')
trainer.fit(model, train_loader, val_loader)

# Run evaluation on the validation set
def evaluate_model(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    jaccard_meter = JaccardIndex(task='multiclass', num_classes=21, ignore_index=255)
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

# Call the evaluation method
evaluate_model(model, val_loader)

# Save the model
model_path = 'cnn.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Finish the W&B run
wandb.finish()
