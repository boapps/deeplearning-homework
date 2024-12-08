import torch
import pytorch_lightning as pl
from torch import nn
from torchmetrics import JaccardIndex
from torchmetrics.segmentation import MeanIoU
import torchvision.models.segmentation as seg_models
import torch.nn.functional as F
# import wandb
import csv
from dataset import create_data_loaders, evaluate_model

# wandb.init(project="segmentation_project", entity="melytanulo-buvarok")

class UNet(pl.LightningModule):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.model = seg_models.fcn_resnet101(pretrained=False, num_classes=num_classes)
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
        # wandb.log({"train_loss": loss})
        
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
        # wandb.log({"val_loss": loss, "val_jaccard": jaccard, "val_mean_iou": mean_iou})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    

train_loader, val_loader, test_loader = create_data_loaders("./img", "./msk")
model = UNet(num_classes=21)
model.load_state_dict(torch.load("../data/cnn_v2.pth"))


def save_per_class_iou(model, dataloader, num_classes, filename='../data/per_class_iou_cnn_v2.csv'):
    model.eval()
    iou_per_class = {i: {'intersection': 0, 'union': 0} for i in range(num_classes)}

    with torch.no_grad():
        for images, masks in dataloader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for cls in range(num_classes):
                intersection = ((preds == cls) & (masks == cls)).sum().item()
                union = ((preds == cls) | (masks == cls)).sum().item()
                iou_per_class[cls]['intersection'] += intersection
                iou_per_class[cls]['union'] += union

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['class', 'iou'])
        sum_iou = 0
        for cls in range(num_classes):
            union = iou_per_class[cls]['union']
            iou = iou_per_class[cls]['intersection'] / union if union > 0 else 0
            writer.writerow([cls, iou])
            sum_iou += iou
        print(f'Mean IoU: {sum_iou / num_classes}')

save_per_class_iou(model, test_loader, num_classes=21)