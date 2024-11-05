import torch
import pytorch_lightning as pl
from torch import nn
from torchmetrics import JaccardIndex
from torchmetrics.segmentation import MeanIoU
import torchvision.models.segmentation as seg_models
import torch.nn.functional as F
# import wandb
from dataset import create_data_loaders, evaluate_model

# wandb.init(project="segmentation_project", entity="melytanulo-buvarok")

class UNet(pl.LightningModule):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.pre_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.model = seg_models.fcn_resnet50(pretrained=False, num_classes=num_classes)
        self.post_conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        self.jaccard = JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=255)
        self.mean_iou = MeanIoU(num_classes=num_classes) # no ignore_index :(

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.model(x)['out']
        x = self.post_conv(x)
        return x
    
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
model.state_dict = torch.load("../data/cnn_v2.pth")
evaluate_model(model, test_loader)