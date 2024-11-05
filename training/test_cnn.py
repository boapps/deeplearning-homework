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
from dataset import create_data_loaders, evaluate_model

train_loader, val_loader, test_loader = create_data_loaders("./img", "./msk")
model = torch.load("cnn.pth")
evaluate_model(model, test_loader)
