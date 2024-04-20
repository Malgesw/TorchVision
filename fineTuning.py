import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.models as models, torchvision.transforms as transforms
import time
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from sklearn import metrics
import inspect
import matplotlib.pyplot as plt
import numpy as np


class MnistResnet18(nn.Module):
    def __init__(self, in_channels=1):
        super(MnistResnet18, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # first layer now take 1 color channel
        fc_outputs = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_outputs, 10)  # last layer now outputs 10 classes instead of 1000

    def forward(self, data):
        return self.model(data)
