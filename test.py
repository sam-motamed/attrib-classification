# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:54:28 2021

@author: 16478
"""


import argparse
import datetime
import json
import os
import numpy as np
from os.path import join
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
import torchvision
from torch.utils.data import Dataset, DataLoader
from helpers import Progressbar, add_scalar_dict
from torchvision import datasets, models, transforms
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable

class MulticlassClassification(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=8)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()
    def forward(self, x):
        return self.sigm(self.base_model(x))

def classify(img_path, net, use_gpu):
    transform =transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    img = Image.open(img_path)
    img = transform(img)
 
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
 
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    return y
use_gpu = torch.cuda.is_available()
model = MulticlassClassification(8)
model.load_state_dict(torch.load('./checkpoint.pth'), strict=False)
model.eval()
img = './test/000427.jpg'
pred = classify(img, model, use_gpu)
print(pred)
