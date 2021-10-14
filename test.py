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
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
dict_race_to_number = {'White' : 0, 
                       'Black': 1, 
                       'Latino_Hispanic': 2, 
                       'East Asian' : 3, 
                       'Southeast Asian' : 4, 
                       'Indian' : 5, 
                       'Middle Eastern' : 6}
class MulticlassClassification(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=7)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()
    def forward(self, x):
        return self.sigm(self.base_model(x))

def classify(img_path, net, use_gpu):
    #labels = ['East_Asian', 'Southeast_Asian','Black','Indian','White','Middle_Eastern', 'Latino']
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
    #race_labels = y[1:]
    max_prob_idx = np.argmax(y)
    '''if classed_race == actual_race:
        print("correct")
        return 1
    else:
        print("missed")
        return 0'''
    return max_prob_idx
use_gpu = torch.cuda.is_available()
model = models.vgg16(pretrained = True)
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096, bias = True),
    nn.ReLU(inplace = True),
    nn.Dropout(0.4),
    nn.Linear(4096, 2048, bias = True),
    nn.ReLU(inplace = True),
    nn.Dropout(0.4),
    nn.Linear(2048, 7)
)
model.load_state_dict(torch.load('./checkpoint.pth'), strict=False)
model.eval()
df = pd.read_csv('./fairface_label_val.csv')
race = df['race']
imm_id = df['file']
correct_class = 0
for idx in range(len(imm_id)):
    if classify(imm_id[idx], model, use_gpu) == dict_race_to_number[race[idx]]:
        correct_class += 1
        print("CORRECT")
    else:
        print("MISCLASSIFIED")
        pass
print(correct_class)

