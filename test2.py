# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 15:54:15 2021

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
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
dict_race_to_number = {'White' : 0, 
                       'Black': 1, 
                       'Latino_Hispanic': 3, 
                       'Asian' : 2, 
                       'Indian' : 4}

missed_dic = {
    'White_as_Asian' : 0,
    'White_as_Indian' : 0,
    'White_as_Black' : 0,
    'White_as_Latino' : 0,
    'Asian_as_White' : 0,
    'Asian_as_Indian' : 0,
    'Asian_as_Black' : 0,
    'Asian_as_Latino' : 0,
    'Black_as_White' : 0,
    'Black_as_Indian' : 0,
    'Black_as_Asian' : 0,
    'Balck_as_Latino' : 0,
    'Indian_as_White' : 0,
    'Indian_as_Asian' : 0,
    'Indian_as_Black' : 0,
    'Indian_as_Latino' : 0,
    'Latino_as_White' : 0,
    'Latino_as_Indian' : 0,
    'Latino_as_Black' : 0,
    'Latino_as_White' : 0,
    }
class Custom(data.Dataset):
    def __init__(self, data_path, attr_path, image_size):
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[0].split()[1:]
        atts = att_list
        self.images = np.loadtxt(attr_path, skiprows=1, usecols=[0], dtype=np.str)
        self.labels = np.loadtxt(attr_path, skiprows=1, usecols = [1], dtype=np.str)
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor( dict_race_to_number[self.labels[index]])
        return img, att
test_dataset = Custom('./val/', './ff-race-val.txt', 128)
test_loader = DataLoader(dataset=test_dataset,batch_size=64, shuffle=True, num_workers=0)

model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('./checkpoint.pth'), strict=False)
model.cuda()
model.eval()
accuracy = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        # run the model on the test set to predict labels
        outputs = model(data)
        # the label with the highest energy will be our prediction
        _, predicted = torch.max(outputs.data, 1)
        for idx in range(len(predicted)):
            if target[idx] == 0 and predicted[idx] == 1:
                missed_dic['White_as_Black'] += 1
            elif target[idx] == 0 and predicted[idx] == 2:
                missed_dic['White_as_Asian'] += 1
            elif target[idx] == 0 and predicted[idx] == 3:
                missed_dic['White_as_Latino'] += 1
            elif target[idx] == 0 and predicted[idx] == 4:
                missed_dic['White_as_Indian'] += 1
            elif target[idx] == 1 and predicted[idx] == 0:
                missed_dic['Black_as_White'] += 1
            elif target[idx] == 1 and predicted[idx] == 2:
                missed_dic['Black_as_Asian'] += 1
            elif target[idx] == 1 and predicted[idx] == 3:
                missed_dic['Black_as_Latino'] += 1
            elif target[idx] == 1 and predicted[idx] == 4:
                missed_dic['Black_as_Indian'] += 1
            elif target[idx] == 2 and predicted[idx] == 0:
                missed_dic['Asian_as_White'] += 1
            elif target[idx] == 2 and predicted[idx] == 1:
                missed_dic['Asian_as_Black'] += 1
            elif target[idx] == 2 and predicted[idx] == 3:
                missed_dic['Asian_as_Latino'] += 1
            elif target[idx] == 2 and predicted[idx] == 4:
                missed_dic['Asian_as_Indian'] += 1
            elif target[idx] == 3 and predicted[idx] == 0:
                missed_dic['Latino_as_White'] += 1
            elif target[idx] == 3 and predicted[idx] == 1:
                missed_dic['Latino_as_Black'] += 1
            elif target[idx] == 3 and predicted[idx] == 2:
                missed_dic['Latino_as_Asian'] += 1
            elif target[idx] == 3 and predicted[idx] == 4:
                missed_dic['Latino_as_Indian'] += 1
            elif target[idx] == 4 and predicted[idx] == 0:
                missed_dic['Indian_as_White'] += 1
            elif target[idx] == 4 and predicted[idx] == 1:
                missed_dic['Indian_as_Black'] += 1
            elif target[idx] == 4 and predicted[idx] == 2:
                missed_dic['Indian_as_Asian'] += 1
            elif target[idx] == 4 and predicted[idx] == 3:
                missed_dic['Indian_as_Latino'] += 1
        total += target.size(0)
        accuracy += (predicted == target).sum().item()
        print(predicted == target)
print(missed_dic)
    # compute the accuracy over all test images
accuracy = (100 * accuracy / total)
print(accuracy)