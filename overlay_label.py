# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:39:10 2021

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
                       'Latino_Hispanic': 4, 
                       'East Asian' : 2, 
                       'Southeast Asian' : 2, 
                       'Indian' : 5, 
                       'Middle Eastern' : 6}

 self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
test_path = './val/'
save_path = './val_with_label'
model.fc = nn.Linear(num_ftrs, 3)
model = DataParallel(model, device_ids=[0])
model.load_state_dict(torch.load('./checkpoint.pth'), strict=False)
model.cuda()
model.eval()
accuracy = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for im_file in os.listdir():
        orig_im = PIL.Image.open(im_file)
        outputs = model(data)
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        _, predicted = torch.max(outputs.data, 1)
        label = dict_race_to_number.keys()[dict_race_to_number.values().index(predicted)]
        draw = PIL.ImageDraw.Draw(orig_im)
        draw.text((10, 10),label)
        original.save(os.path.join(save_dir, im_file))