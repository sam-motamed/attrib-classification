# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 21:12:13 2021

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
from collections import Counter
import pandas as pd

race_list = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
dict_race_to_number = {'White' : 0, 
                       'Black': 1,
                       'Asian' : 2}
def BCELoss_IGNORE(x, y, ignore_idx):
    
    ignored_y = y[[y != ignore_idx]]
    ignored_x = x[[y != ignore_idx]]
    return torch.nn.functional.binary_cross_entropy(ignored_x, ignored_y)
    
    

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
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor( dict_race_to_number[self.labels[index]])
        return img, att
    
    def __len__(self):
        return len(self.images)
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc
def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', type=str, default='./train/')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='./ff-race.txt')
    parser.add_argument('--img_size', dest='img_size', type=int, default=256)
    return parser.parse_args(args)
args = parse()
train_dataset = Custom(args.data_path, args.attr_path, args.img_size)
EPOCHS = 150
BATCH_SIZE = 128
LEARNING_RATE = 0.0003
NUM_FEATURES = len(train_dataset)
NUM_CLASSES = 7
accuracy_stats = {
    'train': []
}
loss_stats = {
    'train': []
}

device = torch.device("cuda:0")
print(device)
print("number of cpu", os.cpu_count())
if torch.cuda.device_count() > 0:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
else:
    print("NO GPU WAS FOUND")
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

model = torch.nn.DataParallel(model, device_ids=[0, 1])
model = model.to(device)
checkpoint_path = os.path.join(os.getcwd(), "checkpoint.pth")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
model.train()
running_loss = 0.0
for e in tqdm(range(1, EPOCHS+1)):
    with tqdm(train_loader, unit="batch") as tepoch:
        # TRAINING
        tepoch.set_description(f"Epoch {e}")
        train_epoch_loss = 0
        train_epoch_acc = 0
        for X_train_batch, y_train_batch in tepoch:
            #y_train_batch = torch.squeeze(y_train_batch)
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=train_loss.item())
        training_state = {'model' : model.state_dict(),'optimizer' : optimizer.state_dict(),'epoch': e}
        torch.save(training_state, checkpoint_path)








