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
import torch.nn.functional as F
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
                       'Black': 1}
def BCELoss_IGNORE(x, y, ignore_idx):
    
    ignored_y = y[[y != ignore_idx]]
    ignored_x = x[[y != ignore_idx]]
    return torch.nn.functional.binary_cross_entropy(ignored_x, ignored_y)
    
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

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
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', type=str, default='./train/')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='./ff-race.txt')
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    return parser.parse_args(args)
args = parse()
train_dataset = Custom(args.data_path, args.attr_path, args.img_size)
EPOCHS = 50
BATCH_SIZE = 164
LEARNING_RATE = 0.0003
NUM_FEATURES = len(train_dataset)
NUM_CLASSES = 2
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
model = models.resnet34(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = torch.nn.DataParallel(model, device_ids=[0, 1])
model = model.to(device)
checkpoint_path = os.path.join(os.getcwd(), "checkpoint.pth")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
model.train()

for e in tqdm(range(1, EPOCHS+1)):
    train_loss = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
        # TRAINING
        tepoch.set_description(f"Epoch {e}")
        train_epoch_loss = 0
        train_epoch_acc = 0
        for data, target in train_loader:
        # move tensors to GPU if CUDA is available
            data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        e, train_loss))
        training_state = {'model' : model.state_dict(),'optimizer' : optimizer.state_dict(),'epoch': e}
        torch.save(training_state, checkpoint_path)

