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
import augly.image as imaugs
COLOR_JITTER_PARAMS = {
    "brightness_factor": 1.2,
    "contrast_factor": 1.2,
    "saturation_factor": 1.2,
}
AUGMENTATIONS = [
    
    imaugs.RandomBrightness(),
    imaugs.ColorJitter(**COLOR_JITTER_PARAMS),
    imaugs.Blur(),
    imaugs.RandomRotation(),
    imaugs.RandomEmojiOverlay(),
    imaugs.Rotate(),
    imaugs.RandomPixelization(),
    imaugs.RandomAspectRatio(),
    imaugs.OneOf(
        [imaugs.RandomNoise(), imaugs.OverlayText(), imaugs.VFlip(), imaugs.RandomRotation(), imaugs.OverlayOntoScreenshot(), imaugs.OverlayEmoji(), imaugs.OverlayText(), imaugs.OverlayStripes(),
         imaugs.Rotate(), ]
    )
]
TRANSFORMS = imaugs.Compose(AUGMENTATIONS)
TENSOR_TRANSFORMS = transforms.Compose(AUGMENTATIONS + [transforms.ToTensor()])

race_list = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
dict_race_to_number = {'White' : 0,
                       'Black': 1,
                       'Asian' : 2,
                       'Latino_Hispanic' : 3,
                       'Indian' : 4}
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
        img = self.tf(TRANSFORMS(Image.open(os.path.join(self.data_path, self.images[index]))).convert('RGB'))
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
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='./ff-race-train.txt')
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    return parser.parse_args(args)
args = parse()
train_dataset = Custom(args.data_path, args.attr_path, args.img_size)
test_dataset = Custom('./val/', './ff-race-val.txt', 128)
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.0003
NUM_FEATURES = len(train_dataset)
NUM_CLASSES = 5
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

model = torchvision.models.resnet34(pretrained=True)
in_ftr  = model.fc.in_features
out_ftr = 5
model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model = model.to(device)
checkpoint_path = os.path.join(os.getcwd(), "checkpoint.pth")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=14)
test_loader = DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
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
torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
model.eval()
accuracy = 0.0
total = 0.0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        # run the model on the test set to predict ilabels
        outputs = model(data)
        # the label with the highest energy will be our prediction
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        accuracy += (predicted == target).sum().item()
        print(predicted == target)
# compute the accuracy over all test images
accuracy = (100 * accuracy / total)
print(accuracy)