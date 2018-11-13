#!/usr/bin/env python
# coding: utf-8

# Imports here
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
import argparse
import setup


# Argparser Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', action='store', dest='path', help='path of directory', required=True)
parser.add_argument('--save_dir', action='store', dest='cp_path', default='checkpoints/', help='path of checkpoint')
parser.add_argument('--arch', action='store', dest='arch', default='vgg16', choices={"vgg16", "densenet161"}, help='architecture of the network')
parser.add_argument('--learning_rate', action='store', nargs='?', default=0.001, type=float, dest='learning_rate', help='(float) learning rate of the network')
parser.add_argument('--epochs', action='store', dest='epochs', default=3, type=int, help='(int) number of epochs while training')
parser.add_argument('--hidden_units', action='store', nargs=2, default=[10240, 1024], dest='hidden_units', type=int,
                    help='Enter 2 hidden units of the network, input -> --hidden_units 256 256 | output-> [512, 256]')
parser.add_argument('--gpu', action='store_true', default=False, dest='boolean_t', help='Set a switch to use GPU')
results = parser.parse_args()


data_dir = results.path
checkpoint_path = results.cp_path
arch = results.arch
hidden_units = results.hidden_units
epochs = results.epochs
lr = results.learning_rate
gpu = results.boolean_t
print_every = 30

if gpu==True:
    using_gpu = torch.cuda.is_available()
    device = 'gpu'
    print('GPU On');
else:
    print('GPU Off');
    device = 'cpu'
    
    
# Loading Dataset
image_trainloader, image_testloader, image_valloader, image_trainset  = setup.loading_data(data_dir)
class_to_idx = image_trainset.class_to_idx

# Network Setup
model, input_size = setup.make_model(arch, hidden_units, lr)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

# Training Model
setup.my_DLM(model, image_trainloader, image_valloader, epochs, print_every, criterion, optimizer, device)

# Testing Model
setup.testing(model, image_testloader)

# Saving Checkpoint
setup.save_checkpoints(model, arch, lr, epochs, input_size, hidden_units, class_to_idx, checkpoint_path)