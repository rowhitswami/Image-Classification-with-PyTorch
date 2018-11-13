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
import json
using_gpu = torch.cuda.is_available()

def loading_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    testval_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_trainset = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_testset = datasets.ImageFolder(test_dir, transform=testval_transforms)
    image_valset = datasets.ImageFolder(valid_dir, transform=testval_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    image_trainloader = torch.utils.data.DataLoader(image_trainset, batch_size=64, shuffle=True)
    image_testloader = torch.utils.data.DataLoader(image_testset, batch_size=64, shuffle=True)
    image_valloader = torch.utils.data.DataLoader(image_valset, batch_size=64, shuffle=True)
    
    return image_trainloader, image_testloader, image_valloader, image_trainset


# Build and train your network
# Freeze parameters so we don't backprop through them
def make_model(arch, hidden_units, lr):
    if arch=="vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088
    else:
        model = models.densenet161(pretrained=True)
        input_size = 2208
    output_size = 102
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('dropout',nn.Dropout(0.5)),
                              ('fc1', nn.Linear(input_size, hidden_units[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                              ('relu2', nn.ReLU()),
                              ('fc3', nn.Linear(hidden_units[1], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    

    return model, input_size


# Training the model
def my_DLM(model, image_trainloader, image_valloader, epochs, print_every, criterion, optimizer, device):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    if device=='gpu':
        model = model.to('cuda')
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(image_trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0
                for i, (inputs,labels) in enumerate(image_valloader):
                            optimizer.zero_grad()
                            inputs, labels = inputs.to('cuda') , labels.to('cuda')
                            model.to('cuda')
                            with torch.no_grad():    
                                outputs = model.forward(inputs)
                                validation_loss = criterion(outputs,labels)
                                ps = torch.exp(outputs).data
                                equality = (labels.data == ps.max(1)[1])
                                accuracy += equality.type_as(torch.FloatTensor()).mean()

                val_loss = validation_loss / len(image_valloader)
                train_ac = accuracy /len(image_valloader)



                print("Epoch: {}/{}... | ".format(e+1, epochs),
                      "Loss: {:.4f} | ".format(running_loss/print_every),
                      "Validation Loss {:.4f} | ".format(val_loss),
                      "Accuracy {:.4f}".format(train_ac))

                running_loss = 0



# Do validation on the test set
def testing(model, dataloader):
    model.eval()
    model.to('cuda')
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _ , prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels.data).sum().item()
        print('Accuracy on the test set: %d %%' % (100 * correct / total))   


# Save the checkpoint 
def save_checkpoints(model, arch, lr, epochs, input_size, hidden_units, class_to_idx, checkpoint_path):
    model.class_to_idx = class_to_idx
    state = {
            'structure' :arch,
            'learning_rate': lr,
            'epochs': epochs,
            'input_size': input_size,
            'hidden_units':hidden_units,
            'state_dict':model.state_dict(),
            'class_to_idx': model.class_to_idx
        }
    torch.save(state, checkpoint_path + 'command_checkpoint.pth')
    print('Checkpoint saved in ', checkpoint_path + 'command_checkpoint.pth')

    

# Write a function that loads a checkpoint and rebuilds the model
def loading_checkpoint(path):
    
    # Loading the parameters
    state = torch.load(path)
    lr = state['learning_rate']
    input_size = state['input_size']
    structure = state['structure']
    hidden_units = state['hidden_units']
    epochs = state['epochs']
    
    # Building the model from checkpoints
    model,_ = make_model(structure, hidden_units, lr)
    model.class_to_idx = state['class_to_idx']
    model.load_state_dict(state['state_dict'])
    return model


# Inference for classification
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
   
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = image_transforms(pil_image)
    return img

# Labeling
def labeling(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name


# Class Prediction
def predict(processed_image, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    model.eval()
    model.cpu()
    processed_image = processed_image.unsqueeze_(0)
    processed_image = processed_image.float()
    
    with torch.no_grad():
        output = model.forward(processed_image)
        probs, classes = torch.topk(input=output, k=topk)
        top_prob = probs.exp()

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in classes.cpu().numpy()[0]]
        
    return top_prob, top_classes