
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import torch.utils.data.dataloader 
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import pandas as pd
import Models.customDataLoader as customDataLoader
from torch.utils.data import DataLoader

DEVELOPMENT = True
if DEVELOPMENT:
    """
    we have a dev csv which is the currently 10% of the data, lets split into train and test so we can pass to the dataloader
    """
    load_dotenv()
    labels_csv = os.getenv("DR_DEV")
    img_dir = os.getenv("DEV_IMAGES")
else:
    labels_csv = os.getenv("DR_LABELS_PATH")
    img_dir = os.getenv("DR_IMAGES")

if labels_csv is not None:
    dataset = pd.read_csv(labels_csv)
train_df, test_df = train_test_split(labels_csv,test_size=0.1, random_state=42, stratify=dataset['level'])



train_loader = customDataLoader(img_dir, train_df, transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])]))

test_loader =  customDataLoader(img_dir, test_df, transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])]))


train = DataLoader(dataset=train_loader, batch_size=32, shuffle=True)
test = DataLoader(dataset=test_loader, batch_size=32, shuffle=True)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device being used is {device}")


weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

"""
we'll freeze the base feature extraction layers for N epochs whilst only training the head classifier
after some epochs some layers will be unfrozen and we can start optimizing params to be better suited to my dataset

below shows we freeze the base layers and add the classifier layer
"""

for parameters in model.features.parameters():
    parameters.requires_grad(False)
    
print(f"number of in features for the last layer is: {model.classifier} ")
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), #probability of neurons being set to zero is 0.2 in the last layer
    torch.nn.Linear(in_features=1280, 
                    out_features=5, # as we only have 5 classes
                    bias=True)).to(device)




def train(model, dataloader, optimizer, loss):
    model.train() #sets the model to training mode
    for _, batch in enumerate(dataloader):
        X,y = batch
        #zero the gradients
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)

        outputs = model(X.to(device))
        loss_function = loss(outputs,y)
        train_loss += loss_function.item()


        """
        here we monitor the training accuracy and loss during training, if the training accuracy isnt improving then we can halt
        """
        predictions = torch.argmax(outputs,dim=1)
         # Count correct predictions in the batch:
        batch_correct = (predictions == y.to(device)).sum().item()
        matches += batch_correct
        total += y.size(0)
        loss_function.backward()
        optimizer.step()


    
    average_loss = train_loss / len(dataloader)  
    train_acc = matches / total
     #the average loss for a single batch and the training accuracy for a full pass of the dataset
    return average_loss, train_acc







        

        





















