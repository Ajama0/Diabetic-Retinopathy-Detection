
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
    if labels_csv is not None:
        df = pd.read_csv(labels_csv)
    train_df, test_df = train_test_split(df,test_size=0.1, random_state=42, stratify=df['level'])

    
else:
    #otherwise if this is production
    img_dir = os.getenv("DR_IMAGES")
    labels_csv = os.getenv("DR_LABELS_PATH")
    if labels_csv is not None:
        df = pd.read_csv(labels_csv)
    
    #now we can split the data for the production set
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['level'])
    test_df, val_df = train_test_split(temp_df, test_size=0.15, random_state=42, stratify=df["level"])
    




train_loader = customDataLoader(img_dir, train_df, transforms.Compose([
    transforms.ToTensor(), #each image will be a 4d tensor
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])]))

test_loader =  customDataLoader(img_dir, test_df, transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])]))

#only create the val loader when we are in production mode
if not DEVELOPMENT:
    val_loader = customDataLoader(img_dir, val_df, transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ]))
    val = DataLoader(val_loader, batch_size=32, shuffle=True)


train = DataLoader(dataset=train_loader, batch_size=32, shuffle=True)
test = DataLoader(dataset=test_loader, batch_size=32, shuffle=True)




device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device being used is {device}")


weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

print(model.summary())

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

        outputs = model(X)
        loss_function = loss(outputs,y)
        train_loss += loss_function.item()

        """
        here we monitor the training accuracy and loss during training, if the training accuracy isnt improving then we can halt
        """


        #softmax activation in the last layer to have values over a P.D
        predictions = torch.argmax(torch.softmax(predictions,dim=1), dim=1)

        # Count correct predictions in the batch:
        batch_correct = (predictions == y).sum().item() / len(y) #accuracy of the batch between 0-1
        matches += batch_correct
        total += y.size(0)
        loss_function.backward()
        optimizer.step()


    #the len(dataloader) is the len(dataset)/32
    average_loss = train_loss / len(dataloader)  
    train_acc = matches / len(dataloader)
    #the average loss and training accuracy for a batch within each EPOCH
    return average_loss, train_acc



def validate(model, dataloader, loss_fn):
    pass

def test(dataloader, model, loss_fn):
    model.eval()  # Sets the model for evaluation.

    total = 0
    correct = 0
    running_loss = 0

    with torch.no_grad():  # No need to calculate the gradients.

        for _,  batch  in enumerate(dataloader):
            X, y = batch
            output = model(X.to(device))  # model's output.
            loss = loss_fn(output, y.to(device)) # loss calculation.
            running_loss += loss.item()

            total += y.size(0)
            predictions = output.argmax(dim=1).cpu().detach()
            correct += (predictions == y).sum().item()

    avg_loss = running_loss / len(dataloader)  # Average loss per batch.

    print(f'\nValidation Loss = {avg_loss:.6f}', end='\t')
    print(f'Accuracy on Validation set = {100 * (correct / total):.6f}% [{correct}/{total}]')  # Prints the Accuracy.

    return avg_loss



        

        





















