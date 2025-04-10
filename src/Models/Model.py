
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
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
from tqdm import tqdm
from Models.customDataset import DiabeticRetinopathyDataset
from torch.utils.data import DataLoader
import copy


DEVELOPMENT = True
if DEVELOPMENT:
    """
    we have a dev csv which is the currently 10% of the data, lets split into train and test so we can pass to the dataloader
    """
    load_dotenv()
    labels_csv = os.getenv("DEV_CSV")
    img_dir = os.getenv("DEV_IMAGES")
    if labels_csv is not None:
        df = pd.read_csv(labels_csv)
    train_df, test_df = train_test_split(df,test_size=0.2, random_state=42, stratify=df['level'])

    
else:
    #otherwise if this is production
    img_dir = os.getenv("DR_IMAGES_PATH")
    labels_csv = os.getenv("DR_LABELS_PATH")
    if labels_csv is not None:
        df = pd.read_csv(labels_csv)
    
    #now we can split the data for the production set
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['level'])
    test_df, val_df = train_test_split(temp_df, test_size=0.15, random_state=42, stratify=df["level"])
    




train_dataset = DiabeticRetinopathyDataset(img_dir, train_df, transforms.Compose([
    transforms.ToTensor(), #each image will be a 4d tensor
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])]))

test_dataset =  DiabeticRetinopathyDataset(img_dir, test_df, transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])]))

#only create the val loader when we are in production mode
if not DEVELOPMENT:
    val_dataset = DiabeticRetinopathyDataset(img_dir, val_df, transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ]))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2)


train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=2)




device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device being used is {device}")


weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

print(model.features)

"""
we'll freeze the base feature extraction layers for N epochs whilst only training the head classifier
after some epochs some layers will be unfrozen and we can start optimizing params to be better suited to my dataset

below shows we freeze the base layers and add the classifier layer
"""

for parameters in model.features.parameters():
    parameters.requires_grad=False
    
#print(f"number of in features for the last layer is: {model.classifier} ")



model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), #probability of neurons being set to zero is 0.2 in the last layer
    torch.nn.Linear(in_features=1280, 
                    out_features=5, # as we only have 5 classes
                    bias=True)).to(device)


def train(model, dataloader, optimizer, loss):
    start = time.time()
    model.train() #sets the model to training mode
    matches = 0
    total = 0
    train_loss = 0.0
    for _, batch in enumerate(tqdm(dataloader, desc="training", leave=False)):
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


        #calculate the predictions by taking largest logit value in output layer
        predictions = torch.argmax(torch.softmax(outputs,dim=1), dim=1)

        # Count correct predictions in the batch:
        batch_correct = (predictions == y).sum().item()
        matches += batch_correct
        total += y.size(0)
        loss_function.backward()
        optimizer.step()


    #the len(dataloader) is the len(dataset)/32
    average_loss = train_loss / len(dataloader)  
    train_acc = matches / total
    time_elapsed = (time.time() - start) / 60
    #the average lossfor a batch within each EPOCH , the accuracy and the total training time for each epoch
    return average_loss, train_acc, time_elapsed



def test(dataloader, model, loss_fn):
    model.eval()  # Sets the model for evaluation mode and ensures dropout layer isnt activated

    total = 0
    correct = 0
    running_loss = 0

    # Lists to store all predictions and ground truth
    all_predictions = []
    all_labels = []


    with torch.inference_mode():  # No need to calculate the gradients.

        for _,  batch  in enumerate(tqdm(dataloader, desc="testing", leave=False)):
            X, y = batch
            output = model(X.to(device))  # model's output.
            loss = loss_fn(output, y.to(device)) # loss calculation.
            running_loss += loss.item()

            total += y.size(0)
            predictions = output.argmax(dim=1)
            correct += (predictions == y).sum().item()

            #here we basically move to cpu (scikit learn doesnt work with gpu's) & add the predictions as a flat list
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())


    test_loss = running_loss / len(dataloader)  # Average loss per batch for each EPOCH .
    test_accuracy = correct / total

    #print(f'\ntest Loss = {avg_loss:.6f}', end='\t')
    #print(f'Accuracy on test set = {100 * (correct / total):.6f}% [{correct}/{total}]')  

    """
    Confusion Matrix and other matrix to be added
    """

    return test_loss, test_accuracy, all_predictions, all_labels


def validate(model, dataloader, loss_fn):

    """
    ARGS:
    Model - Represents the current pretrained model being used
    dataloader - the val dataloader that contains the val instances
    loss_fn - loss funct to see model performance
    """
    running_acc = 0.0
    running_loss = 0.0
    total = 0

    since = time.time()
    model.eval() #set the model to eval mode
    with torch.inference_mode():
        for _ , batch in enumerate(tqdm(dataloader, desc="validation")):
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs,y)
            running_loss+= loss.item()

            predictions = torch.argmax(torch.softmax(outputs,dim=1),dim=1)
            val_matches = (predictions == y).sum().item()
            running_acc+=val_matches
            #we can calculate validation accuracy by using the total 
            total +=y.size(0)

           

        val_acc = running_acc / total
        val_loss = running_loss / len(dataloader)

        time_elapsed = (time.time() - since) / 60
        return val_loss, val_acc, time_elapsed
        


def visualizations(all_predictions, all_labels, results=None, class_names=None):
    """
    Generate visualizations for model performance
    
    Args:
        all_predictions: List of model predictions
        all_labels: List of ground truth labels
        results: Dictionary containing training/validation metrics
        class_names: List of class names (default is None, will use numeric labels)
    """
    # If no class names provided, use numeric labels
    if class_names is None:
        class_names = [str(i) for i in range(5)] 
    
    
    cm = confusion_matrix(all_labels, all_predictions)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    #training/validation curves 
    if results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        
        x_values = results.get('epoch', range(1, len(results['train_acc']) + 1))
        
        #accuracy
        ax1.plot(x_values, results['train_acc'], label='Training Accuracy')
        if results['val_acc']:  
            ax1.plot(x_values[:len(results['val_acc'])], results['val_acc'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # loss
        ax2.plot(x_values, results['train_loss'], label='Training Loss')
        if results['val_loss']:  
            ax2.plot(x_values[:len(results['val_loss'])], results['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()
        


def model_comparions(model):
    """
    pass in a set model and compare the results. 
    """
    pass



def main(model, scheduler, EPOCHS, train_dataloader, use_val_dataloader:bool, optimizer, criterion ,val_dataloader = None):

    """
    The main function which is used during training, we also define the scheduler to decay the learning rate
    """
    
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = 10
    best_model_wts  = copy.deepcopy(model.state_dict())

    results ={
            "epoch" : [],
            "train_acc" : [],
            "train_loss" :[],
            "train_time" :[],
            "val_acc" : [],
            "val_loss":[],
            "val_time" : []
}

    for epoch in tqdm(range(EPOCHS), desc="Epoch", leave=False):
        
        """
        deep copy the model with the best parameters after computing the updated best accuracy in val set
        """
        train_loss, train_accuracy, time_elapsed = train(model=model, dataloader=train_dataloader,
         optimizer=optimizer, loss=criterion)
        

        #scheduler to decay learning rate
        scheduler.step()


        #if we are using the validation set - meaning were using the full dataset and not the 10% of data (for quick prototype)
        if use_val_dataloader and val_dataloader is not None:
            val_loss, val_accuracy, val_time_elapsed = validate(model, dataloader=val_dataloader, loss_fn=criterion)

            
            #early stopping with val loss
            epoch_loss = val_loss
            if(epoch_loss<best_val_loss):
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                #reset the value if the loss starts to improve
                epochs_without_improvement = 0

            else:
                epochs_without_improvement+=1
                print(f"loss didnt improve at : {epoch+1}/{EPOCHS}")


            #if for more than 10 epochs, the loss doesnt improve(decrease) then we can copy the learnable parameters and stop
            if (epochs_without_improvement>=patience): 
               print("Early stopping triggered!!!")
               break

                     
            #here we track the metrics for later visualizations 
            results.get("val_acc").append(val_accuracy)
            results.get("val_loss").append(val_loss)
            results.get("val_time").append(val_time_elapsed)

             
            print(f"Epoch {epoch+1}/{EPOCHS}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Training time: {time_elapsed:.2f} min, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
            f"Val time: {val_time_elapsed:.2f} min")

        else:
            #the val accuracy is not needed here as we are not using a val set.
            print(f"Epoch {epoch+1}/{EPOCHS}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Training time: {time_elapsed}")


        #we always track these results - regardless of val set usage or not
        results.get('epoch').append(epoch+1)
        results.get("train_acc").append(train_accuracy)
        results.get("train_loss").append(train_loss)
        results.get("train_time").append(time_elapsed)

 
    #load the best weights into the model(classifier layer) and save it for later retrieval
    if use_val_dataloader:
        model.load_state_dict(best_model_wts)
        torch.save(best_model_wts, 'best_model.pth') #saves the model locally





    print("-------------------------------------------------------------------")
    print("running it through the test set")
    #once the training loop is completed, we begin the testing. 
    test_loss, test_acc, all_predictions, all_labels = test(test_loader, model=model, loss_fn=criterion)
    print(f"test_loss : {test_loss:.4f}")
    print(f"test accuracy : {test_acc:.4f}")

    class_names = ['No Dr', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    visualizations(all_predictions=all_predictions, all_labels=all_labels, results=results, class_names=class_names)

    
        

   


if __name__ == "__main__":

    #define the optimizer
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    #defines our loss function
    criterion = nn.CrossEntropyLoss()

    """
    This decays the learning rate to 10% of the previous value per N epochs which is 7 epochs in our case
    """
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.1)
    Num_EPOCHS = 20

    if DEVELOPMENT:
        main(model, scheduler=scheduler, EPOCHS=Num_EPOCHS, train_dataloader=train_loader, use_val_dataloader=False,
        optimizer=optimizer, criterion=criterion)
    
    else:
        main(model, scheduler=scheduler, EPOCHS=Num_EPOCHS, train_dataloader=train_loader, use_val_dataloader=True, 
        optimizer=optimizer, criterion=criterion, val_dataloader=val_loader)

    
