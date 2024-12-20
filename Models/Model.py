import os
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import dataloader, Dataset
from torchvision.io import read_image
from torchvision import transforms



#once data is processed set to False to save as pickle file
REBUILD_DATA = True
#path to the trainset images
path = "Dataset/raw/train_images"
#path to 
train_csv_path = "Dataset/raw/train.csv"
IMG_SIZE = 224
training_data = []

count={"No DR" :0 , "Mild DR": 0, "Moderate DR": 0, "Severe DR": 0, "Proliferative DR" : 0}

df = pd.read_csv(train_csv_path)
id_code, diagnosis= df["id_code"][0], df["diagnosis"][0]

print("id_code:", id_code , "diagnosis:" ,diagnosis)

print(df.head)

#access the value of diagnosis per index and incremenet our counter
for key,value in tqdm(df.iterrows(), total=len(df)):
    if value["diagnosis"] == 0:
        count["No DR"]+=1

    elif value["diagnosis"]==1:
        count["Mild DR"]+=1

    elif value["diagnosis"]==2:
        count["Moderate DR"]+=1

    elif value["diagnosis"]==3:
        count["Severe DR"]+=1

    elif value["diagnosis"] ==4:
        count["Proliferative DR"]+=1


print(count)
print(len(df))
print(sum(count.values()))
print(df[0][0])

#visualising the class imbalance with a bar chart
plt.figure(figsize=(10, 6))  
plt.xlabel("Retinopathy Gradings", fontsize=18)
plt.ylabel("Number of Samples", fontsize=18)
plt.title("Class Distribution in Diabetic Retinopathy Dataset", fontsize=20)


plt.bar(count.keys(), count.values())
#plt.show()

# Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(count.values(), labels=count.keys(), autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title("Class Distribution in Diabetic Retinopathy Dataset", fontsize=20)
#plt.show()


class DiabeticRetinopathyDataset(Dataset):

    def __init__(self, img_dir,annotations_file,tranform = None):
        #the directory point to the train images
        self.img_dir  = img_dir
        
        #this refers to the csv file representing image and labels, we can later create an instance and pass in any csv and dataset
        self.img_labels= annotations_file
        #this allows us to perform augmentation and other techniques on our input(images)
        self.transform = tranform
        

    
    #implementing len method to return length of dataset, used when iterating over dataloader
    def __len__(self):
        return len(self.img_labels) 



    def __getitem__(self, idx):
        #Function to be implemented from dataset inheritance, allows us to access an image and label at a given index
        #in the csv file contains ex [0aafff9e:0] representing the image name and diagnosis
        #here we access the absolute path for each image per index using iloc which allows access to the first column value in each index
        img_path = os.path.join(self.img_dir,  f"{self.img_labels.iloc[idx,0]}.png")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #now we want the label corresponding to that specific image at the index
        img_label = self.img_labels.iloc[idx,1]

        #check if any transformations are to be applied to each image
        if self.transform:
            image = self.transform(img)

        #one hot encoding will be performed after passing to to the dataloader
        return image, img_label
    









        
















































 
