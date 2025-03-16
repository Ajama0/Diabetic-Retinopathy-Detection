import os
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms
from pathlib import Path

class DiabeticRetinopathyDataset(Dataset):

    def __init__(self, img_dir,annotations_file,tranform = None):
        #the directory point to the train images
        self.img_dir  = img_dir
        #this refers to the csv file representing image and labels, we can later create an instance and pass in any csv and dataset
        self.annotations_file = annotations_file
        
        #validate csv file before assignment
        self.img_labels= self.csv_check()
        #this allows us to perform augmentation and other techniques on our input(images)
        self.transform = tranform


        #check if the csv file exists
    def csv_check(self):
        #check to ensure the path file is valid
        if not os.path.isfile(self.annotations_file):
            raise FileNotFoundError(f"CSV file '{self.annotations_file}' not found.")
            
        #read file and throw an exception if any error occurs    
        try:
            read_file = pd.read_csv(self.annotations_file)
        except Exception as e:
            print(f"Error reading '{self.annotations_file}': {e}")

        #ensure csv file is not empty
        if read_file.empty:
            raise ValueError(f"File '{self.annotations_file}' is empty.")
            
        return read_file

        
    #implementing len method to return length of dataset, used when iterating over dataloader
    def __len__(self):
        return len(self.img_labels) 



    def __getitem__(self, idx):
        #Function to be implemented from dataset inheritance, allows us to access an image and label at a given index
        #in the csv file contains ex [0aafff9e:0] representing the image name and diagnosis
        #here we access the absolute path for each image per index using iloc which allows access to the first column value in each index
        img_path = os.path.join(self.img_dir,  f"{self.img_labels.iloc[idx,0]}.jpeg")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #now we want the label corresponding to that specific image at the index
        img_label = self.img_labels.iloc[idx,1]

        #check if any transformations are to be applied to each image
        #this will transform our np array into a tensor and normalize the image pixel from [0-255]
        if self.transform:
            image = self.transform(img)

        #one hot encoding will be performed after passing to to the dataloader
        return image, img_label
    



#we can then use the dataloader object and pass in our dataset
    



    

    
    


    









        
















































 
