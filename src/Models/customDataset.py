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

    def __init__(self, img_dir,annotations_file,transform):


        """
        Args:
            img_dir (str): Directory with all the images.
            annotations (str or pd.DataFrame): Path to the CSV file or a DataFrame containing image filenames and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir  = img_dir
        #this refers to the csv file representing image and labels, we can later create an instance and pass in any csv and dataset
        self.annotations_file = annotations_file
        
        #validate csv file before assignment
        self.img_labels= self.csv_check()
        #this allows us to perform augmentation and other techniques on our input(images)
        self.transform = transform


        #check if the csv file exists
    def csv_check(self):
        if isinstance(self.annotations_file, pd.DataFrame):
            return self.annotations_file
        else:
            if not os.path.isfile(self.annotations_file):
                raise FileNotFoundError(f"CSV file '{self.annotations_file}' not found.")
            
            #read file and throw an exception if any error occurs    
            try:
                read_file = pd.read_csv(self.annotations_file)
            except Exception as e:
                print(f"Error reading '{self.annotations_file}': {e}")
            
        return read_file

        

    def __len__(self):
        return len(self.img_labels) 



    def __getitem__(self, idx):
        #Function to be implemented from dataset inheritance, allows us to access an image and label at a given index
        #in the csv file contains ex [0aafff9e:0] representing the image name and diagnosis
        #here we access the absolute path for each image per index using iloc which allows access to the first column value in each index
        img_path = os.path.join(self.img_dir,  f"{self.img_labels.iloc[idx,0]}.jpeg")
        """
        because were only using 10% of the data, the images will be mapped to each row of our dev csv
        """
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #now we want the label corresponding to that specific image at the index
        img_label = self.img_labels.iloc[idx,1]

    
        if self.transform:
            image = self.transform(img)

        return (image, img_label)
    
