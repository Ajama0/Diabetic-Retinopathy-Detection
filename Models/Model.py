import os
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import dataloader, dataset


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

#visualising the class imbalance
plt.figure(figsize=(10, 6))  
plt.xlabel("Retinopathy Gradings", fontsize=18)
plt.ylabel("Number of Samples", fontsize=18)
plt.title("Class Distribution in Diabetic Retinopathy Dataset", fontsize=20)

plt.bar(count.keys(), count.values())
plt.show()

































 
