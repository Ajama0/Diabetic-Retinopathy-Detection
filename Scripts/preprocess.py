import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import io

#resize, crop to remove noise(black borders)
#this flag allows to differentiate between dev dataset and final dataset
DEV_SET = True
class PreProcess:
    def __init__(self, df):
        self.df = df



    """
    images to be cropped will be stored in a new location
    """
    def crop_images(self,path_o, path_n, IMG_SIZE, cropx, cropy ):
        
        #returns a list of files in the directory
        dir = [i for i in os.listdir(path_o)]


        for i in dir:
            total = 0
            img = os.path.join(path_o,f"{i}.jpeg")
            img = cv2.imread(img)
            h, w, c = img.shape #(height, width, channels)
            startx = w//2-(cropx//2)
            starty = h//2-(cropy//2)
            img = img[starty:starty+cropy,startx:startx+cropx]
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            io.imsave(str(path_n + i), img)
            total += 1

    def normalize(X_train, flag=True):
        if(flag):
            X_train = X_train/255

    
    def DirectoryCheck(self, directory):

        if not os.path.exists(directory):
            os.makedirs(directory)



