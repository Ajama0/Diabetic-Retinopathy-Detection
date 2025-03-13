import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import io

#resize, crop to remove noise(black borders)
#this flag allows to differentiate between dev dataset and final dataset
DEV_SET = True
class Crop:
    def __init__(self, df):
        self.df = df



    """
    images to be cropped will be stored in a new location
    dynamic cropping, because some images are taken at different angles, we dont crop from a fixed height and width
    """
    def crop_images(self, path, resize):
        
    

