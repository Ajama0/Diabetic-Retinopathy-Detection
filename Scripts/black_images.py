import os
from tqdm import tqdm
import numpy as np
import cv2
#import the crop file



def calculate_black_images(img):
    height, width, _  = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean = np.array(np.mean(img))
    total_pixels = height * width
    black_pixels = total_pixels - cv2.countNonZero(img)
    black_pct = (black_pixels/total_pixels) * 100
    return black_pct, mean


"""
this function is used to remove the images that are still majority black even after we cropped and have done mask based cropping
"""
def remove_ungradable_images(dataset,path):
    black_images = []
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
        image_path = os.path.join(path,f"{row['image']}.jpeg")
        image = cv2.imread(image_path)
        #now we can make sure only the fundus region beig black is considered as a black image and not its borders
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = preprocess.crop_image(image, tolerance=10)
        black_pixel_percentage,mean = calculate_black_images(image)
        if(black_pixel_percentage>30) or (mean<30):
            #we can inspect these images
            black_images.append(row['image'])
        






