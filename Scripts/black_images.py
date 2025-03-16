import os
from tqdm import tqdm
import numpy as np
import cv2
#import the crop file

"""
as we can see now that we cropped the image, 
we can actually see which images are black based on the fundus instead of the surrounding black borders.
However based on the literature, removal of ungradable images does not help in performance

paper - Rakhlin A. Diabetic Retinopathy detection through integration of Deep Learning classification frame-
work. bioRxiv. 2017;. 
"""
def calculate_black_images(img):
    height, width, _  = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean = np.array(np.mean(img))
    total_pixels = height * width
    black_pixels = total_pixels - cv2.countNonZero(img)
    black_pct = (black_pixels/total_pixels) * 100
    return black_pct, mean


black_images = []
def remove_ungradable_images(dataset,path):
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
        image_path = os.path.join(path,f"{row['image']}.jpeg")
        image = cv2.imread(image_path)
        #now we can make sure only the fundus region beig black is considered as a black image and not its borders
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image(image, tolerance=10)
        black_pixel_percentage,mean = remove_ungradable_images(image)
        if(black_pixel_percentage>30) or (mean<30):
            #we can inspect these images
            black_images.append(np.array(image))
        






