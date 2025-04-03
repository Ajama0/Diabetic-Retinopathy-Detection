import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt


#resize, crop to remove noise(black borders)
#this flag allows to differentiate between dev dataset and final dataset



def crop_image(img, tolerance):
    flag_images = []
    """
    we return a boolean mask that returns 1,0 (T/F) for every pixel value that is greater or less than the tolerance
    the mask.any(1) returns a 1d boolean array that sets the row to true, if that row consists of a single pixel value that is greater than tol
    this way we dont delete any informative features. Only fully black rows and columns are deleted

    for rgb we need to crop each channel indivdually and stack them together

    we can also opt to use a circular crop
    """
    print(img.shape)
    if img.ndim == 2:
        mask = img > tolerance
        #return the index values and return the cropped image
        return img[np.ix_(mask.any(1), mask.any(0))]
        
    elif img.ndim == 3:
        #print("im here")
            
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # now for the coloured image we can return that boolean mask, returning to us a 2d spatial dimenstion of 1&0
        mask = gray_img > tolerance
        #we check if the image is fully black, if all rows are 0, meaning we didnt select any row from the mask, then return the image

            
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if(check_shape==0):
            flag_images.append(img)
        #otherwise if the image is not completely black, then for each RGB channel determine which rows and cols to keep that contain at least  a single pixel above the threshold
        else:
            #we return all rows and columns from each channel and stack them together
            img_c1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img_c2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img_c3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]

            #combine the channels
            #print(img_c1.shape,img_c2.shape,img_c3.shape)
            img = np.stack([img_c1,img_c2,img_c3],axis=-1)
            #print(img.shape)

            return img




def high_boost_filtering(img, sigmaX, resize):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image(image,tolerance=10)
    image = cv2.resize(image, (resize,resize))
    image = cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4,128)
   

    return image





def preprocess_and_save(img_dir, output_dir, df):
    os.makedirs(output_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image = os.path.join(img_dir,f"{row['image']}.jpeg")
        image = high_boost_filtering(image,sigmaX=10, resize=224)

        # image = np.clip(image, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        saved_path = os.path.join(output_dir,f"{row['image']}.jpeg")
        cv2.imwrite(saved_path,image)


"""before saving with imwrite"""
def preprocess_and_save_before(img_dir, df):
    for _, row in (df.sample(1).iterrows()):
        image = os.path.join(img_dir,f"{row['image']}.jpeg")
        image = high_boost_filtering(image,sigmaX=10, resize=512)
    
        plt.imshow(image)
        plt.title("before saving with cv2.imwrite")
        plt.show()


if __name__ == "__main__":
    load_dotenv()
    img_dir = os.getenv("DR_IMAGES_PATH")
    csv = os.getenv("DEV_CSV")
    df = pd.read_csv(csv)
    output_dir = os.getenv("DEV_IMAGES")

    preprocess_and_save(img_dir,output_dir, df)
    #preprocess_and_save_before(img_dir, df)



    

