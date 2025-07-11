import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv




def scaleRadius(img, scale):
    k = img.shape[0]//2
    x = img[int(k), :, :].sum(1)
    r = (x>x.mean()/10).sum()//2
    if r == 0:
        r = 1
    s = scale*1.0/r
    if(s>5):
        #if the scaling factor is greater than 5, (essentially enlargment of the image by 5x) we return None
        return None
    return cv2.resize(img, (0,0), fx=s, fy=s)



def process_single_image(image_path, scale):
    # Read image
    a = cv2.imread(image_path)

    if a is None:
        #if the image is None or null
        print(f"Could not read image: {image_path}")
        with open(os.getenv("CORRUPTED_IMAGES"), "a") as f:
            f.write(f"{image_path}\n")
        return None, None
        
   
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    
    # Scale radius
    a = scaleRadius(a, scale)

    if a is None:
        #if the scaling factor is too high
        print(f"Image has problematic scaling factor: {image_path}")
        with open(os.getenv("CORRUPTED_IMAGES"), "a") as f:
            f.write(f"{image_path}\n")
        return None, None

    
    # Create mask
    b = np.zeros(a.shape)
    x = a.shape[1] / 2
    y = a.shape[0] / 2
    center_coordinates = (int(x), int(y))
    cv2.circle(b, center_coordinates, int(scale * 0.9), (1, 1, 1), -1, 8, 0)
    
    # Apply high-boost filtering with masking
    aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128) * b + 0 * (1 - b)
    return a, aa
    



def crop_image(img, tolerance):

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)


    flag_images = []
    """
    we return a boolean mask that returns 1,0 (T/F) for every pixel value that is greater or less than the tolerance
    the mask.any(1) returns a 1d boolean array that sets the row to true, if that row consists of a single pixel value that is greater than tol
    this way we dont delete any informative features. Only fully black rows and columns are deleted

    for rgb we need to crop each channel indivdually and stack them together

    we can also opt to use a circular crop
    """
    
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


if __name__ == "__main__":
    load_dotenv()

    BIRUNI = True
    if BIRUNI:
        labels_csv = os.getenv("BIRUNI_LABELS_PATH")
        images = os.getenv("BIRUNI_IMAGES_PATH")
        output_dirs = os.getenv("BIRUNI_PREPROCESSED_IMAGES")
    
    else:
        labels_csv = os.getenv("DR_LABELS_PATH")
        images = os.getenv("DR_IMAGES_PATH")
        output_dirs = os.getenv("PREPROCESSED_IMAGES")




    if labels_csv is not None:
        df = pd.read_csv(labels_csv, dtype={"image" : str})
    else:
        print("csv is null ")    

    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try :
            img_path = os.path.join(images,f"{row['image']}.jpg")
            scaled_img, preprocessed = process_single_image(img_path, scale = 500)
            #this returns the preprocessed image which was scaled + high boost filtering
            if(scaled_img is None and preprocessed is None):
                continue
            image = crop_image(preprocessed, tolerance=7)

            saved_path = os.path.join(output_dirs,f"{row['image']}.jpg")
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(saved_path,image)

        except Exception as e:
            print(f"Error preocessing image : {row['image']}.jpg: {e}")
            with open(os.getenv("CORRUPTED_IMAGES"), "a") as f:
                f.write(f"{row['image']}.jpeg\n")
                #we save corruped images to a txt
            continue
     
       
       

