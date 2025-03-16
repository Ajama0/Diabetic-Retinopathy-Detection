import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import io

#resize, crop to remove noise(black borders)
#this flag allows to differentiate between dev dataset and final dataset



def crop_image(img, tolerance=7):
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




def apply_unsharp_masking(img,resize, sigmaX):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image(image,tolerance=10)
    image = cv2.resize(image, (resize,resize))
    image=cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4,128)
    return image



"""the below is just for experiment purposes"""
# cropped_img = crop_image(os.path.join(path,dev_df.iloc[6]["image"]), tolerance=9)

# before_crop = os.path.join(path,dev_df.iloc[6]["image"])

# #the image is actually an rgb
# is_sample_coloured = os.path.join(path,dev_df.iloc[0]["image"])
# print(cv2.imread(is_sample_coloured).shape)



# before_crop = cv2.cvtColor(cv2.imread(before_crop), cv2.COLOR_RGB2GRAY)
# cv2.resize(before_crop, (224,224))
# plt.imshow(before_crop, cmap="gray")
# plt.show()

# cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
# plt.imshow(cropped_img_rgb)
# plt.show()



# def apply_clahe_rgb(img, clipLimit=2.0, tileGridSize=(8, 8)):
#     """
#     Applies CLAHE to an RGB image by converting to LAB color space,
#     equalizing the L channel, and converting back to RGB.
#     """
#     # Convert from RGB to LAB color space
#     lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)

#     # Create a CLAHE object
#     clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

#     # Apply CLAHE to the L (lightness) channel
#     cl = clahe.apply(l)

#     # Merge the CLAHE-enhanced L channel back with a and b
#     merged_lab = cv2.merge((cl, a, b))

#     # Convert LAB back to RGB
#     enhanced_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
#     return enhanced_img

    

