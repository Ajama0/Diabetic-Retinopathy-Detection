
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os

"""
This py file will be used to convert the classes into binary classification

- Classes will be classified into Normal and DR

Normal being class 0

and DR consisting of severity gradings MIld, Moderate, Severe, PDF. (class 1,2,3,4)

"""


def image_exists(img_code):
    img_path = os.path.join(os.getenv('PREPROCESSED_IMAGES'),f"{img_code}.jpeg")
    return os.path.exists(img_path)




def binary_classification(df, output_csv):

    """as we removed some images during the preprocessing step, lets ensure we also remove them in our new binary csv
        we call image_exsits to filter out the df first
    """

    filtered_df = df[df['image'].apply(image_exists)]
    print("filtered the csv")

    df_binary = filtered_df

    if 'level' in df_binary.columns:
        
        #if value of label is greate then 0 set the label for that image to 1, otherwise leave as 0
        try:

            df_binary['level'] = df_binary['level'].apply(lambda x : 1 if x>0 else 0)
            df_binary.to_csv(output_csv, index=False)
            print(f"Binary classification saved to {output_csv}")

        except Exception as e:
            print(f"excetpion occured: {e}")    


    

def offline_augmentation(df, output_dir):
    pass







if __name__ == '__main__':
    load_dotenv()
    multi_csv = os.getenv("DR_LABELS_PATH")

    if multi_csv is not None:
        multi_df = pd.read_csv(multi_csv)

    
    binary_output_csv = "../binary_classification.csv"

    binary_classification(multi_df, binary_output_csv)
