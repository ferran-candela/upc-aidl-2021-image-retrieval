import os, shutil
import pandas as pd

def create_directory(base_dir, name):
    directory = os.path.join(base_dir, name)
    if not os.path.exists(directory):
         os.mkdir(directory)
    return directory


def divide_images_into_df(dataframe, original_image_dir, directory):
    for index, row in dataframe.iterrows():
        src = os.path.join(original_image_dir, str(row['id']) + ".jpg")
        if os.path.isfile(src):
            dst = os.path.join(directory, str(row['id']) + ".jpg")
            shutil.copyfile(src, dst)
        else:        
            dataframe.drop(index, inplace=True)
