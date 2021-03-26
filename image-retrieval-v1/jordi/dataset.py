import os
import shutil

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import numpy as np

from config import DebugConfig

DEBUG = DebugConfig.DEBUG

class FashionProductDataset(Dataset):
    IMAGE_DIR_NAME = 'images'
    IMAGE_FORMAT = '.jpg'

    def __init__(self, base_dir, labels_df, transform=None):
        super().__init__()
        self.base_dir = base_dir
        self.images_path = os.path.join(base_dir, self.IMAGE_DIR_NAME)
        #There are rows with more than 10 columns. ex:6044
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        
        imageid,gender,masterCategory,subCategory,articleType,baseColour, \
            season,year,usage,productDisplayName,masterCategoryEncoded, \
            subCategoryEncoded,articleTypeEncoded,baseColourEncoded = self.labels_df.loc[idx, :]
        path = os.path.join(self.images_path, f"{imageid}.jpg")
        sample = Image.open(path).convert('RGB')
        if self.transform:
            sample = self.transform(sample)

        return sample,articleTypeEncoded

    def get_images_path(self):
        self.images_path
    
    def get_base_path(self):
        self.images_path

class DatasetManager():
    def __init__(self):
        pass

    def Validate_Images_DataFrame(self, df, imgid_colname, image_dir, img_format):
        delete_index = []
        for index, row in df.iterrows():
            imageid = row[imgid_colname]
            path = os.path.join(image_dir, str(imageid ) + img_format)
            if not os.path.isfile(path):
                delete_index.append(index)
        df.drop(df.index[delete_index],inplace=True)
        return df

    def EncodeColumns(self,df):
        #Encode any string columns: Need for training and convert to tensor
        le = LabelEncoder()
        df["masterCategoryEncoded"] =  le.fit_transform(df['masterCategory'])
        df["masterCategoryEncoded"] = df["masterCategoryEncoded"].astype('int64')
        df["subCategoryEncoded"] =  le.fit_transform(df['subCategory'])
        df["subCategoryEncoded"] = df["subCategoryEncoded"].astype('int64')
        df['articleTypeEncoded'] = le.fit_transform(df['articleType'])
        df["articleTypeEncoded"] = df["articleTypeEncoded"].astype('int64')
        df["baseColourEncoded"] =  le.fit_transform(df['baseColour'])
        df["baseColourEncoded"] = df["baseColourEncoded"].astype('int64')
        return df

    def split_dataset(self, dataset_base_dir, original_labels_file, process_dir, clean_process_dir=False, split_train_dir=False, train_size='divide', fixed_validate_test_size=0):

        if train_size == 'all':
            fixed_train_size = -1
        elif train_size == 'divide':
            fixed_train_size = 0
        else:
            fixed_train_size = int(train_size)
            fixed_validate_test_size = int(fixed_validate_test_size)

        base_dir = process_dir
        img_dir = os.path.join(dataset_base_dir, FashionProductDataset.IMAGE_DIR_NAME)
        img_format = FashionProductDataset.IMAGE_FORMAT

        #Validation    
        if not os.path.isfile(original_labels_file) or not os.access(original_labels_file, os.R_OK):    
            print('Labels file is missing or not readable : ' ,original_labels_file)
            return False

        if fixed_train_size > 0:
            if fixed_validate_test_size == 0:
                print('fixed_validate_test_size parameter must be defined')
                return False

        # ******  WARNING ******************
        #Delete all processed data directory
        if clean_process_dir:
            shutil.rmtree(base_dir)
        
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        #If train dataset exists -> load else create it
        if os.path.isfile(os.path.join(base_dir, "train_dataset.csv")):
            train_df = pd.read_csv(os.path.join(base_dir, "train_dataset.csv"), error_bad_lines=False)     
            if fixed_train_size >= 0:
                test_df = pd.read_csv(os.path.join(base_dir, "test_dataset.csv"), error_bad_lines=False)     
                validate_df = pd.read_csv(os.path.join(base_dir, "val_dataset.csv"), error_bad_lines=False)     
        else:
            labels_df = pd.read_csv(original_labels_file, error_bad_lines=False) # header=None, skiprows = 1

            #Validate images exists
            labels_df = self.Validate_Images_DataFrame(labels_df,"id",img_dir,img_format = '.jpg')

            ### FILTER ###
            #classes have minimum 100 images
            minimum_pictures_class = 100
            n_classes = np.sum(labels_df.articleType.value_counts().to_numpy() > minimum_pictures_class)
            if DEBUG: print('num classes ' + str(n_classes))
            classes = labels_df.articleType.value_counts().sort_values(ascending=False)[:n_classes]
            if DEBUG: print(classes)
            labels_df = labels_df[labels_df['articleType'].isin(classes.index)]

            ### ENCODE ###
            #Encode any string columns: Need for training and convert to tensor
            labels_df = self.EncodeColumns(labels_df)

            # Divide labels in train, test and validate
            if fixed_train_size > 0:
                train_df = labels_df.sample(fixed_train_size)
                #train_df = labels_df[:fixed_train_size]
                validate_df = labels_df.sample(fixed_validate_test_size)
                test_df= labels_df.sample(fixed_validate_test_size)
            elif fixed_train_size == -1:
                train_df = labels_df
            else:
                #Divide 60% - 20% - 20%
                train_df, validate_df, test_df = np.split(labels_df.sample(frac=1, random_state=42), 
                                                [int(.6*len(labels_df)), int(.8*len(labels_df))])


            # Save datasets
            train_df.to_csv(os.path.join(base_dir, "train_dataset.csv"),index=False)
            if not fixed_train_size == -1:
                validate_df.to_csv(os.path.join(base_dir, "val_dataset.csv"), index=False)
                test_df.to_csv(os.path.join(base_dir, "test_dataset.csv"), index=False)

            if split_train_dir:
                # Directories for our training, validation and test splits
                train_dir = os.path.join(base_dir, 'train')
                if not os.path.exists(train_dir):
                    os.mkdir(train_dir)
                validation_dir = os.path.join(base_dir, 'validation')
                if not os.path.exists(validation_dir):
                    os.mkdir(validation_dir)
                test_dir = os.path.join(base_dir, 'test')
                if not os.path.exists(test_dir):
                    os.mkdir(test_dir)    

                # Divide images for train, test and validate and copy in destination folder
                for index, row in train_df.iterrows():
                    src = os.path.join(original_image_dir, str(row['id']) + img_format)
                    if os.path.isfile(src):
                        dst = os.path.join(train_dir, str(row['id']) + img_format)
                        shutil.copyfile(src, dst)
                    else:        
                        train_df.drop(index, inplace=True)

                for index, row in validate_df.iterrows():
                    src = os.path.join(original_image_dir, str(row['id']) + img_format)
                    if os.path.isfile(src):
                        dst = os.path.join(validation_dir, str(row['id']) + img_format)
                        shutil.copyfile(src, dst)        
                    else:        
                        validate_df.drop(index, inplace=True)

                for index, row in test_df.iterrows():
                    src = os.path.join(original_image_dir, str(row['id']) + img_format)
                    if os.path.isfile(src):
                        dst = os.path.join(test_dir, str(row['id']) + img_format)
                        shutil.copyfile(src, dst)        
                    else:        
                        test_df.drop(index, inplace=True)

        if DEBUG: print('Total training images:', train_df.shape[0])
        if not fixed_train_size == -1:
            if DEBUG: print('Total test images:', test_df.shape[0])
            if DEBUG: print('Total validation images:', validate_df.shape[0])
            return train_df, test_df, validate_df
        else:
            return train_df, None, None

