import os
import shutil
import numpy as np
import pandas as pd

def Prepare_Data(original_dataset_dir, original_labels_file, process_dir, img_format = '.jpg',clean_process_dir=False, split_train_dir=False, fixed_train_size=0, fixed_validate_test_size=0):

    base_dir = process_dir

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
    
    # Divide labels in train, test and validate
    labels_df = pd.read_csv(original_labels_file, error_bad_lines=False)

    if fixed_train_size > 0:
        train_df = labels_df.sample(fixed_train_size)
        validate_df = labels_df.sample(fixed_validate_test_size)
        test_df= labels_df.sample(fixed_validate_test_size)
    else:
        train_df, validate_df, test_df = np.split(labels_df.sample(frac=1, random_state=42), 
                                            [int(.6*len(labels_df)), int(.8*len(labels_df))])

    # Save datasets
    train_df.to_csv(os.path.join(base_dir, "train_dataset.csv"),index=False)
    validate_df.to_csv(os.path.join(base_dir, "val_dataset.csv"),index=False)
    test_df.to_csv(os.path.join(base_dir, "test_dataset.csv"),index=False)

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

    print('Total training images:', train_df.shape[0])
    print('Total test images:', test_df.shape[0])
    print('Total validation images:', validate_df.shape[0])    

    return train_df, test_df, validate_df
