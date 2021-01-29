import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):

    def __init__(self, images_path, labels_df, transform=None):
        super().__init__()
        self.images_path = images_path
        #There are rows with more than 10 columns. ex:6044
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        
        imageid,gender,masterCategory,subCategory,articleType,baseColour,season,year,usage,productDisplayName = self.labels_df.loc[idx, :]
        path = os.path.join(self.images_path, f"{imageid}.jpg")
        sample = Image.open(path)
        if self.transform:
            sample = self.transform(sample)
        return sample,masterCategory
