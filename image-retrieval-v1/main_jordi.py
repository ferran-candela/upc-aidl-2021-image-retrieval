import os, shutil
import torch
from torch.utils.data import DataLoader
from dataset_jordi import MyDataset
#from model import MyModel
#from utils import accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16

import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# The path to the directory where the original dataset was uncompressed
original_dataset_dir = '/home/manager/upcschool-ai/data/FashionProduct/'
original_image_dir = os.path.join(original_dataset_dir, 'images')
original_labels_file = original_dataset_dir + 'styles.csv'

# The directory where we will store our smaller dataset
base_dir = '/home/manager/upcschool-ai/data/FashionProduct/processed_datalab'

#Delete all processed data directory
shutil.rmtree(base_dir)

if not os.path.exists(base_dir):
    os.mkdir(base_dir)


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

# Divide labels in train, test and validate
labels_df = pd.read_csv(original_labels_file, error_bad_lines=False)
train_df, validate_df, test_df = np.split(labels_df.sample(frac=1, random_state=42), 
                                        [int(.6*len(labels_df)), int(.8*len(labels_df))])
# Divide images for train, test and validate and copy in destination folder
for index, row in train_df.iterrows():
    src = os.path.join(original_image_dir, str(row['id']) + ".jpg")
    if os.path.isfile(src):
        dst = os.path.join(train_dir, str(row['id']) + ".jpg")
        shutil.copyfile(src, dst)
    else:        
        train_df.drop(index, inplace=True)

for index, row in validate_df.iterrows():
    src = os.path.join(original_image_dir, str(row['id']) + ".jpg")
    if os.path.isfile(src):
        dst = os.path.join(validation_dir, str(row['id']) + ".jpg")
        shutil.copyfile(src, dst)        
    else:        
        validate_df.drop(index, inplace=True)

for index, row in test_df.iterrows():
    src = os.path.join(original_image_dir, str(row['id']) + ".jpg")
    if os.path.isfile(src):
        dst = os.path.join(test_dir, str(row['id']) + ".jpg")
        shutil.copyfile(src, dst)        
    else:        
        test_df.drop(index, inplace=True)


#num_files = len([f for f in os.listdir(original_image_dir)if os.path.isfile(os.path.join(original_image_dir, f))])

# Divide images for train, test and validate

#train_size = int(0.6 * num_files)
#test_size = int(0.2 * num_files)
#validate_size = num_files - train_size - test_size

# parent_list = os.listdir(original_image_dir)
# count =0
# for child in parent_list:
#     if count < train_size:
#         folder = train_dir
#     elif count >= train_size and count < train_size + test_size:
#         folder = test_dir
#     else:
#         folder = validation_dir
#     src = os.path.join(original_image_dir, child)
#     dst = os.path.join(folder, child)
#     shutil.copyfile(src, dst)        
#     count = count+1


print('total training images:', len(os.listdir(train_dir)))
print('total test images:', len(os.listdir(test_dir)))
print('total validation images:', len(os.listdir(validation_dir)))

#Test load train dataset and show image
ds = MyDataset(original_image_dir,labels_df)
print('Number of images', len(ds))
plt.figure(figsize=(8, 10))
plt.imshow(ds[0])
plt.xticks([]); plt.yticks([]); plt.grid(False)


#Data preprocessing
# images = 60x80 pixels
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128-32),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

batch_size = 64

train_dataset = MyDataset(train_dir,train_df,transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = MyDataset(validation_dir,validate_df,transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)        

print(train_dataset [0].shape)

# for data_batch, labels_batch in train_loader:
#     print('data batch shape:', data_batch.shape)
#     print('labels batch shape:', labels_batch.shape)
#     break



#VGG16 network, trained on ImageNet
pretrained_model = vgg16(pretrained=True)

# It is very important to put the network into eval mode before extracting features! This 
# turns off things like dropout and using batch statistics in batch norm.
pretrained_model.eval()
pretrained_model.to(device)

# takes the first part of the model, to use it as a feature_extractor
feature_extractor = pretrained_model.features
feature_extractor    

#The method we will use consists in running the convolutional base over our dataset, recording its output to a Numpy array on disk, 
# then using this data as input to a standalone densely-connected classifier. This solution is very fast and cheap to run, because 
# it only requires running the convolutional base once for every input image, and the convolutional base is by far the most expensive 
# part of the pipeline. However, for the exact same reason, this technique would not allow us to leverage data augmentation at all.

# def extract_features(path):
#     transform = transforms.Compose([
#                                 transforms.Resize(150),
#                                 transforms.CenterCrop(150),
#                                 transforms.ToTensor(),
#     ])
#     dataset = MyDataset(path, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=100, shuffle=False) #, num_workers=4
#     features = []
#     labels = []
#     with torch.no_grad():
#         for image_batch, label_batch in dataloader:
#             image_batch, label_batch = image_batch.to(device), label_batch.to(device)
#             features_batch = feature_extractor(image_batch)
#             features.append(features_batch)
#             labels.append(label_batch)
#     features_tensor = torch.cat(features, dim=0)
#     labels_tensor = torch.cat(labels, dim=0)

#     return features_tensor, labels_tensor


# train_features, train_labels = extract_features(train_dir)

#The extracted features are currently of shape (samples, 512, 4, 4). We will feed them to a densely-connected classifier, so first we must flatten them to (samples, 8192):

# train_features = np.reshape(train_features, (-1, 4 * 4 * 512))
# validation_features = np.reshape(validation_features, (-1, 4 * 4 * 512))

# densely-connected classifier 
# feature_classifier = nn.Sequential(
#     nn.Linear(4x4x512, 256),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(256, 1),
#     nn.Sigmoid()
# )


# feature_classifier.to(device)

# optimizer = optim.Adam(feature_classifier.parameters(), lr=0.001)
# criterion = nn.BCELoss()

# from torch.utils.data import TensorDataset

# batch_size = 64
# train_features_dataset = TensorDataset(torch.tensor(train_features), torch.tensor(train_labels))
# train_features_loader = DataLoader(train_features_dataset, batch_size=batch_size, shuffle=True)

# val_features_dataset = TensorDataset(torch.tensor(validation_features), torch.tensor(validation_labels))
# val_features_loader = DataLoader(val_features_dataset, batch_size=batch_size, shuffle=False)

# epochs = 10
# train_accuracies, train_losses, val_accuracies, val_losses = train_model(feature_classifier, optimizer, criterion, train_features_loader, val_features_loader, epochs)