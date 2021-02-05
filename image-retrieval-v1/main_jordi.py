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

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# The path to the directory where the original dataset was uncompressed
original_dataset_dir = '/home/manager/upcschool-ai/data/FashionProduct/'
original_image_dir = os.path.join(original_dataset_dir, 'images')
original_labels_file = original_dataset_dir + 'styles.csv'

# The directory where we will store our smaller dataset
base_dir = '/home/manager/upcschool-ai/data/FashionProduct/processed_datalab'

#Delete all processed data directory
#shutil.rmtree(base_dir)

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

if not os.path.isfile(os.path.join(train_dir, "train_styles.csv")):

    # Divide labels in train, test and validate
    labels_df = pd.read_csv(original_labels_file, error_bad_lines=False)

    #train_df, validate_df, test_df = np.split(labels_df.sample(frac=1, random_state=42), 
    #                                    [int(.6*len(labels_df)), int(.8*len(labels_df))])
    train_df = labels_df.sample(640)
    validate_df = labels_df.sample(128)
    test_df= labels_df.sample(128)

    # Save datasets for don't repeat the copy images
    train_df.to_csv(os.path.join(train_dir, "train_styles.csv"),index=False)
    validate_df.to_csv(os.path.join(validation_dir, "val_styles.csv"),index=False)
    test_df.to_csv(os.path.join(test_dir, "test_styles.csv"),index=False)

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

    #Test load train dataset and show image
    # ds = MyDataset(original_image_dir,labels_df)
    # print('Number of images', len(ds))
    # plt.figure(figsize=(8, 10))
    # plt.imshow(ds[0][0])
    # plt.xticks([]); plt.yticks([]); plt.grid(False)

else:
    train_df = pd.read_csv(os.path.join(train_dir, "train_styles.csv"))
    validate_df = pd.read_csv(os.path.join(validation_dir, "val_styles.csv"))
    test_df = pd.read_csv(os.path.join(test_dir, "test_styles.csv"))

print('total training images:', len(os.listdir(train_dir)))
print('total test images:', len(os.listdir(test_dir)))
print('total validation images:', len(os.listdir(validation_dir)))

#Data preprocessing
# images = 60x80 pixels

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.CenterCrop(128-32),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

batch_size = 64

train_dataset = MyDataset(train_dir,train_df,transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = MyDataset(validation_dir,validate_df,transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)        

print(train_dataset [0][0].shape)

# for data_batch, labels_batch in train_loader:
#     print('data batch shape:', data_batch.shape)
#     print('labels batch shape:', labels_batch.shape)
#     break



#VGG16 network, trained on ImageNet
pretrained_model = vgg16(pretrained=True)

# Tuning the batch norm

# Here we're going to apply a rudimentaty domain adaptation trick. The batch norm statistics for this network match those of the ImageNet dataset. 
# We can use a trick to get them to match our dataset. The idea is to put the network into train mode and do a pass over the dataset without doing any backpropagation. 
# This will cause the network to update the batch norm statistics for the model without modifying the weights. This can sometimes improve results.
pretrained_model.train()
n_batches = len(train_loader)
i = 1
for image_batch, label_batch in train_loader:
    # move batch to device and forward pass through network
    pretrained_model(image_batch.to(device))
    print(f'\rTuning batch norm statistics {i}/{n_batches}', end='', flush=True)
    i += 1


# It is very important to put the network into eval mode before extracting features! This 
# turns off things like dropout and using batch statistics in batch norm.
pretrained_model.eval()
pretrained_model.to(device)


def extract_features(dataloader):
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.CenterCrop(128-32),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    n_batches = len(dataloader)
    i = 1    
    features = []
    with torch.no_grad():
        for image_batch, label_batch in dataloader:
            image_batch = image_batch.to(device)

            batch_features = pretrained_model(image_batch)

            # features to numpy
            batch_features = torch.squeeze(batch_features).cpu().numpy()

            # collect features
            features.append(batch_features)
            print(f'\rProcessed {i} of {n_batches} batches', end='', flush=True)

            i += 1

    # stack the features into a N x D matrix            
    features = np.vstack(features)
    return features


train_features = extract_features(train_loader)

print(f'\nFeatures are {train_features.shape}')

#Postprocessing
# A standard postprocessing pipeline used in retrieval applications is to do L2-normalization,
# PCA whitening, and L2-normalization again. 
# Effectively this decorrelates the features and makes them unit vectors.
train_features = normalize(train_features, norm='l2')
train_features = PCA(128, whiten=True).fit_transform(train_features)
train_features= normalize(train_features, norm='l2')

print(f'\nNormalized features are {train_features.shape}')


#Querying the data - don't normalize

img_ds = MyDataset(train_dir,train_df)

# change this to the index of the image to use as a query!
query_idx = 1


#Similar images will produce similar feature vectors. 
#There are many ways to measure similarity, but the most common are 
#euclidean distance and cosine similarity

# COSINE SIMILARITY
# searching can be done using cosine similarity
# since the features are normalized, this is just a matrix multiplication

query = train_features[query_idx]
print(query.shape)

scores = train_features @ query # Xq
print(scores.shape)

# rank by score, descending, and skip the top match (because it will be the query)
ranking = (-scores).argsort()[1:]

# show the query image
print('Query')
plt.figure(figsize=(2.8,2.8))
plt.imshow(img_ds[query_idx][0])
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.show()

print('Top 10')
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(18, 6))
ax = ax.ravel()
for i in range(10):
    img = img_ds[ranking[i]][0]

    # show the image (remove ticks and grid)
    ax[i].imshow(img)
    ax[i].grid(False) 
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title(i)
plt.show()



#Run evaluation

#compute the similarity matrix
S = train_features @ train_features.T
print(S.shape)

from sklearn.metrics import average_precision_score

def evaluate(S, features, index):
    query = features[index]
    scores = features @ query
        #     aps = []
#     for i, q in enumerate(q_indx):
#         s = S[:, q]
#         y_t = y_true[i]
#         ap = average_precision_score(y_t, s)
#         aps.append(ap)
#     df = pd.DataFrame({'ap': aps}, index=q_indx)
#     return df

# #compute mAP
# df = evaluate(S, y_true, q_indx)
# print(f'mAP: {df.ap.mean():0.04f}')