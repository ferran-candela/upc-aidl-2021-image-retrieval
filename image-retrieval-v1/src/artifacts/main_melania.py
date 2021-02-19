import os, shutil
import torch
from torch.utils.data import DataLoader
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

from dataset import MyDataset
from file_management import create_directory, divide_images_into_df
from feature_extraction import extract_features
from evaluation import make_ground_truth_matrix, create_ground_truth_entries, evaluate

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# The path to the directory where the original dataset was uncompressed
original_dataset_dir = '/Users/melaniasanchezblanco/Documents/UPC_AIDL/Project/Fashion_Product_Small'
original_image_dir = os.path.join(original_dataset_dir, 'images')
original_labels_file = original_dataset_dir + '/styles.csv'

# The directory where we will store our smaller dataset
base_dir = '/Users/melaniasanchezblanco/Documents/UPC_AIDL/Project/processed_datalab'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)


# Directories for our training, validation and test splits
train_dir = create_directory(base_dir, 'train')
validation_dir = create_directory(base_dir, 'validation')
test_dir = create_directory(base_dir, 'test')

# Divide labels in train, test and validate
labels_df = pd.read_csv(original_labels_file, error_bad_lines=False)

if not os.path.isfile(os.path.join(train_dir, "train_styles.csv")):

    train_df = labels_df.sample(640)
    validate_df = labels_df.sample(128)
    test_df= labels_df.sample(128)

    # Save datasets for don't repeat the copy images
    train_df.to_csv(os.path.join(train_dir, "train_styles.csv"),index=False)
    validate_df.to_csv(os.path.join(validation_dir, "val_styles.csv"),index=False)
    test_df.to_csv(os.path.join(test_dir, "test_styles.csv"),index=False)

    # Divide images for train, test and validate and copy in destination folder
    divide_images_into_df(train_df, original_image_dir, train_dir)
    divide_images_into_df(validate_df, original_image_dir, validation_dir)
    divide_images_into_df(test_df, original_image_dir, test_dir)

else:
    train_df = pd.read_csv(os.path.join(train_dir, "train_styles.csv"))
    validate_df = pd.read_csv(os.path.join(validation_dir, "val_styles.csv"))
    test_df = pd.read_csv(os.path.join(test_dir, "test_styles.csv"))

print('total training images:', len(os.listdir(train_dir)))
print('total validation images:', len(os.listdir(validation_dir)))
print('total test images:', len(os.listdir(test_dir)))

# The directory where we have the small style csv with the labels
train_dataset_dir = '/Users/melaniasanchezblanco/Documents/UPC_AIDL/Project/processed_datalab/train'
train_labels_file = train_dataset_dir + '/train_styles.csv'

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

initial_dataset = MyDataset(original_labels_file,labels_df,transform=transform)


print(train_dataset [0][0].shape)


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

# It is very important to put the network into eval mode before extracting features! 
# This turns off things like dropout and using batch statistics in batch norm.
pretrained_model.eval()
pretrained_model.to(device)

train_features = extract_features(train_loader, pretrained_model, n_batches, device)

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

num_evaluation = 100

queries = create_ground_truth_entries(train_labels_file, train_df, num_evaluation)
q_indx, y_true = make_ground_truth_matrix(train_df, queries)

#Compute mean Average Precision (mAP)
df = evaluate(S, y_true, q_indx)
print(f'mAP: {df.ap.mean():0.04f}')