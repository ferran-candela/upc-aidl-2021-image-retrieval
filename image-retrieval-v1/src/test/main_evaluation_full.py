from evaluation import make_ground_truth_matrix, create_ground_truth_entries, evaluate
import pickle
import os, shutil
import pandas as pd

#Read train images
train_labels_path = '/Users/melaniasanchezblanco/Documents/UPC_AIDL/Project/Fashion_Product_Full'
train_image_dir = os.path.join(train_labels_path, 'images')
train_labels_file = train_labels_path + '/styles.csv'

train_df = pd.read_csv(train_labels_file, error_bad_lines=False)


#Get train features
features_path = '/Users/melaniasanchezblanco/Documents/UPC_AIDL/Project/Fashion_Product_Full_features/vgg16/features.pickle'

train_features =  pickle.load( open( features_path, "rb" ) )
print("train features = ", train_features)
print("train features shape = ", train_features.shape)


#Run evaluation

#compute the similarity matrix
S = train_features @ train_features.T
print(S.shape)

num_evaluation = 6000

queries = create_ground_truth_entries(train_labels_file, train_df, num_evaluation)
q_indx, y_true = make_ground_truth_matrix(train_df, queries, num_evaluation)

#Compute mean Average Precision (mAP)
df = evaluate(S, y_true, q_indx)
print(f'mAP: {df.ap.mean():0.04f}')