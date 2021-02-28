from evaluation import make_ground_truth_matrix, create_ground_truth_entries, evaluate
import pickle
import os, shutil
import pandas as pd

def calculate_similarity(features_path):

    train_features =  pickle.load( open( features_path, "rb" ) )
    print("train features = ", train_features)
    print("train features shape = ", train_features.shape)

    #compute the similarity matrix
    similarity_vgg16 = train_features @ train_features.T
    print(similarity_vgg16.shape)

    return similarity_vgg16


def main(config):
    #Read train images
    train_labels_path = config['home_path'] + '/Fashion_Product_Full'
    train_image_dir = os.path.join(train_labels_path, 'images')
    train_labels_file = train_labels_path + '/styles.csv'

    train_df = pd.read_csv(train_labels_file, error_bad_lines=False)
    print("Train dataframe shape = ",train_df.shape)

    #Train features paths
    features_vgg16_path = config['home_path'] + '/Fashion_Product_Full_features/vgg16/features.pickle'
    features_resnet50_path = config['home_path'] + '/Fashion_Product_Full_features/resnet50/features.pickle'
    features_inception_v3_path = config['home_path'] + '/Fashion_Product_Full_features/inception_v3/features.pickle'
    features_inception_resnet_v2_path = config['home_path'] + '/Fashion_Product_Full_features/inception_resnet_v2/features.pickle'

    #Run evaluation

    S = calculate_similarity(features_resnet50_path)

    num_evaluation = 650

    queries = create_ground_truth_entries(train_labels_file, train_df, num_evaluation)
    q_indx, y_true = make_ground_truth_matrix(train_df, queries, num_evaluation)

    #Compute mean Average Precision (mAP)
    df = evaluate(S, y_true, q_indx)
    print(f'mAP: {df.ap.mean():0.04f}')


if __name__ == "__main__":
    config = {
        "home_path" : "/Users/melaniasanchezblanco/Documents/UPC_AIDL/Project"
    }

    main(config)