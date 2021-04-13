import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from models import ModelManager
from features import FeaturesManager
from dataset import DatasetManager
from config import DebugConfig, FoldersConfig, DeviceConfig, RetrievalEvalConfig, ModelTrainConfig
from engine import RetrievalEngine

from utils import ProcessTime, LogFile

device = DeviceConfig.DEVICE
DEBUG = DebugConfig.DEBUG

def create_ground_truth_queries(full_df, test_df):
    entries = []

    N = len(test_df.index)
        
    it = 0
    for index, row in test_df.iterrows():
        id = row[0]
        if (id == 'id'):
            continue
        entry = {}

        entry['id'] = id

        isSameArticleType = full_df['articleType'] == row[3]
        similar_clothes_df = full_df[isSameArticleType]

        entry['gt'] = similar_clothes_df['id'].to_numpy()

        entries.append(entry)

        it += 1
        print(f'\rCreating ground truth queries... {it}/{N}, {id}', end='', flush=True)
        
    return entries

def make_ground_truth_matrix(test_df, full_df, entries):
    n_queries = len(entries)
    q_indx = np.zeros(shape=(n_queries, ), dtype=np.int32)
    y_true = np.zeros(shape=(n_queries, full_df.shape[0]), dtype=np.uint8)

    for it, entry in enumerate(entries):
        if (entry['id'] == 'id'):
            continue
        # lookup query index

        ident = int(entry['id'])
        q_indx[it] = test_df.index[test_df['id'] == ident][0]

        # lookup gt imagesId
        gt = entry['gt']
        gt_ids = [f for f in gt]

        # lookup gt indices
        gt_indices = [full_df.index[full_df['id'] == f][0] for f in gt_ids]
        gt_indices.sort()

        y_true[it][q_indx[it]] = 1
        y_true[it][gt_indices] = 1

        print(f'\rMaking ground truth matrix... {it}/{len(entries)}', end='', flush=True)

    return y_true


def evaluate_deep_fashion(s, y_true):
    aps = []
    for i, y_t in enumerate(y_true):
        ap = average_precision_score(y_t, s)
        aps.append(ap)
    
    #print(f'\nAPs {aps}')
    df = pd.DataFrame({'ap': aps}, index=q_indx)
    return df


def prepare_data_to_evaluate(dataset_base_dir):
    test_df = pd.read_csv(os.path.join(dataset_base_dir, "deep_fashion_with_article_type.csv"), error_bad_lines=False)
    
    article_types = test_df.drop_duplicates(subset = ["articleType"])['articleType'].values.tolist()
    print(article_types)

    test_subset_df = pd.DataFrame()
    for acticle_type in article_types:
        is_type = test_df['articleType'] == acticle_type
        
        test_subset_df = pd.concat([test_subset_df, test_df[is_type].head(30)])

    return test_subset_df


def features_evaluation(features, full_df, test_df, num_queries):

    queries = create_ground_truth_queries(full_df, test_df)
    print('\nMake ground truth matrix...')
    y_true = make_ground_truth_matrix(test_df, full_df, queries)

    # Compute mean Average Precision (mAP)
    print('\nComputing mean Average Precision (mAP)...')
    df = evaluate_deep_fashion(features, y_true)
    print(f'\nmAP: {df.ap.mean():0.04f}')

    mAP = f'{df.ap.mean():0.04f}'

    return mAP


def evaluate_models():
    #Configure device
    device = DeviceConfig.DEVICE

    #create logfile for image retrieval
    fields = ['ModelName', 'DataSetSize', 'UsedFeatures', 'FeaturesSize', 'ProcessTime', 'mAP']
    logfile = LogFile(fields)        
    #Create timer to calculate the process time
    proctimer = ProcessTime()

    model_manager = ModelManager(device, FoldersConfig.WORK_DIR)
    features_manager = FeaturesManager(device, model_manager)

    model_name = 'resnet50_custom'

    # The path of original dataset
    dataset_base_dir = FoldersConfig.DATASET_BASE_DIR
    labels_file = FoldersConfig.DATASET_LABELS_DIR

    # Work directory
    work_dir = FoldersConfig.WORK_DIR
    test_df = prepare_data_to_evaluate(dataset_base_dir=dataset_base_dir)

    num_queries = RetrievalEvalConfig.MAP_N_QUERIES
    print('\n\n## Evaluating model ', model_name, "with num_queries=", str(num_queries))

    print('\n\n## Evaluating NormalizedFeatures ', model_name)

    usedfeatures = 'NormalizedFeatures'
    proctimer.start()

    # Calculate features
    print('\nCalculating features for deep fashion...')

    features = torch.tensor([])

    engine = RetrievalEngine(device, FoldersConfig.WORK_DIR)
    engine.load_models_and_precomputed_features()

    for img_path in test_df.path.values.tolist():
        query_path = engine.get_image_deep_fashion_path(img_path)

        # for model_name in model_names:
        model_name = 'resnet50_custom'
        features = engine.get_query_features(model_name, query_path)

    full_df = pd.read_csv(os.path.join(dataset_base_dir, "styles.csv"), error_bad_lines=False)

    mAP = features_evaluation(features, full_df, test_df, num_queries)

    #LOG
    processtime = proctimer.stop()
    values = {'ModelName': model_name, 
            'DataSetSize': test_df.shape[0],
            'UsedFeatures': usedfeatures,
            'FeaturesSize': features[0].shape[0],
            'ProcessTime': processtime,
            'mAPqueries': num_queries,
            'mAP': mAP,
            'PrecisionHits' : precision
        } 
    logfile.writeLogFile(values)
            
    

    #Print and save logfile    
    logfile.printLogFile()
    logfile.saveLogFile_to_csv("evaluation")

if __name__ == "__main__":
    evaluate_models()