import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from imageretrieval.src.models import ModelManager

from imageretrieval.src.features import FeaturesManager
from imageretrieval.src.dataset import DatasetManager
from imageretrieval.src.config import DebugConfig, FoldersConfig, DeviceConfig, RetrievalEvalConfig, ModelTrainConfig
from imageretrieval.src.engine import RetrievalEngine

from utils import ProcessTime, LogFile

device = DeviceConfig.DEVICE
DEBUG = DebugConfig.DEBUG

def create_ground_truth_queries(full_df, test_df, type, N, imgIdList):
    entries = []

    if type=="List":
        query_df = test_df[test_df.id.isin(imgIdList)]
        N = len(imgIdList)
    else:
        query_df = test_df
        N = len(query_df.index)
        
    it = 0
    for index, row in query_df.iterrows():
        id = row[0]
        if (id == 'id'):
            continue
        entry = {}

        entry['id'] = id

        isSameArticleType = full_df['articleType'] == row[3]
        similar_clothes_df = full_df[isSameArticleType]

        if (similar_clothes_df.size == 0):
            print("IS ZERO")
            print(row[3])
        entry['gt'] = similar_clothes_df['id'].to_numpy()

        entries.append(entry)

        it += 1
        print(f'\rCreating ground truth queries... {it}/{N}, {id}', end='', flush=True)
        
    return entries

def make_ground_truth_matrix(test_df, full_df, entries):
    n_queries = len(entries)
    print("Number of queries =")
    print(n_queries)
    y_true = np.zeros(shape=(n_queries, full_df.shape[0]), dtype=np.uint8)

    for it, entry in enumerate(entries):
        if (entry['id'] == 'id'):
            continue
        # lookup query index

        ident = int(entry['id'])

        # lookup gt imagesId
        gt = entry['gt']
        gt_ids = [f for f in gt]

        # lookup gt indices
        gt_indices = [full_df.index[full_df['id'] == f][0] for f in gt_ids]
        gt_indices.sort()

        y_true[it][gt_indices] = 1

        print(f'\rMaking ground truth matrix... {it}/{len(entries)}', end='', flush=True)

    return y_true


def evaluate_deep_fashion(scores, y_true):
    aps = []
    for i, y_t in enumerate(y_true):
        s = scores[:,i].numpy()
        ap = average_precision_score(y_t.reshape((-1)), s.reshape((-1)))
        aps.append(ap)
    
    #print(f'\nAPs {aps}')

    return np.nanmean(aps)


def prepare_data_to_evaluate(dataset_base_dir, article_types):
    test_df = pd.read_csv(FoldersConfig.DATASET_LABELS_DIR, error_bad_lines=False)
    
    print(article_types)

    test_subset_df = pd.DataFrame()
    for article_type in article_types:
        is_type = test_df['articleType'] == article_type
        
        samples = test_df[is_type]

        if(len(samples) > RetrievalEvalConfig.QUERIES_PER_LABEL):
            samples = samples.sample(RetrievalEvalConfig.QUERIES_PER_LABEL)

        test_subset_df = pd.concat([test_subset_df, samples])

    return test_subset_df


def features_evaluation(scores, full_df, test_df):

    queries = create_ground_truth_queries(full_df, test_df, "None", 0, [])

    y_true = make_ground_truth_matrix(test_df, full_df, queries)

    # Compute mean Average Precision (mAP)
    print('\nComputing mean Average Precision (mAP)...')
    mAP = evaluate_deep_fashion(scores, y_true)
    print(f'\nmAP: {mAP:0.04f}')

    return mAP


def evaluation_hits(full_df, test_df, id_img, ranking):
    # Calculate how many images returned in the ranking are "correct" of the total

    queries = create_ground_truth_queries(full_df, test_df, "List", 0 , [id_img])
    y_true = make_ground_truth_matrix(test_df, full_df, queries)

    imagesIdx = ranking.tolist()
    return round(np.mean(y_true[0][imagesIdx]), 4)


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

    # The path of original dataset
    dataset_base_dir = FoldersConfig.DATASET_BASE_DIR
    labels_file = FoldersConfig.DATASET_LABELS_DIR

    # Work directory
    work_dir = FoldersConfig.WORK_DIR

    model_names = model_manager.get_model_names()

    for model_name in model_names:
        # LOAD FEATURES
        print('\nLoading features from checkpoint...')
        loaded_model_features = features_manager.load_from_norm_features_checkpoint(model_name)
        features = loaded_model_features['normalized_features']

        full_df = loaded_model_features['data']

        labels_df = full_df['articleType']
        dataset_labels = labels_df.unique()

        article_types = dataset_labels.tolist()
        test_df = prepare_data_to_evaluate(dataset_base_dir=dataset_base_dir, article_types=article_types)

        num_queries = len(test_df)
        print('\n\n## Evaluating model ', model_name, "with num_queries=", str(num_queries))

        print('\n\n## Evaluating NormalizedFeatures ', model_name)

        usedfeatures = 'NormalizedFeatures'
        proctimer.start()

        # Calculate features
        print('\nCalculating features for deep fashion...')

        queries = []

        engine = RetrievalEngine(device, FoldersConfig.WORK_DIR)
        engine.load_models_and_precomputed_features()

        test_paths = test_df.path.values.tolist()

        for i, img_path in enumerate(test_paths):
            query_path = engine.get_image_deep_fashion_path(img_path)

            query_features = engine.get_query_features(model_name, query_path)
            queries.append(query_features)
            print(f'\rCalculate features... {i}/{len(test_paths)}', end='', flush=True)

        queries = np.vstack(queries)

        scores = features @ queries.T

        # Compute evaluation Hits
        print('\nComputing evaluation Hits...')
        accuracy = []
        for i,id_img in enumerate(test_df.id.values.tolist()):
            score = scores[:,i].numpy()
            top_k = RetrievalEvalConfig.TOP_K_IMAGE
            ranking = (-score).argsort()[:top_k]
            precision = evaluation_hits(full_df, test_df, id_img, ranking)
            accuracy.append(precision)

        precision = np.mean(accuracy)
        print(f'\nPrecision Hits: {precision:0.04f}')

        # Compute mAP
        mAP = features_evaluation(scores, full_df, test_df)

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