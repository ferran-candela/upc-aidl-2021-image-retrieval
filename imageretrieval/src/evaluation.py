import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from models import ModelManager
from features import FeaturesManager
from config import DebugConfig, FoldersConfig, DeviceConfig, RetrievalEvalConfig

from utils import ProcessTime, LogFile

device = DeviceConfig.DEVICE
DEBUG = DebugConfig.DEBUG

def create_ground_truth_entries(path, dataframe, N):
    entries = []

    with open(path) as myfile:
        lines = [next(myfile) for x in range(N)]

    it = 0
    for row in lines:
        labels = row.rsplit(',')
        id = labels[0]
        if (id == 'id'):
            continue
        entry = {}

        entry['id'] = id

        isSameArticleType = dataframe['articleType'] == labels[4]
        isSimilarSubCategory = dataframe['subCategory'] == labels[3]
        isSimilarColour = dataframe['baseColour'] == labels[5]
        similar_clothes_df = dataframe[isSameArticleType]

        entry['gt'] = similar_clothes_df['id'].to_numpy()

        entries.append(entry)

        it += 1
        print(f'Creating ground truth queries... {it}/{N}', end='', flush=True)
        
    return entries

def create_ground_truth_queries(dataframe, type, N, imgIdxList):
    #type = "Random", "FirstN", "List"
    entries = []

    if type=="FirstN":
        query_df = dataframe[0:N-1]
    elif type=="Random":
        query_df = dataframe.sample(N)
    elif type=="List":
        query_df = dataframe[dataframe.index.isin(imgIdxList)]
    else:
        raise Exception("create_ground_truth_queries: UNKNOW OPTION")
        
    it = 0
    for index, row in query_df.iterrows():
        id = row[0]
        if (id == 'id'):
            continue
        entry = {}

        entry['id'] = id

        isSameArticleType = dataframe['articleType'] == row[4]
        isSimilarSubCategory = dataframe['subCategory'] == row[3]
        isSimilarColour = dataframe['baseColour'] == row[5]
        similar_clothes_df = dataframe[isSameArticleType]

        entry['gt'] = similar_clothes_df['id'].to_numpy()

        entries.append(entry)

        it += 1
        print(f'\rCreating ground truth queries... {it}/{N}', end='', flush=True)
        
    return entries

def make_ground_truth_matrix(dataframe, entries):
    n_queries = len(entries)
    q_indx = np.zeros(shape=(n_queries, ), dtype=np.int32)
    y_true = np.zeros(shape=(n_queries, dataframe.shape[0]), dtype=np.uint8)

    for it, entry in enumerate(entries):
        if (entry['id'] == 'id'):
            continue
        # lookup query index

        ident = int(entry['id'])

        q_indx[it] = dataframe.index[dataframe['id'] == ident][0]

        # lookup gt imagesId
        gt = entry['gt']
        gt_ids = [f for f in gt]

        # lookup gt indices
        gt_indices = [dataframe.index[dataframe['id'] == f][0] for f in gt_ids]
        gt_indices.sort()

        y_true[it][q_indx[it]] = 1
        y_true[it][gt_indices] = 1

    return q_indx, y_true


def evaluate(S, y_true, q_indx):
    aps = []
    for i, q in enumerate(q_indx):
        s = S[:, q]
        y_t = y_true[i]
        ap = average_precision_score(y_t, s)
        aps.append(ap)
    
    #print(f'\nAPs {aps}')
    df = pd.DataFrame({'ap': aps}, index=q_indx)
    return df

def evaluation_hits(labels_df, imgindex, ranking):
    # ranking = index list 
    # imgindex = index image query

    queries = create_ground_truth_queries(labels_df, "List", 0, [imgindex])
    q_indx, y_true = make_ground_truth_matrix(labels_df, queries)

    imagesIdx = ranking.tolist()
    return round(np.mean(y_true[0][imagesIdx]), 4)

def cosine_similarity(features, imgidx, top_k):
    # This gives the same rankings as (negative) Euclidean distance 
    # when the features are L2 normalized (as ours are).
    # The cosine similarity can be efficiently computed for all images 
    # in the dataset using a matrix multiplication!
    query = features[imgidx]
    scores = features @ query 

    # rank by score, descending, and skip the top match (because it will be the query)
    ranking = (-scores).argsort()[1:top_k + 1]
    return ranking

def evaluate_models():
    #create logfile for image retrieval
    fields = ['ModelName', 'DataSetSize', 'UsedFeatures', 'FeaturesSize', 'ProcessTime', 'mAPqueries', 'mAP', 'PrecisionHits']
    logfile = LogFile(fields)        
    #Create timer to calculate the process time
    proctimer = ProcessTime()

    model_manager = ModelManager(device, FoldersConfig.WORK_DIR)
    features_manager = FeaturesManager(device, model_manager)

    model_names = model_manager.get_model_names()


    # TODO: MIRAR EL TEMA DEL AQE
    for model_name in model_names:
        try:
            if(features_manager.is_normalized_feature_saved(model_name)):
                print('\n\n## Evaluating model ', model_name)
                proctimer.start()

                # LOAD FEATURES
                print('\nLoading features from checkpoint...')
                loaded_model_features = features_manager.load_from_norm_features_checkpoint(model_name)

                features = loaded_model_features['normalized_features']
                data = loaded_model_features['data']

                # compute the similarity matrix
                print('\nComputing similarity matrix...')
                S = features @ features.T

                num_queries = RetrievalEvalConfig.MAP_N_QUERIES

                queries = create_ground_truth_queries(data, RetrievalEvalConfig.GT_SELECTION_MODE, num_queries, [])
                print('\nMake ground truth matrix...')
                q_indx, y_true = make_ground_truth_matrix(data, queries)

                # Compute mean Average Precision (mAP)
                print('\nComputing mean Average Precision (mAP)...')
                df = evaluate(S, y_true, q_indx)
                print(f'\nmAP: {df.ap.mean():0.04f}')

                # Compute evaluation Hits
                print('\nComputing evaluation Hits...')
                accuracy = []
                for index in q_indx:
                    ranking = cosine_similarity(features, index, RetrievalEvalConfig.TOP_K_IMAGE)
                    precision = evaluation_hits(data, index, ranking)
                    accuracy.append(precision)
                precision = np.mean(accuracy)
                print(f'\nPrecision Hits: {precision:0.04f}')

                #LOG
                processtime = proctimer.stop()
                values = {'ModelName': model_name, 
                        'DataSetSize': data.shape[0],
                        # 'UsedFeatures': file,
                        'FeaturesSize': features[0].shape[0],
                        'ProcessTime': processtime,
                        'mAPqueries': num_queries,
                        'mAP': f'{df.ap.mean():0.04f}',
                        'PrecisionHits' : precision
                    } 
                logfile.writeLogFile(values)
        except Exception as e:
            print(e)
            processtime = proctimer.stop()
            values = {'ModelName': model_name, 
                    'DataSetSize': data.shape[0],
                    # 'UsedFeatures': file,
                    'FeaturesSize': features[0].shape[0],
                    'ProcessTime': processtime,
                    'mAPqueries': num_queries,
                    'mAP': 'ERROR',
                    'PrecisionHits' : 'ERROR'
                } 
            logfile.writeLogFile(values)
    #Print and save logfile    
    logfile.printLogFile()
    logfile.saveLogFile_to_csv("evaluation")


if __name__ == "__main__":
    evaluate_models()