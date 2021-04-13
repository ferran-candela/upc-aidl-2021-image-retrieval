import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from imageretrieval.src.models import ModelManager
from imageretrieval.src.features import FeaturesManager
from imageretrieval.src.dataset import DatasetManager
from imageretrieval.src.config import DebugConfig, FoldersConfig, DeviceConfig, RetrievalEvalConfig, ModelTrainConfig

from imageretrieval.src.utils import ProcessTime, LogFile

device = DeviceConfig.DEVICE
DEBUG = DebugConfig.DEBUG

def create_ground_truth_queries(full_df, test_df, type, N, imgIdxList):
    #type = "Random", "FirstN", "List"
    entries = []

    if type=="FirstN":
        query_df = test_df[0:N-1]
    elif type=="Random":
        query_df = test_df.sample(N)
    elif type=="List":
        query_df = test_df[test_df.index.isin(imgIdxList)]
        N = len(imgIdxList)
    else:
        raise Exception("create_ground_truth_queries: UNKNOW OPTION")
        
    it = 0
    for index, row in query_df.iterrows():
        id = row[0]
        if (id == 'id'):
            continue
        entry = {}

        entry['id'] = id

        isSameArticleType = full_df['articleType'] == row[4]
        # isSimilarSubCategory = full_df['subCategory'] == row[3]
        # isSimilarColour = full_df['baseColour'] == row[5]
        similar_clothes_df = full_df[isSameArticleType]

        entry['gt'] = similar_clothes_df['id'].to_numpy()

        entries.append(entry)

        it += 1
        print(f'\rCreating ground truth queries... {it}/{N}', end='', flush=True)
        
    return entries

def make_ground_truth_matrix(full_df, entries):
    n_queries = len(entries)
    q_indx = np.zeros(shape=(n_queries, ), dtype=np.int32)
    y_true = np.zeros(shape=(n_queries, full_df.shape[0]), dtype=np.uint8)

    for it, entry in enumerate(entries):
        if (entry['id'] == 'id'):
            continue
        # lookup query index

        ident = int(entry['id'])

        q_indx[it] = full_df.index[full_df['id'] == ident][0]

        # lookup gt imagesId
        gt = entry['gt']
        gt_ids = [f for f in gt]

        # lookup gt indices
        gt_indices = [full_df.index[full_df['id'] == f][0] for f in gt_ids]
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

def evaluation_hits(full_df, test_df, imgindex, ranking):
    # ranking = index list 
    # imgindex = index image query
    # Calculate how many images returned in the ranking are "correct" of the total

    queries = create_ground_truth_queries(full_df, full_df, "List", 0, [imgindex])
    q_indx, y_true = make_ground_truth_matrix(full_df, queries)

    imagesIdx = ranking.tolist()
    return round(np.mean(y_true[0][imagesIdx]), 4)

def cosine_similarity(features, imgidx, top_k):
    # This gives the same rankings as (negative) Euclidean distance 
    # when the features are L2 normalized (as ours are).
    # The cosine similarity can be efficiently computed for all images 
    # in the dataset using a matrix multiplication!
    query = features[imgidx]
    scores = features @ query 
    # Return top K ids
    ranking = (-scores).argsort()[:top_k]
    return ranking

def prepare_data(dataset_base_dir, labels_file, process_dir, train_size, validate_test_size, clean_process_dir):
    dataset_manager = DatasetManager()      

    train_df, test_df, validate_df = dataset_manager.split_dataset(dataset_base_dir=dataset_base_dir,
                                                    original_labels_file=labels_file,
                                                    process_dir=process_dir,
                                                    clean_process_dir=clean_process_dir,
                                                    train_size=train_size,
                                                    fixed_validate_test_size=validate_test_size
                                                    )
        
    train_df.reset_index(drop=True, inplace=True)
    if DEBUG:print(train_df.head(10))
    if not test_df is None:
        test_df.reset_index(drop=True, inplace=True)
        validate_df.reset_index(drop=True, inplace=True)

    return train_df, test_df, validate_df


def features_evaluation(features, full_df, test_df, num_queries):
    # compute the similarity matrix
    print('\nComputing similarity matrix...')
    S = features @ features.T

    queries = create_ground_truth_queries(full_df, test_df, RetrievalEvalConfig.GT_SELECTION_MODE, num_queries, [])
    print('\nMake ground truth matrix...')
    q_indx, y_true = make_ground_truth_matrix(full_df, queries)

    # Compute mean Average Precision (mAP)
    print('\nComputing mean Average Precision (mAP)...')
    df = evaluate(S, y_true, q_indx)
    print(f'\nmAP: {df.ap.mean():0.04f}')

    # Compute evaluation Hits
    print('\nComputing evaluation Hits...')
    accuracy = []
    for index in q_indx:
        ranking = cosine_similarity(features, index, RetrievalEvalConfig.TOP_K_IMAGE)
        precision = evaluation_hits(full_df, test_df, index, ranking)
        accuracy.append(precision)
    precision = np.mean(accuracy)
    print(f'\nPrecision Hits: {precision:0.04f}')

    mAP = f'{df.ap.mean():0.04f}'

    return mAP, precision

def evaluate_models():
    #create logfile for image retrieval
    fields = ['ModelName', 'DataSetSize', 'UsedFeatures', 'FeaturesSize', 'ProcessTime', 'mAPqueries', 'mAP', 'PrecisionHits']
    logfile = LogFile(fields)        
    #Create timer to calculate the process time
    proctimer = ProcessTime()

    model_manager = ModelManager(device, FoldersConfig.WORK_DIR)
    features_manager = FeaturesManager(device, model_manager)

    model_names = model_manager.get_model_names()

    # The path of original dataset
    dataset_base_dir = FoldersConfig.DATASET_BASE_DIR
    labels_file = FoldersConfig.DATASET_LABELS_DIR

    # Work directory
    work_dir = FoldersConfig.WORK_DIR
    train_df, test_df, validate_df = prepare_data(dataset_base_dir=dataset_base_dir,
                                        labels_file=labels_file,
                                        process_dir=work_dir,
                                        clean_process_dir=False,
                                        train_size=ModelTrainConfig.TRAIN_SIZE,
                                        validate_test_size=ModelTrainConfig.TEST_VALIDATE_SIZE)

    for model_name in model_names:
        try:
            num_queries = RetrievalEvalConfig.MAP_N_QUERIES
            print('\n\n## Evaluating model ', model_name, "with num_queries=", str(num_queries))

            if(features_manager.is_normalized_feature_saved(model_name)):
                print('\n\n## Evaluating NormalizedFeatures ', model_name)

                usedfeatures = 'NormalizedFeatures'
                proctimer.start()

                # LOAD FEATURES
                print('\nLoading features from checkpoint...')
                loaded_model_features = features_manager.load_from_norm_features_checkpoint(model_name)
                features = loaded_model_features['normalized_features']
                full_df = loaded_model_features['data']
                mAP, precision = features_evaluation(features, full_df, test_df, num_queries)

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

                # Free memory
                del features
                del full_df
                del loaded_model_features

            if(features_manager.is_aqe_feature_saved(model_name)):
                print('\n\n## Evaluating AQE features ', model_name)

                # LOAD AVERAGE QUERY EXPANSION FEATURES
                proctimer.start()
                usedfeatures = 'AQEFeatures'

                print('\nLoading AQE features from checkpoint...')
                loaded_model_features = features_manager.load_from_aqe_features_checkpoint(model_name)
                features = loaded_model_features['aqe_features']
                full_df = loaded_model_features['data']
                mAP, precision = features_evaluation(features, full_df, test_df, num_queries)

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
                
        except Exception as e:
            print('\n', e)
            processtime = proctimer.stop()
            values = {'ModelName': model_name, 
                    'DataSetSize': test_df.shape[0],
                    'UsedFeatures': usedfeatures,
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