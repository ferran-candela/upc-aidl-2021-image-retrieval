import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

def create_ground_truth_entries(path, dataframe, N):
    entries = []

    with open(path) as myfile:
        lines = [next(myfile) for x in range(N)]

    it = 1
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
        similar_clothes_df = dataframe[isSameArticleType | (isSimilarSubCategory & isSimilarColour)]

        entry['gt'] = similar_clothes_df['id'].to_numpy()

        entries.append(entry)

        print(f'\rCreating ground truth queries... {it}/{N}', end='', flush=True)
        it += 1
        
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
        
    it = 1
    for index, row in query_df.iterrows():
        id = row[0]
        if (id == 'id'):
            continue
        entry = {}

        entry['id'] = id

        isSameArticleType = dataframe['articleType'] == row[4]
        isSimilarSubCategory = dataframe['subCategory'] == row[3]
        isSimilarColour = dataframe['baseColour'] == row[5]
        similar_clothes_df = dataframe[isSameArticleType | (isSimilarSubCategory & isSimilarColour)]

        entry['gt'] = similar_clothes_df['id'].to_numpy()

        entries.append(entry)

        print(f'\rCreating ground truth queries... {it}/{N}', end='', flush=True)
        it += 1
        
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

def evaluation_hits(labels_df, imgindex,ranking):
    # ranking = index list 
    # imgindex = index image query

    queries = create_ground_truth_queries( labels_df, "List", 0, [imgindex])
    q_indx, y_true = make_ground_truth_matrix(labels_df, queries)

    imagesIdx = ranking.tolist()
    return round(np.mean(y_true[0][imagesIdx]),4)


def evaluate_models():
    #create logfile for image retrieval
    fields = ['ModelName', 'DataSetSize', 'UsedFeatures', 'FeaturesSize', 'ProcessTime', 'mAPqueries', 'mAP', 'PrecisionHits']
    logfile = LogFile(fields)        
    #Create timer to calculate the process time
    proctimer = ProcessTime()

    for model_name in pretrained_list_models:

        for file in ['features.pickle','features_aqe.pickle']:

            print(f'\rEvaluate File ... {file}', end='', flush=True)
            proctimer.start()

            # LOAD FEATURES
            model_dir = os.path.join(config["work_dir"], model_name)
            features_file = os.path.join(model_dir , file)
            features = pickle.load(open(features_file , 'rb'))

            #compute the similarity matrix
            S = features @ features.T

            num_queries = config["mAP_n_queries"]

            queries = create_ground_truth_queries( train_df, "FirstN", num_queries,[])
            q_indx, y_true = make_ground_truth_matrix(train_df, queries)

            #Compute mean Average Precision (mAP)
            df = evaluate(S, y_true, q_indx)
            print(f'\rmAP: {df.ap.mean():0.04f}')

            #Compute evaluation Hits
            accuracy = []
            for index in q_indx:
                ranking = pretrained_models.Cosine_Similarity(features,index,config["top_k_image"])
                precision = evaluation_hits(train_df,index,ranking)
                accuracy.append(precision)
            precision = np.mean(accuracy)
            print(f'\rPrecision Hits: {precision:0.04f}')

            #LOG
            processtime = proctimer.stop()
            values = {'ModelName':model_name, 
                    'DataSetSize':train_df.shape[0],
                    'UsedFeatures': file,
                    'FeaturesSize': features[0].shape[0],
                    'ProcessTime': processtime,
                    'mAPqueries': num_queries,
                    'mAP': f'mAP: {df.ap.mean():0.04f}',
                    'PrecisionHits:' : precision
                } 
            logfile.writeLogFile(values)
    #Print and save logfile    
    logfile.printLogFile()
    logfile.saveLogFile_to_csv("evaluation",config)


if __name__ == "__main__":
    evaluate_models()