import numpy as np
import pandas as pd

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

def create_ground_truth_queries(dataframe, type, N):
    #type = "Random", "Fixed"
    entries = []

    if type=="Fixed":
        query_df = dataframe[0:N-1]
    else:
        query_df = dataframe.sample(N)
        
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

from sklearn.metrics import average_precision_score

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
