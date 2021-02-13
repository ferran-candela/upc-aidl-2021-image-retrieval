import numpy as np

def create_ground_truth_entries(path, dataframe, N):
    entries = []

    with open(path) as myfile:
        lines = [next(myfile) for x in range(N)]

    it = 1
    for row in lines:
        labels = row.rsplit(',')
        i = labels[0]
        entry = {}

        entry['id'] = i

        entry['query'] = labels[4]

        isSameArticleType = dataframe['articleType'] == labels[4]
        isSimilarSubCategory = dataframe['subCategory'] == labels[3]
        isSimilarColour = dataframe['baseColour'] == labels[5]
        similar_clothes_df = dataframe[isSameArticleType | (isSimilarSubCategory & isSimilarColour)]

        entry['gt'] = similar_clothes_df['id'].to_numpy()

        entries.append(entry)

        print(f'\rCreating ground truth entries... {it}/{N}', end='', flush=True)
        it += 1
        
    return entries


def make_ground_truth_matrix(dataframe, entries):
    n_queries = len(entries)
    q_indx = np.zeros(shape=(n_queries, ), dtype=np.int32)
    y_true = np.zeros(shape=(n_queries, 5063), dtype=np.uint8)

    it=0
    for entry in entries:
        if (entry['id'] == 'id'):
            continue
        # lookup query index

        i = int(entry['id'])

        q_indx[it] = dataframe.index[dataframe['id'] == i][0]

        # lookup gt filenames
        gt = entry['gt']
        gt_ids = [f for f in gt]

        # lookup gt indices
        gt_indices = [dataframe.index[dataframe['id'] == f][0] for f in gt_ids]
        gt_indices.sort()

        y_true[it][q_indx[it]] = 1
        y_true[it][gt_indices] = 1
        print(f'\rCreating ground truth matrix... {it}/{N}', end='', flush=True)
        it += 1

    return q_indx, y_true

from sklearn.metrics import average_precision_score

def evaluate(S, y_true, q_indx):
    aps = []
    for i, q in enumerate(q_indx):
        s = S[:, q]
        y_t = y_true[i]
        ap = average_precision_score(y_t, s)
        aps.append(ap)
    df = pd.DataFrame({'ap': aps}, index=q_indx)
    return df
