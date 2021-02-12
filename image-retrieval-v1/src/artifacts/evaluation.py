from pyspark.sql.functions import col

def create_ground_truth_entries(path, dataframe):
    entries = {}

    with open(path, 'r') as f:
        lines = [l.strip() for l in f if len(l.strip())]

    for row in lines:
        id, labels = row.rsplit(',', 1)
        entry = entries.setdefault(id, {'query': '', 'gt': []})
        
        entry['query'] = labels[3]
        entry['gt'] = dataframe \
        .filter((col('articleType') == labels[3]) | (col('subCategory') == labels[2] & col('baseColour') == labels[4])) \
        .select(col('id')) \
        .to_numpy()
        
    return entries


def make_ground_truth_matrix(dataset, entries):
    n_queries = len(entries)
    q_indx = np.zeros(shape=(n_queries, ), dtype=np.int32)
    y_true = np.zeros(shape=(n_queries, 5063), dtype=np.uint8)

    for i, q in enumerate(entries):
        entry = entries[q]
     
        filename = i + ".jpg"

        # lookup query index
        q_indx[i] = dataset.index(filename)

        # lookup gt filenames
        gt_filenames = entry['gt']
        gt_filenames = [f + '.jpg' for f in gt_filenames]

        # lookup gt indices
        gt_indices = [dataset.index(f) for f in gt_filenames]
        gt_indices.sort()

        y_true[i][q_indx[i]] = 1
        y_true[i][gt_indices] = 1

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
