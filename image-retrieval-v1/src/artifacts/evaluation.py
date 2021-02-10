def load_ground_truth_entries(path):
    filenames = list_files(path, '.txt', prefix=False)
    filenames.sort()
    entries = {}
    for fn in filenames:
        ident, kind = fn.rsplit('_', 1)
        entry = entries.setdefault(ident, {'query': '', 'gt': [], 'box': None})
        fullpath = os.path.join(path, fn)
        
        with open(fullpath, 'r') as f:
            lines = [l.strip() for l in f if len(l.strip())]
        
        if kind.startswith('query'):
            parts = lines[0].split(' ')
            entry['query'] = parts[0].split('_', 1)[1]
            entry['box'] = list(map(float, parts[1:]))
        
        elif kind.startswith('good') or kind.startswith('ok'):
            entry['gt'].extend(lines)
    return entries


def make_ground_truth_matrix(filenames, entries):
    n_queries = len(entries)
    q_indx = np.zeros(shape=(n_queries, ), dtype=np.int32)
    y_true = np.zeros(shape=(n_queries, 5063), dtype=np.uint8)

    for i, q in enumerate(entries):
        entry = entries[q]
        filename = entry['query'] + '.jpg'
     
        # lookup query index
        q_indx[i] = filenames.index(filename)

        # lookup gt filenames
        gt_filenames = entry['gt']
        gt_filenames = [f + '.jpg' for f in gt_filenames]

        # lookup gt indices
        gt_indices = [filenames.index(f) for f in gt_filenames]
        gt_indices.sort()

        y_true[i][q_indx[i]] = 1
        y_true[i][gt_indices] = 1

    return q_indx, y_true

from sklearn.metrics import average_precision_score

def evaluate(S, features, index):
    query = features[index]
    scores = features @ query
    aps = []
    for i, q in enumerate(q_indx):
        s = S[:, q]
        y_t = y_true[i]
        ap = average_precision_score(y_t, s)
        aps.append(ap)
    df = pd.DataFrame({'ap': aps}, index=q_indx)
    return df
