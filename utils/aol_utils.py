import sys
import numpy as np

from collections import Counter


def save_pickle(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def get_data_aol(data_list):
    c = Counter()
    for data in data_list:
        counter = load_pickle(data)
        print('%s ... # items %d' % (data, len(counter)))
        c.update(counter)

    x, y = zip(*c.most_common())
    return np.asarray(x), np.asarray(y)

def get_data_aol_by_day(data_path):
    data = np.load(data_path)
    return data['queries'], data['counts']

def get_data_aol_by_days(data_list):
    c = Counter()
    for data in data_list:
        print('loading %s' % data)
        x, y = get_data_aol_by_day(data)
        c.update(dict(zip(x, y)))

    x, y = zip(*c.most_common())
    return np.asarray(x), np.asarray(y)

def get_data_aol_feat(data_path):
    data = np.load(data_path)
    query_char_ids = data['query_char_ids']
    counts  = data['counts']
    q_lens  = data['query_lens']

    print('queries', query_char_ids.shape)
    print('counts', counts.shape)
    print('q_lens', q_lens.shape)

    x = np.concatenate((q_lens.reshape(-1, 1), query_char_ids), axis=1)
    y = counts
    assert len(x) == len(y)
    return x, y

def get_data_aol_feat_list(data_paths):
    x = np.array([]).reshape(0, 61)
    y = np.array([])
    for dpath in data_paths:
        xi, yi = get_data_aol_feat(dpath)
        x = np.concatenate((x, xi))
        y = np.concatenate((y, yi))
    return x, y

def get_data_aol_query(data_path):
    data = np.load(data_path)
    queries = data['queries']
    counts  = data['counts']

    print('queries', queries.shape)
    print('counts', counts.shape)

    assert len(queries) == len(counts)
    return queries, counts

def get_data_aol_query_list(data_paths):
    queries = np.array([])
    counts  = np.array([])
    for dpath in data_paths:
        qi, ci = get_data_aol_query(dpath)
        queries = np.concatenate((queries, qi))
        counts = np.concatenate((counts, ci))
        print(dpath)
    return queries, counts
