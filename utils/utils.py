import os
import numpy as np
import subprocess
import pickle

from collections import deque


def git_log():
    return subprocess.check_output(['git', 'log', '-n', '1']).decode('utf-8')

def get_stat(data_name, data_x, data_y):
    s = 'statistics for %s\n' % data_name
    s += 'data #: %d, shape %s\n' % (len(data_x), str(np.asarray(data_x).shape))
    if len(data_y) > 0:
        s += 'positive ratio: %.5f, max %f, min %f\n' % \
            (np.mean(data_y), np.max(data_y), np.min(data_y))
    s += '\n'
    print(s)
    return s

def feat_to_string(v):
    return ''.join([str(int(i)).zfill(3) for i in v])

def format_data_wports(data_x, n_examples):
    data_ip = decimal2binary(data_x[:n_examples, np.arange(8)])
    data_proto = data_x[:n_examples, 8].reshape(-1, 1)
    data_srcport = uint16_to_binary(data_x[:n_examples, 9].reshape(-1, 1))
    data_dstport = uint16_to_binary(data_x[:n_examples, 10].reshape(-1, 1))
    return np.concatenate((data_ip, data_srcport, data_dstport, data_proto), axis=1)

def decimal2binary(x):
    return np.unpackbits(x.astype(np.uint8), axis=1)

def uint16_to_binary(x):
    assert len(x.shape) == 2 and x.shape[1] == 1
    return np.roll(np.unpackbits(x.astype(np.uint16).view(np.uint8), axis=1), 8, axis=1)

def get_data(data_list, feat_idx, n_examples):
    data_x = np.array([]).reshape(0, 8*8+2*16+1)   # src, dst, ip

    data_y = np.array([])
    for data in data_list:
        data = np.load(data).item()
        # NOTE: order of the features are changed!
        data_b = format_data_wports(data['x'], n_examples)
        data_x = np.concatenate((data_x, data_b))
        data_y = np.concatenate((data_y, data['y'][:n_examples]))
    return data_x, data_y

def get_data_list(data_list, feat_idx, n_examples):
    data_x = []
    data_y = []
    for data in data_list:
        data = np.load(data).item()
        # NOTE: order of the features are changed!
        data_b = format_data_wports(data['x'], n_examples)
        data_x.append(data_b)
        data_y.append(data['y'][:n_examples])
    return data_x, data_y

def data_to_string(data):
    ip = feat_to_string(data[:8])
    proto = str(int(data[8])).zfill(3)
    ports = ''.join([str(int(i)).zfill(5) for i in data[9:]])
    return ip + proto + ports

def get_data_str_with_ports(data):
    data = np.load(data).item()
    data_x = data['x']
    data_y = data['y']

    data_x_str = []
    for xi in data_x:
        data_x_str.append(data_to_string(xi))
    return data_x_str, data_y

def get_data_str_with_ports_list(data_list):
    data_x = []
    data_y = np.array([])
    for dpath in data_list:
        x, y = get_data_str_with_ports(dpath)
        data_x += x
        data_y = np.concatenate((data_y, y))
    return data_x, data_y

def keep_latest_files(path, n_keep):
    def sorted_ls(path):
        mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
        return list(sorted(os.listdir(path), key=mtime))

    files = sorted_ls(path)
    if len(files) < n_keep:
        return

    del_list = files[0:(len(files)-n_keep)]
    for dfile in del_list:
        os.remove(path + dfile)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

