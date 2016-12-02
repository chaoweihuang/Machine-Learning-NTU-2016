import numpy as np


def write_submit_file(l, fname):
    if len(l) != 5000000:
        return
    with open(fname, 'w') as fw:
        fw.write('ID,Ans\n')
        fw.write('\n'.join(['%d,%d' % (i, l[i]) for i in range(5000000)]))

def read_submit_file(fname):
    ret = []
    for line in open(fname, 'r'):
        l = line.strip().split(',')
        if not l[0].isdigit():
            continue

        ret.append((int(l[0]), int(l[1])))

    return ret

def get_index(fname='data/check_index.csv'):
    indices = []
    for line in open(fname):
        l = line.strip().split(',')
        if l[0] == 'ID':
            continue
        indices.append((int(l[1]), int(l[2])))

    return indices

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def similarity(v1, v2, normalized=False):
    assert len(v1) == len(v2)
    if not normalized:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    else:
        return np.dot(v1, v2)
