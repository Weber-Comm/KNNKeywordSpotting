import numpy as np
import pickle
from pprint import pprint
from sklearn.exceptions import DataDimensionalityWarning


def checkDataFormat(data, Dim=False, Type=np.ndarray):
    if not type(data) == Type:
        raise TypeError
    if Dim == False:
        pass
    elif not data.ndim == Dim:
        raise DataDimensionalityWarning


def getVecdist(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


class KNNclass:

    def __init__(self, data, label='DefaultLabel'):
        checkDataFormat(data, Dim=2, Type=np.ndarray)
        self.label = label
        self.data = data
        self.shape = self.data.shape
        self.mean = np.mean(data, axis=0)
        self.SD = np.sqrt(np.var(data, axis=0))

    def printinfo(self):
        print('\033[33m\'%s\' shape:\033[0m' % (self.label), self.shape)
        print('\033[33m\'%s\' mean:\033[0m' % (self.label), self.mean)
        print('\033[33m\'%s\' SD:\033[0m' % (self.label), self.SD)


def incrsDim(nparray):
    return np.array([nparray])


def decrsDim(nparray):
    return nparray[0]


def getKNNdist(A, B, ret='all'):
    '''
    Parameters:

    Return:

    '''
    if type(A) == KNNclass:
        dataA = A.data
    elif type(A) == np.ndarray:
        if A.ndim == 1:
            dataA = incrsDim(A)
        else:
            dataA = A
    if type(B) == KNNclass:
        dataB = B.data
    elif type(B) == np.ndarray:
        if B.ndim == 1:
            dataB = incrsDim(B)
        else:
            dataB = B

    dist = np.zeros((dataA.shape[0], dataB.shape[0]))  # 2-D dist

    dist = [np.linalg.norm(dataA[i] - dataB[j])
            for i in range(dataA.shape[0]) for j in range(dataB.shape[0])]

    if dist.shape[0] == 1:
        dist = decrsDim(dist)

    if ret == 'all':
        return dist
        # dist.shape == (A.row, B.row) 2-D if B.row > 1
        # dist.shape == (A.row,) 1-D if B.row == 1
    elif ret == 'mean':
        return np.mean(dist, axis=1)
        # dist.shape == (A.row,) 1-D


def getKNNclass(vec, KNNclasses, N):
    '''
    Parameters:
        vec: 1-D ndarray
        KNNclasses: (KNNclass A, KNNclass B, ...)
            a tuple consists of KNNclasses

    Return:
        confidence: {A.label: probA, B.label: probB, ...}
            a dictionary contains the probability that the vec belongs to KNNclasses

    '''
    dist = [{'label': knnc.label, 'dist': getVecdist(vec, knnvec)}
            for knnc in KNNclasses for knnvec in knnc.data]

    dist.sort(key=lambda i: i['dist'])

    nearest = [i['label'] for i in dist][0:N]

    conf = {knnc.label: nearest.count(knnc.label)/len(nearest)
            for knnc in KNNclasses}

    return conf
