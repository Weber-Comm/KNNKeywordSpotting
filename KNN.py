from distutils.command.config import config
import numpy as np
import pickle
from pprint import pprint


def getVecdist(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


class KNNclass:

    def __init__(self, classdata, classlabel='DefaultLabel'):
        self.label = classlabel
        self.data = classdata
        self.shape = self.data.shape
        self.mean = np.mean(classdata, axis=0)
        self.SD = np.sqrt(np.var(classdata, axis=0))

    def printinfo(self):
        print('\033[33m%s shape:\033[0m' % (self.label), self.shape)
        print('\033[33m%s mean:\033[0m' % (self.label), self.mean)
        print('\033[33m%s SD:\033[0m' % (self.label), self.SD)


def incrsDim(nparray):
    return np.array([nparray])


def decrsDim(nparray):
    return nparray[0]


def getKNNdist(A, B, ret='all'):

    if type(A) == KNNclass:
        dataAc = A.data
    elif type(A) == np.ndarray:
        if A.ndim == 1:
            dataAc = incrsDim(A)
        else:
            dataAc = A
    if type(B) == KNNclass:
        dataBc = B.data
    elif type(B) == np.ndarray:
        if B.ndim == 1:
            dataBc = incrsDim(B)
        else:
            dataBc = B

    dist = np.zeros((dataAc.shape[0], dataBc.shape[0]))  # 2-D dist
    for i in range(dataAc.shape[0]):
        for j in range(dataBc.shape[0]):
            dist[i, j] = np.linalg.norm(dataAc[i] - dataBc[j])

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

    dist = list()
    for knnc in KNNclasses:
        for knnvec in knnc.data:
            dist.append({'label': knnc.label, 'dist': getVecdist(vec, knnvec)})

    dist.sort(key=lambda i: i['dist'])

    nearest = [i['label'] for i in dist][0:N]

    conf = {knnc.label: nearest.count(knnc.label)/len(nearest)
            for knnc in KNNclasses}

    return conf


if __name__ == '__main__':

    A = KNNclass(np.array([[-3, -2, -3], [-1, -7, -5]]), 'A')
    A.printinfo()

    B = KNNclass(np.array([[3, 5, 3], [1, 7, 2], [2, 3, 4]]), 'B')
    B.printinfo()

    print(getKNNclass(np.array([2, 5, 4]), (A, B), N=3))
