from turtle import distance
import matplotlib.pyplot as plt
import numpy as np
import pickle
from KNN import *


def storeData(data,
              filename='untitled',
              dir='.'
              ):
    dir = str(dir)
    if not '.pkl' in filename:
        filename = str(filename) + '.pkl'
    else:
        filename = str(filename)

    file = open(dir + '\\' + filename, 'ab')
    pickle.dump(data, file)
    file.close()


def loadData(filename,
             dir='.'
             ):

    dir = str(dir)
    if not '.pkl' in filename:
        filename = str(filename) + '.pkl'
    else:
        filename = str(filename)

    file = open(dir + '\\' + filename, 'rb')

    data = pickle.load(file)
    file.close()

    return data


def showData(data,
             ROW=4,
             COL=4,
             color='tab:blue'
             ):
    print('\033[33mDatashape:\033[0m', data.shape, end='')
    MAXNUM = ROW * COL - 1

    fig, axs = plt.subplots(ROW, COL, sharex=True, sharey=True)
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])

    for i in range(ROW):
        for j in range(COL):
            if COL*i + j == MAXNUM:
                k = 0
            else:
                k = COL*i + j + 1
            axs[i, j].scatter(data[COL*i+j], data[k], s=4, c=color)
            axs[i, j].set_title("Dim %d and %d" % (COL*i + j + 1, k + 1))

    plt.tight_layout()


def matrix_dist(matrix1, matrix2):
    '''
    Parameters:
        2D - matrix1
        2D - matrix2

    Return:
        2D - matrix:
            distmatrix with row == matrix1's row and col == matrix2's row
            distmatrix[i,j] means vec_dist between matrix1[i] and matrix2[j]
    '''
    dist = np.zeros((matrix1.shape[0], matrix2.shape[0]))
    for i in range(matrix1.shape[0]):
        for j in range(matrix2.shape[0]):
            dist[i, j] = np.linalg.norm(matrix1[i] - matrix2[j])
    return dist


if __name__ == '__main__':
    mfcc_feat_s = KNNclass(loadData(filename='s_', dir='MFCC'), 's')
    mfcc_feat_o = KNNclass(loadData(filename='o_mid_', dir='MFCC'), 'o')
    mfcc_feat_ri = KNNclass(loadData(filename='ri_mid_', dir='MFCC'), 'ri')
    mfcc_feat_s.printinfo()
    mfcc_feat_o.printinfo()
    mfcc_feat_ri.printinfo()

    mfcc_feat_test = KNNclass(
        loadData(filename='sory_test_0501_', dir='MFCC'), 'me')
    # showData(mfcc_feat_ri.T,color='red')

    conf = list()
    for vec in mfcc_feat_test.data:
        conf.append(getKNNclass(
            vec, (mfcc_feat_s, mfcc_feat_o, mfcc_feat_ri), 30))

    plt.scatter(np.arange(mfcc_feat_test.shape[0]), [
                i['s'] for i in conf], s=2, c='tab:blue', alpha=0.5)
    plt.scatter(np.arange(mfcc_feat_test.shape[0]), [
                i['o'] for i in conf], s=2, c='red', alpha=0.5)
    plt.scatter(np.arange(mfcc_feat_test.shape[0]), [
                i['ri'] for i in conf], s=2, c='yellow', alpha=0.5)

    plt.legend(["s", "o", "ri"])
    plt.ylabel("Prob")
    plt.xlabel("Frames")

    plt.show()
