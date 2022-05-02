import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

from KNN import *


def loadFeat(featfilename,
             dir='.',
             N='all'
             ):

    dir = str(dir)
    if not '.feat' in featfilename:
        featfilename = str(featfilename) + '.feat'
    else:
        featfilename = str(featfilename)

    file = open(dir + '\\' + featfilename, 'rb')

    data = pickle.load(file)
    file.close()

    if N == 'all' or N > data.shape[0]:
        return data
    else:
        np.random.seed(100)
        index = np.random.choice(np.arange(data.shape[0]), N)
        return data[index]


def storeData(data,
              filename='untitled',
              dir='.',
              extension='.feat'
              ):
    dir = str(dir)
    if not extension in filename:
        filename = str(filename) + extension
    else:
        filename = str(filename)

    file = open(dir + '\\' + filename, 'ab')
    pickle.dump(data, file)
    file.close()


def showFeat(data,
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
    return [np.linalg.norm(matrix1[i] - matrix2[j])
            for i in range(matrix1.shape[0])
            for j in range(matrix2.shape[0])]


if __name__ == '__main__':
    mfcc_feat_s = KNNclass(
        loadFeat(featfilename='s_', dir='MFCC', N=70), 's')
    mfcc_feat_o = KNNclass(
        loadFeat(featfilename='o_mid_', dir='MFCC', N=50), 'o')
    mfcc_feat_ri = KNNclass(
        loadFeat(featfilename='ri_mid_', dir='MFCC', N=50), 'ri')
    mfcc_feat_t = KNNclass(
        loadFeat(featfilename='t_', dir='MFCC', N=50), 't')
    mfcc_feat_N = KNNclass(loadFeat(featfilename='N_', dir='MFCC', N=27), 'N')
    mfcc_feat_s.printinfo()
    mfcc_feat_o.printinfo()
    mfcc_feat_ri.printinfo()
    mfcc_feat_t.printinfo()
    mfcc_feat_N.printinfo()

    # set testdata
    mfcc_feat_test = KNNclass(
        loadFeat(featfilename='sory_me_part_', dir='MFCC'), 'me')
    # showFeat(mfcc_feat_test.T,color='red')

    # achieve classification confidence for vec

    conf = [getKNNclass(vec,
                        (mfcc_feat_s, mfcc_feat_o, mfcc_feat_ri, mfcc_feat_t,mfcc_feat_N), 27)
            for vec in mfcc_feat_test.data]

    frameindex = np.arange(mfcc_feat_test.shape[0])

    plt.scatter(frameindex, [i['s']
                for i in conf], s=4, c='green', alpha=0.7)
    plt.scatter(frameindex, [i['o'] for i in conf], s=4, c='red', alpha=0.7)
    plt.scatter(frameindex, [i['ri']
                for i in conf], s=4, c='yellow', alpha=0.7)
    plt.scatter(frameindex, [i['t']
                for i in conf], s=4, c='blue', alpha=0.7)
    plt.scatter(frameindex, [i['N'] for i in conf], s=4, c='grey', alpha=0.5)

    plt.legend(["s", "o", "ri", "t", "N"])
    plt.ylabel("Prob")
    plt.xlabel("Frames")

    plt.show()
