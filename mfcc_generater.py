from distutils.log import error
from sqlite3 import DataError
from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import delta

import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os

from dataset_controller import *


if __name__ == '__main__':

    data_set_dir = os.getcwd() + '.'

    sig = np.array([])
    mfcc_feat = np.reshape(np.array([]), (0, 16))
    mfcc_feat_delta = np.reshape(np.array([]), (0, 16))

    for wavfile in os.listdir(data_set_dir):
        if 'test_0501.wav' in wavfile:
            print(wavfile)
            (rate, wavdata) = wav.read(data_set_dir + '\\' + wavfile)
            if not rate == 16000:
                raise DataError
            sig = np.hstack((sig, wavdata))
            mfcc_feat = np.vstack(
                (mfcc_feat, mfcc(wavdata, rate, numcep=17, nfilt=26)[:, 1:]))  # Dim 1 - 16
            mfcc_feat_delta = np.vstack((mfcc_feat_delta, delta(mfcc_feat, 5)))

    # clc(sig,)

    # mfcc_feat = mfcc(sig, rate, numcep=17, nfilt=26)[:,1:]
    # mfcc_feat_delta = delta(mfcc_feat, 1)
    # mfcc_feat=np.hstack((mfcc_feat,mfcc_feat_delta))

    logfbank_feat = logfbank(sig, rate, nfilt=26)

    print(rate, sig.shape)
    print(mfcc_feat.shape)
    print(logfbank_feat.shape)

    plt.subplots_adjust(left=0.15, hspace=0.5)

    plt.subplot(311)
    plt.plot(np.arange(sig.size)/rate, sig)
    plt.xlim([0, sig.size/rate])
    plt.xlabel('Time')
    plt.ylabel('Signal')

    plt.subplot(312)
    branch = 0
    if branch == 1:
        plt.plot(logfbank_feat)
        plt.xlim([0, logfbank_feat.shape[0]])
    else:
        plt.imshow(logfbank_feat.T, origin='lower',
                   aspect='auto', interpolation='nearest')
    plt.xlabel('Frame Index')
    plt.ylabel('Log Filter Banks')

    plt.subplot(313)
    branch = 0
    if branch == 1:
        plt.plot(mfcc_feat)
        plt.xlim([0, mfcc_feat.shape[0]])
    else:
        plt.imshow(mfcc_feat.T, origin='lower',
                   aspect='auto', interpolation='nearest')
    plt.xlabel('Frame Index')
    plt.ylabel('MFCC Coefficient Index')

    plt.show()

    showData(mfcc_feat.T)

    storeData(mfcc_feat, filename='sory_test_0501_', dir='MFCC')
