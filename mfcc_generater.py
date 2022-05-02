from python_speech_features import mfcc
# from python_speech_features import delta
# from python_speech_features import logfbank

import numpy as np
import scipy.io.wavfile as wav
import os
from dataset_controller import *


def getMFCC(str_in_wavfilename, dir='.', ret='MFCC', rateconsistency=16000):
    '''
    Get MFCC from WAVfile
    '''
    data_set_dir = os.getcwd() + '\\' + dir

    sig = np.array([])
    mfcc_feat = np.reshape(np.array([]), (0, 16))
    # mfcc_feat_delta = np.reshape(np.array([]), (0, 16))

    for wavfile in os.listdir(data_set_dir):
        if str_in_wavfilename in wavfile:

            (rate, wavdata) = wav.read(data_set_dir + '\\' + wavfile)
            print('\033[33mWavfile %s opened successfully.\033[0m' %
                  (str_in_wavfilename))
            if not rate == rateconsistency:
                print('\033[33mRate Error!!!\033[0m')
                raise ValueError
            sig = np.hstack((sig, wavdata))
            mfcc_feat = np.vstack(
                (mfcc_feat, mfcc(wavdata, rate, numcep=17, nfilt=26)[:, 1:]))  # Dim 1 - 16
            # mfcc_feat_delta = np.vstack((mfcc_feat_delta, delta(mfcc_feat, 5)))

    if 'MFCC' in ret:
        if 'sig' in ret:
            if 'rate' in ret:
                return (mfcc_feat, sig, rate)
            else:
                return (mfcc_feat, sig)
        else:
            return mfcc_feat
    else:
        return None

if __name__ == '__main__':

    mfcc_feat, sig, rate = getMFCC(
        't_', dir='dataset\\sorybot\\t', ret='MFCC + sig + rate')

    # clc(sig,)

    print('\033[33mSigal rate and frame length:\033[0m', rate, sig.shape)
    print('\033[33mMFCC feature shape:\033[0m', mfcc_feat.shape)

    plt.subplots_adjust(left=0.15, hspace=0.5)

    plt.subplot(311)
    plt.plot(np.arange(sig.size)/rate, sig)
    plt.xlim([0, sig.size/rate])
    plt.xlabel('Time')
    plt.ylabel('Signal')

    CHUNKLEN = 160
    chunks = int(sig.size/CHUNKLEN)  # floor
    energy_chunk = np.zeros(chunks)
    sig_iter = iter(sig)
    for i in range(chunks):
        for _ in range(CHUNKLEN):
            energy_chunk[i] += np.abs(next(sig_iter))

    energy_chunk /= CHUNKLEN

    print(energy_chunk)

    plt.subplot(312)
    plt.plot(np.arange(chunks), energy_chunk)
    # plt.xlim([0, logfbank_feat.shape[0]])
    plt.xlabel('Frame Index')
    plt.ylabel('Energy')

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

    showFeat(mfcc_feat.T)
    plt.show()

    storeData(mfcc_feat, filename='t_', dir='MFCC')