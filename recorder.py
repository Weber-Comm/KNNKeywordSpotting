import wave
import pyaudio
import time
import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


def setColor(color='r'):
    if color == 'r':    # red
        print('\033[31m',sep='',end='')
    elif color == 'y':  # yellow
        print('\033[33m',sep='',end='')
    elif color == 'p':  # puple red
        print('\033[35m',sep='',end='')
    elif color == 'b':  # blue
        print('\033[34m',sep='',end='')
    elif color == 'g':  # green
        print('\033[32m',sep='',end='')
    elif color == 'c':  # cyan
        print('\033[36m',sep='',end='')

def ersColor():
    print('\033[0m',sep='',end='')


def showInfo(func):
    print('\033[31m%s\033[0m' % (__file__))
    print('\033[31mFunction: %s\033[0m' % (func.__name__))
    func()


def recordWAV(path,
              record_second,
              ):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    FRAMERATE = 16000
    CHUNKLEN = 512
    # typical 16chunks per sec for rate=16000 & CHUNKLEN(length)=1024
    # typical 32chunks per sec for rate=16000 & CHUNKLEN(length)=512

    p = pyaudio.PyAudio()  # instantiate obj

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=FRAMERATE,
                    input=True,
                    frames_per_buffer=CHUNKLEN)  # open stream and setting
    wf = wave.open(path, 'wb')  # open .wav and setting
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(FRAMERATE)

    print('\033[33mStart recording...')
    time_start = time.time()

    chunks = int(FRAMERATE * record_second / CHUNKLEN)
    for _ in range(0, chunks):
        data_bytes = stream.read(CHUNKLEN)
        wf.writeframes(data_bytes)  # writing

    time_end = time.time()
    
    setColor('y')
    print('Time cost: ',end='')
    setColor('r')
    print(time_end - time_start, 's', sep='')
    setColor('y')
    print('Recording Complete.')
    ersColor()

    stream.stop_stream()  # close stream
    stream.close()
    p.terminate()
    wf.close()


def showWAV(path):
    (rate, wavdata) = wav.read(path)
    plt.plot(np.arange(wavdata.size)/rate, wavdata)
    plt.xlim([0, wavdata.size/rate])
    plt.xlabel('Time')
    plt.ylabel('Signal')


def getFilePath(filename=None,
                foldername=None,
                extension='.wav'):

    # get projectdir
    projectdir = os.getcwd()

    if foldername == None:
        # get folderdir
        print('\033[33mPlease select a folder:\n>>> \033[0m', end='',)
        foldername = input()

    folderdir = projectdir + '\\' + foldername

    # get files in the folder selected
    files_in_folder = os.listdir(folderdir)
    print(files_in_folder)

    if filename == None:
        # handle name conflict
        print('\033[33mPlease enter filename:\n>>> \033[0m', end='')
        filename = input()

    while(True):
        if filename + extension in files_in_folder:
            print('\033[35mName Conflict, please select another name...\033[0m')
            print('\033[35m>>> \033[0m', end='')
            filename = input()
        else:
            break

    path = (folderdir + '\\' + filename + extension).replace('.\\', '')
    print(path)
    return path


if __name__ == "__main__":

    FileLct = getFilePath(filename='test0502', foldername='.')

    recordWAV(FileLct, 3)
    showWAV(FileLct)
    plt.show()
