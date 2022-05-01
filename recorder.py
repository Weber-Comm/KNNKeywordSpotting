# audioprocess.py v1.4 by weber
import wave
import pyaudio
import time
import os

def wrecord(path,
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
    print('Time cost %.4f' % (time_end - time_start), ' s.', sep='')
    print('Recording complete. \033[0m')

    stream.stop_stream()  # close stream
    stream.close()
    p.terminate()
    wf.close()


def wget_lct():
    # get projectdict
    projectdict = os.getcwd()

    # get folderdict
    print('\033[33mPlease select a folder:\n>> \033[0m', end='',)
    folder = input()
    folderdict = projectdict + '\\' + folder

    # get files in the folder selected
    files_in_folder = os.listdir(folderdict)
    print(files_in_folder)

    # handle name conflict
    print('\033[33mPlease enter filename:\n>> \033[0m', end='')
    filename = input()
    wavname = filename + '.wav'
    '''
    if wavname in files_in_folder:
        print('\033[35mName Conflict, renaming...\033[0m')
        for i in range(1,20):
            wavname = filename + str(i) + '.wav'
            if not wavname in files_in_folder:
                break
    '''

    return folderdict + '\\' + wavname


if __name__ == "__main__":
    print('\033[33m========== recording.py ==========\033[0m')
    FileLct = wget_lct()
    wrecord(FileLct, 3)
