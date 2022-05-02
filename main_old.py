"""
main_old.py
---------------------------------------------------------------
Because of the high cpu occupation while scattering probs figure,
the program can not continuously record voice.
---------------------------------------------------------------
Created by Weber, 22.5.2022.
---------------------------------------------------------------
"""
from dataset_controller import *
from mfcc_generater import *
from recorder import *
from KNN import *
import matplotlib.animation as animation

if __name__ == '__main__':

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    FRAMERATE = 16000
    CHUNKLEN = 400

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=FRAMERATE,
                    input=True,
                    frames_per_buffer=CHUNKLEN)  # open stream and setting
    wf = wave.open('temp.wav', 'wb')  # open .wav and setting
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(FRAMERATE)

    mfcc_feat_s = KNNclass(
        loadFeat(featfilename='s_', dir='MFCC', N=70), 's')
    mfcc_feat_o = KNNclass(
        loadFeat(featfilename='o_mid_', dir='MFCC', N=50), 'o')
    mfcc_feat_ri = KNNclass(
        loadFeat(featfilename='ri_mid_', dir='MFCC', N=50), 'ri')
    mfcc_feat_N = KNNclass(loadFeat(featfilename='N_', dir='MFCC', N=27), 'N')

    setColor('y')
    print('Listening...')
    ersColor()

    time_start = time.time()
    fig = plt.figure()  # 生成画布 #
    plt.ion()  # 打开交互模式 #
    data_cache = [{'s': 0.0, 'o': 0.0, 'ri': 0.0, 'N': 0.0}
                  for _ in range(100)]
    frame_index = np.arange(len(data_cache))

    for _ in range(100):
        try:
            data_bytes = stream.read(CHUNKLEN)
            wf.writeframes(data_bytes)
            # array contain sigal in a frame
            data_int = np.fromstring(data_bytes, dtype=np.int16)

            vol = np.sum(np.abs(data_int))/CHUNKLEN
            frameconf = getKNNclass(
                mfcc(data_int, FRAMERATE, numcep=17, nfilt=26)[
                    :, 1:], (mfcc_feat_s, mfcc_feat_o, mfcc_feat_ri, mfcc_feat_N),
                27)

            data_cache.pop(0)
            data_cache.append(frameconf)

            fig.clf()  # 清空当前Figure对象 #
            fig.suptitle("3d io pic")

            ax = fig.add_subplot(111)
            plt.scatter(frame_index, [i['s']
                        for i in data_cache], s=4, c='green', alpha=0.7)
            plt.scatter(frame_index, [i['o']
                        for i in data_cache], s=4, c='red', alpha=0.7)
            plt.scatter(frame_index, [i['ri']
                        for i in data_cache], s=4, c='yellow', alpha=0.7)
            plt.scatter(frame_index, [i['N']
                        for i in data_cache], s=4, c='grey', alpha=0.5)
            

            ax.legend(["s", "o", "ri", "N"])
            plt.ylabel("Prob")
            plt.xlabel("Frames")

            plt.pause(0.0001)
            # print(frameconf)

        except KeyboardInterrupt:
            break

    time_end = time.time()
    plt.ioff()
    plt.show()

    setColor('y')
    print('Time cost: ', end='')
    setColor('r')
    print(time_end - time_start, 's', sep=' ')
    setColor('y')
    print('Listening Complete.')
    ersColor()

    stream.stop_stream()  # close stream
    stream.close()
    p.terminate()
    wf.close()
