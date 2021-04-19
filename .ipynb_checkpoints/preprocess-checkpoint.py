import os
import numpy as np
import math
from scipy.io import wavfile

# 分帧处理函数
def enframe(path, frame_size: float=0.032, frame_shift: float=0.008):
    '''Enframe the wave data.

    Args:
        path: of the wav files
            e.g. "data/dev/2000.wav"
        frame_size: the length of the frame in terms of seconds
        frame_shift: the intervals of each frame in terms of seconds

    Return:
        frameData: sequence

    '''
    sample_rate, wavData = wavfile.read(path)
    coeff = 0.97#预加重系数
    wlen = len(wavData)
    frameLength:int = math.ceil(frame_size / (1.0/sample_rate))
    step:int = math.ceil(frame_shift / (1.0/sample_rate))
    frameNum:int = math.ceil(wlen / step)

    frameData = np.zeros((frameLength, frameNum))
    #汉明窗
    hamwin = np.hamming(frameLength)

    for i in range(frameNum):
        singleFrame = wavData[i * step : min(i * step+frameLength,wlen)]
        singleFrame = np.append(singleFrame[0], singleFrame[:-1] - coeff*singleFrame[1:])#预加重
        frameData[:len(singleFrame),i] = singleFrame
        frameData[:,i] = hamwin * frameData[:,i]#加窗
    return frameData

def readDataset(dirPath, frame_size: float=0.032, frame_shift: float=0.008):
    '''Read whole data in certain dir
    
    Return:
        dataset: (dictinary) a mapping from a wave ID to its frameData
            e.g. {
                 "1031-133220-0062": [[0, 0, ...], 
                                      [0, 0, ...],
                                      ...........,
                                      [0, 0, ...]] ,
                 "1031-133220-0091": ....
                }
    '''
    files = os.listdir(dirPath)
    dataset = {}
    for filePath in files:
        wavID = filePath.split('.')[0]
        if wavID in dataset:
            raise RuntimeError(f"{filePath} is duplicated")
        dataset[wavID] = enframe(dirPath + '/' + filePath, frame_size, frame_shift)
    return dataset

###################
###minimal test####
###################

def getFrameSample():
    pth = "data/dev/54-121080-0009.wav"
    return enframe(pth)

def getWaveSample():
    pth = "data/dev/54-121080-0009.wav"
    sample_rate, wavData = wavfile.read(pth)
    return sample_rate, wavData

    
if __name__ == "__main__":
    frameData = getFrameSample()
    print( frameData.shape, '\n', frameData)

    dataset = readDataset("data/dev")
    print( dataset['54-121080-0009'] )
