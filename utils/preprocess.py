import os
import numpy as np
import math
from scipy.io import wavfile
from vad_utils import read_label_from_file, prediction_to_vad_label

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
        dataset: (dictinary) a mapping from a wave ID to its **frameData**
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

def makeDataset(dataPath, labelPath, 
                frame_size: float=0.032, frame_shift: float=0.008): 
    '''Create Dataset with features aligned with labels

    Return:
        datalist: (list) frames in each wavefile
        labelist: (list) labels in each wavefile and been padded 
    two outputs are aligned respectively with their ID
    '''
    wavedata = readDataset(dataPath, frame_size, frame_shift)
    wavelabel = read_label_from_file(labelPath, frame_size, frame_shift)
    datalist = []
    labelist = []
    for waveID in wavedata.keys():
        datalist.append(wavedata[waveID])
        frameLen = datalist[-1].shape[1]
        label = wavelabel[waveID]
        labelPad = np.pad(label, (0, np.maximum(frameLen - len(label), 0)))[:frameLen]
        labelist.append(labelPad)
        #if len(labelist) == 2 : print(waveID)
    return datalist, labelist 


def aggregateFeature(frameData):
    zcr = tfe.ZCR(frameData)
    ener = tfe.energy(frameData)
    features = np.stack((zcr,ener))
    return features
    
def makeTrainData( trainPath, labelPath, frame_size: float=0.032, frame_shift: float=0.008):
    dirPath = '../tmpData'
    if os.path.exists(dirPath +'/trainX.npy'):
        trainX = np.load(dirPath + '/trainX.npy')
        trainY = np.load(dirPath + '/trainY.npy')
        return trainX, trainY

    datalist, labelist = makeDataset(trainPath, labelPath, frame_size, frame_shift)
    trainX = np.array([[0],[0]])
    for x in datalist:
        trainX = np.concatenate((trainX,aggregateFeature(x)), axis=1)
    trainY = np.concatenate(labelist)

    # for the sake of running time, we save it 
    if not os.path.exists('../tmpData'):
        os.makedirs('../tmpData') 
    np.save("../tmpData/trainX.npy", trainX)
    np.save("../tmpData/trainY.npy", trainY)
    return trainX[:,1:], trainY 

             
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

############################
#####FUNCTION FOR TEST######
############################
def testfeatures():
    sample = getFrameSample()
    feat = aggregateFeature(sample)
    print( feat.shape, '\n', feat)

    
if __name__ == "__main__":
    #frameData = getFrameSample()
    #print( frameData.shape, '\n', frameData)

    #print( dataset['54-121080-0009'] )
    data, label = makeDataset("data/dev", "data/dev_label.txt")
    print("data[1]'s frame length ", data[1].shape)
    print("label[1]'s frame length: ", label[1].shape)
    print(prediction_to_vad_label(label[1]))
