import numpy as np
from preprocess import getFrameSample

def ZCR(frameData):
    frameNum = frameData.shape[1]
    frameSize = frameData.shape[0]
    zcr = np.zeros((frameNum,1))
    '''
    for i in range(frameNum):
        singleFrame = frameData[:,i]
        temp = singleFrame[:frameSize-1]*singleFrame[1:frameSize]
        temp = np.sign(temp)
        zcr[i] = np.sum(temp<0)
    '''
    tmp = frameData[:][:frameSize-1] * frameData[:][1:frameSize]
    zcr = np.sum(tmp<0, axis=0)
    return zcr

def energy(frameData):
    ener = np.linalg.norm(frameData, ord=2, axis=0)
    return ener

if __name__ == "__main__":
    dataSample = getFrameSample()
