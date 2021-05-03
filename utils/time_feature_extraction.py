import numpy as np
import librosa

def ZCR(frameData):
    frameNum = frameData.shape[1]
    frameSize = frameData.shape[0]
    '''
    for i in range(frameNum):
        singleFrame = frameData[:,i]
        temp = singleFrame[:frameSize-1]*singleFrame[1:frameSize]
        temp = np.sign(temp)
        zcr[i] = np.sum(temp<0)
    '''
    tmp = frameData[:frameSize-1] * frameData[1:frameSize]
    zcr = np.sum(tmp<0, axis=0)
    return zcr

def energy(frameData):
    ener = np.linalg.norm(frameData, ord=2, axis=0)
    return ener

if __name__ == "__main__":
    '''
    dataSample = getFrameSample()
    myzcr = ZCR(dataSample)
    print(myzcr.shape)
    print(myzcr)
    rosaZCR =librosa.zero_crossings(dataSample.T).T.sum(axis=0) 
    print(rosaZCR.shape)
    print(rosaZCR)
    '''
    #from utils.preprocess import getFrameSample
    testdata = np.array( [[1, 2 ,3],
                          [2, 3, -1],
                          [3,-1, 2],
                          [-1, 2,-1],
                          [2, -1, 2]]).T
    print(testdata.shape)
    
    myzcr = ZCR(testdata)
    print(myzcr)
    rosaZCR =librosa.zero_crossings(testdata.T).T.sum(axis=0) 
    print(rosaZCR)
    print(energy(testdata))
