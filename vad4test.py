import numpy as np
import time
from utils.preprocess import makeTrainData, readDataset, aggregateFeature
from utils.evaluate import get_metrics
from state_machine import stateMachine
from utils.smoothing import averageSmooth
from utils.vad_utils import prediction_to_vad_label
from sklearn.linear_model import LogisticRegression as LogiReg

def readTestset(testPath, frame_size: float=0.032, frame_shift: float=0.008):
    '''
    Return:
        testset: (dictinary) a mapping from a wave ID to its **frameData**
            e.g. {
                 "1031-133220-0062": [[0, 0, ...], 
                                      [0, 0, ...],
                                      ...........,
                                      [0, 0, ...]] ,
                 "1031-133220-0091": ....
                }
    '''
    testset = readDataset(testPath, frame_size, frame_shift)
    return testset
    

def trainSmoothLR(classifier, winLen=20):
    # train 
    trainXX, trainY = makeTrainData(trainPath='data/dev', labelPath='data/dev_label.txt', frame_size=0.032, frame_shift=0.008)
    trainXX = trainXX[:,1:]
    trainX = np.copy(trainXX.T)
    for i in [0,1]:
        trainX[:,i] = averageSmooth(trainXX[i,:], winLen)
    classifier.fit(trainX, trainY)

def predict(classifier, winLen, data):
    predata = np.copy(data.T)
    for i in [0,1]:
        predata[:,i] = averageSmooth(data[i,:], winLen)
    label = classifier.predict(predata)   
    return label

if __name__ =='__main__':
    testPath = '/Volumes/T7/vad/wavs/train'    
    labelOutPath = 'train_label_task1_SM.txt'
    lowerTh = [326, 557]
    upperTh = [95, 2425]
    windowLen = 40
    classifier = LogiReg()

    trainSmoothLR(classifier, winLen=20) 

    testset = readTestset(testPath=testPath)
    for index, sound in testset.items():
        feat = aggregateFeature(sound)
        ''' state machine '''
        prediction = stateMachine(feat, lowerTh, upperTh)
        prediction = averageSmooth(prediction, windowLen)
        ''' LR
        prediction = predict(classifier, winLen=20, data=feat)
        '''
        label = prediction_to_vad_label(prediction, threshold = 0.4) 
        with open(labelOutPath, 'a') as f:
            label = index + ' ' + label + '\n'
            f.write(label)


