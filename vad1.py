import numpy as np
import time
from utils.preprocess import makeTrainData
from utils.evaluate import get_metrics
from state_machine import stateMachine
from utils.smoothing import averageSmooth

if __name__ =='__main__':
    
    trainXX, trainY = makeTrainData(trainPath='data/dev', labelPath='data/dev_label.txt', frame_size=0.032, frame_shift=0.008)
    trainXX = trainXX[:,1:]
    trainX = np.copy(trainXX)
    
    '''
    for winLen in [16, 18, 20, 22]:
        print('with preSmooth', winLen)
        for i in [0,1]:
            trainX[i, :] = averageSmooth(trainXX[i,:], 20)
    '''

    lowerTh = [326, 557]
    upperTh = [95, 2425]
    testY = stateMachine(trainX, lowerTh, upperTh)

    auc, eer = get_metrics(testY, trainY)
    print('Before postps, auc = {}, eer = {}'.format(auc,eer))

    # postprocess
    for windowLen in [34,40,48]:
        newY = averageSmooth(testY, windowLen)
        auc, eer = get_metrics(newY, trainY)
        print('With {} len postps, auc = {}, eer = {}'.format(windowLen, auc,eer))

     
