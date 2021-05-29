import numpy as np
import time
from utils.preprocess import makeTrainData
from utils.evaluate import evalPrint
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

    eval(testY, trainY, 'before smooth')

    # postprocess
    for windowLen in [40]:
        newY = averageSmooth(testY, windowLen)
        eval(newY, trainY, 'after smooth')

    for th in [0.2, 0.25, 0.3, 0.35, 0.4]:
        newnewY = newY > th
        eval(newnewY, trainY, 'with th={}'.format(th))
     
