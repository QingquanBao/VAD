import numpy as np
import time
from naive_vad import makeTrainData
from evaluate import get_metrics

def stateMachine(feats, lowerTh, upperTh):
    ''' Based on two threshold
    Input:
        feats: a np.array with shape (featNum, framesLen)
        lowerTh : (list) featNum-d
        upperTh : (list) featNum-d
            e.g. [0.3, 0.4]
    Return:
        predictedLabel: (np.array) with shape (framesLen, )
    '''    
    assert feats.ndim == len(upperTh), '# of features are NOT matched with the # in threshold'
    assert feats.ndim == len(lowerTh), '# of features are NOT matched with the # in threshold'
    if feats.ndim == 1:
        framesLen = feats.shape[0]
    else:
        framesLen = feats.shape[1]

    isVocal = False
    newPredictedLabel = np.zeros(framesLen)
    for i in range(framesLen):
        if isVocal:
            if (feats[:, i] < lowerTh).all():
               isVocal = False
               newPredictedLabel[i] = 0
            else:
               newPredictedLabel[i] = 1
        else:
            if (feats[:, i] > upperTh).all():
               isVocal = True 
               newPredictedLabel[i] = 1
            else:
               newPredictedLabel[i] = 0
    return newPredictedLabel

if __name__ == '__main__':
    startTime = time.time()
    #print('collecting data...')
    trainX, trainY = makeTrainData(trainPath='data/dev', labelPath='data/dev_label.txt', frame_size=0.032, frame_shift=0.008)
    trainX = trainX[:,1:]
    endTime = time.time()
    #print('complete, it takes {:.2f}s'.format(startTime-endTime))
    
    zcrLowerSearchRange = [np.percentile(trainX[0,:], i) for i in range(55,80,5)]
    enrLowerSearchRange = [np.percentile(trainX[1,:], i) for i in range(25,35,3)]
    zcrUpperSearchRange = [np.percentile(trainX[0,:], i) for i in range(35,45,5)]
    enrUpperSearchRange = [np.percentile(trainX[1,:], i) for i in range(48,58,3)]

    searchSize = (len(zcrLowerSearchRange), len(zcrUpperSearchRange), len(enrLowerSearchRange), len(enrUpperSearchRange))

    print('check the search range', zcrLowerSearchRange, zcrUpperSearchRange, '\n ener', enrLowerSearchRange, enrUpperSearchRange)

    AUC = np.zeros(searchSize)
    EER = np.ones(searchSize)

    for i, zl in enumerate(zcrLowerSearchRange):
        endTime = time.time()
        for j, zu in enumerate(zcrUpperSearchRange):
            for p, el in enumerate(enrLowerSearchRange):
                for q, eu in enumerate(enrUpperSearchRange):
                    Y = stateMachine(trainX, [zl, el], [zu, eu])
                    AUC[i,j,p,q], EER[i,j,p,q] = get_metrics(Y, trainY)
        startTime = time.time()
        print('zl = {}, it takes {}s'.format(zl, startTime-endTime))

    np.save('AUCsearch6.npy', AUC)
    np.save('EERsearch6.npy', EER)
                     
    print("auc: {}".format(np.amax(AUC)), "eer: {}".format(np.amin(EER)))
    print("in {}".format(np.unravel_index(np.argmax(AUC, axis=None), AUC.shape)), 'in {}'.format(np.unravel_index(np.argmin(EER, axis=None), EER.shape))) 
