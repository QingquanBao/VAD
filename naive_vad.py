import os
import time
import numpy as np
import matplotlib.pyplot as plt
import preprocess as pps
import time_feature_extraction as tfe
from sklearn.linear_model import LogisticRegression as LogiReg
from sklearn.linear_model import LinearRegression 
from evaluate import get_metrics

def aggregateFeature(frameData):
    zcr = tfe.ZCR(frameData)
    ener = tfe.energy(frameData)
    features = np.stack((zcr,ener))
    return features
    
def makeTrainData( trainPath, labelPath, frame_size: float=0.032, frame_shift: float=0.008):
    dirPath = 'tmpData'
    if os.path.exists(dirPath +'/trainX.npy'):
        trainX = np.load(dirPath + '/trainX.npy')
        trainY = np.load(dirPath + '/trainY.npy')
        return trainX, trainY

    datalist, labelist = pps.makeDataset(trainPath, labelPath, frame_size, frame_shift)
    trainX = np.array([[0],[0]])
    for x in datalist:
        trainX = np.concatenate((trainX,aggregateFeature(x)), axis=1)
    trainY = np.concatenate(labelist)

    # for the sake of running time, we save it 
    if not os.path.exists('tmpData'):
        os.makedirs('tmpData') 
    np.save("tmpData/trainX.npy", trainX)
    np.save("tmpData/trainY.npy", trainY)
    return trainX[:,1:], trainY 

def stateMachine(predictLabel, upperTh, lowerTh):
    isVocal = False
    newPredictedLabel = np.zeros_like(predictLabel)
    for i in range(predictLabel.shape[0]):
        if isVocal:
            if predictLabel[i] < lowerTh:
               isVocal = False
               newPredictedLabel[i] = 0
            else:
               newPredictedLabel[i] = 1
        else:
            if predictLabel[i] > upperTh:
               isVocal = True 
               newPredictedLabel[i] = 1
            else:
               newPredictedLabel[i] = 0
    return newPredictedLabel
             
############################
#####FUNCTION FOR TEST######
############################
def testfeatures():
    sample = pps.getFrameSample()
    feat = aggregateFeature(sample)
    print( feat.shape, '\n', feat)


if __name__ == "__main__":
    #testfeatures()
    # use Softmax to classify whether a frame is vocal or not
    classifier = LinearRegression()
    # train 
    startTime = time.time()
    print('collecting data...')
    trainX, trainY = makeTrainData(trainPath='data/dev', labelPath='data/dev_label.txt', frame_size=0.032, frame_shift=0.008)
    trainX = trainX[:,1:].T
    endTime = time.time()
    print('complete, it takes {:.2f}s'.format(startTime-endTime))
    print('start training')
    startTime = time.time()
    classifier.fit(trainX, trainY)
    print("training completed with {:.2f}s,\n the coeffs are ".format(endTime-startTime), classifier.coef_)

    # test with the training data
    testY = classifier.predict(trainX)
    print('testY shape: ', testY.shape[0])
    print('with less {}: '.format(0.7), (testY<0.7).sum())
    print('with more 0.85: ', (testY>=0.85).sum())
    auc, eer = get_metrics(testY, trainY) 
    print("auc: {}".format(auc), "eer: {}".format(eer))


    # give some baselines
    allTrueY = np.ones_like(trainY)
    auc1, eer1 = get_metrics(allTrueY, trainY)
    print('baseline1 (All True): \n',"auc: {}".format(auc1), "eer: {}".format(eer1))
    allFalseY= np.zeros_like(trainY)
    auc2, eer2 = get_metrics(allFalseY, trainY)
    print('baseline2 (All False): \n',"auc: {}".format(auc2), "eer: {}".format(eer2))


    # add some postprocess
    # here is statemachine
    print('postprocessing....')
    startTime = time.time()
    newY = stateMachine(testY, upperTh = 0.85, lowerTh = 0.7)
    endTime = time.time() 
    print('done! it takes {:.4f}s'.format(endTime-startTime))

    auc, eer = get_metrics(newY, trainY) 
    print("auc: {}".format(auc), "eer: {}".format(eer))
    
    
    # generate the prediction label for the test wavefile
