import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LogiReg
from sklearn.linear_model import LinearRegression 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from utils.evaluate import get_metrics
import utils.preprocess as pps
import utils.time_feature_extraction as tfe



#from state_machine import stateMachine
if __name__ == "__main__":
    lda = LDA()
    classifier = LogiReg()
    # train 
    startTime = time.time()
    #print('collecting data...')
    trainX, trainY = pps.makeTrainData(trainPath='data/dev', labelPath='data/dev_label.txt', frame_size=0.032, frame_shift=0.008)
    trainX = trainX[:,1:].T
    endTime = time.time()
    #print('complete, it takes {:.2f}s'.format(startTime-endTime))
    #print('start training')
    startTime = time.time()
    trainX_lda = lda.fit_transform(trainX, trainY)
    classifier.fit(trainX_lda, trainY)
    #print("training completed with {:.2f}s,\n the coeffs are ".format(endTime-startTime), classifier.coef_)

    # test with the training data
    #testY = classifier.predict(trainX_lda)
    testY = lda.predict(trainX)
    #print('testY shape: ', testY.shape[0])
    print('with less {}: '.format(0.7), (testY<0.7).sum())
    print('with more 0.85: ', (testY>=0.85).sum())
    auc, eer = get_metrics(testY, trainY) 
    print("auc: {}".format(auc), "eer: {}".format(eer))


    # give some baselines
    allTrueY = np.ones_like(trainY)
    auc1, eer1 = get_metrics(allTrueY, trainY)
    #print('baseline1 (All True): \n',"auc: {}".format(auc1), "eer: {}".format(eer1))
    allFalseY= np.zeros_like(trainY)
    auc2, eer2 = get_metrics(allFalseY, trainY)
    #print('baseline2 (All False): \n',"auc: {}".format(auc2), "eer: {}".format(eer2))


    '''
    # add some postprocess
    # here is statemachine
    print('postprocessing....')
    startTime = time.time()
    gridsearchAUC = np.zeros((16,16))
    gridsearchEER = np.zeros((16,16))
    for i, uTh in enumerate(np.arange(0.8,0.95,0.01)):
        for j, lTh in enumerate(np.arange(0.65,0.8,0.01)):
            newY = stateMachine(testY, upperTh = uTh, lowerTh = lTh)
            gridsearchAUC[i][j], gridsearchEER[i][j] = get_metrics(newY, trainY) 
    
    endTime = time.time() 
    print('done! it takes {:.4f}s'.format(endTime-startTime))

    print(gridsearchAUC, gridsearchEER)

    print("auc: {}".format(np.amax(gridsearchAUC)), "eer: {}".format(np.amin(gridsearchEER)))
    print("in {}".format(np.argmax(gridsearchAUC)), 'in {}'.format(np.argmin(gridsearchEER))) 
    
    '''
    # generate the prediction label for the test wavefile
