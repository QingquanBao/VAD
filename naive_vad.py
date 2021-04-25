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
from state_machine import stateMachine
from utils.smoothing import averageSmooth


if __name__ == "__main__":
    print('LDA + Smooth + LR :')
    lda = LDA()
    classifier = LogiReg()
    # train 
    trainXX, trainY = pps.makeTrainData(trainPath='data/dev', labelPath='data/dev_label.txt', frame_size=0.032, frame_shift=0.008)
    trainXX = trainXX[:,1:]
    trainX = np.copy(trainXX.T)
    for winLen in [24,28,32,36,40,60]:
        for i in [0,1]:
            trainX[:,i] = averageSmooth(trainXX[i,:], winLen)
     
        trainX_lda = lda.fit_transform(trainX, trainY)
        #classifier.fit(trainX_lda, trainY)
        #print("training completed with {:.2f}s,\n the coeffs are ".format(endTime-startTime), classifier.coef_)

        # test with the training data
        testY = lda.predict(trainX)
        #testY = classifier.predict(trainX_lda)
        #print('with less {}: '.format(0.7), (testY<0.7).sum())
        #print('with more 0.85: ', (testY>=0.85).sum())
        auc, eer = get_metrics(testY, trainY) 
        print('with windowlen = {}'.format(winLen),"auc: {}".format(auc), "eer: {}".format(eer))


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
