import numpy as np
import time
from tqdm import tqdm
from utils.preprocess import makeTrainData, readDataset, aggregateFeature
from utils.evaluate import get_metrics
from model.state_machine import stateMachine
from utils.smoothing import averageSmooth
from utils.vad_utils import prediction_to_vad_label
from sklearn.linear_model import LogisticRegression as LogiReg
from sklearn.mixture import GaussianMixture as GMM
from model.GMM_Classifier import GMMClassifier as myGMM
from utils.spectralFeature import getMFCC, spectralData

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

def trainGMM(model):
    NMFCC = 20
    RANDOM_SEED = 1337

    specdata, label = spectralData('/Volumes/T7/vad/wavs/train', labelDirPath='/Volumes/T7/vad/data/train_label.txt', NMFCC=NMFCC)
    model.fit(specdata.T, label)
    trainy = model.predict(specdata.T)
    evalPrint(trainy, label, 'TRAIN GMM without smoothed (winlen=30) spectral feat , NMFCC={}, N_COMPONENTS={} \t'.format(NMFCC, N_COMPONENTS), 'log.txt')

def predictGMM(model, x, winlen=30, th=0.35):
    y = model.predict(x)
    newy = averageSmooth(y, winlen)
    finaly = newy > th 
    return finaly

if __name__ =='__main__':
    testPath = 'wav/test'    
    labelOutPath = 'train_label_task2_GMM.txt'
    # para of State Machine
    lowerTh = [326, 557]
    upperTh = [95, 2425]
    # para of LR
    windowLen = 40
    # para of GMM
    N_COMPONENTS = 3
    RANDOM_SEED = 1337
    
    model = 'GMM'
    featType = 'MFCC'
    testset = readTestset(testPath=testPath)

    if ( model == 'LR'):
        classifier = LogiReg()
        trainSmoothLR(classifier, winLen=20) 
    elif (model == 'GMM'):
        classifier = myGMM( [GMM(n_components=N_COMPONENTS, covariance_type='full', random_state=RANDOM_SEED),
                    GMM(n_components=N_COMPONENTS, covariance_type='full', random_state=RANDOM_SEED)])
        trainGMM(classifier)

    for index, sound in tqdm(testset.items()):
        if (featType == 'Time'):
            feat = aggregateFeature(sound)
        elif (featType == 'MFCC'):
            feat = getMFCC(sound).T
            

        if (model =='StateMachine'):   
            prediction = stateMachine(feat, lowerTh, upperTh)
            prediction = averageSmooth(prediction, windowLen)
        elif (model =='LR'):
            prediction = predict(classifier, winLen=20, data=feat)
        elif (model =='GMM'):
            prediction = predictGMM(gmm, x=feat, winlen=30, th=0.35)

        label = prediction_to_vad_label(prediction, threshold = 0.4) 
        with open(labelOutPath, 'a') as f:
            label = index + ' ' + label + '\n'
            f.write(label)


