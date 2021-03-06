import os
import numpy as np
import time
import argparse
from tqdm import tqdm
import torch
from utils.preprocess import makeTrainData, readDataset, aggregateFeature
from utils.spectralFeature import getMFCC, spectralData, getmelFeature
from utils.evaluate import get_metrics, evalPrint
from utils.smoothing import averageSmooth
from utils.vad_utils import prediction_to_vad_label, read_label_from_file
from sklearn.linear_model import LogisticRegression as LogiReg
from model.state_machine import stateMachine
from sklearn.mixture import GaussianMixture as GMM
from model.GMM_Classifier import GMMClassifier as myGMM
from model.lstm import RNN

def parse_args():
    parser = argparse.ArgumentParser(description='Label the test file')
    parser.add_argument('--model', type=str, default='LSTM',
                        choices=['LR', 'GMM', 'LSTM', 'StateMachine'])
    parser.add_argument('--featType', type=str, default='MEL',
                        choices=['Time', 'MEL', 'MFCC'])
    parser.add_argument('--gpu_id', type=str, default="2,3")
    parser.add_argument('--testdirPath', type=str, default='wav/dev')
    parser.add_argument('--outPath', type=str, default='dev_label_task2_LSTM.txt')
    parser.add_argument('--testlabel', type=bool, default=False,
                        help='whether the test files have truth label')
    return parser.parse_args()

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

def trainGMM(model, useHistory=False, histPath=None):
    NMFCC = 20


    specdata, label = spectralData('/Volumes/T7/vad/wavs/train', labelDirPath='/Volumes/T7/vad/data/train_label.txt', NMFCC=NMFCC)
    model.fit(specdata.T, label)
    trainy = model.predict(specdata.T)
    evalPrint(trainy, label, 'TRAIN GMM without smoothed (winlen=30) spectral feat , NMFCC={}, N_COMPONENTS={} \t'.format(NMFCC, N_COMPONENTS))

def predictGMM(model, x, winlen=30):
    y = model.predict(x)
    newy = averageSmooth(y, winlen)
    return y

def alignLabel(label, frames):
    label_pad = np.pad(label, (0, np.maximum(frames - len(label), 0)))[:frames]
    return label_pad

if __name__ =='__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id   #????????????????????????gpu??????

    # para of State Machine
    lowerTh = [326, 557]
    upperTh = [95, 2425]
    # para of LR
    windowLen = 40
    # para of GMM
    N_COMPONENTS = 2
    RANDOM_SEED = 1337
    
    testset = readTestset(testPath=args.testdirPath)
    if args.testlabel == True:
        print ('use test label')
        devlabeldirpath = 'wav/dev_label.txt'
        wavelabel = read_label_from_file(devlabeldirpath)

    # Set the model
    if ( args.model == 'LR'):
        classifier = LogiReg()
        trainSmoothLR(classifier, winLen=20) 
    elif (args.model == 'GMM'):
        classifier = myGMM( [GMM(n_components=N_COMPONENTS, covariance_type='full', random_state=RANDOM_SEED),
                    GMM(n_components=N_COMPONENTS, covariance_type='full', random_state=RANDOM_SEED)])
        trainGMM(classifier)
    elif (args.model == 'LSTM'):
        assert (args.featType == 'MEL'), 'Do not support {} + LSTM'.format(args.featType)
        classifier = RNN(40)
        classifier.load_state_dict(torch.load('model/epoch127.pt'))
        classifier.eval()
    
    # Start predicting
    for index, sound in tqdm(testset.items()):
        # GET Feature in feat
        if (args.featType == 'Time'):
            feat = aggregateFeature(sound)
        elif (args.featType == 'MFCC' or 'MEL'):
            feat = getmelFeature(sound) 
            if (args.featType == 'MFCC'):
                feat = getMFCC(feat.T, n_mfcc=20).T
            
        # Predict label file-wise
        if (args.model =='StateMachine'):   
            prediction = stateMachine(feat, lowerTh, upperTh)
            prediction = averageSmooth(prediction, windowLen)
        elif (args.model =='LR'):
            prediction = predict(classifier, winLen=20, data=feat)
        elif (args.model =='GMM'):
            prediction = predictGMM(classifier, x=feat, winlen=40)
        elif (args.model =='LSTM'):
            testx = torch.tensor(feat).reshape(1,feat.shape[0],-1).to(torch.float32)
            prediction = classifier(testx)
            prediction = prediction.squeeze().detach().numpy()
            prediction = averageSmooth(prediction, 15)

        # Transfer to format like '1.1,2.2 2.5,3.7'
        label = prediction_to_vad_label(prediction, threshold = 0.5) 
        with open(args.outPath, 'a') as f:
            if args.testlabel == True:
                thislabel = np.array(wavelabel[index])
                thislabel = alignLabel(thislabel, len(prediction))
                acc = ((prediction > 0.5 ) == thislabel).sum() / len(prediction)
                label = index + ' ' + label + ' acc={}'.format(acc) + '\n'
            else:
                label = index + ' ' + label + '\n'
                
            f.write(label)


