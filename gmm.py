import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.linear_model import LogisticRegression as LogiReg
import utils.preprocess as pps
from utils.spectralFeature import spectralData
from utils.evaluate import evalPrint
from model.GMM_Classifier import GMMClassifier as myGMM
from utils.smoothing import averageSmooth

NMFCC = 20
N_COMPONENTS = 4
RANDOM_SEED = 1337


specdata, label = spectralData('/Volumes/T7/vad/wavs/train', labelDirPath='/Volumes/T7/vad/data/train_label.txt', NMFCC=NMFCC)
testX = np.load('tmpData/dev/mfcc_20_specX.npy')
testY = np.load('tmpData/dev/specY.npy')
print('data collection completed')

for N_COMPONENTS in [2, 3]:
    model = myGMM( [GMM(n_components=N_COMPONENTS, covariance_type='full', random_state=RANDOM_SEED),
                    GMM(n_components=N_COMPONENTS, covariance_type='full', random_state=RANDOM_SEED)])

    model.fit(specdata.T, label)
    trainy = model.predict(specdata.T)
    evalPrint(trainy, label, 'TRAIN GMM without smoothed (winlen=30) spectral feat , NMFCC={}, N_COMPONENTS={} \t'.format(NMFCC, N_COMPONENTS))
    pred_test = model.predict(testX.T)
    evalPrint(pred_test, label, 'TEST GMM without smoothed (winlen=30) spectral feat , NMFCC={}, N_COMPONENTS={} \t'.format(NMFCC, N_COMPONENTS))
    

    winlen = 30 
    y = averageSmooth(y, winlen)
    pred_test = averageSmooth(pred_test, winlen)
    evalPrint(trainy, label, 'TRAIN GMM with smoothed (winlen=30) spectral feat , NMFCC={}, N_COMPONENTS={} \t'.format(NMFCC, N_COMPONENTS))
    evalPrint(trainy, label, 'TEST GMM with smoothed (winlen=30) spectral feat , NMFCC={}, N_COMPONENTS={} \t'.format(NMFCC, N_COMPONENTS))


    for th in [ 0.25, 0.35, 0.45]:
        newy = y > th
        new_testy = pred_test > th
        evalPrint(newy, label, 'TRAIN GMM with spectral feat and post Smooth winlen={}, NMFCC={}, N_COMPONENTS={}, and th={} \t'.format(winlen, NMFCC, N_COMPONENTS, th))
        evalPrint(new_testy, label, 'TEST GMM with spectral feat and post Smooth winlen={}, NMFCC={}, N_COMPONENTS={}, and th={} \t'.format(winlen, NMFCC, N_COMPONENTS, th))

