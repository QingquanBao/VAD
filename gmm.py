import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression as LogiReg
import utils.preprocess as pps
from utils.spectralFeature import spectralData
from utils.evaluate import evalPrint

N_COMPONENTS = 3
RANDOM_SEED = 1337

model = GaussianMixture(n_components=N_COMPONENTS,covariance_type='full', random_state=RANDOM_SEED)
#model = LogiReg()

specdata, label = spectralData('data/dev', labelDirPath='data/dev_label.txt')
'''
print( specdata.shape, label.shape)
print(label[0:40])
'''
model.fit(specdata.T, label)
y = model.predict(specdata.T)
evalPrint(y, label, 'GMM with spectral feat')
