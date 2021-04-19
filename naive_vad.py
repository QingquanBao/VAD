import numpy as np
import matplotlib.pyplot as plt
import preprocess as pps
import time_feature_extraction as tfe
from sklearn.linear_model import Perceptron

def aggregateFeature(frameData):
    zcr = tfe.ZCR(frameData)
    ener = tfe.energy(frameData)
    features = np.concatenate((zcr,ener), axis=1)
    return features

def predictVAD(frameData, label):
    features = aggregateFeature(framedata)
    
    # use linear classifier
    classifier = Perceptron(random_state=10086)
    # train
    classifier.fit(features, label)
    
    
