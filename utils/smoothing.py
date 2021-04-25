import numpy as np

def averageSmooth(data, windowLen=40):
    return np.convolve(data, np.ones((windowLen,))/windowLen, mode='same')

