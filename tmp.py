from tqdm import tqdm
import os
import numpy as np
from utils.spectralFeature import getMFCC

        
mel = np.load("tmpData/mel_" + '40' + "_specX.npy")
mfcc = getMFCC(mel.T, n_mfcc=20)
np.save("tmpData/mfcc_" + '20'  + "_specX.npy", mfcc) 
