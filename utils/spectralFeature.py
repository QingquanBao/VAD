import os
import numpy as np
import librosa
from . import preprocess as pps
from .vad_utils import read_label_from_file, prediction_to_vad_label

def getSpecFeature(frameData, sampleRate,
                   frame_size: float=0.032, frame_shift: float=0.008,
                   NFFT=512, NMELS=40):
    fftData = np.fft.rfft(frameData.T)
    periodogram = np.abs(fftData) ** 2 / NFFT
    melFBank = librosa.filters.mel(sr=sampleRate, n_fft=NFFT, n_mels=NMELS)
    spectralData = np.dot(periodogram, melFBank.T)
    return spectralData

# since the implementation in `preprocess.py` ocupying too much memory
# We use another way to read training data
def spectralData(waveDirPath: str, frame_size: float=0.032, frame_shift: float=0.008,
                 NFFT=512, NMELS=40, labelDirPath=None):
    '''Generate data with spectral feature (and its coresponding label)
    
    Return:
        spectralData: np.ndarray[shape=(frameNum, NFFT/2 + 1)]
    '''
    tmpsavePath = '../tmpData'
    if os.path.exists(tmpsavePath +    '/specX.npy'):
        trainX = np.load(tmpsavePath + '/specX.npy')
        trainY = np.load(tmpsavePath + '/specY.npy')
        return trainX, trainY

    wavfiles = os.listdir(waveDirPath)
    specData = np.zeros((1, NFFT/2 + 1 ))

    if labelDirPath != None:
        wavelabel = read_label_from_file(labelDirPath, frame_size, frame_shift)
        labelist = []
         
    for fileName in wavfiles:
        wavID = fileName.split('.')[0]
        frameData, sampleRate = pps.enframe(dirPath + '/' + fileName, frame_size, frame_shift)
        spectralData = getSpecFeature(frameData, sampleRate, frame_size, frame_shift, NFFT, NMELS)
        specData = np.concatenate((specData, spectralData), axis=0)
        
        if labelDirPath != None:
            frameNum = frameData.shape[1]
            label = wavelabel[waveID]
            labelPad = np.pad(label, (0, np.maximum(frameNum - len(label), 0)))[:frameNum]
            labelist.append(labelPad)
    
    label = np.concatenate(labelist)
    np.save("../tmpData/specX.npy", specData[1:])
    np.save("../tmpData/specY.npy", label)

    return specData[1:],label 
            
