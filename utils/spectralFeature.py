import os
import numpy as np
import librosa
from . import preprocess as pps
from .vad_utils import read_label_from_file, prediction_to_vad_label

def getmelFeature(frameData, sampleRate,
                   frame_size: float=0.032, frame_shift: float=0.008,
                   NFFT=512, NMELS=40):
    fftData = np.fft.rfft(frameData.T)
    periodogram = np.abs(fftData) ** 2 / NFFT
    melFBank = librosa.filters.mel(sr=sampleRate, n_fft=NFFT, n_mels=NMELS)
    melData = np.dot(periodogram, melFBank.T)
    return melData

def getMFCC(melData, n_mfcc=40):
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melData), n_mfcc=n_mfcc)
    return mfcc

# since the implementation in `preprocess.py` ocupying too much memory
# We use another way to read training data
def spectralData(waveDirPath: str, frame_size: float=0.032, frame_shift: float=0.008,
                 NFFT=512, NMELS=40, NMFCC=20, labelDirPath=None, featType='mfcc'):
    '''Generate data with spectral feature (and its coresponding label)
    
    Return:
        spectralData: np.ndarray[shape=(frameNum, NFFT/2 + 1)]
            or mfcc with shape=(N_MFCC=40, frameNum)
        label
    '''
    assert featType == 'mfcc' or featType =='mel', 'No such kind of feat{}'.format(featType)

    tmpsavePath = 'tmpData'
    if (featType == 'mfcc'):
        if os.path.exists(tmpsavePath + '/' + featType +'_' + str(NMELS) +'_specX.npy'):
            trainX = np.load(tmpsavePath + '/' + featType + '_' + str(NMELS) +'_specX.npy')
            trainY = np.load(tmpsavePath + '/' + 'specY.npy')
            return trainX, trainY
    else:
        if os.path.exists(tmpsavePath + '/' + featType +'_' + str(NMFCC)+ '_specX.npy'):
            trainX = np.load(tmpsavePath + '/' + featType + '_' + str(NMFCC) +'_specX.npy')
            trainY = np.load(tmpsavePath + '/' + 'specY.npy')
            return trainX, trainY

    wavfiles = os.listdir(waveDirPath)
    specData = np.zeros((1, NMELS ))

    if labelDirPath != None:
        wavelabel = read_label_from_file(labelDirPath, frame_size, frame_shift)
        labelist = []
         
    for fileName in wavfiles:
        waveID = fileName.split('.')[0]
        frameData, sampleRate = pps.enframe(waveDirPath + '/' + fileName, frame_size, frame_shift)
        spectralData = getmelFeature(frameData, sampleRate, frame_size, frame_shift, NFFT, NMELS)
        specData = np.concatenate((specData, spectralData), axis=0)
        
        if labelDirPath != None:
            frameNum = frameData.shape[1]
            label = wavelabel[waveID]
            labelPad = np.pad(label, (0, np.maximum(frameNum - len(label), 0)))[:frameNum]
            labelist.append(labelPad)
    
    label = np.concatenate(labelist)
    np.save("tmpData/specY.npy", label)
    np.save("tmpData/mel_" + str(NMELS) + "_specX.npy", specData[1:])
    if (featType == 'mel'):
        return specData[1:], label 
    else:
        mfcc = getMFCC(specData[1:].T, n_mfcc=NMFCC)
        np.save("tmpData/mfcc_" + str(NMFCC) + "_specX.npy", mfcc) 
        return mfcc, label 
        

            
if __name__ == '__main__':
   specdata, label = spectralData('../data/dev', labelDirPath='../data/dev_label.txt')
   print(specdata.shape, label.shape)
