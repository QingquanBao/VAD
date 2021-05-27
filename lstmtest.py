import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.evaluate import evalPrint
from utils.smoothing import averageSmooth
from model.lstm import RNN

testX = np.load('tmpData/dev/mel_40_specX.npy')
testY = np.load('tmpData/dev/specY.npy')
testx = torch.tensor(testX).cuda().reshape(1,testX.shape[0],-1).to(torch.float32)

rnn = RNN(40).cuda()
rnn.load_state_dict(torch.load('model/epoch127.pt'))
rnn.eval()

pred = rnn(testx)
pred = pred.squeeze().detach().cpu().numpy()
predi = averageSmooth(pred, 20)

evalPrint(predi, testY, 'lstm without smooth') 

for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
    finaly = predi > th
    evalPrint(finaly, testY, 'lstm with smooth in th={}'.format(th))
