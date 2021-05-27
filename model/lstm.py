import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class RNN(torch.nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=20, num_layers=2, batch_first=True)
        self.out = torch.nn.Sequential(torch.nn.Linear(20,10),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(10,1),
                                       torch.nn.Sigmoid())

    def forward(self, x):
        rnn_out, (h_n,h_c) = self.rnn(x, None)
        out_ = self.out(rnn_out)
        return out_

class SoundData(Dataset):
    def __init__(self, data, label):
        self.size = int(data.shape[0] / 1024 )
        self.data = data[:self.size*1024]
        self.label = label[:self.size*1024]  

    def __getitem__(self, index):
        start = index * 1024
        end = start + 1024
        return self.data[start:end], self.label[start:end]

    def __len__(self):
        return self.size

if __name__ == '__main__':
    # If u want to use it, please put this file in the father dir
    from utils.evaluate import evalPrint
    epoch_NUM = 128
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    melX = np.load('tmpData/mel_40_specX.npy')
    melY = np.load('tmpData/specY.npy')
    testX = np.load('tmpData/dev/mel_40_specX.npy')
    testY = np.load('tmpData/dev/specY.npy')
    testx = torch.tensor(testX).cuda().reshape(1,testX.shape[0],-1).to(torch.float32)

    meldata = SoundData(melX, melY)
    dataloader = DataLoader(meldata, batch_size=128)

    rnn = RNN(40).cuda()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.009)
    loss_func = torch.nn.MSELoss()

    for epoch in tqdm(range(epoch_NUM)):
        for i, [feat, label] in enumerate(dataloader):
            feat = feat.to(torch.float32).cuda()
            label = label.to(torch.float32).cuda()
            y_hat = rnn(feat).squeeze()
            loss = loss_func(y_hat, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 16 == 15 :      
            testy = rnn(testx).squeeze().detach().cpu().numpy()
            evalPrint(testy, testY, 'lstm epoch{} '.format(epoch))
        if epoch % 32 == 31 :
            torch.save(rnn.state_dict(), 'model/epoch{}.pt'.format(epoch))

    finaly = testy > 0.5
    evalPrint(finaly, testY, 'lstm final ')

    
    '''
    testPath = 'wav/dev'    
    devlabeldirpath = 'wav/dev_label.txt'
    testset = readTestset(testPath=testPath)
    wavelabel = read_label_from_file(devlabeldirpath)
    for index, sound in tqdm(testset.items()):
        pred = rnn(getmelFeature(sound))
    '''
