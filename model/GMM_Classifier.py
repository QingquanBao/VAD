from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import joblib 

class GMMClassifier():
    def __init__(self, model: list):
        ''' Based on Baysian infernece to classify
            If u want to implement a K classifier, u should input M sklearn.mixture.GaussianMixture.
            
        Input: model (list): a series of GMM models defines in sklearn.mixture.GaussianMixture
        '''
        self.model = model
        
    def fit(self, X, y):
        ''' Just like the API in sklearn
        Input: 
            X (np.ndarray[shape:(N, D)]): data
            y (np.ndarray[shape:(N,  )]): label of the coresponding data.
                                          By default, we only accept labels from 0 to K-1, 
                                            where M is the number of labels
        '''
        unique_label = np.unique(y)
        for label in unique_label:
            x = X[y==label]
            self.model[int(label)].fit(x)
            
    def predict(self, X):
        K = len(self.model)
        result = np.zeros((X.shape[0],K))
        for i, gmm in enumerate(self.model):
            result[:,i] = gmm.score_samples(X)
        
        return np.argmax(result, axis=1)
    
    def save(self, outPath:str, index:str):
        for i, gmm in enumerate(self.model):
            joblib.dump(gmm, outPath + '/' + index + 'gmm_{}.pkl'.format(i))

    def load(self, savePath:str):
        for i, gmm in enumerate(self.model):
            self.model[i] = joblib.load(savePath + '{}.pkl'.format(i))
        
            
        
