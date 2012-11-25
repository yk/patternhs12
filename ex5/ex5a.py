from numpy.core.fromnumeric import mean
from numpy.lib.scimath import sqrt
from numpy.linalg import pinv
from numpy.ma.extras import dot
from random import shuffle
from scipy.io import loadmat

def getRootMeanSquaredErrors(calculated,real):
    d = calculated-real
    d = d**2
    m = mean(d)
    sm = sqrt(m)
    return sm

class LinearRegression:
    def train(self,data,labels):
        self.w = dot(pinv(data.newbyteorder('=')),labels)
        self.trainingError = self.classify(data, labels)
        
    def classify(self,data,labels):
        res = dot(data,self.w)
        return getRootMeanSquaredErrors(res, labels)

if __name__ == '__main__':
    data = loadmat('data/cereals.mat')['data']
    shuffle(data)
    r = 0.5
    N = int(data.shape[0]*r)
    trainLabels,trainData = data[:N,:1],data[:N,1:]
    testLabels,testData=data[N:,:1],data[N:,1:]
    regr = LinearRegression()
    regr.train(trainData,trainLabels)
    err = regr.classify(testData, testLabels)
    print 'trainingError: ', regr.trainingError, ', testError: ', err