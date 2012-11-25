from classifiers import DummyClassifier, SVMClassifier, \
    SumOfSquareErrorClassifier, NonLinearSVM
from numpy import float64
from numpy.core.fromnumeric import transpose
from numpy.core.numeric import count_nonzero, arange, ndarray, inf
from numpy.lib.shape_base import vsplit
from numpy.linalg.linalg import norm
from numpy.ma.core import exp
from numpy.ma.extras import dot
from numpy.random import shuffle
from scipy.io.matlab.mio import loadmat, savemat
import os
import pylab
import sys

def splitZipData(name,Ntrain=None,Nclassify=None):
    data = loadmat('data/' + name + '.mat')
    trainData = transpose(data[name+'_train'])
    testData = transpose(data[name+'_test'])
    shuffle(trainData)
    trainData = trainData[:Ntrain]
    shuffle(testData)
    testData = testData[:Nclassify]
    trainingLabels,trainingSet = trainData[:,0],trainData[:,1:]
    testLabels,testSet = testData[:,0],testData[:,1:]
    return trainingSet,trainingLabels,testSet,testLabels

# def splitCerealData(rTrain = 0.5):
#    data = loadmat('data/cereals.mat')['data']
#    shuffle(data)
#    N = int(data.shape[0]*rTrain)
#    trainingSet = data[:N,1:]
#    trainingLabels = data[:N,:1]
#    testSet = data[N:,1:]
#    testLabels = data[N:,:1]
#    return trainingSet,trainingLabels,testSet,testLabels

if __name__ == '__main__':
    #a3
    linearKernel = lambda x,xp:dot(x,xp)
    polyKernel = lambda x,xp:(dot(x,xp)+1.0)**2.0
    rbfKernel = lambda x,xp:exp((norm(x-xp)**2.0)/(-2.0*9.0))
    
    for s in ['zip13','zip38']:
        kernels = dict((k,NonLinearSVM(C, eval(k))) for k,C in zip(['linearKernel','polyKernel','rbfKernel'],[0.1,0.1,1.0]))
        testErrors = dict()
        
        trainS,trainL,testS,testL = splitZipData(s,300,300)
        for n,k in kernels.viewitems():
            k.train(trainS,trainL)
            mcr = k.test(testS,testL)
            testErrors[n] = mcr
        
        print s
        print 'Kernel\t\t\ttrainingError\ttestError\tnumber of support vectors'
        for n,k in kernels.viewitems():
            print n,'\t\t',"%5f"%(k.trainingError),'\t\t',"%5f"%(testErrors[n]),'\t',k.numSupportVectors
