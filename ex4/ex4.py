from classifiers import DummyClassifier, SVMClassifier, \
    SumOfSquareErrorClassifier
from numpy import float64
from numpy.core.fromnumeric import transpose
from numpy.core.numeric import count_nonzero, arange, ndarray, inf
from numpy.lib.shape_base import vsplit
from numpy.random import shuffle
from scipy.io.matlab.mio import loadmat, savemat
import os
import pylab
import sys


def doTest(dataSetName, classifier):
    allData = loadmat('data/' + dataSetName + '.mat')
    trainLblsAndData = transpose(allData[dataSetName + '_train'])
    testLblsAndData = transpose(allData[dataSetName + '_test'])
    trainLabels, trainData = trainLblsAndData[:, 0], trainLblsAndData[:, 1:]
    testLabels, testData = testLblsAndData[:, 0], testLblsAndData[:, 1:]
    classifier.train(trainData, trainLabels)
    mcR = classifier.test(testData, testLabels)
    return mcR
    
def shuffleSkin():
    skin = loadmat('data/skinorig.mat')
    st = transpose(skin['skin_train'])
    shuffle(st)
    skin['skin_train'] = transpose(st[:2000])
    savemat('data/skin.mat',skin)

if __name__ == '__main__':
    suppressOut = True
    if suppressOut:
        devnull = open('/dev/null', 'w')
        oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)
    shuffleSkin()
    dataSets = ['toy', 'zip13', 'zip38', 'skin']
    svmsC = [15.0,10.0,0.8,0.35]
    ssqMcrs = [doTest(e, SumOfSquareErrorClassifier()) for e in dataSets]
    #svmMcrs = [doTest(e[0], SVMClassifier(C=e[1])) for e in zip(dataSets, svms)]
    svms = [SVMClassifier(C=e) for e in svmsC[:len(svmsC)]]
    svmMcrs = [doTest(e[0], e[1]) for e in zip(dataSets, svms)]
    if suppressOut:
        os.dup2(oldstdout_fno, 1)
    print '\n\nmisclassification rates'
    print 'sum of squares error:'
    for e in zip(dataSets, ssqMcrs):
        print str(e[0]) + ": " + str(e[1])
    print 'support vector machine:'
    for e in zip(dataSets, svmMcrs,svms):
        print str(e[0]) + ": " + str(e[1]) + " number of support vectors: " + str(e[2].numSupportVectors)
    
