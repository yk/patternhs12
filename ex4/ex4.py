from classifiers import DummyClassifier, SVMClassifier, \
    SumOfSquareErrorClassifier
from numpy import float64
from numpy.core.fromnumeric import transpose
from numpy.core.numeric import count_nonzero, arange, ndarray, inf
from numpy.lib.shape_base import vsplit
from numpy.random import shuffle
from scipy.io.matlab.mio import loadmat, savemat
import pylab


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
    skin['skin_train'] = transpose(st[:1000])
    savemat('data/skin.mat',skin)

if __name__ == '__main__':
    shuffleSkin()
    dataSets = ['toy', 'zip13', 'zip38', 'skin']
    svms = [15.0,100.0,inf,1.0]
    ssqMcrs = [doTest(e, SumOfSquareErrorClassifier()) for e in dataSets]
    svmMcrs = [doTest(e[0], SVMClassifier(C=e[1])) for e in zip(dataSets, svms)]
    print '\n\nmisclassification rates'
    print 'sum of squares error:'
    for e in zip(dataSets, ssqMcrs):
        print str(e[0]) + ": " + str(e[1])
    print 'support vector machine:'
    for e in zip(dataSets, svmMcrs):
        print str(e[0]) + ": " + str(e[1])
    
