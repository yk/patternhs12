from classifiers import DummyClassifier, SVMClassifier
from numpy import float64
from numpy.core.fromnumeric import transpose
from numpy.core.numeric import count_nonzero, arange, ndarray, inf
from numpy.lib.shape_base import vsplit
from scipy.io.matlab.mio import loadmat
import pylab


def doTest(dataSetName, classifier):
    allData = loadmat('data/' + dataSetName + '.mat')
    trainLblsAndData = transpose(allData[dataSetName+'_train'])
    testLblsAndData = transpose(allData[dataSetName+'_test'])
    trainLabels, trainData = trainLblsAndData[:,0],trainLblsAndData[:,1:]
    testLabels,testData = testLblsAndData[:,0],testLblsAndData[:,1:]
    classifier.train(trainData,trainLabels)
    mcR = classifier.test(testData,testLabels)
    return mcR
    

if __name__=='__main__':
    myClassifier = DummyClassifier()
    micRate1 = doTest('toy',myClassifier)
    print micRate1
    svm = SVMClassifier(C=15.0)
    micRate2 = doTest('toy',svm)
    print micRate2
    svm2 = SVMClassifier(C=inf)
    micRate3 = doTest('zip13',svm2)
    print micRate3