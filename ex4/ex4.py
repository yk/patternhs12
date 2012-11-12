from classifiers import DummyClassifier, SumOfSquareErrorClassifier
from numpy import float64
from numpy.core.fromnumeric import transpose
from numpy.core.numeric import count_nonzero
from numpy.lib.shape_base import vsplit
from scipy.io.matlab.mio import loadmat


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