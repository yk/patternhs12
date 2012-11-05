from numpy import float64
from numpy.core.fromnumeric import transpose
from numpy.core.numeric import count_nonzero
from scipy.io.matlab.mio import loadmat
from numpy.lib.shape_base import vsplit


def doTest(dataSetName, classifier):
    allData = loadmat('data/' + dataSetName + '.mat')
    trainLblsAndData = transpose(allData[dataSetName+'_train'])
    testLblsAndData = transpose(allData[dataSetName+'_test'])
    trainLabels, trainData = vsplit(trainLblsAndData, 1)
    testLabels,testData = vsplit(testLblsAndData, 1)
    classifier.train(trainData,trainLabels)
    mcR = classifier.test(testData,testLabels)
    return mcR
    

if __name__=='__main__':
    myClassifier = LinearClassifier()
    micRate1 = doTest('toy',myClassifier)