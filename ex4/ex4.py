from numpy import float64
from numpy.core.fromnumeric import transpose
from numpy.core.numeric import count_nonzero
from scipy.io.matlab.mio import loadmat
from numpy.lib.shape_base import vsplit

def calculateMisclassificationRate(ours, real):
    return 1.0 - count_nonzero(ours + real).astype(float64) / ours.size

class LinearClassifier:
    def train(self, data, labels):
        '''input: training data and corresponding labels(1,-1), output: nothing'''
        pass
    
    def test(self, data, labels):
        ourLabels = self.classify(data)
        return calculateMisclassificationRate(ourLabels, labels)
    
    def classify(self, data):
        '''input: data to classify, output: array of labels (1/-1)'''
        pass

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