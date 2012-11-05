from numpy import float64
from numpy.core.numeric import ndarray, count_nonzero
def calculateMisclassificationRate(ours, real):
    return 1.0 - count_nonzero(ours + real)*1.0 / ours.size

class LinearClassifier:
    def train(self, data, labels):
        '''input: training data and corresponding labels(1,-1), output: nothing'''
        pass
    
    def test(self, data, labels):
        ourLabels = self.classify(data)
        return calculateMisclassificationRate(ourLabels.flatten(), labels)
    
    def classify(self, data):
        '''input: data to classify, output: array of labels (1/-1)'''
        pass
    
class DummyClassifier(LinearClassifier):
    def classify(self, data):
        labels = ndarray((data.shape[0],1),dtype=bool)
        labels.fill(True)
        return labels
        
    def train(self, data, labels):
        return 0
    
class LogisticRegressionClassifier(LinearClassifier):
    def classify(self, data):
        pass
    
    def train(self, data, labels):
        pass
    