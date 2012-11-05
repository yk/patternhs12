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
    
class LogisticRegressionClassifier(LinearClassifier):
    def classify(self, data):
        pass
    
    def test(self, data, labels):
        pass
    