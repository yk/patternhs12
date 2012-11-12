from cvxopt.coneprog import qp
from numpy import float64
from numpy.core.fromnumeric import transpose, nonzero
from numpy.core.numeric import ndarray, count_nonzero, ones, zeros, Inf, \
    identity, inf, where
from numpy.lib.shape_base import tile
from numpy.lib.twodim_base import diag
from numpy.ma.core import max
from numpy.ma.extras import dot
from numpy.matrixlib.defmatrix import mat
from openopt.oo import QP
from openopt.solvers.CVXOPT.cvxopt_misc import matrix
import numpy

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

class SVMClassifier(LinearClassifier):
    def __init__(self,C):
        self.C = C
        
    def classify(self, data):
        t = dot(data,self.w.reshape((-1,1))) + self.w0
        return t > 0
    
    def train(self, data, labels):
        l = labels.reshape((-1,1))
        xy = data * l
        H = dot(xy,transpose(xy))
        f = -1.0*ones(labels.shape)
        lb = zeros(labels.shape)
        ub = self.C * ones(labels.shape)
        Aeq = labels
        beq = 0.0
        p = QP(matrix(H),f.tolist(),lb=lb.tolist(),ub=ub.tolist(),Aeq=Aeq.tolist(),beq=beq)
        r = p.solve('cvxopt_qp')
        r.xf[where(r.xf<1e-3)] = 0
        self.w = dot(r.xf*labels,data)
        nonzeroindexes = where(r.xf>1e-4)[0]
        l1 = nonzeroindexes[0]
        self.w0 = 1.0/labels[l1]-dot(self.w,data[l1])
        
    
class LogisticRegressionClassifier(LinearClassifier):
    def classify(self, data):
        pass
    
    def train(self, data, labels):
        pass
    