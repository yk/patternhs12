from cvxopt.coneprog import qp
from numpy import float64, int
from numpy.core.fromnumeric import transpose
from numpy.core.numeric import count_nonzero, ndarray, ones, zeros, where, \
    asarray
from numpy.core.shape_base import hstack
from numpy.linalg.linalg import pinv
from numpy.ma.extras import dot
from numpy.random import sample
from openopt.oo import QP
from openopt.solvers.CVXOPT.cvxopt_misc import matrix
import os
import sys

def calculateMisclassificationRate(ours, real):
    return 1.0 - count_nonzero(ours + real)*1.0 / ours.size

class Classifier:
    def __train__(self, data, labels):
        '''input: training data and corresponding labels(1,-1), output: nothing'''
        pass
    
    def train(self,data,labels):
        l = labels.flatten()
        self.__train__(data, l)
        self.trainingError = self.test(data, l)
    
    def test(self, data, labels):
        l = labels.flatten()
        ourLabels = self.__classify__(data)
        return calculateMisclassificationRate(ourLabels.flatten(), l)
    
    def __classify__(self, data):
        '''input: data to __classify__, output: array of labels (1/-1)'''
        pass
    
class DummyClassifier(Classifier):
    def __classify__(self, data):
        labels = ndarray((data.shape[0],1),dtype=bool)
        labels.fill(True)
        return labels
        
    def __train__(self, data, labels):
        return 0

class SVMClassifier(Classifier):
    def __init__(self,C):
        self.C = C
        
    def __classify__(self, data):
        t = dot(data,self.w.reshape((-1,1))) + self.w0
        return t > 0
    
    def __train__(self, data, labels):
        l = labels.reshape((-1,1))
        xy = data * l
        H = dot(xy,transpose(xy))
        f = -1.0*ones(labels.shape)
        lb = zeros(labels.shape)
        ub = self.C * ones(labels.shape)
        Aeq = labels
        beq = 0.0
        devnull = open('/dev/null', 'w')
        oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)
        p = QP(matrix(H),f.tolist(),lb=lb.tolist(),ub=ub.tolist(),Aeq=Aeq.tolist(),beq=beq)
        r = p.solve('cvxopt_qp')
        os.dup2(oldstdout_fno, 1)
        lim = 1e-4
        r.xf[where(r.xf<lim)] = 0
        self.w = dot(r.xf*labels,data)
        nonzeroindexes = where(r.xf>lim)[0]
        l1 = nonzeroindexes[0]
        self.w0 = 1.0/labels[l1]-dot(self.w,data[l1])
        self.numSupportVectors = len(nonzeroindexes)
        
class NonLinearSVM(Classifier):
    def __init__(self,C,kernelFunc):
        self.C = C
        self.__kernelFunc__ = kernelFunc
    
    def __classify__(self, data):
        labels = asarray([sum([self.__lambdas__[i]*self.__trainingLabels__[i]*self.__kernelFunc__(self.__trainingData__[i],x)  for i in range(len(self.__lambdas__))]) for x in data])
        labels = labels.reshape((-1,1))
        return labels>0
        
    
    def __train__(self, data, labels):
        l = labels.reshape((-1,1))
        self.__trainingData__ = data
        self.__trainingLabels__ = l
        N = len(l)
        H = zeros((N,N))
        for i in range(N):
            for j in range(N):
                H[i,j] = self.__trainingLabels__[i]*self.__trainingLabels__[j]*self.__kernelFunc__(self.__trainingData__[i],self.__trainingData__[j])
        f = -1.0*ones(labels.shape)
        lb = zeros(labels.shape)
        ub = self.C * ones(labels.shape)
        Aeq = labels
        beq = 0.0
        suppressOut = True
        if suppressOut:
            devnull = open('/dev/null', 'w')
            oldstdout_fno = os.dup(sys.stdout.fileno())
            os.dup2(devnull.fileno(), 1)
        p = QP(matrix(H),f.tolist(),lb=lb.tolist(),ub=ub.tolist(),Aeq=Aeq.tolist(),beq=beq)
        r = p.solve('cvxopt_qp')
        if suppressOut:
            os.dup2(oldstdout_fno, 1)
        lim = 1e-4
        r.xf[where(abs(r.xf)<lim)] = 0
        self.__lambdas__ = r.xf
        nonzeroindexes = where(r.xf>lim)[0]
#        l1 = nonzeroindexes[0]
#        self.w0 = 1.0/labels[l1]-dot(self.w,data[l1])
        self.numSupportVectors = len(nonzeroindexes)
    
class LogisticRegressionClassifier(Classifier):
    def __classify__(self, data):
        pass
    
    def __train__(self, data, labels):
        pass
    
class SumOfSquareErrorClassifier(Classifier):
    def __classify__(self, data):
        t=dot(hstack((data, ones((data.shape[0],1)))),self.w)
        return t>0
    
    def __train__(self, data, labels):
        o=ones((data.shape[0],1))
        h=hstack((data, o))
        pseudoX=pinv(h)
        self.w=dot(pseudoX, labels.reshape((-1,1)))
