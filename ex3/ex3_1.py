from numpy import *
from numpy.core.numeric import asarray, identity, ndarray, zeros
from numpy.ma.extras import dot
from pypr.clustering.gmm import mulnormpdf, logmulnormpdf
from scipy.cluster.vq import kmeans2
from scipy.io.matlab.mio import loadmat

#class Gaussian:
#    def __init__(self, mean, covm):
#        self.mean = mean
#        self.covm = covm
#    def getP(self, sample):
#        prob = mulnormpdf(sample, self.mean, self.covm)
#        return prob

class GaussianMM:
    def __init__(self, mean):
        self.mean = mean
        self.N, self.dim = mean.shape
        self.covm = asarray([identity(self.dim)] * self.N)
        self.c = asarray([1.0 / self.N] * self.N)
    
    def getP(self,sample):
        p = ndarray((sample.shape[0]))
        for k in range(0,self.N):
            p += self.c[k]*mulnormpdf(sample, self.mean[k], self.covm[k])
        return p
    
    def getLogP(self,sample):
        p = zeros((sample.shape[0]))
        for k in range(0,self.N):
            p += self.c[k]*logmulnormpdf(sample, self.mean[k], self.covm[k])
        return p

def gmmEM(data, K, it):
    centroid = kmeans2(data, K)[0]
    N = data.shape[0]
    gmm = GaussianMM(centroid)
    while it > 0:
        it = it - 1
        # e-step
        gausses = ndarray((K, N), dtype='float64')
        for k in range(0, K):
            gausses[k] = gmm.c[k]*mulnormpdf(data, gmm.mean[k], gmm.covm[k])
        sums = sum(gausses, axis=0)
        gausses /= sums
        # m step
        sg = sum(gausses, axis=1)
        gmm.c = ones(sg.shape) / N * sg
        for k in range(0, K):
            gmm.mean[k] = sum(data * gausses[k].reshape((-1,1)), axis=0) / sg[k]
            d = data - gmm.mean[k]
            d1 = d.transpose()*gausses[k]
            gmm.covm[k]=dot(d1,d)/sg[k]
    return gmm

if __name__ == '__main__':
    data = loadmat('data/skin.mat')['sdata'].reshape((-1, 3))
    gmm = gmmEM(data,3,1)
    #em(data,3,max_iter=2)
