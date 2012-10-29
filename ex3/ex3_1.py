from numpy.core.numeric import asarray, identity, ndarray
from numpy.ma.extras import dot
from pypr.clustering.gmm import mulnormpdf
from scipy.cluster.vq import kmeans2
from scipy.io.matlab.mio import loadmat
from numpy import sum

class Gaussian:
    def __init__(self, mean, covm):
        self.mean = mean
        self.covm = covm
    def getP(self, sample):
        prob = mulnormpdf(sample, self.mean, self.covm)
        return prob

class GaussianMM:
    def __init__(self, mean):
        self.mean = mean
        self.N, self.dim = mean.shape
        self.covm = asarray([identity(self.dim)] * self.N)
        self.c = asarray([1.0 / self.N] * self.N)

def gmmEM(data, K, it):
    centroid = kmeans2(data, K)[0]
    N = data.shape[0]
    gmm = GaussianMM(centroid)
    while it > 0:
        it = it - 1
        # e-step
        gausses = ndarray((K, N), dtype='float64')
        for k in range(0, K):
            #gauss = Gaussian(gmm.mean[k], gmm.covm[k])
            gausses[k, :] = mulnormpdf(data, gmm.mean[k], gmm.covm[k])
        sums = sum(gausses, axis=0)
        gausses /= sums
        # m step
        for k in range(0, K):
            sgk = sum(gausses[k, :])
            gmm.c[k] = 1.0 / N * sgk
            gmm.mean[k] = sum(data * gausses[k].reshape((-1,1)), axis=0) / sgk
            d = data - gmm.mean[k]
            d1 = gausses[k]*d.reshape((3,-1))
            gmm.covm[k] = dot(d1, d) / sgk
    return gmm

if __name__ == '__main__':
    data = loadmat('../ex1/data/skin.mat')['sdata'].reshape((-1, 3))
    gmm = gmmEM(data,3,1)
    #em(data,3,max_iter=2)
