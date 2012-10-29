from numpy.core.fromnumeric import transpose
from numpy.core.numeric import asarray, identity, ndarray, zeros, fromfile
from numpy.core.shape_base import hstack
from numpy.lib.scimath import sqrt
from numpy.lib.shape_base import hsplit
from numpy.linalg.linalg import det, inv
from numpy.ma.core import exp
from numpy.ma.extras import dot
from numpy.matrixlib.defmatrix import mat
from scipy.cluster.vq import kmeans2
from scipy.constants.constants import pi
from scipy.io.matlab.mio import loadmat

class Likelihood:
    _samples = []
    
    #def __init__(self, fileName=None, key=None, recalc=False, samples = None):
        #self.readSamples(fileName, key,recalc,samples)
    
    def readSamples(self, fileName, key,recalc=False,samples=None):
        fn = fileName + ".pre"
        try:
            if recalc: raise IOError()
            with open(fn): pass
            print "precalculated file present"
            self.mu, self.cov = hsplit(mat(fromfile(fn).reshape((3,-1))),[1])
        except IOError:
            if samples != None:
                self._samples = samples
                print "got samples: " , self._samples
            else:
                print "no file present, calculating..."
                smpls = loadmat(fileName)[key]
                print "loaded from mat file"
                self._samples = mat(smpls)
                print "reshaped into samples"
            self.mu = sum(self._samples, axis=1) / self._samples.shape[1]
            print "mu=", str(self.mu)
            sampdiffmu = self._samples - self.mu
            self.cov = sampdiffmu*sampdiffmu.T / self._samples.shape[1]
            print"cov=", str(self.cov)
            mat(hstack((self.mu,self.cov))).tofile(fn)
        self._invCov = self.cov.I
        self._detCov = det(self.cov)
        self._multConst = 1 / sqrt((2 * pi) ** 3 * self._detCov)
        
    def setParams(self,mu,cov):
        self.mu = mu
        self.cov = cov
        self._invCov = inv(self.cov)
        self._detCov = det(self.cov)
        self._multConst = 1 / sqrt((2 * pi) ** 3 * self._detCov)
        
        
    def getP(self, sample):
        p = self._multConst * exp(-(1 / 2.0) * (transpose(sample - self.mu)*self._invCov*(sample - self.mu)))
        return p

class GaussianMM:
    def __init__(self,data,K,it):
        self.data = data
        self.N, self.dims = data.shape
        self.K = K
        self.it = it
        self.__gmmEm__()
        
    def __gmmEm__(self):
        self.mean = kmeans2(self.data, self.K)[0]
        self.c = asarray([1.0/self.K]*self.K)
        self.covm = asarray([identity(self.K)]*self.K)
        self.p = ndarray((self.N,self.K),dtype='float32')
        while self.it > 0:
            self.it -=1
            self.__calculateP__()
            #self.__Estep__()
            self.__Mstep__()
    
    def __Estep__(self):
        pass
            
    
    def __Mstep__(self):
        for k in range(0,self.K):
            self.c[k,:] = 1.0/self.N * sum(self.p,axis=1)
            self.mean[k,:] = sum(self.p*self.data, axis=1)/sum(self.p,axis=1)
            self.covm[k,:,:] = sum(self.p*dot(self.data - self.mean[k,:],transpose(self.data - self.mean[k,:])),axis=1)/sum(self.p,axis=1)
    
    def __calculateP__(self):
        p = mat(self.p.shape)
        for k in range(0,self.K):
            llh = Likelihood()
            llh.setParams(self.mean[k,:],self.covm[k,:,:])
            for i in range(0,self.N):
                p[k,i] = llh.getP(self.data[i,:])
        sp = sum(p,axis=0)
        for k in range(0,self.K):
            for i in range(0,self.N):
                self.p[k,i] = p[k,i]/sp[i]

if __name__ == '__main__':
    gmm = GaussianMM(loadmat('../ex1/data/skin.mat')['sdata'].reshape((-1,3)), 3, 10)
    