from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import max, min, float64, sum
from numpy.core.numeric import asarray, identity, ndarray, zeros, count_nonzero, ones
from numpy.ma.extras import dot
from numpy.oldnumeric.rng import random_sample
from pypr.clustering.gmm import mulnormpdf, logmulnormpdf, gauss_ellipse_2d, gm_assign_to_cluster
from scipy.cluster.vq import kmeans2
from scipy.io.matlab.mio import loadmat
import pylab

class GaussianMM:
    def __init__(self, mean):
        self.mean = mean
        self.N, self.dim = mean.shape
        self.covm = asarray([identity(self.dim)] * self.N)
        self.c = asarray([1.0 / self.N] * self.N)
    
    def getP(self,sample):
        p = zeros(sample.shape[0])
        for k in range(0,self.N):
            p += self.c[k]*mulnormpdf(sample, self.mean[k], self.covm[k])
        return p.reshape((-1,1))
    
    def draw(self,data):
        if self.dim == 2:
            fig = pylab.figure()
            ass = gm_assign_to_cluster(data,self.mean,self.covm,self.c)
            clrs = 'brgy'
            for i in range(self.N):
                pts = data[ass == i]
                #pts = pts[::100]
                pylab.plot(pts[:,0],pts[:,1],'.',c=clrs[i])
                x1, x2 = gauss_ellipse_2d(self.mean[i], self.covm[i])
                pylab.plot(x1, x2, 'k', linewidth=2)
            pylab.show()
        elif self.dim == 3:
            fig = pylab.figure()
            ax = Axes3D(fig)
            ass = gm_assign_to_cluster(data,self.mean,self.covm,self.c)
            clrs = 'brgy'
            for i in range(self.N):
                pts = data[ass == i].astype(float64)
                pts = pts[::100]
                ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=clrs[i])
            pylab.show()

def gmmEM(data, K, it,show=False,usekmeans=True):
    #data += finfo(float128).eps*100
    centroid = kmeans2(data, K)[0] if usekmeans else ((max(data) - min(data))*random_sample((K,data.shape[1])) + min(data))
    N = data.shape[0]
    gmm = GaussianMM(centroid)
    if show: gmm.draw(data)
    while it > 0:
        print it," iterations remaining"
        it = it - 1
        # e-step
        gausses = zeros((K, N), dtype = data.dtype)
        for k in range(0, K):
            gausses[k] = gmm.c[k]*mulnormpdf(data, gmm.mean[k], gmm.covm[k])
        sums = sum(gausses, axis=0)
        if count_nonzero(sums) != sums.size:
            raise "Divide by Zero"
        gausses /= sums
        # m step
        sg = sum(gausses, axis=1)
        if count_nonzero(sg) != sg.size:
            raise "Divide by Zero"
        gmm.c = ones(sg.shape) / N * sg
        for k in range(0, K):
            gmm.mean[k] = sum(data * gausses[k].reshape((-1,1)), axis=0) / sg[k]
            d = data - gmm.mean[k]
            d1 = d.transpose()*gausses[k]
            gmm.covm[k]=dot(d1,d)/sg[k]
        if show: gmm.draw(data)
    return gmm

if __name__ == '__main__':
    data = loadmat('data/gmmdata.mat')['gmmdata'].reshape((-1, 2))
    gmm = gmmEM(data,3,30,show=True,usekmeans=True)
