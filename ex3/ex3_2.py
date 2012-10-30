from ex3_1 import gmmEM
from numpy.core.numeric import seterr
from numpy.lib.shape_base import dstack
from scipy.io.matlab.mio import loadmat
from scipy.misc.pilutil import imread
import pylab
from pypr.clustering.gmm import em
import numpy

def getBlackWhiteFromBinary(img):
    return dstack((img,img,img))

if __name__ == '__main__':
    seterr(all='warn')
    skindata, nonskindata = loadmat('data/skin.mat')['sdata'].reshape((-1, 3)).astype(float), loadmat('data/nonskin.mat')['ndata'].reshape((-1, 3)).astype(float)
    gmmskin, gmmnonskin = gmmEM(skindata, 2, 5), gmmEM(nonskindata, 2, 5)
    #g1,g2 = em(skindata,2,max_iter=1),em(nonskindata,2,max_iter=1)
    img = imread('data/image.png').astype(float) / 255.0
    imshape = img.shape
    img = img.reshape((-1, 3))
    #for sample in img:
        #gmmskin.getLogP(sample)
    skinp, nonskinp = gmmskin.getLogP(img), gmmnonskin.getLogP(img)
    res = (skinp > nonskinp)
    res = res.reshape((imshape[0],imshape[1]))
    res = getBlackWhiteFromBinary(res)
    pylab.figure()
    pylab.imshow(res)
    pylab.show()
