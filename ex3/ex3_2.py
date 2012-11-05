from ex3_1 import gmmEM
from numpy.core.numeric import seterr, float128, float64, count_nonzero
from numpy.lib.shape_base import dstack
from numpy.ma.core import logical_not, logical_and
from pypr.clustering.gmm import em
from scipy.io.matlab.mio import loadmat
from scipy.misc.pilutil import imread
import numpy
import pylab

class GroundTruth:
    mask = None
    notMask = None
    positives = None
    negatives = None
    
    def __init__(self,fileName):
        self.mask = imread(fileName) > 0
        self.notMask = logical_not(self.mask)
        self.positives = count_nonzero(self.mask)*1.0
        self.negatives = (self.mask.size - self.positives)*1.0
        
    def checkClassification(self,binary):
        fp = count_nonzero(logical_and(self.notMask, binary))
        tp = count_nonzero(logical_and(self.mask,binary))
        return fp,tp

def getBlackWhiteFromBinary(img):
    return dstack((img,img,img))

if __name__ == '__main__':
    seterr(all='warn')
    skindata, nonskindata = loadmat('data/skin.mat')['sdata'].reshape((-1, 3)).astype(float128), loadmat('data/nonskin.mat')['ndata'].reshape((-1, 3)).astype(float128)
    iters = 10
    show = False
    usekmeans = False
    gmmskin, gmmnonskin = gmmEM(skindata, 2, iters,show,usekmeans), gmmEM(nonskindata, 2, iters,show,usekmeans)
    img = imread('data/image.png').astype(float128) / 255.0
    imshape = img.shape
    img = img.reshape((-1, 3))
    skinp, nonskinp = gmmskin.getP(img), gmmnonskin.getP(img)
    res = skinp > nonskinp
    res = res.reshape((imshape[0],imshape[1]))
    res = logical_not(res)
    
    gt = GroundTruth('data/mask.png')
    fp, tp = gt.checkClassification(res)
    print "false positive ratio: ", fp*1.0/gt.negatives
    print "true positive ratio: ", tp*1.0/gt.positives
    
    res = getBlackWhiteFromBinary(res)
    pylab.figure()
    pylab.imshow(res)
    pylab.show()
