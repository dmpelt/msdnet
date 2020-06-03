#-----------------------------------------------------------------------
#Copyright 2019 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/msdnet/
#License: MIT
#
#This file is part of MSDNet, a Python implementation of the
#Mixed-Scale Dense Convolutional Neural Network.
#-----------------------------------------------------------------------

"""Module implementing network operations on CPU."""

import ctypes
import numpy as np
from pathlib import Path
import os
import concurrent.futures as cf
import glob
import sys

if sys.platform == 'darwin':
    libpath = Path(__file__).parent / 'libcoperations.dylib'
    lib = ctypes.CDLL(str(libpath))
elif os.name == 'nt':
    libpath = Path(__file__).parent.parent.parent.parent / 'bin' / 'coperations.dll'
    if not libpath.exists():
        libpath = Path(__file__).parent.parent / 'bin' / 'coperations.dll'
    lib = ctypes.WinDLL(str(libpath))
else:
    libpath = Path(__file__).parent / 'libcoperations.so'
    lib = ctypes.CDLL(str(libpath))

aslong = ctypes.c_uint64
asuint = ctypes.c_uint
asfloat = ctypes.c_float

cfloatp = ctypes.POINTER(ctypes.c_float)
def asfloatp(arr):
    return arr.ctypes.data_as(cfloatp)

cdoublep = ctypes.POINTER(ctypes.c_double)
def asdoublep(arr):
    return arr.ctypes.data_as(cdoublep)


cintp = ctypes.POINTER(ctypes.c_int32)
def asintp(arr):
    return arr.ctypes.data_as(cintp)

lib.sum.restype = ctypes.c_float
lib.masksum.restype = ctypes.c_float
lib.std.restype = ctypes.c_float
lib.multsum.restype = ctypes.c_float
lib.squaresum.restype = ctypes.c_longdouble
lib.gradientmap2d.restype = ctypes.c_float

def relu(inp):
    lib.relu(asfloatp(inp.ravel()), aslong(inp.size))

def leakyrelu(inp, w):
    lib.leakyrelu(asfloatp(inp.ravel()), aslong(inp.size), asfloat(w))

def sum(inp):
    return lib.sum(asfloatp(inp.ravel()), aslong(inp.size))

def masksum(inp, msk):
    return lib.masksum(asfloatp(inp.ravel()), asfloatp(msk.ravel()), aslong(inp.shape[1]*inp.shape[2]), aslong(inp.shape[0]))

def std(inp, mn):
    return lib.std(asfloatp(inp.ravel()), asfloat(mn), aslong(inp.size))

def multsum(a, b):
    return lib.multsum(asfloatp(a.ravel()), asfloatp(b.ravel()), aslong(a.size))

def softmax(inp):
    lib.softmax(asfloatp(inp.ravel()), aslong(inp[0].size), asuint(inp.shape[0]))

def softmaxderiv(out, err, act):
    lib.softmaxderiv(asfloatp(out.ravel()), asfloatp(err.ravel()), asfloatp(act.ravel()), aslong(err.shape[1]*err.shape[2]), asuint(err.shape[0]))

def squaresum(a):
    return lib.squaresum(asfloatp(a.ravel()),aslong(a.size))

def diff(out, a, b):
    lib.diff(asfloatp(out.ravel()), asfloatp(a.ravel()), asfloatp(b.ravel()), aslong(a.size))

def squarediff(out, a, b):
    lib.squarediff(asfloatp(out.ravel()), asfloatp(a.ravel()), asfloatp(b.ravel()), aslong(a.size))

def crossentropylog(out, im, tar):
    lib.crossentropylog(asfloatp(out.ravel()), asfloatp(im.ravel()), asfloatp(tar.ravel()), aslong(im.size))

def crossentropyderiv(out, im, tar):
    lib.crossentropyderiv(asfloatp(out.ravel()), asfloatp(im.ravel()), asfloatp(tar.ravel()), aslong(im.size))

def relu2(inp, out):
    lib.relu2(asfloatp(inp.ravel()), asfloatp(out.ravel()), aslong(inp.size))

def leakyrelu2(inp, out, w):
    lib.leakyrelu2(asfloatp(inp.ravel()), asfloatp(out.ravel()), aslong(inp.size), asfloat(w))

def combine(inp, out, w):
    lib.combine(asfloatp(inp.ravel()), asfloatp(out.ravel()), aslong(inp.size), asfloat(w))

def conv2d(inp, out, f, d):
    shx = indexlist(d, out.shape[0])
    shy = indexlist(d, out.shape[1])
    lib.conv2d(asfloatp(inp.ravel()), asfloatp(out.ravel()), asfloatp(f.ravel()), asuint(inp.shape[0]), asuint(inp.shape[1]), asintp(shx.ravel()), asintp(shy.ravel()))

def filtergradient2d(inp, delta, ux, uy):
    return lib.gradientmap2d(asfloatp(inp.ravel()), asfloatp(delta.ravel()), asuint(inp.shape[0]), asuint(inp.shape[1]), asintp(ux.ravel()), asintp(uy.ravel()))


# Utility functions
idx_list = {}
def indexlist(d, shp):
    if (d, shp) in idx_list:
        return idx_list[(d, shp)]
    out = np.zeros((shp, 2), dtype=np.int32)
    out[:, 0] = range(-d,shp-d)
    out[:, 1] = range(d,shp+d)
    out[out<0] *=-1
    msk = out>=shp
    out[msk] = 2*shp - out[msk] - 2
    idx_list[(d, shp)] = out
    return out

def setthreads(nthrds):
    lib.set_threads(asuint(nthrds))

# Try to set number of threads to number of physical cores
try:
    import psutil
    ncpu = psutil.cpu_count(logical=False)
    try:
        naff = len(psutil.Process().cpu_affinity())
        if naff < ncpu:
            ncpu = naff
    except AttributeError:
        pass
    setthreads(ncpu)
except ImportError:
    pass


# Internal data object for intermediate images
class ImageData(object):
    """Object that represents a set of 2D images on CPU.
    
    :param shape: total shape of all images
    :param dl: list of dilations in the network
    :param nin: number of input images of network
    """

    def __init__(self, shape, dl, nin):
        self.arr = np.zeros(shape, dtype=np.float32)
        self.dl = dl
        self.nin = nin
    
    def setimages(self, ims):
        """Set data to set of images.
        
        :param ims: set of images
        """
        self.arr[:ims.shape[0]] = ims
    
    def setscalars(self, scl, start=0):
        """Set each image to a scalar.
        
        :param scl: scalar values
        """
        self.arr[start:] = scl
    
    def fill(self, val, start=None, end=None):
        """Set image data to single scalar value.
        
        :param val: scalar value
        """
        self.arr[start:end] = val
    
    def copy(self, start=None, end=None):
        """Return copy of image data."""
        return self.arr[start:end].copy()
    
    def get(self, start=None, end=None):
        """Return image data."""
        return self.arr[start:end]
    
    def add(self, val, i):
        """Add scalar to single image.
        
        :param val: scalar to add
        :param i: index of image to add value to
        """
        self.arr[i] += val
    
    def mult(self, val, i):
        """Multiply single image with value.
        
        :param val: value
        :param i: index of image to multiply
        """
        combine(self.arr[i], self.arr[i], val-1)
    
    def prepare_forw_conv(self, f):
        """Prepare for forward convolutions.
        
        :param f: convolution filters
        """
        self.forw_f = f

    def forw_conv(self, i, outidx, dl):
        """Perform forward convolutions
        
        :param i: image index to compute
        :param outidx: image index to write output to
        :param dl: dilation list
        """
        f = self.forw_f[i]
        for j in range(outidx):
            conv2d(self.arr[j], self.arr[outidx], f[j], dl)
    
    def prepare_back_conv(self, f):
        """Prepare for backward convolutions.
        
        :param f: convolution filters
        """
        self.back_f = f

    def back_conv(self, outidx, dl):
        """Perform backward convolutions
        
        :param outidx: image index to write output to
        :param dl: dilation list
        """
        f = self.back_f[outidx]
        for i in range(outidx+1, self.arr.shape[0]):
            conv2d(self.arr[i], self.arr[outidx], f[i-outidx-1], dl[i])
    
    @property
    def shape(self):
        return self.arr.shape

    def relu(self, i):
        """Apply ReLU to single image."""
        relu(self.arr[i])
    
    def relu2(self, i, dat, j):
        """Apply backpropagation ReLU to single image."""
        relu2(dat.arr[j], self.arr[i])
      
    def combine_all_all(self, dat, w):
        """Compute linear combinations of images."""
        for i in range(self.arr.shape[0]):
            for j in range(dat.arr.shape[0]):
                combine(dat.arr[j], self.arr[i], w[i,j])
    
    def prepare_gradient(self):
        """Prepare for gradient computation."""
        self.uxs = {}
        self.uys = {}
        shp = self.arr[0].shape
        for d in self.dl:
            for q in [-1,0,1]:
                if not q*d in self.uxs:
                    tmp = np.zeros(shp[0], dtype=np.int32)
                    tmp[:] = range(q*d, shp[0]+q*d)
                    tmp[tmp<0] = -tmp[tmp<0]
                    tmp[tmp>=shp[0]] = 2*shp[0] - tmp[tmp>=shp[0]] - 2
                    self.uxs[q*d] = tmp
                if not q*d in self.uys:
                    tmp = np.zeros(shp[1], dtype=np.int32)
                    tmp[:] = range(q*d, shp[1]+q*d)
                    tmp[tmp<0] = -tmp[tmp<0]
                    tmp[tmp>=shp[1]] = 2*shp[1] - tmp[tmp>=shp[1]] - 2
                    self.uys[q*d] = tmp
    
    def filtergradientfull(self, ims):
        """Compute gradients for filters."""
        gs = []
        for i in range(len(self.dl)):
            d = self.dl[i]
            for j in range(self.nin+i):
                for q in [-1,0,1]:
                    for r in [-1,0,1]:
                        gs.append(filtergradient2d(ims.arr[j], self.arr[i], self.uxs[q*d], self.uys[r*d]))
        return np.array(gs)
    
    def weightgradientall(self, delta):
        """Compute gradients for weights."""
        out = np.zeros((delta.shape[0], self.arr.shape[0]),dtype=np.float32)
        for i in range(delta.shape[0]):
            for j in range(self.arr.shape[0]):
                out[i,j] = multsum(self.arr[j], delta.arr[i])
        return out
    
    def sumall(self):
        """Compute image sums."""
        out = np.zeros(self.arr.shape[0], dtype=np.float32)
        for i in range(self.arr.shape[0]):
            out[i] = sum(self.arr[i])
        return out
    
    def softmax(self):
        """Compute softmax."""
        softmax(self.arr)
