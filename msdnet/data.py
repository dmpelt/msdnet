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

"""
Module for image data input.

Below, :math:`N_{c}` is the number of image channels, and :math:`N_{x} \\times N_{y}` the image dimensions in pixels.
"""

import numpy as np
import abc
import imageio
import random
import collections

class DataPoint(abc.ABC):
    """Base class for a single data point (input image with corresponding target image)"""

    @property
    def input(self):
        """Input image"""
        return self.getinputarray().astype(np.float32)
    
    @property
    def target(self):
        """Target image"""
        return self.gettargetarray().astype(np.float32)
    
    @property
    def mask(self):
        """Mask image"""
        return self.getmaskarray()

    @abc.abstractmethod
    def getinputarray(self):
        """Return input image."""
        pass
    
    @abc.abstractmethod
    def gettargetarray(self):
        """Return target image."""
        pass
    
    def getmaskarray(self):
        """Return mask image."""
        return None
    
    def getall(self):
        """Return input image, target image, and mask image (when given)."""
        return self.input, self.target, self.mask

class OnlyAllDataPoint(DataPoint):
    """
    Base class for a single data point (input image with corresponding target image)
    that can only return all images at once (i.e. `getall`).
    """

    errormsg = "Only getall supported"

    def getinputarray(self):
        raise RuntimeError(OnlyAllDataPoint.errormsg)
    
    def gettargetarray(self):
        raise RuntimeError(OnlyAllDataPoint.errormsg)
    
    def getmaskarray(self):
        raise RuntimeError(OnlyAllDataPoint.errormsg)
    
    @abc.abstractmethod
    def getall(self):
        pass

class BatchProvider(object):
    """Object that returns small random batches of datapoints.
    
    :param dlist: List of :class:`DataPoint`.
    :param batchsize: Number of datapoints per batch.
    :param seed: (optional) Random seed.
    """

    def __init__(self, dlist, batchsize, seed=None):
        self.d = dlist
        self.nd = len(self.d)
        self.rndm = np.random.RandomState(seed)
        self.idx = np.arange(self.nd,dtype=np.int)
        self.rndm.shuffle(self.idx)
        self.bsize = batchsize
        self.i = 0
    
    def getbatch(self):
        """Return batch of datapoints."""
        batch = []
        while len(batch)<self.bsize:
            if self.i>=self.nd:
                self.i = 0
                self.rndm.shuffle(self.idx)
            batch.append(self.d[self.idx[self.i]])
            self.i+=1
        return batch

class ArrayDataPoint(DataPoint):
    """Datapoint with numpy array image data.
    
    :param inputarray: numpy array with input image (size: :math:`N_{c} \\times N_{x} \\times N_{y}`)
    :param targetarray: (optional) numpy array with target image (size: :math:`N_{c} \\times N_{x} \\times N_{y}`)
    :param maskarray: (optional) numpy array with mask image (size: :math:`N_{c} \\times N_{x} \\times N_{y}`)
    """
    def __init__(self, inputarray, targetarray=None, maskarray=None):
        self.iarr = inputarray.astype(np.float32)
        if not targetarray is None:
            self.tarr = targetarray.astype(np.float32)
        else:
            self.tarr = None
        if not maskarray is None:
            self.marr = maskarray.astype(np.float32)
        else:
            self.marr = None
    
    def getinputarray(self):
        return self.iarr
    
    def gettargetarray(self):
        return self.tarr

    def getmaskarray(self):
        return self.marr

class ImageFileDataPoint(DataPoint):
    """Datapoint with image files. Supported are: TIFFs and most standard image formats (e.g. PNG and JPEG).
    
    :param inputfile: file name of input image
    :param targetfile: (optional) file name of target image
    :param maskfile: (optional) file name of mask image
    """
    def __init__(self, inputfile, targetfile=None, maskfile=None):
        self.ifn = inputfile
        self.tfn = targetfile
        self.mfn = maskfile
    
    def __fix_image_dimensions(self, im):
        if len(im.shape)==2:
            return im[np.newaxis]
        if im.shape[2]<im.shape[0]:
            return np.ascontiguousarray(im.swapaxes(1,2).swapaxes(0,1))
        return im
    
    def __readimage(self, fn):
        try:
            im = imageio.volread(fn)
        except Exception:
            # require tifffile only if imageio fails
            from skimage.external.tifffile import tifffile
            im = tifffile.imread(fn)
        return self.__fix_image_dimensions(im)

    def getinputarray(self):
        return self.__readimage(self.ifn)
    
    def gettargetarray(self):
        return self.__readimage(self.tfn)
    
    def getmaskarray(self):
        if self.mfn is None:
            return None
        try:
            im = imageio.volread(self.mfn, flatten=True)
        except TypeError:
            try:
                im = imageio.volread(self.mfn)
            except Exception:
                # require tifffile only if imageio fails
                from skimage.external.tifffile import tifffile
                im = tifffile.imread(self.mfn)
        return im

class OneHotDataPoint(DataPoint):
    """Datapoint that converts a data point with a labeled image to
    one-hot images.
    
    :param datapoint: input :class:`DataPoint`
    :param labels: list of numberical labels in label image
    :param maskunlabeled: (optional) whether to mask out unlabeled pixels
    """
    def __init__(self, datapoint, labels, maskunlabeled=True):
        self.dp = datapoint
        self.l = labels
        self.munl = maskunlabeled
    
    def getinputarray(self):
        return self.dp.getinputarray()
    
    def getmaskarray(self):
        if self.munl:
            tar = self.gettargetarray()
            return tar.sum(0)
        return self.dp.getmaskarray()
    
    def gettargetarray(self):
        im = self.dp.gettargetarray()
        oh = np.zeros((len(self.l), *im.shape[1:]),dtype=np.float32)
        for i, l in enumerate(self.l):
            oh[i] = (im[0]==l)
        return oh

class SlabDataPoint(DataPoint):
    """Datapoint that represents a slab of data points.
    
    :param datapoints: list of :class:`DataPoint`.
    :param flip: (optional) whether to augment data by also flipping slab.
    """
    def __init__(self, datapoints, flip=False):
        self.nd = len(datapoints)
        if self.nd%2==0:
            raise ValueError('Number of datapoints must be odd')
        self.dp = list(datapoints)
        self.flip = flip
        self.curflip = random.randint(0,1)
    
    def getinputarray(self):
        self.curflip = (self.curflip+1)%2
        if self.flip and self.curflip==1:
            return np.vstack([d.getinputarray() for d in reversed(self.dp)])
        else:
            return np.vstack([d.getinputarray() for d in self.dp])
    
    def gettargetarray(self):
        return self.dp[self.nd//2].gettargetarray()
    
    def getmaskarray(self):
        return self.dp[self.nd//2].getmaskarray()

def convert_to_slabs(datapoints, n_above_and_below, flip=False, reflective_boundary=True):
    """Convert a list of datapoints (representing a 3D volume) to
    a list of datapoints of slabs.
    
    :param datapoints: list of :class:`DataPoint`.
    :param n_above_and_below: number of slices to take above and below each slice.
    :param flip: (optional) whether to augment data by also flipping slab.
    :param reflective_boundary: (optional) whether to use reflective boundary at top and bottom
    :return: list of :class:`DataPoint`.
    """
    n = 2*n_above_and_below+1

    slablist = []

    if reflective_boundary:
        curlist = collections.deque([],n)
        idxlist = list(range(n_above_and_below,0,-1)) + [0,] + list(range(1,n_above_and_below+1))
        for i in idxlist:
            curlist.append(datapoints[i])
        slablist.append(SlabDataPoint(curlist, flip=flip))
        for i in range(n_above_and_below+1, n):
            curlist.append(datapoints[i])
            slablist.append(SlabDataPoint(curlist, flip=flip))    
    else:
        curlist = collections.deque(datapoints[:n], n)
        slablist.append(SlabDataPoint(curlist, flip=flip))
    for i in range(n, len(datapoints)):
        curlist.append(datapoints[i])
        slablist.append(SlabDataPoint(curlist, flip=flip))
    if reflective_boundary:
        for i in range(n_above_and_below):
            curlist.append(datapoints[len(datapoints)-2-i])
            slablist.append(SlabDataPoint(curlist, flip=flip))
    return slablist


class RotateAndFlipDataPoint(OnlyAllDataPoint):
    """Datapoint that augments input datapoint with rotations and flips.
    
    :param datapoint: input :class:`DataPoint`.
    """
    def __init__(self, datapoint):
        self.dp = datapoint
        self.__resetlist()

    def __resetlist(self):
        self.lst = list(range(8))
        random.shuffle(self.lst)
    
    def getall(self):
        inp = self.dp.input
        tar = self.dp.target
        msk = self.dp.mask
        c = self.lst.pop()
        if len(self.lst)==0:
            self.__resetlist()
        if c==1:
            inp, tar = inp[:,::-1], tar[:,::-1]
        elif c==2:
            inp, tar = inp[:,:,::-1], tar[:,:,::-1]
        elif c==3:
            inp, tar = inp[:,::-1,::-1], tar[:,::-1,::-1]
        elif c==4:
            inp, tar = np.rot90(inp,1,axes=(1,2)), np.rot90(tar,1,axes=(1,2))
        elif c==5:
            inp, tar = np.rot90(inp,3,axes=(1,2)), np.rot90(tar,3,axes=(1,2))
        elif c==6:
            inp, tar = np.rot90(inp,1,axes=(1,2))[:,::-1], np.rot90(tar,1,axes=(1,2))[:,::-1]
        elif c==7:
            inp, tar = np.rot90(inp,3,axes=(1,2))[:,::-1], np.rot90(tar,3,axes=(1,2))[:,::-1]
        inp = np.ascontiguousarray(inp)
        tar = np.ascontiguousarray(tar)
        if not msk is None:
            if c==1:
                msk = msk[::-1]
            elif c==2:
                msk = msk[:,::-1]
            elif c==3:
                msk = msk[::-1,::-1]
            elif c==4:
                msk = np.rot90(msk,1)
            elif c==5:
                msk = np.rot90(msk,3)
            elif c==6:
                msk = np.rot90(msk,1)[::-1]
            elif c==7:
                msk = np.rot90(msk,3)[::-1]
            msk = np.ascontiguousarray(msk)
        return inp, tar, msk

