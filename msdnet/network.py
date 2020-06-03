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

"""Module implementing neural networks."""

from . import operations
from . import store
import numpy as np
import abc
import threading

class Network(abc.ABC):
    """Base class for a neural network."""

    @abc.abstractmethod
    def forward(self, im, returnoutput=True):
        """Compute a forward pass of the network.

        :param im: input image (channels x rows x columns)
        
        :param returnoutput: whether to return the output image (default: True)
        
        :return: output image (channels x rows x columns)
        """
        pass
    
    @abc.abstractmethod
    def backward(self, im):
        """Compute a backpropagation pass of the network. Sensitivity maps of
        each intermediate image are stored within the network.

        :param im: error gradient image (channels x rows x columns)
        """
        pass
    
    @abc.abstractmethod
    def gradient_zero(self):
        """Set all gradient variables to zero.
        """
        pass
    
    @abc.abstractmethod
    def gradient(self):
        """Compute gradient variables using computed sensitivity maps.
        """
        pass
    
    @abc.abstractmethod
    def getgradients(self):
        """Return a flat array with all gradient variables.

        :return: all gradient variables
        """
        pass
    
    def updategradients(self, u):
        """Update variables of network within a thread.

        :param u: update variables
        """
        # Run __updategradients in thread to prevent SIGINT
        thrd = threading.Thread(target=self.updategradients_internal, args=(u,))
        thrd.start()
        thrd.join()
    
    @abc.abstractmethod
    def updategradients_internal(self, u):
        """Update variables of network.

        :param u: update variables
        """
        pass

    
    @abc.abstractmethod
    def to_dict(self):
        """Return a dictionary containing all network variables and parameters.

        :return: all network variables and parameters
        """
        pass
    
    @abc.abstractmethod
    def load_dict(self, dct):
        """Set all network variables and parameters from dictionary.

        :param dct: all network variables and parameters
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def from_dict(cls, dct, gpu=True):
        """Initialize network and all network variables and parameters from dictionary.

        :param dct: all network variables and parameters
        """
        pass
    
    @classmethod
    def from_file(cls, fn, gpu=True, groupname='network'):
        """Initialize network and all network variables and parameters from file.

        :param fn: filename
        :param gpu: (optional) whether to use GPU or CPU
        """
        dct = store.get_dict(fn, groupname)
        return cls.from_dict(dct, gpu=gpu)
    
    def to_file(self, fn, groupname='network'):
        """Save all network variables and parameters to file.

        :param fn: filename
        """
        store.store_dict(fn, groupname, self.to_dict())
    
    def normalizeinout(self, datapoints):
        """Normalize input and output of network to zero mean and unit variance.

        :param datapoints: list of datapoints to compute normalization factors with.
        """
        self.normalizeinput(datapoints)
        self.normalizeoutput(datapoints)

class MSDNet(Network):
    """Main implementation of a Mixed-Scale Dense network.
    
    :param d: depth of network (width is always 1)
    :param dil: :class:`.dilations.Dilations` class defining dilations
    :param nin: number of input channels
    :param nout: number of output channels
    :param gpu: (optional) whether to use GPU or CPU
    """
    
    def __init__(self, d, dil, nin, nout, gpu=True):
        
        self.d = d
        self.nin = nin
        self.nout = nout
        
        # Fill dilation list
        if dil:
            dil.reset()
            self.dl = np.array([dil.nextdil() for i in range(d)],dtype=np.int32)
        
        # Set up temporary images, force creation in first calls
        self.ims = np.zeros(1)
        self.delta = np.zeros(1)
        self.indelta = np.zeros(1)

        self.fshape = (3,3)
        self.axesslc = (slice(None), None, None)
        self.revf = (slice(None,None,-1),slice(None,None,-1))
        self.ndim = 2
        
        if gpu:
            from . import gpuoperations
            self.dataobject = gpuoperations.GPUImageData
        else:
            self.dataobject = operations.ImageData
        
        # Set up filters
        self.f = []
        for i in range(d):
            self.f.append(np.zeros((nin+i,*self.fshape),dtype=np.float32))
        self.fg = [np.zeros_like(k) for k in self.f]
        
        # Set up weights
        self.w = np.zeros((nout,nin+d),dtype=np.float32)
        self.wg = np.zeros_like(self.w)
        
        # Set up offsets
        self.o = np.zeros(d, dtype=np.float32)
        self.og = np.zeros_like(self.o)
        self.oo = np.zeros(nout, dtype=np.float32)
        self.oog = np.zeros_like(self.oo)

    def forward(self, im, returnoutput=True):
        if self.nin==1 and len(im.shape)==self.ndim:
            im = im[np.newaxis]
        if im.shape[0]!=self.nin:
            raise ValueError("Number of input channels ({}) does not match expected number ({}).".format(im.shape[0], self.nin))
        if im.shape[1:]!=self.ims.shape[1:]:
            self.ims = self.dataobject((self.d+self.nin, *im.shape[1:]),self.dl,self.nin)
            self.out = self.dataobject((self.nout, *im.shape[1:]),self.dl,self.nin)
        self.ims.setimages(im)
        self.scaleinput()
        self.ims.setscalars(self.o[self.axesslc], start=self.nin)
        self.ims.prepare_forw_conv(self.f)
        for i in range(self.d):
            self.ims.forw_conv(i, self.nin+i, self.dl[i])
            self.ims.relu(self.nin+i)
        self.out.setscalars(self.oo[self.axesslc])
        self.out.combine_all_all(self.ims, self.w)
        self.scaleoutput()
        if returnoutput:
            return self.out.copy()
    
    def backward(self, im, inputdelta=False):
        if im.shape[1:]!=self.delta.shape[1:]:
            self.delta = self.dataobject((self.d, *im.shape[1:]), self.dl, self.nin)
            self.delta.fill(0)
            self.deltaout = self.dataobject(im.shape, self.dl, self.nin)
            self.delta.prepare_gradient()
        else:
            self.delta.fill(0)
        self.deltaout.setimages(im)
        self.scaleoutputback()
        wt = self.w[:,self.nin:].transpose().copy()
        self.delta.combine_all_all(self.deltaout, wt)
        self.delta.relu2(self.delta.shape[0]-1, self.ims, self.ims.shape[0]-1)

        back_f = {}
        for i in reversed(range(self.d-1)):
            fb = np.zeros((self.d-i-1,*self.fshape),dtype=np.float32)
            for j in range(i+1,self.d):
                fb[j-i-1] = self.f[j][self.nin+i][self.revf]
            back_f[i] = fb
        self.delta.prepare_back_conv(back_f)

        for i in reversed(range(self.d-1)):
            self.delta.back_conv(i,self.dl)
            self.delta.relu2(i, self.ims, self.nin+i)
        
        if inputdelta:
            if im.shape[1:]!=self.indelta.shape[1:]:
                self.indelta = np.zeros((self.nin, *im.shape[1:]), dtype=np.float32)
            self.indelta.fill(0)
            do = self.deltaout.get()
            de = self.delta.get()
            for i in range(self.nin):
                fb = np.zeros((self.d,*self.fshape),dtype=np.float32)
                for j in range(self.d):
                    fb[j] = self.f[j][i][self.revf]
                for j in range(self.nout):
                    operations.combine(do[j], self.indelta[i], self.w[j,i])
                for j in range(self.d):
                    operations.conv2d(de[j], self.indelta[i], fb[j], self.dl[j])
            

    
    def initialize(self):
        """Initialize network parameters."""
        for f in self.f:
            f[:] = np.sqrt(2/(f[0].size*(self.nin+self.d-1)+self.nout))*np.random.normal(size=f.shape)
        self.o[:]=0
        self.w[:]=0
        self.oo[:]=0
    
    def gradient_zero(self):
        self.wg[:]=0
        for fg in self.fg:
            fg[:]=0
        self.og[:]=0
        self.oog[:]=0
    
    def gradient(self):
        self.oog += self.deltaout.sumall()
        self.og += self.delta.sumall()
       
        self.wg += self.ims.weightgradientall(self.deltaout)

        self.filtergradient()
    
    def filtergradient(self):
        """Compute filter gradient values."""
        fg = self.delta.filtergradientfull(self.ims).reshape((-1,3,3))
        idx = 0
        for i in range(self.d):
            self.fg[i] += fg[idx:idx+self.nin+i]
            idx+=self.nin+i
      
    def getgradients(self):
        fgu = np.hstack([f.ravel() for f in self.fg])
        return np.hstack([self.wg.ravel(),fgu.ravel(),self.og.ravel(),self.oog.ravel()]).ravel()
    
    def updategradients_internal(self, u):
        def update(v, idx):
            v = v.ravel()
            end = idx + len(v)
            v -= u[idx:end]
            return end
        idx = update(self.w, 0)
        for f in self.f:
            idx = update(f, idx)
        idx = update(self.o, idx)
        idx = update(self.oo, idx)
    
    def setinputscale(self, gamma, offset):
        """Set input normalization values."""
        self.gam_in = gamma
        self.off_in = offset
    
    def setoutputscale(self, gamma, offset):
        """Set output normalization values."""
        self.gam_out = gamma
        self.off_out = offset
    
    def scaleinput(self):
        """Normalize input image."""
        try:
            for i in range(self.nin):
                self.ims.mult(self.gam_in[i], i)
                self.ims.add(self.off_in[i], i)
        except AttributeError:
            pass
    
    def scaleoutput(self):
        """Rescale output image."""
        try:
            for i in range(self.nout):
                self.out.mult(self.gam_out[i], i)
                self.out.add(self.off_out[i], i)
        except AttributeError:
            pass
    
    def scaleoutputback(self):
        """Rescale output image during backpropagation."""
        try:
            for i in range(self.nout):
                self.deltaout.mult(1/self.gam_out[i], i)
        except AttributeError:
            pass
    
    def normalizeinput(self, datapoints):
        """Normalize input of network to zero mean and unit variance.

        :param datapoints: list of datapoints to compute normalization factors with.
        """
        nd = len(datapoints)
        allmeans = []
        allstds = []
        for d in datapoints:
            inp, _, _ = d.getall()
            means = []
            stds = []
            for im in inp:
                mn = operations.sum(im)/im.size
                std = operations.std(im, mn)
                means.append(mn)
                stds.append(std)
            allmeans.append(means)
            allstds.append(stds)
        mean = np.array(allmeans).mean(0)
        std = np.array(allstds).mean(0)

        self.gam_in = (1/std).astype(np.float32)
        self.off_in = (-mean/std).astype(np.float32)
    
    def normalizeoutput(self, datapoints):
        """Normalize output of network to zero mean and unit variance.

        :param datapoints: list of datapoints to compute normalization factors with.
        """
        nd = len(datapoints)
        allmeans = []
        allstds = []
        for d in datapoints:
            _, inp, _ = d.getall()
            means = []
            stds = []
            for im in inp:
                mn = operations.sum(im)/im.size
                std = operations.std(im, mn)
                means.append(mn)
                stds.append(std)
            allmeans.append(means)
            allstds.append(stds)
        mean = np.array(allmeans).mean(0)
        std = np.array(allstds).mean(0)

        self.gam_out = (std).astype(np.float32)
        self.off_out = (mean).astype(np.float32)
    
    def to_dict(self):
        dct = {}
        dct['d'] = self.d
        dct['nin'] = self.nin
        dct['nout'] = self.nout
        dct['dl'] = self.dl.copy()
        dct['w'] = self.w.copy()
        dct['o'] = self.o.copy()
        dct['oo'] = self.oo.copy()
        try:
            dct['gam_in'] = self.gam_in
            dct['off_in'] = self.off_in
        except AttributeError:
            pass
        try:
            dct['gam_out'] = self.gam_out
            dct['off_out'] = self.off_out
        except AttributeError:
            pass
        dctf = {}
        for i in range(self.d):
            dctf['{:05d}'.format(i)] = self.f[i].copy()
        dct['f'] = dctf
        return dct

    
    def load_dict(self, dct):
        self.dl = dct['dl'].copy()
        self.w[:] = dct['w']
        self.o[:] = dct['o']
        self.oo[:] = dct['oo']
        try:
            self.gam_in = dct['gam_in']
            self.off_in = dct['off_in']
        except KeyError:
            pass
        try:
            self.gam_out = dct['gam_out']
            self.off_out = dct['off_out']
        except KeyError:
            pass
        dctf = dct['f']
        for i in range(self.d):
            self.f[i] = dctf['{:05d}'.format(i)]
        pass
    
    @classmethod
    def from_dict(cls, dct, gpu=True):
        n = cls(dct['d'], None, dct['nin'], dct['nout'], gpu=gpu)
        n.load_dict(dct)
        return n
    
class SegmentationMSDNet(MSDNet):
    """Main implementation of a Mixed-Scale Dense network for segmentation.
    
    Same parameters as :class:`MSDNet`, with additional:

    :param softmaxderiv: whether to compute derivative of the softmax layer (default: True)
    """
    def __init__(self, *args, softmaxderiv=True, **kwargs):
        self.deriv = softmaxderiv
        super().__init__(*args,**kwargs)
    
    def forward(self, im, returnoutput=True):
        super().forward(im, returnoutput=False)
        self.out.softmax()
        if returnoutput:
            return self.out.copy()
    
    def backward(self, im, inputdelta=False):
        if self.deriv:
            tmp = np.zeros_like(im)
            act = self.out.copy()
            operations.softmaxderiv(tmp, im, act)
            im = tmp
        super().backward(im, inputdelta=inputdelta)

    
    def normalizeoutput(self, datapoints):
        pass
