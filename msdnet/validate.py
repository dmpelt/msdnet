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
Module for defining and processing validation sets.
"""

from . import store
from . import operations
from . import loss
import abc
import numpy as np


class Validation(abc.ABC):
    """Base class for processing a validation set."""

    @abc.abstractmethod
    def validate(self, n):
        """Compute validation metrics.
        
        :param n: :class:`.network.Network` to validate with
        :return: True if validation metric is lower than best validation error encountered, False otherwise.
        """
        pass
    
    @abc.abstractmethod
    def to_dict(self):
        """Compute validation metrics."""
        pass
    
    @abc.abstractmethod
    def load_dict(self, dct):
        """Return a dictionary containing all network variables and parameters.

        :return: all network variables and parameters
        """

        pass
    
    @classmethod
    @abc.abstractmethod
    def from_dict(cls, dct):
        """Initialize Validation object from dictionary.

        :param dct: dictionary with all parameters
        """
        pass
    
    @classmethod
    def from_file(cls, fn):
        """Initialize Validation object from file.

        :param fn: filename
        """
        dct = store.get_dict(fn, 'validation')
        return cls.from_dict(dct)
    
    def to_file(self, fn):
        """Save all Validation object parameters to file.

        :param fn: filename
        """
        store.store_dict(fn, 'validation', self.to_dict())


class LossValidation(Validation):
    """Validation object that computes simple difference metrics.

    :param data: list of :class:`.data.DataPoint` objects to validate with.
    :param keep: (optional) whether to keep the best, worst, and typical result in memory.
    """
    def __init__(self, data, loss=None, keep=True):
        self.d = data
        self.keep = keep
        self.best = np.Inf
        self.loss = loss
    
    def errorfunc(self, output, target, msk):
        """Error function used for validation.

        :param output: network output image.
        :param target: target image.
        :param mask: mask image to indicate where to compute error function for.

        :return: error function value.
        """
        lv = self.loss.lossvalue(output, target, msk)
        if msk is None:
            npix = target.size
        else:
            npix = target.shape[0]*(msk>0).sum()
        return lv/npix
    
    def getbest(self):
        """Return the input, target, and network output for best result.

        :return: list of images (input, target, network output)
        """

        d = self.d[self.idx[0]]
        out = []
        out.append(d.input)
        out.append(d.target)
        if self.keep:
            out.append(self.outputs[0])
        else:
            out.append(self.n.forward(d.input))
        return out
    
    def getworst(self):
        """Return the input, target, and network output for worst result.

        :return: list of images (input, target, network output)
        """
        d = self.d[self.idx[1]]
        out = []
        out.append(d.input)
        out.append(d.target)
        if self.keep:
            out.append(self.outputs[1])
        else:
            out.append(self.n.forward(d.input))
        return out
    
    def getmedian(self):
        """Return the input, target, and network output for median result.

        :return: list of images (input, target, network output)
        """
        d = self.d[self.idx[2]]
        out = []
        out.append(d.input)
        out.append(d.target)
        if self.keep:
            out.append(self.outputs[2])
        else:
            out.append(self.n.forward(d.input))
        return out
    
    def validate(self, n):
        self.n = n
        errs = np.zeros(len(self.d))
        if self.keep:
            self.outputs = [0,0,0]
        low = np.Inf
        high = -np.Inf
        self.idx = [0,0,0]
        for i,d in enumerate(self.d):
            out = self.n.forward(d.input)
            err = self.errorfunc(out, d.target, d.mask)
            errs[i] = err
            if err<low:
                low = err
                self.idx[0] = i
                if self.keep:
                    self.outputs[0] = out
            if err>high:
                high = err
                self.idx[1] = i
                if self.keep:
                    self.outputs[1] = out
        
        median = np.argsort(errs)[errs.shape[0]//2]
        self.idx[2] = median
        if self.keep:
            if median==self.idx[0]:
                self.outputs[2] = self.outputs[0]
            elif median==self.idx[1]:
                self.outputs[2] = self.outputs[1]
            else:
                self.outputs[2] = self.n.forward(self.d[median].input)
        error = errs.mean()
        self.curerr = error
        if error<self.best:
            self.best = error
            return True
        return False
    
    def to_dict(self):
        dct = {}
        dct['best'] = self.best
        dct['keep'] = self.keep
        return dct
    
    def load_dict(self, dct):
        self.best = dct['best']
        self.keep = dct['keep']
    
    @classmethod
    def from_dict(cls, dct):
        v = cls(None, None)
        v.load_dict(dct)
        return v

# For backwards compatibility, uses L2 norm
class MSEValidation(LossValidation):
    def __init__(self, data, keep=True):
        super().__init__(data, loss=loss.L2Loss(), keep=keep)
