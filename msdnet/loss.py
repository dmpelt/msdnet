#-----------------------------------------------------------------------
#Copyright 2020 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/msdnet/
#License: MIT
#
#This file is part of MSDNet, a Python implementation of the
#Mixed-Scale Dense Convolutional Neural Network.
#-----------------------------------------------------------------------

"""Module for training and validation loss functions."""

from . import operations
import abc
import numpy as np

class Loss(abc.ABC):
    '''Base loss class
    
    Computes loss function and its derivative.
    '''

    @abc.abstractmethod
    def loss(self, im, tar):
        '''Computes loss function for each pixel. To be implemented by each class.

        :param im: network output image
        :param tar: target image
        :returns: image of loss function values
        '''
        pass

    def lossvalue(self, im, tar, msk):
        '''Computes loss function.

        :param im: network output image
        :param tar: target image
        :param msk: mask image (or None)
        :return: loss function value
        '''
        vals = self.loss(im, tar)
        if msk is None:
            return operations.sum(vals)
        else:
            return operations.masksum(vals, msk)

    @abc.abstractmethod
    def deriv(self, im, tar):
        '''Computes derivative of loss function. To be implemented by each class.

        :param im: network output image
        :param tar: target image
        :return: image of loss function derivative values
        '''
        pass


class L2Loss(Loss):
    '''Computes L2-norm loss function.
    '''

    def loss(self, im, tar):
        err = np.zeros_like(tar)
        operations.squarediff(err, im, tar)
        return err
        

    def deriv(self, im, tar):
        err = np.zeros_like(tar)
        operations.diff(err, im, tar)
        return err

class CrossEntropyLoss(Loss):
    '''Computes cross entropy loss function for one-hot data.
    '''

    def loss(self, im, tar):
        err = np.zeros_like(tar)
        operations.crossentropylog(err, im, tar)
        return err
        

    def deriv(self, im, tar):
        err = np.zeros_like(tar)
        operations.crossentropyderiv(err, im , tar)
        return err

