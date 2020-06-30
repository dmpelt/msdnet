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

"""Module for defining a stopping criterion for training."""

import abc
import time

class StoppingCriterion(abc.ABC):
    '''Base stopping criterion class
    '''

    @abc.abstractmethod
    def check(self, ntrainimages, better_val):
        '''Decides whether to continue training or not. To be implemented by each class.

        :param ntrainimages: number of training images processed since last check
        :param better_val: whether a better validation loss was found since last check
        :returns: True if training has to be stopped, False otherwise
        '''
        pass

    def reset(self):
        '''Reset and initialize the stopping criterion instance.
        '''
        pass


class NeverStop(StoppingCriterion):
    '''Never stop training, i.e. train until process is killed.
    '''

    def check(self, ntrainimages, better_val):
        return False
    
class NonImprovingValidationSteps(StoppingCriterion):
    '''Stop after a chosen number of non-improving validation steps.
    '''

    def __init__(self, maxsteps):
        '''
        :param maxsteps: Maximum number of non-improving validation steps.
        '''
        self.max = maxsteps

    def reset(self):
        self.cur = 0
    
    def check(self, ntrainimages, better_val):
        if better_val:
            self.cur = 0
            return False
        else:
            self.cur += 1
            if self.cur >= self.max:
                return True
            else:
                return False

class MaxTime(StoppingCriterion):
    '''Stop after a certain number of hours of training.
    '''

    def __init__(self, hours):
        '''
        :param hours: number of hours to train.
        '''
        self.s = hours*60*60
    
    def reset(self):
        self.starttime = time.monotonic()
    
    def check(self, ntrainimages, better_val):
        if time.monotonic() - self.starttime > self.s:
            return True
        else:
            return False

class MaxEpochs(StoppingCriterion):
    '''Stop after a certain number of epochs.
    '''

    def __init__(self, epochsize, maxepochs):
        '''
        :param epochsize: number of training images in each epoch.
        :param maxepochs: number of epochs to train.
        '''
        self.esize = epochsize
        self.max = maxepochs
    
    def reset(self):
        self.nims = 0
    
    def check(self, ntrainimages, better_val):
        self.nims += ntrainimages
        if self.nims/self.esize >= self.max:
            return True
        else:
            return False