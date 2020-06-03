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

"""Module for training networks."""

from . import store, loss
import numpy as np
import abc
import tqdm

class TrainAlgorithm(abc.ABC):
    """Base class implementing a training algorithm."""
    @abc.abstractmethod
    def step(self, n, dlist):
        """Take a single algorithm step.
        
        :param n: :class:`.network.Network` to train with
        :param dlist: list of :class:`.data.DataPoint` to train with
        """
        pass
    
    @abc.abstractmethod
    def to_dict(self):
        """Save algorithm state to dictionary"""
        pass
    
    @abc.abstractmethod
    def load_dict(self, dct):
        """Load algorithm state from dictionary"""
        pass
    
    @classmethod
    @abc.abstractmethod
    def from_dict(cls, dct):
        """Load algorithm from dictionary"""
        pass
    
    @classmethod
    def from_file(cls, fn):
        """Load algorithm from file"""
        dct = store.get_dict(fn, 'trainalgorithm')
        return cls.from_dict(dct)
    
    def to_file(self, fn):
        """Save algorithm state to file"""
        store.store_dict(fn, 'trainalgorithm', self.to_dict())

class AdamAlgorithm(TrainAlgorithm):
    """Implementation of the ADAM algorithm.
    
    :param network: :class:`.network.Network` to train with
    :param a: ADAM parameter
    :param b1: ADAM parameter
    :param b2: ADAM parameter
    :param e: ADAM parameter
    """
    def __init__(self,  network, loss = None, a = 0.001, b1 = 0.9, b2 = 0.999, e = 10**-8):
        self.a = a
        self.b1 = b1
        self.b1t = b1
        self.b2 = b2
        self.b2t = b2
        self.e = e
        self.loss = loss
        if network:
            self.npars = network.getgradients().shape[0]
            self.m = np.zeros(self.npars)
            self.v = np.zeros(self.npars)
    
    def step(self, n, dlist):
        n.gradient_zero()
        tpix = 0
        for d in dlist:
            inp, tar, msk = d.getall()
            out = n.forward(inp)
            err = self.loss.deriv(out, tar)
            if msk is None:
                tpix += err.size
            else:
                msk = (msk == 0)
                err[:, msk] = 0
                tpix += err.size - err.shape[0]*msk.sum()
            n.backward(err)
            n.gradient()
        g = n.getgradients()
        g/=tpix
        self.m *= self.b1
        self.m += (1-self.b1)*g
        self.v *= self.b2
        self.v += (1-self.b2)*(g**2)
        mhat = self.m/(1-self.b1t)
        vhat = self.v/(1-self.b2t)
        self.b1t *= self.b1
        self.b2t *= self.b2
        upd = self.a * mhat/(np.sqrt(vhat) + self.e)
        n.updategradients(upd)
    
    def to_dict(self):
        dct = {}
        dct['a'] = self.a
        dct['b1'] = self.b1
        dct['b1t'] = self.b1t
        dct['b2'] = self.b2
        dct['b2t'] = self.b2t
        dct['e'] = self.e
        dct['npars'] = self.npars
        dct['m'] = self.m.copy()
        dct['v'] = self.v.copy()
        return dct
    
    def load_dict(self, dct):
        self.a = dct['a']
        self.b1 = dct['b1']
        self.b1t = dct['b1t']
        self.b2 = dct['b2']
        self.b2t = dct['b2t']
        self.e = dct['e']
        self.npars = dct['npars']
        self.m = dct['m'].copy()
        self.v = dct['v'].copy()
    
    @classmethod
    def from_dict(cls, dct):
        t = cls(None)
        t.load_dict(dct)
        return t

def restore_training(fn, netclass, trainclass, valclass, valdata, gpu=True):
    """Restore training from file.

    :param fn: filename to load
    :param netclass: :class:`.network.Network` class to use
    :param trainclass: :class:`TrainAlgorithm` class to use
    :param valclass: :class:`.validate.Validation` class to use
    :param valdata: list of :class:`.data.DataPoint` to validate with
    :param gpu: (optional) whether to use GPU or CPU
    :return: network object, training algorithm object, and validation object
    """
    n = netclass.from_file(fn, groupname='checkpoint', gpu=gpu)
    t = trainclass.from_file(fn)
    v = valclass.from_file(fn)
    v.d = valdata
    return n, t, v


def train(network, trainalg, validation, dataprov, outputfile, val_every=None, loggers=None, stopcrit=np.Inf, progress=False):
    """Train network.

    :param network: :class:`.network.Network` to train with   
    :param trainalg: :class:`TrainAlgorithm` object that performs training.
    :param validation: :class:`.validate.Validation` object that performs validation.
    :param dataprov: :class:`.data.BatchProvider` object that generates training batches.
    :param outputfile: file to store trained network parameters in
    :param val_every: (optional) number of training steps before each validation step
    :param loggers: (optional) list of :class:`loggers.Logger` objects to perform logging.
    :param stopcric: (optional) number of validations steps without improvement before stopping training
    :param progress: (optional) whether to show progress during training
    """
    if not val_every:
        val_every = len(validation.d)
    
    parts = outputfile.split('.')
    if len(parts)>1:
        checkpointfile = '.'.join(parts[:-1]) + '.checkpoint'
    else:
        checkpointfile = outputfile + '.checkpoint'
    nworse = 0
    nstep = 0
    if progress:
        pbar = tqdm.tqdm(total=val_every)
    while True:
        trainalg.step(network, dataprov.getbatch())
        nstep+=1
        if progress:
            pbar.update()
        if nstep>=val_every:
            if progress:
                pbar.clear()
                pbar.close()
            nstep=0
            network.to_file(checkpointfile, groupname='checkpoint')
            trainalg.to_file(checkpointfile)
            validation.to_file(checkpointfile)    
            if validation.validate(network):
                network.to_file(outputfile)
                nworse=0
            else:
                nworse+=1
                if nworse>=stopcrit:
                    return
            if loggers:
                try:
                    for log in loggers:
                        log.log(validation)
                except TypeError:
                    loggers.log(validation)
            if progress:
                pbar = tqdm.tqdm(total=val_every)
