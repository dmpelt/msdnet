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
Module for defining how to pick dilations in the MS-D networks.
"""

import abc

class Dilations(abc.ABC):
    """Base class implementing dilations."""

    @abc.abstractmethod
    def reset(self):
        """Reset object to initial state."""
        pass
    
    @abc.abstractmethod
    def nextdil(self):
        """Return next dilation factor.
        
        :return: next dilation factor.
        """
        pass

class IncrementDilations(Dilations):
    """Dilations that increase by 1 until a certain limit (as in paper).
    
    :param maxv: maximum dilation
    :param minv: (optional) minimum dilation
    :param start: (optional) starting dilation
    """

    def __init__(self, maxv, minv=1, start=1):
        self.mx = maxv
        self.mn = minv
        self.start = start
        self.reset()
    def reset(self):
        self.i = self.start
    def nextdil(self):
        vl = self.i
        self.i+=1
        if self.i>self.mx:
            self.i = self.mn
        return vl