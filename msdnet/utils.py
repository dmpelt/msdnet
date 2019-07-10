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

"""Module with miscellaneous utility functions."""

from . import data

import numpy as np
import glob

def augment_and_average_prediction(n, inp):
    """Augment input by rotating and flipping, and
    average network output to improve results.
    
    :param n: MS-D network :class:`.network.Network`.
    :param inp: input array (size: :math:`N_{c} \\times N_{x} \\times N_{y}`).
    :return: averaged output array.
    """

    out = n.forward(np.ascontiguousarray(inp))
    for i in range(1,4):
        inp_in = np.ascontiguousarray(np.rot90(inp, i, (1,2)))
        out += np.rot90(n.forward(inp_in), -i, (1,2))
    inp_in = np.ascontiguousarray(inp[:,::-1])
    out += n.forward(inp_in)[:,::-1]
    inp_in = np.ascontiguousarray(inp[:,:,::-1])
    out += n.forward(inp_in)[:,:,::-1]
    inp_in = np.ascontiguousarray(np.rot90(inp,1,(1,2))[:,::-1])
    out += np.rot90(n.forward(inp_in)[:,::-1], -1, (1,2))
    inp_in = np.ascontiguousarray(np.rot90(inp,3,(1,2))[:,::-1])
    out += np.rot90(n.forward(inp_in)[:,::-1], -3, (1,2))
    out /= 8
    return out

def load_simple_data(input_files, target_files, augment=False, labels=None, maskunlabeled=True):
    """Load DataPoints in a simple format: a set of input file names and matching
    target file names (both specified by wildcards). The input file names and target
    file names should 'sort' the same.

    :param input_files: string with path specification of input files (e.g. "input/*.tiff")
    :param target_files: string with path specification of input files (e.g. "target/*.tiff")
    :param augment: (optional) whether to augment images by rotating and flipping.
    :param labels: (optional) list of numerical labels in label image. If specified, returned
                   DataPoints are converted to one-hot encoded DataPoints for segmentation.
    :param maskunlabeled: (optional) whether to mask out unlabeled pixels if labels is specified
    :return: list of :class:`.data.DataPoint` objects.
    """

    flsin = sorted(glob.glob(input_files))
    flstg = sorted(glob.glob(target_files))

    if len(flsin) != len(flstg):
        raise ValueError("Number of input images ({}) does not match number of target images ({}).".format(len(flsin), len(flstg)))
    
    dats = []
    for i in range(len(flsin)):
        d = data.ImageFileDataPoint(flsin[i],flstg[i])
        if labels:
            d = data.OneHotDataPoint(d,labels,maskunlabeled=maskunlabeled)
        if augment:
            d = data.RotateAndFlipDataPoint(d)
        dats.append(d)
    
    return dats