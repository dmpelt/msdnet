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
Example 01: Train a network for regression
==========================================

This script trains a MS-D network for regression (i.e. denoising/artifact removal)
Run generatedata.py first to generate required training data.
"""

# Import code
import msdnet
from pathlib import Path

# Define dilations in [1,10] as in paper.
dilations = msdnet.dilations.IncrementDilations(10)

# Create main network object for regression, with 100 layers,
# [1,10] dilations, 1 input channel, 1 output channel, using
# the GPU (set gpu=False to use CPU)
n = msdnet.network.MSDNet(100, dilations, 1, 1, gpu=True)

# Initialize network parameters
n.initialize()

# Define training data
# First, create lists of input files (noisy) and target files (noiseless)
flsin = sorted((Path('train') / 'noisy').glob('*.tiff'))
flstg = sorted((Path('train') / 'noiseless').glob('*.tiff'))
# Create list of datapoints (i.e. input/target pairs)
dats = []
for i in range(len(flsin)):
    # Create datapoint with file names
    d = msdnet.data.ImageFileDataPoint(str(flsin[i]),str(flstg[i]))
    # Augment data by rotating and flipping
    d_augm = msdnet.data.RotateAndFlipDataPoint(d)
    # Add augmented datapoint to list
    dats.append(d_augm)
# Note: The above can also be achieved using a utility function for such 'simple' cases:
# dats = msdnet.utils.load_simple_data('train/noisy/*.tiff', 'train/noiseless/*.tiff', augment=True)

# Normalize input and output of network to zero mean and unit variance using
# training data images
n.normalizeinout(dats)

# Use image batches of a single image
bprov = msdnet.data.BatchProvider(dats,1)

# Define validation data (not using augmentation)
flsin = sorted((Path('val') / 'noisy').glob('*.tiff'))
flstg = sorted((Path('val') / 'noiseless').glob('*.tiff'))
datsv = []
for i in range(len(flsin)):
    d = msdnet.data.ImageFileDataPoint(str(flsin[i]),str(flstg[i]))
    datsv.append(d)
# Note: The above can also be achieved using a utility function for such 'simple' cases:
# datsv = msdnet.utils.load_simple_data('val/noisy/*.tiff', 'val/noiseless/*.tiff', augment=False)

# Select loss function
l2loss = msdnet.loss.L2Loss()

# Validate with loss function
val = msdnet.validate.LossValidation(datsv, loss=l2loss)

# Use ADAM training algorithms
t = msdnet.train.AdamAlgorithm(n, loss=l2loss)

# Log error metrics to console
consolelog = msdnet.loggers.ConsoleLogger()
# Log error metrics to file
filelog = msdnet.loggers.FileLogger('log_regr.txt')
# Log typical, worst, and best images to image files
imagelog = msdnet.loggers.ImageLogger('log_regr', onlyifbetter=True)

# Train network until program is stopped manually
# Network parameters are saved in regr_params.h5
# Validation is run after every len(datsv) (=25)
# training steps.
msdnet.train.train(n, t, val, bprov, 'regr_params.h5',loggers=[consolelog,filelog,imagelog], val_every=len(datsv))
