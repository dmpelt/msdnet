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
Example 02: Apply trained network for regression
================================================

This script applies a trained MS-D network for regression (i.e. denoising/artifact removal)
Run generatedata.py first to generate required training data and train_regr.py to train
a network.
"""

# Import code
import msdnet
from pathlib import Path
import imageio

# Make folder for output
outfolder = Path('results')
outfolder.mkdir(exist_ok=True)

# Load network from file
n = msdnet.network.MSDNet.from_file('regr_params.h5', gpu=True)

# Process all test images
flsin = sorted((Path('test') / 'noisy').glob('*.tiff'))
for i in range(len(flsin)):
    # Create datapoint with only input image
    d = msdnet.data.ImageFileDataPoint(str(flsin[i]))
    # Compute network output
    output = n.forward(d.input)
    # Save network output to file
    imageio.imsave(outfolder / 'regr_{:05d}.tiff'.format(i), output[0])
