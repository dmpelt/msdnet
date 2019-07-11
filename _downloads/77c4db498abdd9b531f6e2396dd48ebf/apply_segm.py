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
Example 04: Apply trained network for segmentation
==================================================

This script applies a trained MS-D network for segmentation (i.e. labeling)
Run generatedata.py first to generate required training data and train_segm.py to train
a network.
"""

# Import code
import msdnet
import glob
import tifffile
import os
import numpy as np

# Make folder for output
os.makedirs('results', exist_ok=True)

# Load network from file
n = msdnet.network.SegmentationMSDNet.from_file('segm_params.h5', gpu=True)

# Process all test images
flsin = sorted(glob.glob('test/noisy/*.tiff'))
for i in range(len(flsin)):
    # Create datapoint with only input image
    d = msdnet.data.ImageFileDataPoint(flsin[i])
    # Compute network output
    output = n.forward(d.input)
    # Save labels with maximum probability to file (i.e. prediceted labels for each pixel)
    tifffile.imsave('results/segm_label_{:05d}.tiff'.format(i), np.argmax(output,0).astype(np.uint8))
    # Save probability map of a single channel (here, channel 2) to file
    tifffile.imsave('results/segm_prob_lab2_{:05d}.tiff'.format(i), output[2])
