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
Example 08: Apply trained network for segmentation (tomography)
===============================================================

This script applies a trained MS-D network for segmentation (i.e. labeling)
Run generatedata_tomography.py first to generate required training data and train_segm_tomography.py to train
a network.
"""

# Import code
import msdnet
import glob
import tifffile
import os
import numpy as np

# Make folder for output
os.makedirs('tomo_results', exist_ok=True)

# Load network from file
n = msdnet.network.SegmentationMSDNet.from_file('tomo_segm_params.h5', gpu=True)

# Process all test images
flsin = sorted(glob.glob('tomo_test/lowqual/*.tiff'))
dats = [msdnet.data.ImageFileDataPoint(f) for f in flsin]
# Convert input slices to input slabs (i.e. multiple slices as input)
dats = msdnet.data.convert_to_slabs(dats, 2, flip=False)
for i in range(len(flsin)):
    # Compute network output
    output = n.forward(dats[i].input)
    # Save labels with maximum probability to file (i.e. prediceted labels for each pixel)
    tifffile.imsave('tomo_results/segm_label_{:05d}.tiff'.format(i), np.argmax(output,0).astype(np.uint8))
    # Save probability map of a single channel (here, channel 2) to file
    tifffile.imsave('tomo_results/segm_prob_lab2_{:05d}.tiff'.format(i), output[2])
