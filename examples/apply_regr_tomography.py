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
Example 06: Apply trained network for regression (tomography)
=============================================================

This script applies a trained MS-D network for regression (i.e. denoising/artifact removal)
Run generatedata_tomography.py first to generate required training data and train_regr_tomography.py to train
a network.
"""

# Import code
import msdnet
from pathlib import Path
import tifffile

# Make folder for output
outfolder = Path('tomo_results')
outfolder.mkdir(exist_ok=True)

# Load network from file
n = msdnet.network.MSDNet.from_file('tomo_regr_params.h5', gpu=True)

# Process all test images
flsin = sorted((Path('tomo_test') / 'lowqual').glob('*.tiff'))
dats = [msdnet.data.ImageFileDataPoint(str(f)) for f in flsin]
# Convert input slices to input slabs (i.e. multiple slices as input)
dats = msdnet.data.convert_to_slabs(dats, 2, flip=False)
for i in range(len(flsin)):
    # Compute network output
    output = n.forward(dats[i].input)
    # Save network output to file
    tifffile.imsave(outfolder / 'regr_{:05d}.tiff'.format(i), output[0])
