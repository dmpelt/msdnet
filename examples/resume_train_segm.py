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
Example 09: Resume training a network for segmentation
======================================================

This script resumes an earlier training of a MS-D network for 
segmentation (i.e. labeling)
Run generatedata.py first to generate required training data, and
train_segm.py to generate a partially trained network.
"""

# Import code
import msdnet
from pathlib import Path

# Define training data
# First, create lists of input files (noisy) and target files (labels)
flsin = sorted((Path('train') / 'noisy').glob('*.tiff'))
flstg = sorted((Path('train') / 'label').glob('*.tiff'))
# Create list of datapoints (i.e. input/target pairs)
dats = []
for i in range(len(flsin)):
    # Create datapoint with file names
    d = msdnet.data.ImageFileDataPoint(str(flsin[i]),str(flstg[i]))
    # Convert datapoint to one-hot, using labels 0, 1, 2, 3, and 4,
    # which are the labels given in each label TIFF file.
    d_oh = msdnet.data.OneHotDataPoint(d, [0,1,2,3,4])
    # Augment data by rotating and flipping
    d_augm = msdnet.data.RotateAndFlipDataPoint(d_oh)
    # Add augmented datapoint to list
    dats.append(d_augm)
# Note: The above can also be achieved using a utility function for such 'simple' cases:
# dats = msdnet.utils.load_simple_data('train/noisy/*.tiff', 'train/label/*.tiff', augment=True, labels=[0,1,2,3,4])

# Use image batches of a single image
bprov = msdnet.data.BatchProvider(dats,1)

# Define validation data (not using augmentation)
flsin = sorted((Path('val') / 'noisy').glob('*.tiff'))
flstg = sorted((Path('val') / 'label').glob('*.tiff'))
datsv = []
for i in range(len(flsin)):
    d = msdnet.data.ImageFileDataPoint(str(flsin[i]),str(flstg[i]))
    d_oh = msdnet.data.OneHotDataPoint(d, [0,1,2,3,4])
    datsv.append(d_oh)
# Note: The above can also be achieved using a utility function for such 'simple' cases:
# datsv = msdnet.utils.load_simple_data('train/noisy/*.tiff', 'train/label/*.tiff', augment=False, labels=[0,1,2,3,4])

# Load network, training algorithm, and validation object from checkpoint of previous training
n, t, val = msdnet.train.restore_training('segm_params.checkpoint', msdnet.network.SegmentationMSDNet, msdnet.train.AdamAlgorithm, msdnet.validate.MSEValidation, datsv, gpu=True)

# Log error metrics to console
consolelog = msdnet.loggers.ConsoleLogger()
# Log error metrics to file
filelog = msdnet.loggers.FileLogger('log_segm.txt')
# Log typical, worst, and best images to image files
imagelog = msdnet.loggers.ImageLabelLogger('log_segm', onlyifbetter=True)
# Log typical, worst, and best images to image files
# Output probability map for a single channel (in this case, channel 3)
singlechannellog = msdnet.loggers.ImageLogger('log_segm_singlechannel', chan_out=3, onlyifbetter=True)

# Train network until program is stopped manually
# Network parameters are saved in segm_params.h5
# Validation is run after every len(datsv) (=25)
# training steps.
msdnet.train.train(n, t, val, bprov, 'segm_params_resumed.h5',loggers=[consolelog,filelog,imagelog,singlechannellog], val_every=len(datsv))
