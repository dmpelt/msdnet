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
Supplementary 02: Generate data for tomography examples
=======================================================


This script generates tomographic reconstructions of phantom samples:
1 sample for training, 1 for validation, and 1 for testing
"""

import numpy as np
import tifffile
from pathlib import Path
import astra

n = 256
nang = 256
ang = np.linspace(0, np.pi, nang, False)
np.random.seed(12345)

# ASTRA objects
proj_geom = astra.create_proj_geom('parallel', 1, int(np.sqrt(2)*n), ang)
vol_geom = astra.create_vol_geom((n,n))
pid = astra.create_projector('strip', proj_geom, vol_geom)
w = astra.OpTomo(pid)

for tpe in ['tomo_train', 'tomo_val', 'tomo_test']:
    xfac = 0.75 + np.random.random()*1.25
    yfac = 0.75 + np.random.random()*1.25
    zfac = 0.75 + np.random.random()*1.25

    xx,yy,zz = np.mgrid[-1.5*xfac:1.5*xfac:1j*n, -1.5*yfac:1.5*yfac:1j*n, -1.5*zfac:1.5*zfac:1j*n]
    ph = np.zeros((n,n,n),dtype=np.float32)
    ph_label = np.zeros((n,n,n),dtype=np.uint8)
    msk = np.logical_and(np.abs(xx)<=1,np.logical_and(np.abs(zz)<=1,np.abs(yy)<=1))
    ph[msk]=1
    ph_label[msk]=1
    msk = xx**2+yy**2+zz**2<=1
    ph[msk]=0
    ph_label[msk]=2
    q = 1/np.sqrt(3)
    msk = np.logical_and(np.abs(xx)<=q,np.logical_and(np.abs(zz)<=q,np.abs(yy)<=q))
    ph[msk]=1
    ph_label[msk]=3

    tpe_path = Path(tpe)
    tpe_path.mkdir(exist_ok=True)
    (tpe_path / 'lowqual').mkdir(exist_ok=True)
    (tpe_path / 'highqual').mkdir(exist_ok=True)
    (tpe_path / 'label').mkdir(exist_ok=True)
    
    for j in range(n):
        sinogram = w*ph[j]
        sinogram_hq = sinogram + np.random.normal(size=sinogram.shape, scale=n/1000)
        sinogram_lq = sinogram + np.random.normal(size=sinogram.shape, scale=n/10)
        rec_hq = w.reconstruct('FBP', sinogram_hq)
        rec_lq = w.reconstruct('FBP', sinogram_lq)
        tifffile.imsave(tpe_path / 'lowqual' / '{:05d}.tiff'.format(j), rec_lq)
        tifffile.imsave(tpe_path / 'highqual' / '{:05d}.tiff'.format(j), rec_hq)
        tifffile.imsave(tpe_path / 'label' / '{:05d}.tiff'.format(j), ph_label[j])
        
