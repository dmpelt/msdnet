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

"""Module for storing parameters in HDF5 files"""

import h5py
import numpy as np
import threading

def get_dict(fn, grpname):
    """Get dictionary from HDF5 file.

    :param fn: filename
    :param grpname: group name inside HDF5 file
    :return: loaded dictionary
    """
    dct = {}
    with h5py.File(fn, 'r') as f:
        grp = f[grpname]
        __store_grp_in_dict(grp, dct)
    return dct

def __store_grp_in_dict(grp, dct):
    for key, val in grp.attrs.items():
        try:
            dct[key] = val.item()
        except ValueError: # Fix for old network files with python lists in attributes
            dct[key] = val
    
    for key, val in grp.items():
        if isinstance(val, h5py.Group):
            newdct = {}
            __store_grp_in_dict(val, newdct)
            dct[key] = newdct
        elif isinstance(val, h5py.Dataset):
            dct[key] = val[:]

def store_dict(fn, grpname, dct):
    """Store dictionary in HDF5 file.

    :param fn: filename
    :param grpname: group name inside HDF5 file
    :param dct: dictionary to store
    """
    thrd = threading.Thread(target=__store_dict, args=(fn,grpname,dct))
    thrd.start()
    thrd.join()

def __store_dict(fn, grpname, dct):
    with h5py.File(fn, 'a') as f:
        if grpname in f:
            del f[grpname]
        grp = f.create_group(grpname)
        __store_dict_in_grp(grp, dct)

def __store_dict_in_grp(grp, dct):
    a = grp.attrs
    for key, val in dct.items():
        if isinstance(val, dict):
            newgrp = grp.create_group(key)
            __store_dict_in_grp(newgrp, val)
        elif isinstance(val, (np.ndarray, list)):
            grp.create_dataset(key, data=val)
        else:
            a[key] = val
