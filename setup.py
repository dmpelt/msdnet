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

from skbuild import setup
from setuptools import find_packages
setup(
    name='msdnet',
    packages=find_packages(),
    version=open('VERSION').read().strip(),
    include_package_data=True,
    cmake_languages=['C',],
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
)