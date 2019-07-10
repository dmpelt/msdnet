Python implementation of the Mixed-Scale Dense Convolutional Neural Network.

* [\[Latest Release\]](https://github.com/dmpelt/msdnet/releases/latest)
* [\[Version history\]](https://github.com/dmpelt/msdnet/blob/master/CHANGELOG.md)
* [\[Bug Tracker\]](https://github.com/dmpelt/msdnet/issues)
* [\[Documentation\]](https://dmpelt.github.io/msdnet/)

If you use this code in a publication, we would appreciate it if you would refer to:


* Pelt, D. M., & Sethian, J. A. (2018). A mixed-scale dense convolutional neural network for image analysis. Proceedings of the National Academy of Sciences, 115(2), 254-259.

If you use this code to improve tomographic reconstruction, we would appreciate it if you would refer to:

* Pelt, D. M., Batenburg, K. J., & Sethian, J. A. (2018). Improving Tomographic Reconstruction from Limited Data Using Mixed-Scale Dense Convolutional Neural Networks. Journal of Imaging, 4(11), 128.

Development of the Mixed-Scale Dense Convolutional Neural Network method was supported by CAMERA, jointly funded by The Office of Advanced Scientific Research (ASCR) and the Office of Basic Energy Sciences (BES) within the United States Department of Energy's Office of Science. Development of the Python implementation is supported by Centrum Wiskunde & Informatica (CWI), with financial support provided by The Netherlands Organisation for Scientific Research (NWO), project number 016.Veni.192.235.

# Installation

To install this code in a conda environment, run:

```bash
conda install -c defaults -c conda-forge -c dmpelt msdnet
```

In other environments, the code can be installed by running:

```bash
python setup.py install
```

The code requires the following Python modules: numpy, scipy, tifffile, scikit-image, psutil, h5py, tqdm, numba >=0.41 .
For compiling the code, the scikit-build module is required.

# Usage

Please see the included example scripts for usage information. The scripts and further documentation can be found here: https://dmpelt.github.io/msdnet/.

