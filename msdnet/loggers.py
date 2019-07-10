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

"""Module for logging progress during training."""

import abc
import skimage.transform as skt
import imageio
import numpy as np
import os.path

class Logger(abc.ABC):
    '''Base logger class
    
    Logs progress of validation set during training.
    '''

    def __init__(self, *args, onlyifbetter=False, **kwargs):
        self.onlyifbetter = onlyifbetter
        self.initialize(*args, **kwargs)

    @abc.abstractmethod
    def makelog(self, v):
        '''Logs progress of validation set. To be implemented by each class.

        :param v: validation set
        '''
        pass
    
    @abc.abstractmethod
    def initialize(self, *args, **kwargs):
        '''Initialize logger.'''
        pass
    
    def log(self, v):
        '''Logs progress of validation set.

        :param v: validation set
        '''
        if self.onlyifbetter and v.best != v.curerr:
            return
        self.makelog(v)

class ConsoleLogger(Logger):
    '''Output error values to the console.'''
    def initialize(self):
        pass

    def makelog(self, v):
        print('Current error: ', v.curerr, ', Best error: ', v.best)

class FileLogger(Logger):
    '''Output error values to a file.'''
    def initialize(self, fn):
        '''Initialize logger.
        
        :param fn: Filename to log error values to.
        '''
        self.fn = fn
        with open(fn,'w') as _:
            pass
    
    def makelog(self, v):
        with open(self.fn, 'a') as f:
            f.write('Current error: {}, Best error: {}\n'.format(v.curerr, v.best))

header_image = None
header_dict = {}
def getheaderimage(width):
    """Return image header.
    
    :param width: width of image
    """
    global header_image
    if width in header_dict:
        return header_dict[width]
    if header_image is None:
        header_filename = os.path.join(os.path.dirname(__file__), 'image_logger_header.png')
        header_image = imageio.imread(header_filename).astype(np.float32)/255
        header_dict[header_image.shape[1]] = header_image
        if width == header_image.shape[1]:
            return header_image
    res_image = skt.rescale(header_image, width/1536, preserve_range=True, mode='constant', anti_aliasing=True, multichannel=False)
    header_dict[width] = res_image
    return res_image
    
    

def stitchimages(ims, imsize=None, scaleoutput=True):
    """Stitch three images (input, target, output).
    
    :param ims: list of images to stitch
    :param imsize: (optional) Maximum size of image
    :param scaleoutput: (optional) whether to scale output image to target image range
    """
    if imsize:
        sz = ims[0].shape
        fc = imsize/max(sz)
        if fc<1:
            osz = (int(fc*sz[0]), int(fc*sz[1]))
            sims = []
            sims.append(skt.resize(ims[0],osz,preserve_range=True, mode='constant', anti_aliasing=True))
            sims.append(skt.resize(ims[1],osz,preserve_range=True, mode='constant', anti_aliasing=True))
            sims.append(skt.resize(ims[2],osz,preserve_range=True, mode='constant', anti_aliasing=True))
        else:
            sims = [ims[0].copy(), ims[1].copy(), ims[2].copy()]
    else:
        sims = [ims[0].copy(), ims[1].copy(), ims[2].copy()]
    mn = sims[0].min()
    mx = sims[0].max()
    sims[0] -= mn
    if mx>mn:
        sims[0]/=(mx-mn)
    if scaleoutput:
        mn = sims[1].min()
        mx = sims[1].max()
        sims[1] -= mn
        sims[2] -= mn
        if mx>mn:
            sims[1]/=(mx-mn)
            sims[2]/=(mx-mn)
            sims[2][sims[2]<0]=0
            sims[2][sims[2]>1]=1
    allims = np.hstack(sims)
    head_im = getheaderimage(allims.shape[1])
    if len(allims.shape) != len(head_im.shape):
        head_im = np.repeat(head_im[...,np.newaxis], 3, axis=2)
    return (np.vstack((head_im,allims))*255).astype(np.uint8)

class ImageLogger(Logger):
    '''Output best, worst, and typical images for validation set.'''
    def initialize(self, fn, chan_in=0, chan_out=0, imsize=512):
        """Initialize logger.

        :param fn: base filename to output images to.
        :param chan_in: input channel to show
        :param chan_out: output channel to show
        :param imsize: maximum image size to output
        """
        self.fn = fn
        self.ci = chan_in
        self.co = chan_out
        self.imsize = imsize
        for tpe in ['best', 'worst', 'typical']:
            with open(fn+'_'+tpe+'.png','w') as _:
                pass
    
    def toimage(self, ims):
        inp, tar, out = ims
        return stitchimages([inp[self.ci], tar[self.co], out[self.co]], self.imsize)

    def makelog(self, v):
        imageio.imsave(self.fn+'_best.png',self.toimage(v.getbest()))
        imageio.imsave(self.fn+'_worst.png',self.toimage(v.getworst()))
        imageio.imsave(self.fn+'_typical.png',self.toimage(v.getmedian()))

class ImageLabelLogger(Logger):
    '''Output best, worst, and typical images for validation set for segmentation problems.'''
    def initialize(self, fn, chan_in=0, imsize=512):
        """Initialize logger.

        :param fn: base filename to output images to.
        :param chan_in: input channel to show
        :param imsize: maximum image size to output
        """
        self.fn = fn
        self.ci = chan_in
        self.imsize = imsize
        for tpe in ['best', 'worst', 'typical']:
            with open(fn+'_'+tpe+'.png','w') as _:
                pass
        self.colors = [
            [0,0,0],
            [31,120,180],
            [51,160,44],
            [227,26,28],
            [255,127,0],
            [106,61,154],
            [255,255,153],
            [177,89,40],
            [166,206,227],
            [178,223,138],
            [251,154,153],
            [253,191,111],
            [202,178,214]
        ]
     
    def toimage(self, ims):
        inp, tar, out = ims
        tar = np.argmax(tar,axis=0)
        out = np.argmax(out,axis=0)
        tm = tar.max()
        om = out.max()
        if tm>=13 or om>=13:
            tar = tar.astype(np.float32)
            out = out.astype(np.float32)
            inp = inp[self.ci]
            return stitchimages([inp, tar, out], self.imsize, scaleoutput=True)
        else:
            inp2 = np.zeros((*tar.shape,3),dtype=np.float32)
            tar2 = np.zeros((*tar.shape,3),dtype=np.float32)
            out2 = np.zeros((*tar.shape,3),dtype=np.float32)
            inp2[...,0] = inp[self.ci]
            inp2[...,1] = inp[self.ci]
            inp2[...,2] = inp[self.ci]
            for i in range(tm+1):
                tar2[tar==i] = self.colors[i]
            for i in range(om+1):
                out2[out==i] = self.colors[i]
            return stitchimages([inp2, tar2/255, out2/255], self.imsize, scaleoutput=False)

        

    def makelog(self, v):
        imageio.imsave(self.fn+'_best.png',self.toimage(v.getbest()))
        imageio.imsave(self.fn+'_worst.png',self.toimage(v.getworst()))
        imageio.imsave(self.fn+'_typical.png',self.toimage(v.getmedian()))
