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

"""Module implementing network operations on GPU using Numba."""

import numpy as np
from numba import cuda, float32, int32
import math

def get1dgridsize(sz, tpb = 1024):
    """Return CUDA grid size for 1d arrays.
    
    :param sz: input array size
    :param tpb: (optional) threads per block
    """
    return (sz + (tpb - 1)) // tpb, tpb

def get2dgridsize(sz, tpb = (8, 8)):
    """Return CUDA grid size for 2d arrays.
    
    :param sz: input array size
    :param tpb: (optional) threads per block
    """
    bpg0 = (sz[0] + (tpb[0] - 1)) // tpb[0]
    bpg1 = (sz[1] + (tpb[1] - 1)) // tpb[1]
    return (bpg0, bpg1), tpb


class GPUImageData(object):
    """Object that represents a set of 2D images on GPU.
    
    :param shape: total shape of all images
    :param dl: list of dilations in the network
    :param nin: number of input images of network
    """
    def __init__(self, shape, dl, nin):
        self.arr = cuda.device_array(shape, dtype=np.float32)
        self.dlg = cuda.to_device(dl.astype(np.uint8))
        dlgt = np.zeros((len(dl),nin+len(dl)),dtype=np.uint8)
        for i,d in enumerate(dl):
            dlgt[i] = d
        self.dlgt = cuda.to_device(dlgt)
        dlgb = np.zeros((len(dl),len(dl)),dtype=np.uint8)
        for i in range(len(dl)):
            dlgb[i,:len(dl)-i-1] = dl[i+1:]
        self.dlgb = cuda.to_device(dlgb)

        self.set_block_size((shape[-2],shape[-1]))
        self.shape = shape
        self.nin = nin
    
    def set_block_size(self, imshape):
        """Set CUDA grid sizes to be used."""
        self.bpg1d, self.tpb1d = get1dgridsize(imshape[0]*imshape[1])
        self.bpg2d, self.tpb2d = get2dgridsize(imshape)
    
    def setimages(self, ims):
        """Set data to set of images.
        
        :param ims: set of images
        """
        bpg, tpb = get1dgridsize(ims.size)
        imsg = cuda.to_device(ims)
        setimages_cuda[bpg, tpb](imsg.ravel(), self.arr.ravel())
    
    def setscalars(self, scl, start=0):
        """Set each image to a scalar.
        
        :param scl: scalar values
        """
        bpg, tpb = get1dgridsize(self.arr[start:].size)
        sclr = cuda.to_device(scl.ravel())
        set_scalar_cuda[bpg, tpb](sclr, self.arr[start:].ravel(), self.arr[0].size)
    
    def fill(self, val, start=None, end=None):
        """Set image data to single scalar value.
        
        :param val: scalar value
        """
        bpg, tpb = get1dgridsize(self.arr[start:end].size)
        fill_cuda[bpg, tpb](np.float32(val), self.arr[start:end].ravel())
    
    def copy(self, start=None, end=None):
        """Return copy of image data."""
        return self.arr[start:end].copy_to_host()
    
    def get(self, start=None, end=None):
        """Return image data."""
        return self.arr[start:end].copy_to_host()
    
    def add(self, val, i):
        """Add scalar to single image.
        
        :param val: scalar to add
        :param i: index of image to add value to
        """
        bpg, tpb = get1dgridsize(self.arr[i].size)
        add_cuda[bpg, tpb](np.float32(val), self.arr[i].ravel())
    
    def mult(self, val, i):
        """Multiply single image with value.
        
        :param val: value
        :param i: index of image to multiply
        """
        bpg, tpb = get1dgridsize(self.arr[i].size)
        mult_cuda[bpg, tpb](np.float32(val), self.arr[i].ravel())
    
    def prepare_forw_conv(self, f):
        """Prepare for forward convolutions.
        
        :param f: convolution filters
        """
        self.forw_idx = np.zeros((len(f),2),dtype=np.uint32)
        idx = 0
        for i, fi in enumerate(f):
            self.forw_idx[i] = idx, idx+fi.size
            idx += fi.size
        ff = np.zeros(idx,dtype=np.float32)
        for i, fi in enumerate(f):
            l, r = self.forw_idx[i]
            ff[l:r] = fi.ravel()
        self.forw_fg = cuda.to_device(ff)

    def forw_conv(self, i, outidx, dl):
        """Perform forward convolutions
        
        :param i: image index to compute
        :param outidx: image index to write output to
        :param dl: dilation list
        """
        l, r = self.forw_idx[i]
        conv2d[self.bpg2d, self.tpb2d, 0,4*(r-l)](self.arr, 0, outidx, outidx, self.forw_fg, l, r, self.dlgt, i)
    
    def prepare_back_conv(self, f):
        """Prepare for backward convolutions.
        
        :param f: convolution filters
        """
        self.back_idx = {}
        idx = 0
        for key, val in f.items():
            self.back_idx[key] = idx, idx + val.size
            idx += val.size
        ff = np.zeros(idx,dtype=np.float32)
        for key, val in f.items():
            l, r = self.back_idx[key]
            ff[l:r] = val.ravel()
        self.back_fg = cuda.to_device(ff)
    
    def back_conv(self, outidx, dl):
        """Perform backward convolutions
        
        :param outidx: image index to write output to
        :param dl: dilation list
        """
        l, r = self.back_idx[outidx]
        conv2d[self.bpg2d, self.tpb2d, 0, 4*(r-l)](self.arr, outidx+1, self.shape[0], outidx, self.back_fg, l, r, self.dlgb, outidx)

    def relu(self, i):
        """Apply ReLU to single image."""
        relu2d_cuda[self.bpg2d, self.tpb2d](self.arr, i)
    
    def relu2(self, i, dat, j):
        """Apply backpropagation ReLU to single image."""
        relu2_2d_cuda[self.bpg2d, self.tpb2d](dat.arr, self.arr, j, i)
    
    def combine_all_all(self, dat, w):
        """Compute linear combinations of images."""
        wg = cuda.to_device(w)
        comb_all_all_cuda[self.bpg2d, self.tpb2d](dat.arr, self.arr, wg)
    
    def prepare_gradient(self):
        """Prepare for gradient computation."""
        inlist = []
        dellist = []
        for i in range(self.arr.shape[0]):
            inlist.extend(range(self.nin+i))
            dellist.extend([i]*(self.nin+i))
        self.inlist = cuda.to_device(np.array(inlist).astype(np.uint32))
        self.dellist = cuda.to_device(np.array(dellist).astype(np.uint32))
        self.nf = len(inlist)
        self.gr = cuda.to_device(np.zeros(self.nf*9,dtype=np.float32))

    
    def filtergradientfull(self, ims):
        """Compute gradients for filters."""
        bpg, tpb = get1dgridsize(9*self.nf)
        filtergradientfull[bpg,tpb](ims.arr, self.arr, self.dlg, self.gr, self.inlist, self.dellist)
        q = self.gr.copy_to_host()
        return q
      
    def weightgradientall(self, delta):
        """Compute gradients for weights."""
        tmp = cuda.device_array(24*self.shape[0]*delta.shape[0])
        fastmult[24,1024](delta.arr,self.arr,tmp)
        return tmp.copy_to_host().reshape((delta.shape[0],self.arr.shape[0],24)).sum(2)
    
    def sumall(self):
        """Compute image sums."""
        tmp = cuda.device_array(24*self.shape[0])
        fastsumall[24,1024](self.arr,tmp)
        return tmp.copy_to_host().reshape((self.arr.shape[0],24)).sum(1)
  
    def softmax(self):
        """Compute softmax."""
        softmax[self.bpg2d, self.tpb2d](self.arr)


@cuda.jit(fastmath=True)
def setimages_cuda(inp, out):
    i = cuda.grid(1)
    if i<inp.size:
        out[i] = inp[i]

@cuda.jit(fastmath=True)
def add_cuda(val, out):
    i = cuda.grid(1)
    if i<out.size:
        out[i] += val

@cuda.jit(fastmath=True)
def fill_cuda(val, out):
    i = cuda.grid(1)
    if i<out.size:
        out[i] = val

@cuda.jit(fastmath=True)
def set_scalar_cuda(val, out, size):
    i = cuda.grid(1)
    if i<out.size:
        j = i//size
        out[i] = val[j]

@cuda.jit(fastmath=True)
def mult_cuda(val, out):
    i = cuda.grid(1)
    if i<out.size:
        out[i] *= val

@cuda.jit(fastmath=True)
def mult_arr_cuda(in1, in2, ii, jj, out):
    i, j = cuda.grid(2)
    if i<out.shape[0] and j<out.shape[1]:
        out[i,j] = in1[ii, i, j]*in2[jj, i, j]

@cuda.jit(fastmath=True)
def comb_cuda(inp, out, w):
    i = cuda.grid(1)
    if i<out.size:
        out[i] += w*inp[i]

@cuda.jit(fastmath=True)
def comb_all_cuda(inp, out, w):
    i, j = cuda.grid(2)
    if i<out.shape[0] and j<out.shape[1]:
        tmp = float32(0)
        for k in range(w.shape[0]):
            tmp += w[k] * inp[k,i, j]
        out[i, j] += tmp

@cuda.jit(fastmath=True)
def comb_all_all_cuda(inp, out, w):
    i, j = cuda.grid(2)
    if i<out.shape[1] and j<out.shape[2]:
        for l in range(out.shape[0]):
            tmp = float32(0)
            for k in range(inp.shape[0]):
                tmp += w[l, k] * inp[k,i, j]
            out[l, i, j] += tmp

@cuda.jit(fastmath=True)
def relu_cuda(data):
    i = cuda.grid(1)
    if i<data.size:
        if data[i]<0:
            data[i]=0

@cuda.jit(fastmath=True)
def relu2d_cuda(data, k):
    i, j = cuda.grid(2)
    if i<data.shape[1] and j<data.shape[2]:
        if data[k,i,j]<0:
            data[k,i,j]=0

@cuda.jit(fastmath=True)
def relu2_cuda(inp, out):
    i = cuda.grid(1)
    if i<inp.size:
        if inp[i]<=0:
            out[i]=0

@cuda.jit(fastmath=True)
def relu2_2d_cuda(inp, out, k, l):
    i, j = cuda.grid(2)
    if i<inp.shape[1] and j<inp.shape[2]:
        if inp[k,i,j]<=0:
            out[l,i,j]=0

@cuda.jit(fastmath=True)
def conv2d(arr, il, ir, ao, fin, fl, fr, dlin, dli):
    inp = arr[il:ir]
    out = arr[ao]
    f = fin[fl:fr]
    dl = dlin[dli]
    fshared = cuda.shared.array(shape=0, dtype=float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bdx = cuda.blockDim.x
    bdy = cuda.blockDim.y
    tid = ty*bdx+tx
    nth = bdx*bdy
    for i in range(tid,f.size,nth):
        fshared[i] = f[i]
    cuda.syncthreads()
    do=-1
    xc,yc = cuda.grid(2)
    if xc<out.shape[0] and yc<out.shape[1]:
        tmp = float32(0)
        idx = int32(0)
        for j in range(inp.shape[0]):
            if do!=dl[j]:
                do=dl[j]
                d=dl[j]
                if xc>=d:
                    xl = xc-d
                else:
                    xl = d-xc
                if xc<out.shape[0]-d:
                    xr = xc+d
                else:
                    xr = 2*out.shape[0] - (xc+d + 2)
                if yc>=d:
                    yl = yc-d
                else:
                    yl = d-yc
                if yc<out.shape[1]-d:
                    yr = yc+d
                else:
                    yr = 2*out.shape[1] - (yc+d + 2)
            tmp = cuda.fma(inp[j,xl,yl],fshared[idx], tmp)
            tmp = cuda.fma(inp[j,xl,yc],fshared[idx+1], tmp)
            tmp = cuda.fma(inp[j,xl,yr],fshared[idx+2], tmp)
            tmp = cuda.fma(inp[j,xc,yl],fshared[idx+3], tmp)
            tmp = cuda.fma(inp[j,xc,yc],fshared[idx+4], tmp)
            tmp = cuda.fma(inp[j,xc,yr],fshared[idx+5], tmp)
            tmp = cuda.fma(inp[j,xr,yl],fshared[idx+6], tmp)
            tmp = cuda.fma(inp[j,xr,yc],fshared[idx+7], tmp)
            tmp = cuda.fma(inp[j,xr,yr],fshared[idx+8], tmp)
            idx+=9
        out[xc,yc] += tmp

@cuda.jit(fastmath=True)
def filtergradientfull(inp, delta, dl, gr, inlist, dellist):
    idx = cuda.grid(1)
    f = idx % 9
    idx2 = idx // 9
    if idx2 >= dellist.shape[0]:
        return
    j = dellist[idx2]
    i = inlist[idx2]
    fi = f // 3
    fj = f % 3
    ii = inp[i]
    jj = delta[j]

    d = dl[j]
    l = (fi-1)*d
    u = (fj-1)*d

    tmp = float32(0)
    for q in range(inp.shape[1]):
        xc = q+l
        if xc<0:
            xc = -xc
        if xc>=inp.shape[1]:
            xc = 2*inp.shape[1] - (xc + 2)
        for r in range(inp.shape[2]):
            yc = r+u
            if yc<0:
                yc = -yc
            if yc>=inp.shape[2]:
                yc = 2*inp.shape[2] - (yc + 2)
            tmp += ii[xc,yc]*jj[q,r]
    gr[idx] = tmp

def fastmult_impl(a, b, out):
    tx = int32(cuda.threadIdx.x)
    gtx = tx + cuda.blockIdx.x * 1024
    gsize = 1024 * cuda.gridDim.x
    sz2 = a[0].size
    nc = a[0].shape[1]
    fshared = cuda.shared.array(shape=1024, dtype=float32)
    fidx = 0
    for ai in range(a.shape[0]):
        for bi in range(b.shape[0]):
            sumv = float32(0)
            for i in range(gtx,sz2,gsize):
                sumv += a[ai,i//nc,i%nc]*b[bi,i//nc,i%nc]
            fshared[tx] = sumv
            cuda.syncthreads()
            sz = int32(512)
            while sz>0:
                if tx<sz:
                    fshared[tx] += fshared[tx+sz]
                cuda.syncthreads()
                sz//=2
            if tx==0:
                out[cuda.blockIdx.x + fidx] = fshared[0]
            fidx += cuda.gridDim.x

def fastsumall_impl(a, out):
    tx = int32(cuda.threadIdx.x)
    gtx = tx + cuda.blockIdx.x * 1024
    gsize = 1024 * cuda.gridDim.x
    sz2 = a[0].size
    nc = a[0].shape[1]
    fshared = cuda.shared.array(shape=1024, dtype=float32)
    fidx = 0
    for ai in range(a.shape[0]):
        sumv = float32(0)
        for i in range(gtx,sz2,gsize):
            sumv += a[ai,i//nc,i%nc]
        fshared[tx] = sumv
        cuda.syncthreads()
        sz = int32(512)
        while sz>0:
            if tx<sz:
                fshared[tx] += fshared[tx+sz]
            cuda.syncthreads()
            sz//=2
        if tx==0:
            out[cuda.blockIdx.x + fidx] = fshared[0]
        fidx += cuda.gridDim.x

maxregisters = 64
fastsumall = cuda.jit(fastsumall_impl, fastmath=True, max_registers=maxregisters)
fastmult = cuda.jit(fastmult_impl, fastmath=True, max_registers=maxregisters)
while maxregisters>16:
    tmp = cuda.to_device(np.zeros((1,1,1),dtype=np.float32))
    out = cuda.to_device(np.zeros(1024,dtype=np.float32))
    try:
        fastsumall[24,1024](tmp,out)
        fastmult[24,1024](tmp,tmp,out)
    except cuda.cudadrv.driver.CudaAPIError:
        maxregisters -= 16
        fastsumall = cuda.jit(fastsumall_impl, fastmath=True, max_registers=maxregisters)
        fastmult = cuda.jit(fastmult_impl, fastmath=True, max_registers=maxregisters)
        print('Lowering maximum number of CUDA registers to ', maxregisters)
        continue
    break

@cuda.jit(fastmath=True)
def softmax(inp):
    x, y = cuda.grid(2)
    if x>=inp.shape[1] or y>=inp.shape[2]:
        return
    nim = inp.shape[0]
    mx = inp[0, x, y]
    for j in range(1,nim):
        if inp[j,x, y]>mx:
            mx = inp[j,x,y]
    sm = 0
    for j in range(nim):
        inp[j,x,y] = math.exp(inp[j,x,y] - mx)
        sm += inp[j,x,y]
    for j in range(nim):
        inp[j,x,y] /= sm
