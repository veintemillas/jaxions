#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import re, os
import h5py
import datetime
import glob
import matplotlib.colors as col
from numpy import fft
import time

from pyaxions import jaxions as pa

##### generate custom colormap
def mkcmap():
    white = '#ffffff'
    yellow = '#FFFF00'
    turq = '#00868B'
    black = '#000000'
    red = '#ff0000'
    mblue = '#03A89E' #'#0000ff'
    lblue =  '#00BFFF'
    dorchid = '#68228B'
#     light '#BF3EFF'
    anglemap = col.LinearSegmentedColormap.from_list(
        'anglemap', [black,black,dorchid,turq,yellow,yellow], N=100, gamma=1)
    return anglemap

logdensmap = mkcmap()

def fft15sec(data):
    start = time.time()
    ou = np.fft.fftn(data)
    end = time.time()
    print('fft took %1.3f s' %(end-start))
    return ou

def sisa (n):
    bas = np.zeros((n,n,n),dtype=float)
    return np.concatenate((np.arange(0,1+n/2,dtype=float),np.arange(1-n/2,0,1,dtype=float)))

def modtab (n):
    bas = np.zeros((n,n,n),dtype=float)
    linap = np.concatenate((np.arange(0,1+n/2,dtype=float),np.arange(1-n/2,0,1,dtype=float)))
    x_pos = bas + linap.reshape((n,1,1))
    y_pos = bas + linap.reshape((1,n,1))
    z_pos = bas + linap.reshape((1,1,n))
    return np.sqrt(x_pos**2 + y_pos**2 + z_pos**2)

def binpsp(pou,moda):
    print('Input [arrays of |delta_k|, and |k|] shall be reshaped as n3')
    print('Returns arrays of <|delta_k|^2>, stddev, bias_k, nmod')
    print('Estimates 4 min for 512 (no threads...)')
    nmax=int(moda.max())+1
    bino2=np.zeros(nmax)
    binoc2=np.zeros(nmax)
    bias2=np.zeros(nmax)
    nmod2=np.zeros(nmax)

    start = time.time()
    for nb in range(0,nmax):
        mask  = (nb< moda) * (moda <= nb+1)
        fuss  = pou[mask]
        sumin = len(fuss)
        sumi  = np.sum(fuss**2)
        sumic = np.sum(fuss**4)
        bino2[nb] = sumi
        binoc2[nb]= sumic
        nmod2[nb] = sumin
        bias2[nb] = np.mean(moda[mask])-nb
        print('n = %d done (n_mask=%d)'%(nb,sumin))
    end = time.time()
    print('bin took ',(end-start)/60,'min')
    binostd= np.sqrt(np.abs((binoc2/nmod2) - (bino2/nmod2)**2))
    return bino2, binostd, bias2, nmod2

def stamode(pou,moda,nb):
    print('Input [arrays of |delta_k|, and |k|] reshaped as n3 and an Integer')
    print('Returns array of values of <|delta_k|^2>')
    mask  = (nb< moda) * (moda <= nb+1)
    fuss  = pou[mask]
    return fuss**2
