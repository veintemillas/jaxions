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

def ifft15sec(ou):
    start = time.time()
    data = np.fft.ifftn(ou)
    end = time.time()
    print('ifft took %1.3f s' %(end-start))
    return data

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

# def reduce(data,n, nr):
#     redf = n//nr
#     ou = np.fft.rfftn(data)
#     si = sisa(n)
#     fil = (1/nr)**2
#     puessi = np.exp(-fil*si**2)
#     mn=n//2+1
#     ou *= (puessi.reshape(n,1,1))
#     ou *= (puessi.reshape(1,n,1))
#     ou *= (puessi[0:mn].reshape(1,1,mn))
#     dataf = np.fft.irfftn(ou)
#     return dataf[::redf,::redf,::redf]

def reduce(data,n, nr, len=1):
    print('[reduce] reducing data[n,n,n] to nr^3 by fft smoothing')
    redn = n//nr
    start = time.time()
    ou = np.fft.rfftn(data)
    end = time.time()
    print('[reduce] fft took %1.3f s' %(end-start))
    si = sisa(n)
    fil = (len*redn/n)**2
    start = time.time()
    puessi = np.exp(-fil*si**2)
    mn=n//2+1
    ou *= (puessi.reshape(n,1,1))
    ou *= (puessi.reshape(1,n,1))
    ou *= (puessi[0:mn].reshape(1,1,mn))
    end = time.time()
    print('[reduce] filtering took %1.3f s' %(end-start))
    start = time.time()
    dataf = np.fft.irfftn(ou)
    end = time.time()
    print('[reduce] ifft took %1.3f s' %(end-start))
    return dataf[::redn,::redn,::redn]

def coormap(n):
    print('[coormap] feed me with n and I give you a n3 nparray of coordinate positions')
    nu = np.zeros((n,n,n))
    x = np.arange(0,n)
    xg = nu+np.reshape(x,(n,1,1))
    yg = nu+np.reshape(x,(1,n,1))
    zg = nu+np.reshape(x,(1,1,n))
    cop = np.stack([xg,yg,zg], axis=3)
    return np.reshape(cop,(n**3,3))

def posfromdat(condata):
    print('[posfromdat] feed me a contrast data file and I return a coormap and the con n3 nparrays')
    n=len(condata)
    coma = coormap(n)
    auxd = np.reshape(condata,n**3)
    return coma, auxd

def filtera(coma, auxd,con):
    start=time.time()
    mask = auxd>con
    end=time.time()
    print('building took %f sec'%(end-start))
    return coma[mask]#, auxd[mask]

# to randomize points
def addrand(coma, auxd, n, sig=1):
    print('[addrand] adds random noise to a collection of points')
    start=time.time()
    comar = coma + 0.5*sig*np.random.randn(len(auxd),3)
    comar[comar[:,0] < 0] = comar[comar[:,0] < 0]+[n,0,0]
    comar[comar[:,1] < 0] = comar[comar[:,1] < 0]+[0,n,0]
    comar[comar[:,2] < 0] = comar[comar[:,2] < 0]+[0,0,n]
    comar[comar[:,0] > n] = comar[comar[:,0] > n]-[n,0,0]
    comar[comar[:,1] > n] = comar[comar[:,1] > n]-[0,n,0]
    comar[comar[:,2] > n] = comar[comar[:,2] > n]-[0,0,n]
    end=time.time()
    print('add rand took %f sec'%(end-start))
    return comar
