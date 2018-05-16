# -*- coding: utf-8 -*-

from pyaxions import jaxions as pa
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

from sympy import integer_nthroot

import os,re,sys

import h5py

# MOVE TRANSITION FILES
if os.path.exists('./axion.m.10000'):
    os.rename('./axion.m.10000','./../axion.m.10000')
if os.path.exists('./axion.m.10001'):
    os.rename('./axion.m.10001','./../axion.m.10001')


filename = './' + sys.argv[-1]
fileHdf5 = h5py.File(filename, "r")


filename = './' + sys.argv[-1]
fileHdf5 = h5py.File(filename, "r")


print("you can type dens/redens after the file to choose map if possible")
if len(sys.argv) == 2:
    filename = './' + sys.argv[-1]
    fileHdf5 = h5py.File(filename, "r")
    an_contrastmap = 'energy/density' in fileHdf5
    re_contrastmap = 'energy/redensity' in fileHdf5
    if re_contrastmap :
        an_contrastmap = False
elif len(sys.argv) == 3:
    filename = './' + sys.argv[-2]
    fileHdf5 = h5py.File(filename, "r")
    dens0redens = sys.argv[-1]
    if dens0redens == 'dens':
        an_contrastmap = 'energy/density' in fileHdf5
        re_contrastmap = False
    elif dens0redens == 'redens':
        re_contrastmap = 'energy/redensity' in fileHdf5
        an_contrastmap = False

if an_contrastmap:
    print('Contrast found')
    con = pa.gm(filename,'3Dmapefull',True)
    Lx    = len(con)
    Ly    = len(con)
    Lz    = len(con)
    sizeL = pa.gm(filename,'L')
    z     = pa.gm(filename,'z')
    print('Size =  (',Lx,'x',Ly,'x',Lz,') in file ',filename)

if re_contrastmap:
    print('Reduced Contrast found')
    con = pa.gm(filename,'3Dmape')
    Lx    = len(con)
    Ly    = len(con)
    Lz    = len(con)
    sizeL = pa.gm(filename,'L')
    z     = pa.gm(filename,'z')
    print('Size =  (',Lx,'x',Ly,'x',Lz,') in file ',filename)

mena = np.mean(con)
con  = con/mena

print('Max contrast = ', con.max())

L2 = 1

x = np.linspace(-L2, L2, Lx).reshape(Lx,1,1)
y = np.linspace(-L2, L2, Ly).reshape(1,Ly,1)
z = np.linspace(-L2, L2, Lz).reshape(1,1,Lz)
rh2 = 1-np.clip(np.sqrt(x**2 + y**2 +z**2),0,1)

d2 = np.empty(con.shape + (4,), dtype=np.ubyte)
positive = np.clip(con, 0, 10)**2

d2[..., 0] = positive * (255./positive.max())
d2[..., 1] = d2[..., 0]
d2[..., 2] = d2[..., 0]
d2[..., 3] = d2[..., 0]
d2[..., 3] = (d2[..., 3].astype(float) / 255.) **2 * 255
#d2[..., 3] = rh2 * 255

d2[:, 0, 0] = [255,0,0,100]
d2[0, :, 0] = [0,255,0,100]
d2[0, 0, :] = [0,0,255,100]


from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = Lx*2
w.show()
w.setWindowTitle('Logathmic density contrast')


v = gl.GLVolumeItem(d2)
v.translate(-int(Lx/2),-int(Ly/2),-int(Lz/2))
w.addItem(v)

ax = gl.GLAxisItem()
w.addItem(ax)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
