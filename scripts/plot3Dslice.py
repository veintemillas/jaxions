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

print("You can type dens/redens AFTER the file name to choose the full or reduced energy map (if both present)")
print("By default reduced maps are printed.")
if len(sys.argv) == 2:
    filename = './' + sys.argv[-1]
    fileHdf5 = h5py.File(filename, "r")
    an_contrastmap = 'energy/density' in fileHdf5
    re_contrastmap = ('energy/redensity' in fileHdf5) or ('energy/rdensity' in fileHdf5)

elif len(sys.argv) == 3:
    filename = './' + sys.argv[-2]
    fileHdf5 = h5py.File(filename, "r")
    dens0redens = sys.argv[-1]
    if dens0redens == 'dens':
        an_contrastmap = 'energy/density' in fileHdf5
        re_contrastmap = False
    elif dens0redens == 'redens':
        re_contrastmap = 'energy/rdensity' in fileHdf5
        an_contrastmap = False

    # make a choice, if both reduced and full maps exist print the reduced

if re_contrastmap:
    if an_contrastmap:
        print('Both Full and Reduced Contrast found, displaying reduced map')
        an_contrastmap = False
    else :
        print('Reduced Contrast found')
    con = pa.gm(filename,'3Dmaper')
    Lx    = fileHdf5['energy/rdensity'].attrs[u'Size']
    Ly    = fileHdf5['energy/rdensity'].attrs[u'Size']
    Lz    = fileHdf5['energy/rdensity'].attrs[u'Depth']
    sizeL = pa.gm(filename,'L')
    z     = pa.gm(filename,'z')
    print('Size =  (',Lx,'x',Ly,'x',Lz,') in file ',filename)

if an_contrastmap:
    print('Contrast found')
    con = pa.gm(filename,'3Dmapefull',True)
    Lx    = fileHdf5['energy/density'].attrs[u'Size']
    Ly    = fileHdf5['energy/density'].attrs[u'Size']
    Lz    = fileHdf5['energy/density'].attrs[u'Depth']
    sizeL = pa.gm(filename,'L')
    z     = pa.gm(filename,'z')
    print('Size =  (',Lx,'x',Ly,'x',Lz,') in file ',filename)


mena = np.mean(con)
con  = con/mena


print('Average density  = ', mena)
print('Maximum contrast = ', con.max())

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


# -*- coding: utf-8 -*-
"""
Demonstrate a simple data-slicing task: given 3D data (displayed at top), select
a 2D plane and interpolate data along that plane to generate a slice image
(displayed at bottom).


"""

## Add path to library (just for examples; you do not need this)


app = QtGui.QApplication([])

## Create window with two ImageView widgets
win = QtGui.QMainWindow()
win.resize(800,800)
win.setWindowTitle('pyqtgraph example: DataSlicing')
cw = QtGui.QWidget()
win.setCentralWidget(cw)
l = QtGui.QGridLayout()
cw.setLayout(l)
imv1 = pg.ImageView()
imv2 = pg.ImageView()
l.addWidget(imv1, 0, 0)
l.addWidget(imv2, 1, 0)
win.show()

roi = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
imv1.addItem(roi)

# x1 = np.linspace(-30, 10, 128)[:, np.newaxis, np.newaxis]
# x2 = np.linspace(-20, 20, 128)[:, np.newaxis, np.newaxis]
# y = np.linspace(-30, 10, 128)[np.newaxis, :, np.newaxis]
# z = np.linspace(-20, 20, 128)[np.newaxis, np.newaxis, :]
# d1 = np.sqrt(x1**2 + y**2 + z**2)
# d2 = 2*np.sqrt(x1[::-1]**2 + y**2 + z**2)
# d3 = 4*np.sqrt(x2**2 + y[:,::-1]**2 + z**2)
# data = (np.sin(d1) / d1**2) + (np.sin(d2) / d2**2) + (np.sin(d3) / d3**2)

def update():
    global con, imv1, imv2
    d2 = roi.getArrayRegion(con, imv1.imageItem, axes=(1,2))
    imv2.setImage(d2)

roi.sigRegionChanged.connect(update)


## Display the data
imv1.setImage(con)
imv1.setHistogramRange(-0.01, 0.01)
imv1.setLevels(-0.003, 0.003)

update()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
