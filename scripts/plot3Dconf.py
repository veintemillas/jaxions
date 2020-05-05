# -*- coding: utf-8 -*-


from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

import os,re,sys

import h5py


fileHdf5 = h5py.File('./' + sys.argv[-1], "r")

an_m = 'm' in fileHdf5

Lx    = fileHdf5["/"].attrs.get("Size")
Ly    = fileHdf5["/"].attrs.get("Size")
Lz    = fileHdf5["/"].attrs.get("Depth")
sizeL = fileHdf5["/"].attrs.get("Physical size")
z     = fileHdf5["/"].attrs.get("z")

if 'Field type' in fileHdf5.attrs:
    ftype = fileHdf5.attrs.get('Field type').decode()
else :
    ftype = 'who knows'

if ftype == 'Axion' :
    if (fileHdf5['potential'].attrs.get('VQcd type').decode() == 'VQcd Moore') and (sys.argv[-2] == 'm'):
        print('Mooron!')
        Lx    = fileHdf5["/"].attrs.get("SizeX")
        Ly    = fileHdf5["/"].attrs.get("SizeY")
        con   = np.mod(fileHdf5[sys.argv[-2]].value.reshape(Lz,Ly,Lx)+np.pi,2*np.pi)-np.pi
    else :
        con   = fileHdf5[sys.argv[-2]].value.reshape(Ly,Lx,Lz)
if ftype == 'Naxion':
    con   = fileHdf5['m'].value.reshape(Ly,Lx,Lz,2)
    if   sys.argv[-1] == 'm':
        con   = con[:,:,:,0]
    elif sys.argv[-1] == 'v':
        con   = con[:,:,:,1]
elif ftype == 'Saxion':
    if sys.argv[-2] == 'm':
        con   = np.array(fileHdf5[sys.argv[-2]].value.reshape(Ly,Lx,Lz,2))
        con   = np.arctan2(con[:,:,:,0],con[:,:,:,1])
    elif sys.argv[-2] == 'mr':
        con   = np.array(fileHdf5[sys.argv[-2]].value.reshape(Ly,Lx,Lz,2))
        con   = con[:,:,:,0]
    elif sys.argv[-2] == 'mi':
        con   = np.array(fileHdf5[sys.argv[-2]].value.reshape(Ly,Lx,Lz,2))
        con   = con[:,:,:,1]
    elif sys.argv[-2] == 'rho':
        con   = np.array(fileHdf5['m'].value.reshape(Ly,Lx,Lz,2))
        con   = np.sqrt(con[:,:,:,0]**2+con[:,:,:,1]**2)
    elif sys.argv[-2] == 'v':
        con   = np.array(fileHdf5[sys.argv[-2]].value.reshape(Ly,Lx,Lz,2))
        con   = np.sqrt(con[:,:,:,0]**2+con[:,:,:,1]**2)
    elif sys.argv[-2] == 'vr':
        con   = np.array(fileHdf5[sys.argv[-2]].value.reshape(Ly,Lx,Lz,2))
        con   = con[:,:,:,0]
    elif sys.argv[-2] == 'vi':
        con   = np.array(fileHdf5[sys.argv[-2]].value.reshape(Ly,Lx,Lz,2))
        con   = con[:,:,:,1]
    elif sys.argv[-2] == 'vt':
        m   = np.array(fileHdf5['m'].value.reshape(Ly,Lx,Lz,2))
        v   = np.array(fileHdf5['v'].value.reshape(Ly,Lx,Lz,2))
        con   = (m[:,:,:,0]*v[:,:,:,1]-m[:,:,:,1]*v[:,:,:,0])/(m[:,:,:,0]**2+v[:,:,:,1]**2)
    elif sys.argv[-2] == 'vtn':
        m   = np.array(fileHdf5['m'].value.reshape(Ly,Lx,Lz,2))
        v   = np.array(fileHdf5['v'].value.reshape(Ly,Lx,Lz,2))
        con   = (m[:,:,:,0]*v[:,:,:,1]-m[:,:,:,1]*v[:,:,:,0])
elif ftype == 'Paxion':
    con   = np.array(fileHdf5[sys.argv[-2]].value.reshape(Ly,Lx,Lz))

if ftype == 'who knows':
    Lx    = fileHdf5["/lattice_data"].attrs.get("Tx")
    Ly    = fileHdf5["/lattice_data"].attrs.get("Ty")
    Lz    = fileHdf5["/lattice_data"].attrs.get("Tz")
    fa = sys.argv[-2]
    if fa in ['Ax','Ay','Az']:
        con   = np.array(fileHdf5[sys.argv[-2]].value.reshape(Ly,Lx,Lz))
    elif fa[0] == 'E':
        i = int(fa[1])
        con   = fileHdf5['E'][()]
        num = (np.arange(len(con)) // 8) % 3
        con = np.reshape(con[num == i],(Lz,Ly,Lx))

print('Size =  (',Lx,'x',Ly,'x',Lz,') in file ',fileHdf5)

print('range is',con.min(),con.max())
L2 = 1

x = np.linspace(-L2, L2, Lx).reshape(Lx,1,1)
y = np.linspace(-L2, L2, Ly).reshape(1,Ly,1)
z = np.linspace(-L2, L2, Lz).reshape(1,1,Lz)
rh2 = 1-np.clip(np.sqrt(x**2 + y**2 +z**2),0,1)

d2 = np.empty(con.shape + (4,), dtype=np.ubyte)
positive = np.clip(con, 0, 10)**2

d2[..., 0] = con * (255./con.max())
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
