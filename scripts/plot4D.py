# -*- coding: utf-8 -*-
"""
This example demonstrates the use of ImageView, which is a high-level widget for
displaying and analyzing 2D and 3D data. ImageView provides:

  1. A zoomable region (ViewBox) for displaying the image
  2. A combination histogram and gradient editor (HistogramLUTItem) for
     controlling the visual appearance of the image
  3. A timeline for selecting the currently displayed frame (for 3D data only).
  4. Tools for very basic analysis of image data (see ROI and Norm buttons)

"""
## Add path to library (just for examples; you do not need this)
#import initExample

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from pyaxions import jaxions as pa

from matplotlib import cm

import os,re,sys
import h5py

maskthreshold = 0.5

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

app = QtGui.QApplication([])

## Create window with ImageView widget
win = QtGui.QMainWindow()
win.resize(1600,1600)
imv = pg.ImageView()
win.setCentralWidget(imv)
win.show()
win.setWindowTitle('pyqtgraph example: ImageView')

## Create random 3D data set with noisy signals
# img = pg.gaussianFilter(np.random.normal(size=(200, 200)), (5, 5)) * 20 + 100
# img = img[np.newaxis,:,:]
# decay = np.exp(-np.linspace(0,0.3,100))[:,np.newaxis,np.newaxis]
# data = np.random.normal(size=(100, 200, 200))
# data += img * decay
# data += 2

print('modes: theta [default], vtheta, saxion, vsaxion, saxion, eA, eP, real, imag')
mode = 'theta'
mapa = 'map/m'
if len(sys.argv) == 2:
    if (sys.argv[-1] == 'eA'):
        mode = 'eA'
        mapa = 'map/E'
        print('Axion energy map')
    elif (sys.argv[-1] == 'eP'):
        mode = 'eP'
        mapa = 'map/P'
        print('Axion energy projection map')
    elif (sys.argv[-1] == 'S'):
        mode = 'S'
        mapa = 'map/m'
        print('Saxion')
    if (sys.argv[-1] == 'vA'):
        mode = 'vA'
        mapa = 'map/v'
        print('Axion velocity')
    elif (sys.argv[-1] == 'M'):
        mode = 'M'
        mapa = 'map/m'
        print('Saxion-masked')
    elif (sys.argv[-1] == 'dens'):
        mode = 'den'
        mapa = 'map/v'
        print('Density from m,v')
    elif (sys.argv[-1] == 'real'):
        mode = 'real'
        mapa = 'map/m'
        print('real part of m/|m|')
    elif (sys.argv[-1] == 'imag'):
        mode = 'imag'
        mapa = 'map/m'
        print('imaginary part of m/|m|')
    elif (sys.argv[-1] == 'N'):
        mode = 'Naxion'
        mapa = 'map/m'
        print('|theta|')

if len(sys.argv) == 3:
    if (sys.argv[-2] == 'map'):
        mode = 'map'
        mapa = 'map/'+sys.argv[-1]
        print('mode map -> ',mapa)

prefileMeas = sorted([x for x in [y for y in os.listdir("./")] if re.search("axion.m.[0-9]{5}$", x)])
fileMeas = []

for maes in prefileMeas:
	try:
		with h5py.File(maes, 'r') as f:
			if (mapa in f) :
				fileMeas.append(maes)
	except:
		print('Error opening file: %s'%maes)

fileHdf5 = h5py.File(fileMeas[0], "r")
Lx = fileHdf5["/"].attrs.get("Size")
Ly = fileHdf5["/"].attrs.get("Size")
Lz = fileHdf5["/"].attrs.get("Depth")
# z = fileHdf5["/"].attrs.get("z")

fileHdf5.close()
allData = []
zData = []
for meas in fileMeas:
#			print(meas)
    fileHdf5 = h5py.File(meas, "r")
    zR = fileHdf5["/"].attrs.get("z")
    R  = fileHdf5["/"].attrs.get("R")
    fl = fileHdf5["/"].attrs.get("Field type").decode()
    mA = fileHdf5["/"].attrs.get("Axion mass")

    if (mode == 'theta') and pa.gm(meas,'map?'):
        if fl == "Saxion":
            mTmp  = fileHdf5['map']['m'].value.reshape(Ly,Lx,2)
            # aData = (np.arctan2(mTmp[:,:,1], mTmp[:,:,0]) + 2*np.pi)/(4.*np.pi)
            aData = (np.arctan2(mTmp[:,:,1], mTmp[:,:,0]))
            # rData = np.sqrt(mTmp[:,:,0]**2 + mTmp[:,:,1]**2)
            # rMax = np.amax(rData)
            # rData = rData/zR
        elif fl == "Axion":
            aData = fileHdf5['map']['m'].value.reshape(Ly,Lx)
            aData = aData/zR
            # rData = np.ones(aData.shape)
            # pData = np.ones(aData.shape)*(2*np.pi)
            # aData = (aData + pData)/(4.*np.pi)
        elif fl == "Naxion":
            aData = fileHdf5['map']['m'].value.reshape(Ly,Lx,2)
            # aData = np.sqrt(aData[:,:,0]**2 + aData[:,:,1]**2)
            aData = aData[:,:,0]/np.sqrt(mA*R)/R
            # missing normalisation
    if (mode == 'vA') and pa.gm(meas,'map?'):
        if fl == "Saxion":
            mTmp  = fileHdf5['map']['m'].value.reshape(Ly,Lx,2)
            mTmp2  = fileHdf5['map']['v'].value.reshape(Ly,Lx,2)
            # aData = ((mTmp2/mTmp))[:,:,1]
            aData = (mTmp2[:,:,1]*mTmp[:,:,0]-mTmp2[:,:,0]*mTmp[:,:,1])/(mTmp[:,:,0]**2+mTmp[:,:,1]**2)

        elif fl == "Axion":
            mTmp = fileHdf5['map']['m'].value.reshape(Ly,Lx)
            mTmp2 = fileHdf5['map']['v'].value.reshape(Ly,Lx)
            aData = (mTmp2-mTmp/zR)/zR
    elif mode == 'eA' and pa.gm(meas,'2Dmape?'):
            # avi = pa.gm(meas,'eA')
            # aData = ((fileHdf5['map']['E'].value.reshape(Ly,Lx)/avi -1))**2
            # aData = fileHdf5['map']['E'].value.reshape(Ly,Lx)/avi
            aData = fileHdf5['map']['E'].value.reshape(Ly,Lx)
            aData = aData/aData.mean()
    elif mode == 'eP' and pa.gm(meas,'2DmapP?'):
            avi = pa.gm(meas,'eA')
            if fl == "Paxion":
                avi = pa.gm(meas,'eAK')
            # aData = ((fileHdf5['map']['E'].value.reshape(Ly,Lx)/avi -1))**2
            # aData = fileHdf5['map']['P'].value.reshape(Ly,Lx)/avi
            aData = fileHdf5['map']['P'].value.reshape(Ly,Lx)/avi/pa.gm(meas,'sizeN')

    elif (mode == 'S') and (fl == "Saxion") and pa.gm(meas,'map?'):
            aData = pa.gm(meas,'maprho')
    elif (mode == 'M') and (fl == "Saxion") and pa.gm(meas,'map?'):
            aData = pa.gm(meas,'maprho')
            mask = (aData < maskthreshold)
            aData = (1-mask)
    if (mode == 'den') and pa.gm(meas,'map?'):
        if fl == "Saxion":
            mTmp  = fileHdf5['map']['m'].value.reshape(Ly,Lx,2)
            aData = 1-np.arg(mTmp[:,:,0])/np.abs(mTmp[:,:])
            # rData = np.sqrt(mTmp[:,:,0]**2 + mTmp[:,:,1]**2)
            # rMax = np.amax(rData)
            # rData = rData/zR
        elif fl == "Axion":
            aData = fileHdf5['map']['m'].value.reshape(Ly,Lx)
            aData = aData/zR
            rData = np.ones(aData.shape)
            pData = np.ones(aData.shape)*(2*np.pi)
            aData = (aData + pData)/(4.*np.pi)
    if (mode == 'real') and pa.gm(meas,'map?'):
        if fl == "Saxion":
            mTmp  = fileHdf5['map']['m'].value.reshape(Ly,Lx,2)
            aData = mTmp[:,:,1]/zR
            # rData = np.sqrt(mTmp[:,:,0]**2 + mTmp[:,:,1]**2)
            # rMax = np.amax(rData)
            # rData = rData/zR
        elif fl == "Axion":
            aData = fileHdf5['map']['m'].value.reshape(Ly,Lx)
            aData = np.cos(aData/zR)
    if (mode == 'Naxion') and pa.gm(meas,'map?'):
        if fl == "Saxion":
            mTmp  = fileHdf5['map']['m'].value.reshape(Ly,Lx,2)
            mAmA  = fileHdf5["/"].attrs.get("Axion mass")
            aData = np.sqrt((mTmp[:,:,0]**2 + mTmp[:,:,1]**2)/(mAmA*R**3))
    if (mode == 'map'):
        aData  = fileHdf5[mapa].value.reshape(Ly,Lx)

        # possible but not coded yet
        # elif fl == "Axion":
        #     aData = fileHdf5['map']['m'].value.reshape(Ly,Lx)
        #     aData = np.cos(aData/zR)

    #				iData = np.trunc(aData/(2*np.pi))
    #				aData = aData - iData*(2*np.pi)
    #				aData = aData - pData
    #				pm = np.amax(aData)
    #				print ("AMax %f" % pm)
    		# rMax  = zR

    allData.append(aData)
    zData.append(zR)
    fileHdf5.close()
    # size = size + 1
    # print(meas,zR)

## Add time-varying signal
# sig = np.zeros(data.shape[0])
# sig[30:] += np.exp(-np.linspace(1,10, 70))
# sig[40:] += np.exp(-np.linspace(1,10, 60))
# sig[70:] += np.exp(-np.linspace(1,10, 30))
#
# sig = sig[:,np.newaxis,np.newaxis] * 3
# data[:,50:60,30:40] += sig

allData=np.array(allData)
zData=np.array(zData)

print(allData[0].shape)
## Display the data and assign each frame a time value from 1.0 to 3.0
# imv.setImage(data, xvals=np.linspace(1., 3., data.shape[0]))
imv.setImage(allData, xvals=zData, autoLevels=True)

## Set a custom color map
if mode == 'Axion':
    colors = [
        (255, 255, 255),(255, 200, 200),
        (255, 0, 0), (0, 0, 0), (0, 0, 255),
        (200, 200, 255), (255, 255, 255) ]
    cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 7), color=colors)
    imv.setColorMap(cmap)

# elif mode == 'eA':
    # colors = []
    # for a in range(0,7):
    #     b=1-(a/7)**2
    #     colors.append((255*b,255*b,255*b))
    # cmap = pg.ColorMap(pos=np.linspace(0., 1, len(colors)), color=colors)
    # imv.setColorMap(cmap)



## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
