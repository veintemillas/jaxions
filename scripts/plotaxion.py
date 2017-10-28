# -*- coding: utf-8 -*-


from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

import os,re,sys

import h5py

prefileMeas = sorted([x for x in [y for y in os.listdir("./")] if re.search("axion.m.[0-9]{5}$", x)])
fileMeas = []
for maes in prefileMeas:
	f = h5py.File(maes, 'r')
	if 'map' in f:
		fileMeas.append(maes)
		
tl = len(fileMeas)

fileHdf5 = h5py.File(fileMeas[0], "r")

Lx    = fileHdf5["/"].attrs.get("Size")
Ly    = fileHdf5["/"].attrs.get("Size")
Lz    = fileHdf5["/"].attrs.get("Depth")
sizeL = fileHdf5["/"].attrs.get("Physical size")

z = fileHdf5["/"].attrs.get("z")

fileHdf5.close()

print('Expected ',tl,' ',Lx,'x',Ly, ' pics (L=,',sizeL,')')

allData = np.empty((tl,Lx,Ly))

## Create a GL View widget to display data
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.resize(1000,1000)
w.show()
w.setWindowTitle('pyqtgraph example: GLSurfacePlot')

L2 = 1
HH =0.3

w.setCameraPosition(distance=2*L2)

## Add a grid to the view
#g = gl.GLGridItem()
#g.scale(2,2,2)
#g.setSpacing(sizeL/10)
#g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
#w.addItem(g)

xGrid = gl.GLGridItem()
yGrid = gl.GLGridItem()
zGrid = gl.GLGridItem()

w.addItem(xGrid)
w.addItem(yGrid)
w.addItem(zGrid)

xGrid.rotate(90, 0, 1, 0)
yGrid.rotate(90, 1, 0, 0)


xGrid.translate(-L2, 0, 0)
yGrid.translate(0, -L2, 0)
zGrid.translate(0, 0, 0)

xGrid.scale(0.1*HH, 0.1*L2, 0.1*L2)
yGrid.scale(0.1*L2, 0.1*HH, 0.1*L2)
zGrid.scale(0.1*L2, 0.1*L2, 0)
step  = 1
tStep = 100
pause = False

timer = pg.QtCore.QTimer()

# Read all data
i = 0
size = 0

#		if os.path.exists("./Strings.PyDat"):
#			fp = gzip.open("./Strings.PyDat", "rb")
#			allData = pickle.load(fp)
#			fp.close()
#			size = len(allData)
#		else:
for meas in fileMeas:
#			print(meas)
	fileHdf5 = h5py.File(meas, "r")

	Lx = fileHdf5["/"].attrs.get("Size")
	Ly = fileHdf5["/"].attrs.get("Size")
	Lz = fileHdf5["/"].attrs.get("Depth")
	zR = fileHdf5["/"].attrs.get("z")

	fl = fileHdf5["/"].attrs.get("Field type").decode()

	if Lx != Lx or Ly != Ly or Lz != Lz:
		print("Error: Size mismatch (%d %d %d) vs (%d %d %d)\nAre you mixing files?\n" % (Lx, Ly, Lz, Lx, Ly, Lz))
		exit()

	if fl == "Saxion":
		mTmp  = fileHdf5['map']['m'].value.reshape(Ly,Lx,2)
		aData = np.arctan2(mTmp[:,:,1], mTmp[:,:,0])*HH/(np.pi)

	elif fl == "Axion":
		aData = fileHdf5['map']['m'].value.reshape(Ly,Lx)
	#				pm = np.amax(aData)
	#				print ("BMax %f" % pm)
		aData = (aData/zR)*HH/(np.pi)

	else:
		print("Unrecognized field type %s" % fl)
		exit()

	allData[size] = aData
	fileHdf5.close()

	size = size + 1

#			fp = gzip.open("Strings.PyDat", "wb")
#			pickle.dump(allData, fp, protocol=2)
#			fp.close()

## Animated example
## compute surface vertex data

x = np.linspace(-L2, L2, Lx).reshape(Lx,1)
y = np.linspace(-L2, L2, Ly).reshape(1,Ly)

## precompute height values for all frames
#phi = np.arange(0, np.pi*2, np.pi/20.)
#z = np.sin(d[np.newaxis,...] + phi.reshape(phi.shape[0], 1, 1)) / d2[np.newaxis,...]


## create a surface plot, tell it to use the 'heightColor' shader
## since this does not require normal vectors to render (thus we
## can set computeNormals=False to save time when the mesh updates)
# p4 = gl.GLSurfacePlotItem(x=x[:,0], y = y[0,:], shader='heightColor', computeNormals=False, smooth=False)
# p4.shader()['colorMap'] = np.array([0.2, 2, 0.5, 0.2, 1, 1, 0.2, 0, 2])
p4 = gl.GLSurfacePlotItem(x=x[:,0], y = y[0,:], shader='heightColor', computeNormals=False, smooth=True)
#p4.shader()['colorMap'] = np.array([0.2, 2, 0.5, 0.2, 1, 1, 0.2, 0, 2])
p4.shader()['colorMap'] = np.linspace(0,1,100)

p4.translate(0, 0, 0)
w.addItem(p4)

index = 0
def update():
    global p4, allData, index
    index += 1
    p4.setData(z=allData[index%allData.shape[0]])

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
