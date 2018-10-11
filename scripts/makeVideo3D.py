# -*- coding: utf-8 -*-

from pyaxions import jaxions as pa
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

from sympy import integer_nthroot

import os,re,sys

import h5py

def     makeColor():
	colTable = []

	for color in range(0,255):
		colTable.append(np.array([color,0,255-color]))

	return  colTable

# Make color table

col = makeColor()

class   GLViewWithText(gl.GLViewWidget):
	z = 0.0
	def     updateZ(self,z):
		self.z = z
	def     paintGL(self, *args, **kwds):
		gl.GLViewWidget.paintGL(self, *args, **kwds)
		self.qglColor(QtCore.Qt.white)
		self.renderText(0,0,1.5, "z = %f" % self.z)

class   Plot3D():
	def     __init__(self):
		# Gathers list of files

		fileMeas = sorted([x for x in [y for y in os.listdir("./")] if re.search("axion.m.[0-9]{5}$", x)])

		# Filters the files so only those with an energy density map are listed

		fFiles = {}
		Lx     = -1
		Lz     = -1

		for cFile in fileMeas:
			fileHdf5 = h5py.File(cFile, "r")
			cTheta = 'energy/density/cTheta' in fileHdf5
			if cTheta:
				Lx = fileHdf5["/energy/density/"].attrs.get("Size")
				Lz = fileHdf5["/energy/density/"].attrs.get("Depth")
				cHash = str(Lx)+str(Lz)
				if cHash in fFiles.keys():
					fFiles[cHash].append(cFile)
				else:
					fFiles[cHash] = []
					fFiles[cHash].append(cFile)
			fileHdf5.close()

		maxLen = 0
		cKey = ''
		for key in fFiles.keys():
			if len(fFiles[key]) > maxLen:
				maxLen = len(fFiles[key])
				cKey = key

		self.allFiles = fFiles[key]
		self.nFiles   = len(self.allFiles)

		aFHdf5 = h5py.File(self.allFiles[0], "r")

		self.Lx = aFHdf5["/energy/density/"].attrs.get("Size")
		self.Lz = aFHdf5["/energy/density/"].attrs.get("Depth")

		self.step  = 1
		self.tStep = 100
		self.pause = False

		self.current = 0

		self.timer = pg.QtCore.QTimer()


		pg.setConfigOptions(antialias=True)

		self.app  = QtGui.QApplication([])
		self.view = GLViewWithText() #gl.GLViewWidget()

		self.view.show()

		xGrid = gl.GLGridItem()
		yGrid = gl.GLGridItem()
		zGrid = gl.GLGridItem()

		self.view.addItem(xGrid)
		self.view.addItem(yGrid)
		self.view.addItem(zGrid)

		xGrid.rotate(90, 0, 1, 0)
		yGrid.rotate(90, 1, 0, 0)

		xGrid.translate(-1.0, 0, 0)
		yGrid.translate(0, -1.0, 0)
		zGrid.translate(0, 0, -1.0)

		xGrid.scale(0.1, 0.1, 0.1)
		yGrid.scale(0.1, 0.1, 0.1)
		zGrid.scale(0.1, 0.1, 0.1)

		conData   = aFHdf5['/energy/density']['cTheta'].value.reshape(self.Lx,self.Lx,self.Lz)
		meanData  = np.mean(conData)
		conData	  = conData/meanData
		self.data = np.zeros(conData.shape + (4,), dtype=np.ubyte)
		pData     = np.clip(conData, 0, 10)**2
		self.data[..., 0] = pData * (255./pData.max())
		self.data[..., 1] = self.data[..., 0]
		self.data[..., 2] = self.data[..., 0]
		self.data[..., 3] = ((self.data[...,0].astype(float)/255.)**2)*255

		self.z    = aFHdf5['/'].attrs.get("z")

		self.view.updateZ(self.z)
		self.plt = gl.GLVolumeItem(self.data)
		self.view.addItem(self.plt)
		self.plt.scale(2./float(self.Lx), 2./float(self.Lx), 2./float(self.Lz))
		self.plt.translate(-1.0,-1.0,-1.0)

		aFHdf5.close()

	def     update(self):
		fileHdf5  = h5py.File(self.allFiles[self.current], "r")
		conData   = fileHdf5['/energy/density']['cTheta'].value.reshape(self.Lx,self.Lx,self.Lz)

		meanData  = np.mean(conData)
		conData	  = conData/meanData
		self.data = np.zeros(conData.shape + (4,), dtype=np.ubyte)
		pData     = np.clip(conData, 0, 10)**2
		self.data[..., 0] = pData * (255./pData.max())
		self.data[..., 1] = self.data[..., 0]
		self.data[..., 2] = self.data[..., 0]
		self.data[..., 3] = ((self.data[...,0].astype(float)/255.)**2)*255

		self.z    = fileHdf5['/'].attrs.get("z")
		self.view.updateZ(self.z)
		self.plt.setData(data=self.data)
		#self.view.grabFrameBuffer().save("img.%04d.png" % self.current)
		img = self.view.renderToArray((1920, 1080))
		pg.makeQImage(img).save("img.%04d.png" % self.current)

		self.current = (self.current+self.step)
		fileHdf5.close()
		if self.current >= self.nFiles:
			QtGui.QApplication.instance().closeAllWindows()


	def     start(self):
		self.timer.timeout.connect(self.update)
		self.timer.start(self.tStep)
		if      (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
			QtGui.QApplication.instance().exec_()

	def     setData(self,i):
		self.current = i

# Plot 3D

if      __name__ == '__main__':

	p = Plot3D()
	p.start()

	os.system("ffmpeg -r 2 -i img.\%04d.png -vcodec mpeg4 -y axion.mp4")
	os.system("rm -f img.????.png")
	print("Check axion.mp4")
