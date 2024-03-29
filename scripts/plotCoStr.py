#!/usr/bin/python3

import os,re,sys

import h5py, pickle, gzip
import numpy as np

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import pyqtgraph as pg
import pyqtgraph.opengl as gl

pString = np.array([255, 0, 0])	# Red for strings+
nString = np.array([0, 255, 0])	# Green for strings-
cWalls  = np.array([0, 0, 255])	# Blue for walls

def	makeColor():
	colTable = []

	for color in range(0,256):
		col = np.array([0,0,0])

		if (color & 7) != 0:
			col = col + pString

		if (color & 56) != 0:
			col = col + nString

		if ((color & 63) == 0) and (color & 64):
			col = col + cWalls

		colTable.append(col)

	return	colTable

# Make color table

col = makeColor()

class	GLViewWithText(gl.GLViewWidget):
	z = 0.0
	def	updateZ(self,z):
		self.z = z
	def	paintGL(self, *args, **kwds):
		gl.GLViewWidget.paintGL(self, *args, **kwds)
		self.qglColor(QtCore.Qt.white)
		self.renderText(0,0,1.5, "z = %f" % self.z)

class	Plot3D():
	def	__init__(self):
		fileMeas = sorted([x for x in [y for y in os.listdir("./")] if re.search("axion.m.[0-9]{5}$", x)])

		usableFiles = []
		print('files with string coordinates? ... ',end='')

		for myFile in fileMeas:
			try:
				with h5py.File(myFile, 'r') as fileHdf5:
					if "/string/codata" in fileHdf5:
						usableFiles.append(myFile)
			except:
				print('Error opening file: %s'%myFile)

		print('from ', usableFiles[0], ' to ', usableFiles[-1])

		self.allData = []

		self.step  = 1
		self.tStep = 100
		self.pause = False

		self.timer = pg.QtCore.QTimer()

		# Read all data

		self.i = 0
		self.size = 0

		if os.path.exists("./CoStrings.PyDat"):
			print("Using pickle file")
			fp = gzip.open("./CoStrings.PyDat", "rb")
			self.Lx = pickle.load(fp)
			self.Ly = pickle.load(fp)
			self.Lz = pickle.load(fp)
			self.allData = pickle.load(fp)
			fp.close()
			self.size = len(self.allData)
		else:
			print("Reading measurement files")

			fileHdf5 = h5py.File(usableFiles[0], "r")
			# self.Lx = fileHdf5["/string"].attrs.get("Size")
			# self.Ly = fileHdf5["/string"].attrs.get("Size")
			# self.Lz = fileHdf5["/string"].attrs.get("Depth")
			self.Lx = fileHdf5["/string"].attrs.get("nmax")
			self.Ly = self.Lx
			self.Lz = self.Lx

			# self.z = fileHdf5["/"].attrs.get("z")

			fileHdf5.close()
			i = 0
			for meas in usableFiles:
				fileHdf5 = h5py.File(meas, "r")

				# Lx = fileHdf5["/string"].attrs.get("Size")
				# Ly = fileHdf5["/string"].attrs.get("Size")
				# Lz = fileHdf5["/string"].attrs.get("Depth")
				# zR = fileHdf5["/"].attrs.get("z")
				Lx = fileHdf5["/string"].attrs.get("nmax")
				Ly = Lx
				Lz = Lx
				zR = fileHdf5["/"].attrs.get("z")

				# fl = fileHdf5["/"].attrs.get("Field type").decode()
				#
				# if fl == "Axion":
				# 	continue

				if self.Lx != Lx or self.Ly != Ly or self.Lz != Lz:
					print("Error: Size mismatch (%d %d %d) vs (%d %d %d)\nAre you mixing files?\n" % (Lx, Ly, Lz, self.Lx, self.Ly, self.Lz))
					exit()

				if "/string/codata" in fileHdf5:
					# strData  = np.bitwise_and(fileHdf5['string']['data'].value.reshape(Lx,Ly,Lz), 63)
					strData = np.array(fileHdf5["string/codata"])
					lena = len(strData)//3
					strData =  strData.reshape(lena,3)

					self.allData.append([strData, zR])
					self.size = self.size + 1

				fileHdf5.close()
				i = i+1
			fp = gzip.open("CosStrings.PyDat", "wb")
			pickle.dump(Lx, fp, protocol=4)
			pickle.dump(Ly, fp, protocol=4)
			pickle.dump(Lz, fp, protocol=4)
			pickle.dump(self.allData, fp, protocol=4)
			fp.close()
		print('data loaded')

		pg.setConfigOptions(antialias=True)

		self.app  = QtWidgets.QApplication([])
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

		data = self.allData[0]
		self.sizst = 2
		self.view.updateZ(data[1])
		self.plt = gl.GLScatterPlotItem(pos=data[0], size=self.sizst)
		self.view.addItem(self.plt)
		self.plt.scale(2./float(self.Lx), 2./float(self.Ly), 2./float(self.Lz))
		self.plt.translate(-1.0,-1.0,-1.0)

		self.baseKeyPress = self.view.keyPressEvent
		print('shot!')

	def	update(self):
		data = self.allData[self.i]
		self.view.updateZ(data[1])
		self.plt.setData(pos=data[0])
		self.i = (self.i+self.step)%self.size

	def	start(self):
		self.timer.timeout.connect(self.update)
		self.timer.start(self.tStep)
		self.view.keyPressEvent = self.keyPressEvent
		if	(sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
			QtWidgets.QApplication.instance().exec_()

	def	setData(self,i):
		self.data = allData[i]

	def	keyPressEvent(self, event):
		key = event.key()

		self.baseKeyPress(event)

		if key == QtCore.Qt.Key_Space :
			if self.pause:
				self.pause = False
				self.timer.start(self.tStep)
			else:
				self.pause = True
				self.timer.stop()
		elif key == QtCore.Qt.Key_M:
			self.tStep = self.tStep + 10
			if self.tStep > 1500:
				self.tStep = 1500
			self.timer.setInterval(self.tStep)
		elif key == QtCore.Qt.Key_N:
			self.tStep = self.tStep - 10
			if self.tStep < 20:
				self.tStep = 20
			self.timer.setInterval(self.tStep)
		elif key == QtCore.Qt.Key_R:
			self.step = -self.step



# Plot 3D

if	__name__ == '__main__':

	noWalls = False

	for arg in sys.argv[1:]:
		if arg == "noWalls":
			noWalls = True

	p = Plot3D()
	p.start()
