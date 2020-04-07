#!/usr/bin/python

import os,re,sys

import h5py, pickle, gzip
import numpy as np

from pyqtgraph.Qt import QtCore, QtGui

import pyqtgraph as pg
import pyqtgraph.opengl as gl

class	Plot2D():
	def	__init__(self,map='map'):
		prefileMeas = sorted([x for x in [y for y in os.listdir("./")] if re.search("axion.m.[0-9]{5}$", x)])
		fileMeas = []
		for maes in prefileMeas:
			try:
				with h5py.File(maes, 'r') as f:
					if (map in f) :
						fileMeas.append(maes)
			except:
				print('Error opening file: %s'%maes)

		fileHdf5 = h5py.File(fileMeas[0], "r")

		self.Lx = fileHdf5["/"].attrs.get("Size")
		self.Ly = fileHdf5["/"].attrs.get("Size")
		self.Lz = fileHdf5["/"].attrs.get("Depth")

		self.z = fileHdf5["/"].attrs.get("z")
		self.R = fileHdf5["/"].attrs.get("R")

		fileHdf5.close()

		self.allData = []

		self.step  = 1
		self.tStep = 100
		self.pause = False

		self.timer = pg.QtCore.QTimer()

		# Read all data

		self.i = 0
		self.size = 0

#		if os.path.exists("./Strings.PyDat"):
#			fp = gzip.open("./Strings.PyDat", "rb")
#			self.allData = pickle.load(fp)
#			fp.close()
#			self.size = len(self.allData)
#		else:
		for meas in fileMeas:
#			print(meas)
			fileHdf5 = h5py.File(meas, "r")

			Lx = fileHdf5["/"].attrs.get("Size")
			Ly = fileHdf5["/"].attrs.get("Size")
			Lz = fileHdf5["/"].attrs.get("Depth")
			zR = fileHdf5["/"].attrs.get("z")
			R  = fileHdf5["/"].attrs.get("R")
			# if 'R' in fileHdf5:
			# 	R = fileHdf5["/"].attrs.get("R")

			fl = fileHdf5["/"].attrs.get("Field type").decode()

			if self.Lx != Lx or self.Ly != Ly or self.Lz != Lz:
				print("Error: Size mismatch (%d %d %d) vs (%d %d %d)\nAre you mixing files?\n" % (Lx, Ly, Lz, self.Lx, self.Ly, self.Lz))
				exit()

			if fl == "Saxion":
				mTmp  = fileHdf5[map]['m'].value.reshape(Ly,Lx,2)
				aData = (np.arctan2(mTmp[:,:,1], mTmp[:,:,0]) + 2*np.pi)/(4.*np.pi)
				if sys.argv[-1] == 'vel':
					vTmp  = fileHdf5[map]['v'].value.reshape(Ly,Lx,2)
					rData = vTmp[:,:,1]*mTmp[:,:,0]-vTmp[:,:,0]*mTmp[:,:,1]
					rMax = np.amax(rData)
				else :
					rData = np.sqrt(mTmp[:,:,0]**2 + mTmp[:,:,1]**2)
					rMax = np.amax(rData)
					rData = rData/R

			elif fl == "Axion":
				aData = fileHdf5[map]['m'].value.reshape(Ly,Lx)
#				pm = np.amax(aData)
#				print ("BMax %f" % pm)
				aData = aData/R
				rData = np.ones(aData.shape)
				pData = np.ones(aData.shape)*(2*np.pi)
				aData = (aData + pData)/(4.*np.pi)
#				iData = np.trunc(aData/(2*np.pi))
#				aData = aData - iData*(2*np.pi)
#				aData = aData - pData
#				pm = np.amax(aData)
#				print ("AMax %f" % pm)
				rMax  = R
			elif fl == "Naxion":
				mTmp  = fileHdf5[map]['m'].value.reshape(Ly,Lx,2)
				mAmA  = fileHdf5["/"].attrs.get("Axion mass")
				rData = np.sqrt((mTmp[:,:,0]**2 + mTmp[:,:,1]**2)) # /(mAmA*R**3))
				rMax = np.amax(rData)
				aData = (np.arctan2(mTmp[:,:,1], mTmp[:,:,0]) + 2*np.pi)/(4.*np.pi)
			elif fl == "Paxion":
				mTmp1  = fileHdf5[map]['m'].value.reshape(Ly,Lx)
				mTmp2  = fileHdf5[map]['v'].value.reshape(Ly,Lx)
				mAmA  = fileHdf5["/"].attrs.get("Axion mass")
				rData = np.sqrt((mTmp1[:,:]**2 + mTmp2[:,:]**2)) #/(mAmA*R**3))
				rMax = np.amax(rData)
				rData = rData/rMax
				aData = (np.arctan2(mTmp2[:,:], mTmp1[:,:]) + 2*np.pi)/(4.*np.pi)

			else:
				print("Unrecognized field type %s" % fl)
				exit()

			self.allData.append([rData, aData, zR, rMax])
			fileHdf5.close()

			self.size = self.size + 1

#			fp = gzip.open("Strings.PyDat", "wb")
#			pickle.dump(self.allData, fp, protocol=2)
#			fp.close()


		self.app  = QtGui.QApplication([])
		self.pWin = pg.GraphicsLayoutWidget()
		self.pWin.setWindowTitle('Axion / Saxion evolution')
		self.pWin.resize(1600,1600)
		pg.setConfigOptions(antialias=True)

		self.aPlot = self.pWin.addPlot(row=0, col=0)
		self.sPlot = self.pWin.addPlot(row=0, col=1)

		self.aPlot.setXRange(0,Lx*1.2)
		self.aPlot.setYRange(0,Lx*1.2)
		self.sPlot.setXRange(0,Lx*1.2)
		self.sPlot.setYRange(0,Lx*1.2)

		self.zAtxt = pg.TextItem("z=0.000000")
		self.zStxt = pg.TextItem("z=0.000000")

		data = self.allData[0]

		aPos = np.array([0.00, 0.10, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0 ])
		aCol = ['#006600', 'b', 'w', 'r', 'k', 'b', 'w', 'r', '#006600']

		vb = self.aPlot.getViewBox()

		aSzeX = vb.size().width()*0.96
		aSzeY = vb.size().height()*0.8

		self.aMap  = pg.ColorMap(aPos, np.array([pg.colorTuple(pg.Color(c)) for c in aCol]))
		self.aLut  = self.aMap.getLookupTable()
		self.aLeg  = pg.GradientLegend((aSzeX/20, aSzeY), (aSzeX, aSzeY/12.))
		self.aLeg.setLabels({ "-pi": 0.0, "-pi/2": 0.25, "0.0": 0.50, "+pi/2": 0.75, "+pi": 1.00 })
		self.aLeg.setParentItem(self.aPlot)
		self.aLeg.gradient.setColorAt(0.00, QtGui.QColor(255,255,255))
		self.aLeg.gradient.setColorAt(0.25, QtGui.QColor(255,  0,  0))
		self.aLeg.gradient.setColorAt(0.50, QtGui.QColor(  0,  0,  0))
		self.aLeg.gradient.setColorAt(0.75, QtGui.QColor(  0,  0,255))
		self.aLeg.gradient.setColorAt(1.00, QtGui.QColor(255,255,255))

		self.aImg = pg.ImageItem(lut=self.aLut)
		self.aPlot.addItem(self.aImg)
		self.aPlot.addItem(self.zAtxt)

#		sPos = np.linspace(0.0, data[3], 5)
		# sPos = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
		# sLab = ["%.2f" % mod for mod in sPos]
		# sCol = ['w', 'r', 'y', 'c', 'k']
		sPos = np.array([0.00, 1.00])
		sLab = ["%.2f" % mod for mod in sPos]
		sCol = ['k', 'w']

		vs = self.sPlot.getViewBox()

		sSzeX = vs.size().width()*0.96
		sSzeY = vs.size().height()*0.8

		self.sMap  = pg.ColorMap(sPos, np.array([pg.colorTuple(pg.Color(c)) for c in sCol]))
		self.sLut  = self.sMap.getLookupTable()
		self.sLeg  = pg.GradientLegend((sSzeX/20, sSzeY), (aSzeX/0.96 + sSzeX, sSzeY/12.))
		# self.sLeg.setLabels({ sLab[0]: 0.0, sLab[1]: 0.25, sLab[2]: 0.50, sLab[3]: 0.75, sLab[4]: 1.00 })
		self.sLeg.setLabels(dict(zip(sLab, sPos)))
		self.sLeg.setParentItem(self.sPlot)
		for s,c in zip(sPos,sCol):
			self.sLeg.gradient.setColorAt(s, QtGui.QColor(c))
		# self.sLeg.gradient.setColorAt(s, QtGui.QColor(255,255,255))
		# self.sLeg.gradient.setColorAt(0.25, QtGui.QColor(255,  0,  0))
		# self.sLeg.gradient.setColorAt(0.50, QtGui.QColor(255,255,  0))
		# self.sLeg.gradient.setColorAt(0.75, QtGui.QColor(  0,255,255))
		# self.sLeg.gradient.setColorAt(1.00, QtGui.QColor(  0,  0,  0))

		self.sImg = pg.ImageItem(lut=self.sLut)
		self.sPlot.addItem(self.sImg)
		self.sPlot.addItem(self.zStxt)


		self.sImg.setImage(data[0], levels=(0.,1.))
		self.aImg.setImage(data[1], levels=(0.,1.))

		self.pWin.show()


		self.baseKeyPress = self.pWin.keyPressEvent

	def	update(self):
		data = self.allData[self.i]
		self.sImg.setImage(data[0], levels=(0.,1.))
		self.aImg.setImage(data[1], levels=(0.,1.))
		self.zStxt.setText("z = %.3e" % data[2])
		self.zAtxt.setText("z = %.3e" % data[2])

		vb = self.aPlot.getViewBox()

		aSzeX = vb.size().width()*0.96
		aSzeY = vb.size().height()*0.8

		self.aLeg.size   = (aSzeX/20, aSzeY)
		self.aLeg.offset = (aSzeX,    aSzeY/12.)

		vs = self.sPlot.getViewBox()

		sSzeX = vs.size().width()*0.96
		sSzeY = vs.size().height()*0.8

		self.sLeg.size   = (sSzeX/20, sSzeY)
		self.sLeg.offset = (aSzeX/0.96 + sSzeX, sSzeY/12.)

		sPos = np.linspace(0.0, data[3], 5)
		sLab = ["%.2f" % mod for mod in sPos]
		self.sLeg.gradient.setColorAt(0.00, QtGui.QColor(255,255,255))
		self.sLeg.gradient.setColorAt(0.25, QtGui.QColor(255,  0,  0))
		self.sLeg.gradient.setColorAt(0.50, QtGui.QColor(255,255,  0))
		self.sLeg.gradient.setColorAt(0.75, QtGui.QColor(  0,255,255))
		self.sLeg.gradient.setColorAt(1.00, QtGui.QColor(  0,  0,  0))
		self.sLeg.setLabels({ sLab[0]: 0.0, sLab[1]: 0.25, sLab[2]: 0.50, sLab[3]: 0.75, sLab[4]: 1.00 })

		self.i = (self.i+self.step)%self.size

	def	start(self):
		self.timer.timeout.connect(self.update)
		self.timer.start(self.tStep)
		self.pWin.keyPressEvent = self.keyPressEvent

		if	(sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
			QtGui.QApplication.instance().exec_()

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


# Plot 2D

if	__name__ == '__main__':

	map = 'map'
	if sys.argv[-1] == 'mapp':
		map = 'mapp'
		print('mode mapp')
	p = Plot2D(map)
	p.start()
