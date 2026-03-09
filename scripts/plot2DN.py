#!/usr/bin/python

import os,re,sys

import h5py, pickle, gzip
import numpy as np

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from matplotlib import colors

class	Plot2D():
	def	__init__(self,map='map',field=''):
		prefileMeas = sorted([x for x in [y for y in os.listdir("./")] if re.search("axion.m.[0-9]{5}$", x)])
		fileMeas = []
		for maes in prefileMeas:
			try:
				with h5py.File(maes, 'r') as f:
					if (map in f) :
						fileMeas.append(maes)
					if (map == 'Moore'):
						if ('m' in f) :
							fileMeas.append(maes)
			except:
				print('Error opening file: %s'%maes)

		print('%s > %s'%(fileMeas[0],fileMeas[-1]))

		fileHdf5 = h5py.File(fileMeas[0], "r")

		self.Lx = fileHdf5["/"].attrs.get("Size")
		self.Ly = fileHdf5["/"].attrs.get("Size")
		self.Lz = fileHdf5["/"].attrs.get("Depth")

		print('%dx%dx%d'%(self.Lx,self.Ly,self.Lz))

		self.z = fileHdf5["/"].attrs.get("z")
		if ('R' in fileHdf5.attrs) :
			self.R = fileHdf5["/"].attrs.get("R")
		else :
			self.R = self.z

		fileHdf5.close()

		self.allData = []

		self.step  = 1
		self.tStep = 100
		self.pause = False

		self.timer = pg.QtCore.QTimer()

		# Read all data

		self.i = 0
		self.size = 0

		self.L1 = self.Lx
		self.L2 = self.Ly

		if map == 'mapp':
			self.L1 = self.Ly
			self.L2 = self.Lz

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

			if ('R' in fileHdf5.attrs) :
				R  = fileHdf5["/"].attrs.get("R")
			else :
				R = zR

			L1=Lx
			L2=Ly

			if map == 'mapp':
				L1=Ly
				L2=Lz

			fl = fileHdf5["/"].attrs.get("Field type").decode()

			if self.Lx != Lx or self.Ly != Ly or self.Lz != Lz:
				print("Error: Size mismatch (%d %d %d) vs (%d %d %d)\nAre you mixing files?\n" % (Lx, Ly, Lz, self.Lx, self.Ly, self.Lz))
				exit()



			if fl == "Saxion":
				mTmp1  = fileHdf5[map]['m'][()].reshape(L2,L1,2)
				mTmp2  = fileHdf5[map]['m2'][()].reshape(L2,L1,2)
				# remove shift?
				shift = fileHdf5["/potential"].attrs.get("Shift")
				# mTmp[:,:,0] -= shift*R
				# mTmp1[:,:,0] -= shift*R
				aData = ((2*np.arctan2(mTmp1[:,:,1], mTmp1[:,:,0])
				          + np.arctan2(mTmp2[:,:,1], mTmp2[:,:,0])+np.pi)%(2*np.pi))/(2.*np.pi)
				rData = np.sqrt((mTmp1[:,:,0]**2 + mTmp1[:,:,1]**2)*(mTmp2[:,:,0]**2 + mTmp2[:,:,1]**2))/R
				rMax = np.amax(rData)
				rData = rData/R*0.75


			else:
				print("Unrecognized field type %s" % fl)
				exit()

			self.allData.append([rData, aData, zR, rMax, fl])
			fileHdf5.close()

			self.size = self.size + 1

#			fp = gzip.open("Strings.PyDat", "wb")
#			pickle.dump(self.allData, fp, protocol=2)
#			fp.close()


		self.app  = QtWidgets.QApplication([])
		self.pWin = pg.GraphicsLayoutWidget()

		data = self.allData[0]
		fl = data[4]
		if fl == 'Saxion':
			self.pWin.setWindowTitle('Axion / Saxion evolution')
		if fl == 'Axion':
			self.pWin.setWindowTitle('Axion / energy evolution')
		if fl == 'Paxion':
			self.pWin.setWindowTitle('Paxion phase / density evolution')

		self.pWin.resize(1600,1600)
		pg.setConfigOptions(antialias=True)

		self.aPlot = self.pWin.addPlot(row=0, col=0)
		self.sPlot = self.pWin.addPlot(row=0, col=1)

		L12 = L1
		if L2>L1:
			L12=L2
		self.aPlot.setXRange(0,L12*1.2)
		self.aPlot.setYRange(0,L12*1.2)
		self.sPlot.setXRange(0,L12*1.2)
		self.sPlot.setYRange(0,L12*1.2)

		self.zAtxt = pg.TextItem("ct=0.000000")
		self.zStxt = pg.TextItem("ct=0.000000")


		aPos = np.array([0.00, 0.30, 0.5, 0.70, 1.0 ])
		aCol = ['w', 'r', 'k', 'b', 'w']

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
		if fl == 'Axion' or fl == 'Saxion':
			sPos = np.array([0.00, 0.25, 0.5, 0.75, 1.0])
			sCol = ['w', 'r', 'y', 'k', 'm']
		else :
			# sPos = np.array([0.00, 1.00])
			# sCol = ['k', 'w']
			sPos = np.array([0.00, 0.5, 1.00])
			sCol = ['k', 'g', 'w']
		sLab = ["%.2f" % mod for mod in sPos]

		vs = self.sPlot.getViewBox()

		sSzeX = vs.size().width()*0.96
		sSzeY = vs.size().height()*0.8

		self.sMap  = pg.ColorMap(sPos, np.array([pg.colorTuple(pg.Color(c)) for c in sCol]))
		self.sLut  = self.sMap.getLookupTable()
		self.sLeg  = pg.GradientLegend((sSzeX/20, sSzeY), (sSzeX/0.96 + sSzeX, sSzeY/12.))
		# self.sLeg.setLabels({ sLab[0]: 0.0, sLab[1]: 0.25, sLab[2]: 0.50, sLab[3]: 0.75, sLab[4]: 1.00 })
		self.sLeg.setLabels(dict(zip(sLab, sPos)))
		self.sLeg.setParentItem(self.sPlot)

		self.cola = dict(zip(sCol, [QtGui.QColor(c) for c in sCol]))
		for s,c in zip(sPos,sCol):
			# print(QtGui.QColor(c))
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
		fl   = data[4]
		self.sImg.setImage(data[0], levels=(0.,1.))
		self.aImg.setImage(data[1], levels=(0.,1.))
		self.zStxt.setText("ct = %.3e" % data[2])
		self.zAtxt.setText("ct = %.3e" % data[2])

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

		if fl == 'Axion' or fl == 'Saxion':
			sPos = np.array([0.00, 0.25, 0.5, 0.75, 1.0])
			sCol = ['w', 'r', 'y', 'k', 'm']
			sLab = ["%.2f" % (mod/0.75) for mod in sPos]
			self.sLeg.gradient.setColorAt(0.00, QtGui.QColor(255,255,255))
			self.sLeg.gradient.setColorAt(0.25, QtGui.QColor(255,  0,  0))
			self.sLeg.gradient.setColorAt(0.50, QtGui.QColor(255,255,  0))
			self.sLeg.gradient.setColorAt(0.75, QtGui.QColor(  0,0,0))
			self.sLeg.gradient.setColorAt(1.00, QtGui.QColor(255,  0,255))
			# for s,c in zip(sPos,sCol):
			# 	self.sLeg.gradient.setColorAt(s, self.cola[c])

			# self.sLeg.gradient.setColorAt(1.50, QtGui.QColor(255,  0,255))

			# self.sLeg.setLabels({ sLab[0]: 0.0, sLab[1]: 0.25, sLab[2]: 0.50, sLab[3]: 0.75, sLab[4]: 1.00, sLab[5]: 1.50 })
			self.sLeg.setLabels(dict(zip(sLab, sPos)))

		else :
			sPos = np.array([0.00, 0.5, 1.00])
			sCol = ['k', 'g', 'w']
			sLab = ["%.2f" % mod for mod in sPos]
			self.sLeg.gradient.setColorAt(0.00, QtGui.QColor(0,0,0))
			self.sLeg.gradient.setColorAt(0.50, QtGui.QColor(0,255,  0))
			self.sLeg.gradient.setColorAt(1.00, QtGui.QColor(  255,  255,  255))
			self.sLeg.setLabels({ sLab[0]: 0.0, sLab[1]: 0.50, sLab[2]: 1.00 })

		self.sLeg.setLabels(dict(zip(sLab, sPos)))
		# for s,c in zip(sPos,sCoQ):
		# 	self.sLeg.gradient.setColorAt(s, c)


		self.i = (self.i+self.step)%self.size

	def	start(self):
		self.timer.timeout.connect(self.update)
		self.timer.start(self.tStep)
		self.pWin.keyPressEvent = self.keyPressEvent

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


# Plot 2D

if	__name__ == '__main__':

	map = 'map'
	if sys.argv[-1] == 'mapp':
		map = 'mapp'
		print('mode mapp')
	elif sys.argv[-1] == 'Moore':
		map = 'Moore'
		print('mode Moore')
	p = Plot2D(map)
	p.start()
