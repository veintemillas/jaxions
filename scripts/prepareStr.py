#!/usr/bin/python3

import os,re,sys

import h5py, pickle, gzip
import numpy as np

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

class	Plot3D():
	def	__init__(self):
		fileMeas = sorted([x for x in [y for y in os.listdir("./")] if re.search("axion.m.[0-9]{5}$", x)])

		cucu = []
		print('files with strings? ... ',end='')
		for morsa in fileMeas:
			fileHdf5 = h5py.File(morsa, "r")
			if "/string/data" in fileHdf5:
				cucu.append(morsa)
		print(cucu)
		print('from ', cucu[0], ' to ', cucu[-1])

		self.allData = []

		self.step  = 1
		self.tStep = 100
		self.pause = False

		# Read all data

		self.i = 0
		self.size = 0

		if os.path.exists("./Strings.PyDat"):
			print("already done!")
		else:
			print("Reading measurement files")

			fileHdf5 = h5py.File(fileMeas[0], "r")
			self.Lx = fileHdf5["/"].attrs.get("Size")
			self.Ly = fileHdf5["/"].attrs.get("Size")
			self.Lz = fileHdf5["/"].attrs.get("Depth")

			self.z = fileHdf5["/"].attrs.get("z")

			fileHdf5.close()

			for meas in fileMeas:
				fileHdf5 = h5py.File(meas, "r")

				Lx = fileHdf5["/"].attrs.get("Size")
				Ly = fileHdf5["/"].attrs.get("Size")
				Lz = fileHdf5["/"].attrs.get("Depth")
				zR = fileHdf5["/"].attrs.get("z")

				fl = fileHdf5["/"].attrs.get("Field type").decode()

				if fl == "Axion":
					continue

				if self.Lx != Lx or self.Ly != Ly or self.Lz != Lz:
					print("Error: Size mismatch (%d %d %d) vs (%d %d %d)\nAre you mixing files?\n" % (Lx, Ly, Lz, self.Lx, self.Ly, self.Lz))
					exit()

				if "/string/data" in fileHdf5:
					if noWalls == False:
						strData  = fileHdf5['string']['data'].value.reshape(Lx,Ly,Lz)
						print(meas + ' + walls')
					else:
						strData  = np.bitwise_and(fileHdf5['string']['data'].value.reshape(Lx,Ly,Lz), 63)
						print(meas + ' nowalls')
					z, y, x = strData.nonzero()

					pos = np.array([z,y,x]).transpose()
					color = np.array([col[strData[tuple(p)]] for p in pos])

					self.allData.append([pos, color, zR])
					self.size = self.size + 1

				fileHdf5.close()

			fp = gzip.open("Strings.PyDat", "wb")
			pickle.dump(Lx, fp, protocol=4)
			pickle.dump(Ly, fp, protocol=4)
			pickle.dump(Lz, fp, protocol=4)
			pickle.dump(self.allData, fp, protocol=4)
			fp.close()
		print('data loaded')



noWalls = False

for arg in sys.argv[1:]:
	if arg == "noWalls":
		noWalls = True

p = Plot3D()
