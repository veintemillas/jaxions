#!/usr/bin/python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import re, os
import h5py
import datetime
mark=f"{datetime.datetime.now():%Y-%m-%d}"
from uuid import getnode as get_mac
mac = get_mac()

# LATEX OUTPUT

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rc('font', family='serif')


# CREATE DIR FOR PICS
if not os.path.exists('./pics'):
    os.makedirs('./pics')

# MOVE TRANSITION FILES
if os.path.exists('./m/axion.m.10000'):
    os.rename('./m/axion.m.10000','./axion.m.10000')
if os.path.exists('./m/axion.m.10001'):
    os.rename('./m/axion.m.10001','./axion.m.10001')


# HDF5 DATASETS
fileMeas = sorted([x for x in [y for y in os.listdir("./m/")] if re.search("axion.m.[0-9]{5}$", x)])
f = h5py.File('./m/'+fileMeas[0], 'r')

# SIMULATION DATA
sizeN = f.attrs[u'Size']
sizeL = f.attrs[u'Physical size']
if 'nQcd' in f['/potential/'].attrs:
    nqcd = f['/potential/'].attrs[u'nQcd']
    print('new format!')
elif 'nQcd' in f:
    nqcd = f.attrs[u'nQcd']
    print('old format')
else :
    nqcd = 7.0

# ID
ups = 'N'+str(sizeN)+' L'+str(sizeL)+' n'+str(nqcd)+' ('+mark+')'+str(mac)
print('ID = '+ups)
print()
for item in f.attrs.keys():
    print(item + ":", f.attrs[item])

# SAMPLE POINT EVOLUTION (+THETA, NSTRINGS...)
# FORMAT :
# z, m_axion, Lambda, m1, m2, v1, v2, #strings, maxtheta, shift
# z, m_axion, theta, vtheta, maxtheta

if os.path.exists('./sample.txt'):

    with open('./sample.txt') as f:
        lines=f.readlines()
        l10 = 0
        l5 = 0
        for line in lines:
            myarray = np.fromstring(line, dtype=float, sep=' ')
            l = len(myarray)
            if l==11:
                l10 = l10 +1
            elif l==6:
                l5 = l5 +1
        print('length %d %d',l10,l5)
        arrayA = np.genfromtxt('./sample.txt',skip_header=l10)
        arrayS = np.genfromtxt('./sample.txt',skip_footer=l5)

    axiondata = len(arrayA) >0

    if l10 >1 :
        ztab1 = arrayS[:,0]
        Rtab1 = arrayS[:,1]
#UNSHIFTED
        Thtab1 = np.arctan2(arrayS[:,5],arrayS[:,4])
        Rhtab1 = arrayS[:,4]**2 + arrayS[:,5]**2
        #theta_z
        VThtab1 = (arrayS[:,4]*arrayS[:,7]-arrayS[:,5]*arrayS[:,6])/(Rhtab1)
        #ctheta_z
        #VThtab1 = ztab1*(arrayS[:,3]*arrayS[:,6]-arrayS[:,4]*arrayS[:,5])/(Rhtab1) + Thtab1
        Rhtab1 = np.sqrt(Rhtab1)/Rtab1

#SHIFTED
        arrayS[:,4] = arrayS[:,4]-Rtab1*arrayS[:,10]
        Thtab1_shift = np.arctan2(arrayS[:,5],arrayS[:,4])
        Rhtab1_shift = arrayS[:,4]**2 + arrayS[:,5]**2
        #theta_z
        VThtab1_shift = (arrayS[:,4]*arrayS[:,7]-arrayS[:,5]*arrayS[:,6])/(Rhtab1_shift)
        #ctheta_z
        #VThtab1_shift = ztab1*(arrayS[:,3]*arrayS[:,6]-arrayS[:,4]*arrayS[:,5])/(Rhtab1_shift) + Thtab1_shift
        Rhtab1_shift = np.sqrt(Rhtab1_shift)/Rtab1


    if axiondata:
        ztab2 = arrayA[:,0]
        Rtab2 = arrayA[:,1]
        Thtab2 = arrayA[:,3]/Rtab2
        #theta_z
        VThtab2 = (arrayA[:,4]-Thtab2)/Rtab2
        #ctheta_z
        #VThtab2 = arrayA[:,3]
    if l10 >1 :
        strings = arrayS[:,8]
        fix = [[ztab1[0],strings[0]]]
        i = 0
        for i in range(0, len(ztab1)-1):
            if strings[i] != strings[i+1]:
                fix.append([ztab1[i+1],strings[i+1]])
        stringo = np.asarray(fix)

        co = (sizeL/sizeN)*(1/6)*(1/sizeL)**3


from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtWidgets.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsLayoutWidget(title="Evolution idx=0") #pg.GraphicsWindow(title="Evolution idx=0")
win.resize(1000,600)
win.setWindowTitle('jaxions evolution')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

p1 = win.addPlot(title=r'theta evolution')

# p1.PlotItem.('left',r'$\theta$')
if l10 >1 :
    p1.plot(ztab1,Thtab1,pen=(100,100,100))
    p1.plot(ztab1,Thtab1_shift,pen=(255,255,255))
if axiondata:
    p1.plot(ztab2,Thtab2,pen=(255,255,0))
p1.setLabel('left',text='theta')
p1.setLabel('bottom',text='time')


p2 = win.addPlot(title=r'theta_t evolution')

# p1.PlotItem.('left',r'$\theta$')
if l10 >1 :
    p2.plot(ztab1,VThtab1,pen=(100,100,100))
    p2.plot(ztab1,VThtab1_shift,pen=(255,255,255))
if axiondata:
    p2.plot(ztab2,VThtab2,pen=(255,255,0))
p2.setLabel('left',text='theta_t')
p2.setLabel('bottom',text='time')


if l10 >1 :
    win.nextRow()

    p3 = win.addPlot(title=r'rho evolution')
    p3.plot(ztab1,Rhtab1,pen=(200,0,0),name='unshifted')
    p3.plot(ztab1,Rhtab1-arrayS[:,10],pen=(100,100,100),name='unshifted')
    p3.plot(ztab1,Rhtab1_shift,pen=(255,255,255),name='shifted')

    p3.setLabel('left',text='rho/v')
    p3.setLabel('bottom',text='time')

    p4 = win.addPlot(title=r'string evolution')

    # p1.PlotItem.('left',r'$\theta$')

    p4.plot(stringo[1:,0],co*stringo[1:,1]*stringo[1:,0]**2,pen=(255,255,255),symbolBrush=(153,255,204))
    p4.setLabel('left',text='Length/Volume')
    p4.setLabel('bottom',text='time')
win.show()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()
