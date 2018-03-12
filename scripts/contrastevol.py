# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import re, os, sys
import h5py
import datetime
from pyaxions import jaxions as pa
mark=f"{datetime.datetime.now():%Y-%m-%d}"
from uuid import getnode as get_mac
mac = get_mac()

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rc('font', family='serif')

# MOVE TRANSITION FILES
if os.path.exists('./m/axion.m.10000'):
    os.rename('./m/axion.m.10000','./axion.m.10000')
if os.path.exists('./m/axion.m.10001'):
    os.rename('./m/axion.m.10001','./axion.m.10001')

# HDF5 DATASETS TO INCLUDE FIRST AND LAST
fileMeas = pa.findmfiles('./m/')

mylist = []
sel = True
firstlast = True
firstnumber = 1

for meas in fileMeas:
    if pa.gm(meas,'bincon?'):
        mylist.append(meas)
if len(mylist)==0:
    print('No single file with conbin!')
    sys.exit()

# LOOK FOR ARGUMENTS OF THE FUNCTION TO COMPLETE THE SETS PLOTTED
if len(sys.argv) == 1:
    if len(mylist) == 1:
        mylist = [ mylist[0] ]
    elif len(mylist) > 1:
        mylist = [ mylist[0], mylist[-1] ]
else:
    for input in sys.argv[1:]:
        if input == 'all':
            sel = False
        if input == 'every10':
            sel = False
            freqout = len(mylist)//10
            if freqout ==0 :
                freqout +=1
            mylistaux = mylist[::freqout]
            if mylistaux[-1] != mylist[-1]:
                mylistaux=mylistaux + [mylist[-1]]
            mylist = mylistaux

    if sys.argv[1] == 'only':
            firstlast = False
            firstnumber = 2

    if sel :
        if firstlast:
            mylist = [mylist[0],mylist[-1]]
        else:
            mylist = []
        for input in sys.argv[firstnumber:]:
            filename = './m/axion.m.'+ input.zfill(5)
            if pa.gm(filename,'binconB?'):
                mylist.append(filename)

# SIMULATION DATA
sizeN = pa.gm(mylist[0],'N')
sizeL = pa.gm(mylist[0],'L')
nqcd  = pa.gm(mylist[0],'nqcd')

# SIMULATION NAME?
simname = ''
dirpath = os.getcwd()
foldername = os.path.basename(dirpath)
if len(foldername) > 4:
    simname = foldername[4:]
# ID
ups = simname+' : N'+str(sizeN)+' L'+str(sizeL)+' n'+str(nqcd)+' ('+mark+')'
print('ID = '+ups)

# CREATE DIR FOR PICS
if not os.path.exists('./pics'):
    os.makedirs('./pics')


## PROCESS BIN CONT DATA

for meas in mylist:
    lis = pa.conbin(meas, 100)
    plt.loglog(lis[:,0],lis[:,1],label=r'$\tau=%.1f$' %(pa.gm(meas,'time')), linewidth=0.2,marker='.',markersize=0.1)

## FINAL PLOT

#plt.loglog(contab,auxtab,       c='b',linewidth=0.3,marker='.',markersize=0.1)
rc('text', usetex=False)
plt.title(ups)
rc('text', usetex=True)
plt.ylabel(r'$dP/d\delta$')
plt.xlabel(r'$\rho/\langle\rho\rangle$')
plt.legend(loc='lower left',title='- -')
plt.savefig("pics/contrastbin_all.pdf")
print("->pics/contrastbin_all.pdf")
#plt.show()
