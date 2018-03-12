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
    if pa.gm(meas,'nsp?'):
        mylist.append(meas)
if len(mylist)==0:
    print('No single file with nSpecrum!')
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
            if pa.gm(filename,'nsp?'):
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

# CALCULATE NUMBER OF MODES
nmodes = pa.phasespacedensityBOX(sizeN)
nmax = len(nmodes)
klist  = (0.5+np.arange(nmax))*2*math.pi/sizeL
#klist[0]=0

plt.clf()

for meas in mylist:
    nT = (klist**3)*pa.gm(meas,'nsp')/nmodes
    # nG = pa.gm(meas,'nspG')/nmodes
    # nV = pa.gm(meas,'nspV')/nmodes

    plt.loglog(klist,nT,linewidth=0.1,label=r'$\tau$={%.1f}'%(pa.gm(meas,'time')))

rc('text', usetex=False)
plt.title(ups)
rc('text', usetex=True)
#plt.ylim([0.00000001,100])
plt.ylabel(r'$(k/k_1)^3 n_k$')
plt.xlabel(r'comoving {$k [1/R_1 H_1]$}')
plt.legend(loc='lower left')
plt.savefig("pics/occnumber_all.pdf")
print("->pics/occnumber_all.pdf")
