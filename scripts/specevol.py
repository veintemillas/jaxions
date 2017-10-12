# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import re, os, sys
import h5py
import datetime
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
fileMeas = sorted([x for x in [y for y in os.listdir("./m/")] if re.search("axion.m.[0-9]{5}$", x)])

mylist = []
sel = True
firstlast = True
firstnumber = 1

for meas in fileMeas:
    f = h5py.File('./m/'+ meas, 'r')
    if 'nSpectrum' in f:
        mylist.append(meas)


# LOOK FOR ARGUMENTS OF THE FUNCTION TO COMPLETE THE SETS PLOTTED
if len(sys.argv) == 1:
    mylist = [mylist[0],mylist[-1]]
else:
    for input in sys.argv[1:]:
        if input == 'all':
            sel = False

    if sys.argv[1] == 'only':
            firstlast = False
            firstnumber = 2

    if sel :
        if firstlast:
            mylist = [mylist[0],mylist[-1]]
        else:
            mylist = []
        for input in sys.argv[firstnumber:]:
            f = h5py.File('./m/axion.m.'+ input.zfill(5), 'r')
            if 'nSpectrum' in f:
                mylist.append('axion.m.' + input.zfill(5))



# SIMULATION DATA FROM FIRST ENTRY
f = h5py.File('./m/'+ mylist[0], 'r')

sizeN = f.attrs[u'Size']
sizeL = f.attrs[u'Physical size']
nqcd = f.attrs[u'nQcd']


# ID
ups = 'N'+str(sizeN)+' L'+str(sizeL)+' n'+str(nqcd)+' ('+mark+')'+str(mac)
print('ID = '+ups)
print()
for item in f.attrs.keys():
    print(item + ":", f.attrs[item])


# CREATE DIR FOR PICS
if not os.path.exists('./pics'):
    os.makedirs('./pics')


# CALCULATE NUMBER OF MODES
n2=int(sizeN/2)
powmax = f['nSpectrum/sK/data/'].size
ktab = (0.5+np.arange(powmax))*2*math.pi/sizeL

def funi(x,a,b):
    return a + b*x

from math import exp, log10, fabs, atan, log, atan2

def volu( rR ):
    if rR <= 1.0:
        return (4*math.pi/3)*rR**3

    elif 1.0 < rR <= math.sqrt(2.):
        return (2*math.pi/3)*(9*rR**2-4*rR**3-3)

    elif math.sqrt(2.) < rR < math.sqrt(3.):
        a2 = rR**2-2
        a = math.sqrt(a2)
        b = 8*a - 4*(3*rR**2 -1)*(atan(a)-atan(1/a))
        return b - (8/3)*(rR**3)*atan2(a*(6*rR + 4*rR**3 -2*rR**5),6*rR**4-2-12*rR**2)

    elif  math.sqrt(3) < rR:
        return 8.
vecvolu=np.vectorize(volu)
foca = np.arange(0,powmax)/n2
foca2 = np.arange(1,powmax+1)/n2
nmodes2 = (n2**3)*(vecvolu(foca2)-vecvolu(foca))


## FINAL PLOT
N3 = sizeN*sizeN*sizeN
plt.clf()

mylist = sorted(mylist)


for meas in mylist:
    #print(meas)
    f = h5py.File('./m/'+ meas, 'r')
    time = f.attrs[u'z']

    # larvaP = np.reshape(f['pSpectrum/sP'],(powmax))
    # av = larvaP/nmodes2
    #
    # plt.loglog(ktab,(ktab**3)*av/(math.pi**2),label=r'$\tau$={%.2f}'%(time))

    larvaK = np.reshape(f['nSpectrum/sK/data/'],(powmax))
    larvaG = np.reshape(f['nSpectrum/sG/data/'],(powmax))
    larvaV = np.reshape(f['nSpectrum/sV/data/'],(powmax))

    #OCUPATION NUMBER is more interesting D.9 of notes
    nK = larvaK/nmodes2
    nG = larvaG/nmodes2
    nV = larvaV/nmodes2


#        plt.loglog(ktab[:lima],koni*ktab[:lima]**indi,c='gray',linewidth=1,label=r'$%.1f\, k^{%.1f}$' % (koni,indi))
    # plt.loglog(ktab[:-2],nK[:-2],c='r',linewidth=0.6,marker='.',markersize=0.1,label='K')
    # plt.loglog(ktab[:-2],nG[:-2],c='b',linewidth=0.6,marker='.',markersize=0.1,label='G')
    # plt.loglog(ktab[:-2],nV[:-2],c='k',linewidth=0.6,marker='.',markersize=0.1,label='V')
    plt.loglog(ktab[:-2],(nK+nV+nG)[:-2],linewidth=0.1,label=r'$\tau$={%.1f}'%(time))


plt.title(ups)
#plt.ylim([0.00000001,100])
plt.ylabel(r'$n_k$')
plt.xlabel(r'comoving {$k [1/R_1 H_1]$}')
plt.legend(loc='lower left',title=r'$\tau$={%.1f}'%(time))
plt.savefig("pics/occnumber_all.pdf")