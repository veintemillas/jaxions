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
    if 'bins/contB' in f:
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
            if 'bins/contB' in f:
                mylist.append('axion.m.' + input.zfill(5))



# SIMULATION DATA FROM FIRST ENTRY
f = h5py.File('./m/'+ mylist[0], 'r')


sizeL = f.attrs[u'Physical size']
sizeN = f.attrs[u'Size']
N3 = sizeN*sizeN*sizeN
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


# CREATE DIR FOR PICS
if not os.path.exists('./pics'):
    os.makedirs('./pics')


## PROCESS BIN CONT DATA

mylist = sorted(mylist)

for meas in mylist:

    f = h5py.File('./m/'+ meas, 'r')
    time = f.attrs[u'z']

    sizeN = f.attrs[u'Size']
    N3 = sizeN*sizeN*sizeN

    numBIN = f['bins/contB'].attrs[u'Size']
    tc = np.reshape(f['bins/contB/data'],(numBIN))

    avdens = f['energy'].attrs[u'Axion Gr X']
    avdens += f['energy'].attrs[u'Axion Gr Y']
    avdens += f['energy'].attrs[u'Axion Gr Z']
    avdens += f['energy'].attrs[u'Axion Kinetic']
    avdens += f['energy'].attrs[u'Axion Potential']

    maxcon = f['bins/contB'].attrs[u'Maximum']
    mincon = f['bins/contB'].attrs[u'Minimum']

    bino = tc*N3*(maxcon-mincon)/numBIN
    numbins = numBIN

    i0 = 0
    while bino[i0] < 1.99:
            i0 = i0 + 1
    bino = bino[i0:]

    sum = 0
    parsum = 0
    nsubbin = 0
    minimum = 10
    lista=[]


    # JOIN BINS ALL ALONG to have a minimum of 10 points
    for bin in range(0,len(bino)):
        # adds bin value to partial bin
        parsum += bino[bin]
        nsubbin += 1
        if nsubbin == 1:
                # records starting bin
                inbin = bin
        if parsum < 10:
            # if parsum if smaller than 10 we will continue
            # adding bins
            sum += 1
        else:
            enbin = bin

            low = 10**((maxcon-mincon)*(i0+inbin)/numbins + mincon)
            med = 10**((maxcon-mincon)*(i0+(inbin+enbin+1)/2)/numbins + mincon)
            sup = 10**((maxcon-mincon)*(i0+enbin+1)/numbins + mincon)
            lista.append([med,parsum/(sup-low)])

            parsum = 0
            nsubbin = 0

    lis = np.array(lista)
    plt.loglog(lis[:,0],lis[:,1]/N3,label=r'$\tau=%.1f$' % time, linewidth=0.2,marker='.',markersize=0.1)



## FINAL PLOT

#plt.loglog(contab,auxtab,       c='b',linewidth=0.3,marker='.',markersize=0.1)
plt.ylabel(r'$dP/d\delta$')
plt.xlabel(r'$\rho/\langle\rho\rangle$')
plt.title(ups)
plt.legend(loc='lower left',title='- -')
plt.savefig("pics/contrastbin_all.pdf")
#plt.show()
