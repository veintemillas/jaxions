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
    if 'bins/testTh' or 'bins/theta' in f:
        mylist.append(meas)

# LOOK FOR ARGUMENTS OF THE FUNCTION TO COMPLETE THE SETS PLOTTED
if len(sys.argv) == 1:
    if len(mylist) == 1:
        mylist = [ mylist[0] ]
    else:
        mylist = [ mylist[0], mylist[-1] ]
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
            if 'bins/testTh'or 'bins/theta' in f:
                mylist.append('axion.m.' + input.zfill(5))

# SIMULATION DATA FROM FIRST ENTRY
f = h5py.File('./m/'+ mylist[0], 'r')
sizeL = f.attrs[u'Physical size']
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

## FINAL PLOT
N3 = sizeN*sizeN*sizeN
plt.clf()

mylist = sorted(mylist)

for meas in mylist:
    f = h5py.File('./m/'+ meas, 'r')
    time = f.attrs[u'z']

    if 'bins/testTh' in f :
        if 'bins/testTh/data' in f:

            Thmax = f['bins/testTh'].attrs[u'Maximum']
            Thmin = f['bins/testTh'].attrs[u'Minimum']
            siza = f['bins/testTh'].attrs[u'Size']

            dat = np.reshape(f['bins/testTh/data'],(siza))

        else:

            Thmax = f['bins/'].attrs[u'Maximum']
            Thmin = f['bins/'].attrs[u'Minimum']
            siza = f['bins/'].attrs[u'Size']

            dat = np.reshape(f['bins/testTh'],(siza))

        thbin = np.linspace(Thmin,Thmax,siza)
        plt.semilogy(thbin,dat,linewidth=0.1,label=r'$\tau$={%.1f}'%(time))
    elif 'bins/theta' in f :

        Thmax = f['bins/'].attrs[u'Maximum']
        Thmin = f['bins/'].attrs[u'Minimum']
        siza = f['bins/'].attrs[u'Size']

        dat = np.reshape(f['bins/theta/data'],(siza))

        thbin = np.linspace(Thmin,Thmax,siza)
        norma = siza/(N3*(Thmax-Thmin))
        plt.semilogy(thbin,dat*norma,linewidth=0.1,label=r'$\tau$={%.1f}'%(time))


plt.title(ups)
#plt.ylim([0.00000001,100])
plt.ylabel(r'$dP/d\theta$')
plt.xlabel(r'$\theta$')
plt.legend(loc='upper left')
plt.savefig("pics/theta_all.pdf")

# TRANSITION

mylist =[]
plt.clf()

if os.path.exists('./axion.m.10001'):
    mylist.append('axion.m.10000')
if os.path.exists('./axion.m.10001'):
    mylist.append('axion.m.10001')

# for meas in mylist:
#     f = h5py.File('./'+ meas, 'r')
#     time = f.attrs[u'z']
#     Thmax = f['bins/'].attrs[u'Maximum']
#     Thmin = f['bins/'].attrs[u'Minimum']
#     siza = f['bins/'].attrs[u'Size']
#
#     # dat = np.reshape(f['bins/testTh'],(siza))
#     thbin = np.linspace(Thmin,Thmax,siza)
#     # plt.semilogy(thbin,dat,linewidth=0.1,label=r'$\tau$={%.1f}'%(time))
#     norma = siza/(N3*(Thmax-Thmin))
#     if 'bins/testTh' in f :
#         dat = np.reshape(f['bins/testTh'],(siza))
#         plt.semilogy(thbin,dat,linewidth=0.1,label=r'$\tau$={%.1f}'%(time))
#     elif 'bins/theta' in f :
#         dat = np.reshape(f['bins/theta/data'],(siza))
#         plt.semilogy(thbin,dat*norma,linewidth=0.1,label=r'$\tau$={%.1f}'%(time))

for meas in mylist:
    f = h5py.File('./'+ meas, 'r')
    time = f.attrs[u'z']

    if 'bins/testTh' in f :
        if 'bins/testTh/data' in f:

            Thmax = f['bins/testTh'].attrs[u'Maximum']
            Thmin = f['bins/testTh'].attrs[u'Minimum']
            siza = f['bins/testTh'].attrs[u'Size']

            dat = np.reshape(f['bins/testTh/data'],(siza))

        else:

            Thmax = f['bins/'].attrs[u'Maximum']
            Thmin = f['bins/'].attrs[u'Minimum']
            siza = f['bins/'].attrs[u'Size']

            dat = np.reshape(f['bins/testTh'],(siza))

        thbin = np.linspace(Thmin,Thmax,siza)
        plt.semilogy(thbin,dat,linewidth=0.1,label=r'$\tau$={%.1f}'%(time))
    elif 'bins/theta' in f :

        Thmax = f['bins/'].attrs[u'Maximum']
        Thmin = f['bins/'].attrs[u'Minimum']
        siza = f['bins/'].attrs[u'Size']

        dat = np.reshape(f['bins/theta/data'],(siza))

        thbin = np.linspace(Thmin,Thmax,siza)
        norma = siza/(N3*(Thmax-Thmin))
        plt.semilogy(thbin,dat*norma,linewidth=0.1,label=r'$\tau$={%.1f}'%(time))

plt.title(ups)
#plt.ylim([0.00000001,100])
plt.ylabel(r'$dP/d\theta$')
plt.xlabel(r'$\theta$')
plt.legend(loc='upper left')
plt.savefig("pics/theta_trans.pdf")
