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




## FINAL ANAL
N3 = sizeN*sizeN*sizeN
an_firstspec = True
plt.clf()

for meas in fileMeas:

    f = h5py.File('./m/'+ meas, 'r')
    # for item in f.attrs.keys():
    #     print(item + ":", f.attrs[item])
    # print()

    # print()

    time = f.attrs[u'z']
    an_cont = False
    an_nspec = False
    an_pspec = False

    for item in list(f):
        if item == 'bins':
            #print('contrast bin posible')
            an_cont = True
        if item == 'nSpectrum':
            #print('nSpec bin posible')
            an_nspec = True
        if item == 'pSpectrum':
            #print('pSpec bin posible')
            an_pspec = True

    # NUMBER SPECTRUM
    if an_pspec:
        print('analysiing '+  meas )

        if an_firstspec:
            from scipy.optimize import curve_fit
            n2=int(sizeN/2)
            powmax = f['pSpectrum/sP'].size
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

        larvaP = np.reshape(f['pSpectrum/sP'],(powmax))
        av = larvaP/nmodes2

        plt.loglog(ktab,(ktab**3)*av/(math.pi**2),label=r'$\tau$={%.2f}'%(time))


plt.loglog(ktab,np.ones(powmax))
plt.title(ups)
#plt.ylim([0.00000001,100])
plt.ylabel(r'$\Delta^2_k$')
plt.xlabel(r'comoving {$k [1/R_1 H_1]$}')
plt.legend(loc='lower left',title=r'$\tau$={%.2f}'%(time))
plt.savefig("pics/powerspectrum_all.pdf")
#plt.show()
