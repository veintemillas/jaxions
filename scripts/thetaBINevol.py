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

zt = []
for meas in fileMeas:
    if pa.gm(meas,'bintheta?'):
        mylist.append(meas)
        zt.append(pa.gm(meas,'time'))
if len(mylist)==0:
    print('No single file with contheta!')
    sys.exit()

ordi = np.argsort(np.array(zt))
omylist = [mylist[i] for i in ordi]
mylist = omylist

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
        if input == 'every10':
            sel = False
            freqout = len(mylist)//10
            if freqout ==0 :
                freqout +=1
            mylistaux = mylist[::freqout]
            if mylistaux[-1] != mylist[-1] :
                mylistaux=mylistaux + [mylist[-1]]
            mylist = mylistaux

    if sys.argv[1] == 'only':
            firstlast = False
            firstnumber = 2

    # todo thetaB -> pa.gm(binthetaB)
    if sel :
        if firstlast:
            mylist = [mylist[0],mylist[-1]]
        else:
            mylist = []
        for input in sys.argv[firstnumber:]:
            f = h5py.File('./m/axion.m.'+ input.zfill(5), 'r')
            if 'bins/thetaB'  in f:
                mylist.append('axion.m.' + input.zfill(5))


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

## FINAL PLOT

plt.clf()

for meas in mylist:
    # f = h5py.File('./m/'+ meas, 'r')
    # time = f.attrs[u'z']
    #
    # if 'bins/thetaB/data' in f:
    #
    #     Thmax = f['bins/thetaB'].attrs[u'Maximum']
    #     Thmin = f['bins/thetaB'].attrs[u'Minimum']
    #     siza = f['bins/thetaB'].attrs[u'Size']
    #
    #     dat = np.reshape(f['bins/thetaB/data'],(siza))
    #
    #     thbin = np.linspace(Thmin,Thmax,siza)
    #     plt.semilogy(thbin,dat,linewidth=0.1,label=r'$\tau$={%.1f}'%(time))
    bas = pa.thetabin(meas,10)
    # Thmax = pa.gm(meas,'binthetaBmax')
    # Thmin = pa.gm(meas,'binthetaBmin')
    # dat = pa.gm(meas,'binthetaB')
    # siza = len(dat)
    # thbin = np.linspace(Thmin,Thmax,siza)
    plt.semilogy(bas[:,0],bas[:,1],linewidth=0.1,label=r'$\tau$={%.1f}'%(pa.gm(meas,'ct')))

rc('text', usetex=False)
plt.title(ups)
rc('text', usetex=True)
#plt.ylim([0.00000001,100])
plt.ylabel(r'$dP/d\theta$')
plt.xlabel(r'$\theta$')
plt.legend(loc='upper left')
plt.savefig("pics/theta_all.pdf")
print('->pics/theta_all.pdf')

# TRANSITION

plt.clf()

if (pa.gm(omylist[0],'ftype')=='Saxion') and (pa.gm(omylist[-1],'ftype')=='Axion'):
    lastSax = omylist[0]
    for meas in fileMeas:
        if pa.gm(meas,'ftype') =='Saxion':
            lastSax = meas
        else:
            firstAx = meas
            break
    print("Saxion (%s)to Axion (%s) transition (%s->%s)"%(fileMeas[0],fileMeas[-1],lastSax,firstAx))

    for meas in [lastSax,firstAx]:
        bas = pa.thetabin(meas,10)
        plt.semilogy(bas[:,0],bas[:,1],linewidth=0.1,label=r'$\tau$={%.1f}'%(pa.gm(meas,'ct')))
        # Thmax = pa.gm(meas,'binthetaBmax')
        # Thmin = pa.gm(meas,'binthetaBmin')
        # dat = pa.gm(meas,'binthetaB')
        # siza = len(dat)
        # thbin = np.linspace(Thmin,Thmax,siza)
        # plt.semilogy(thbin,dat,linewidth=0.1,label=r'$\tau$={%.1f}'%(pa.gm(meas,'ct')))
    rc('text', usetex=False)
    plt.title(ups)
    rc('text', usetex=True)
    #plt.ylim([0.00000001,100])
    plt.ylabel(r'$dP/d\theta$')
    plt.xlabel(r'$\theta$')
    plt.legend(loc='upper left')
    plt.savefig("pics/theta_trans.pdf")
    print('->pics/theta_trans.pdf')


# if len(mylist)==0:
#     print('No 10000 files')
# else :
#
#
# mylist =[]
# plt.clf()
#
# if os.path.exists('./axion.m.10000'):
#     mylist.append('axion.m.10000')
# if os.path.exists('./axion.m.10001'):
#     mylist.append('axion.m.10001')
#
# if len(mylist)==0:
#     print('No 10000 files')
# else :
#     for meas in mylist:
#         Thmax = pa.gm(meas,'binthetaBmax')
#         Thmin = pa.gm(meas,'binthetaBmin')
#         dat = pa.gm(meas,'binthetaB')
#         siza = len(dat)
#         thbin = np.linspace(Thmin,Thmax,siza)
#         plt.semilogy(thbin,dat,linewidth=0.1,label=r'$\tau$={%.1f}'%(pa.gm(meas,'ct')))
#     rc('text', usetex=False)
#     plt.title(ups)
#     rc('text', usetex=True)
#     #plt.ylim([0.00000001,100])
#     plt.ylabel(r'$dP/d\theta$')
#     plt.xlabel(r'$\theta$')
#     plt.legend(loc='upper left')
#     plt.savefig("pics/theta_trans.pdf")
#     print('->pics/theta_trans.pdf')
