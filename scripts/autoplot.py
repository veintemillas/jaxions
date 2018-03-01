#!/usr/bin/python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import re, os
import h5py
import datetime
from pyaxions import jaxions as pa
import importlib
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
for item in f['/ic/'].attrs.keys():
    print(item + ":", f['/ic/'].attrs[item])
for item in f['/potential/'].attrs.keys():
    print(item + ":", f['/potential/'].attrs[item])

# SAMPLE POINT EVOLUTION (+THETA, NSTRINGS...)
# FORMAT :
# z, m_axion, Lambda, m1, m2, v1, v2, #strings, maxtheta, shift
# z, m_axion, theta, vtheta, maxtheta

# CREATE DIR FOR PICS
if not os.path.exists('./pics'):
    os.makedirs('./pics')

# MOVE TRANSITION FILES
if os.path.exists('./m/axion.m.10000'):
    os.rename('./m/axion.m.10000','./axion.m.10000')
if os.path.exists('./m/axion.m.10001'):
    os.rename('./m/axion.m.10001','./axion.m.10001')


if os.path.exists('./sample.txt'):

    with open('./sample.txt') as f:
        lines=f.readlines()
        l10 = 0
        l5 = 0
        for line in lines:
            myarray = np.fromstring(line, dtype=float, sep=' ')
            l = len(myarray)
            if l==10:
                l10 = l10 +1
            elif l==5:
                l5 = l5 +1

        arrayA = np.genfromtxt('./sample.txt',skip_header=l10)
        arrayS = np.genfromtxt('./sample.txt',skip_footer=l5)

    axiondata = len(arrayA) >0

    if l10 >1 :
        ztab1 = arrayS[:,0]
        Thtab1 = np.arctan2(arrayS[:,4],arrayS[:,3])
        Rhtab1 = np.sqrt(arrayS[:,3]**2 + arrayS[:,4]**2)/ztab1
        VThtab1 = Thtab1 + (arrayS[:,3]*arrayS[:,6]-arrayS[:,4]*arrayS[:,5])/(ztab1*Rhtab1**2)
    if axiondata:
        ztab2 = arrayA[:,0]
        Thtab2 = arrayA[:,2]/ztab2
        VThtab2 = arrayA[:,3]

    # THETA EVOLUTION
    plt.clf()
    if l10 >1 :
        plt.plot(ztab1,Thtab1,linewidth=0.1,marker='.',markersize=0.1)
    if axiondata:
        plt.plot(ztab2,Thtab2,linewidth=0.1,marker='.',markersize=0.1)
    plt.ylim([-3.15,3.15])
    plt.ylabel(r'$\theta$')
    plt.xlabel(r'$\tau$')
    plt.title(ups)
    plt.savefig("pics/point_theta.pdf")
    #plt.show()

    # RHO EVOLUTION
    plt.clf()
    if l10 >1 :
        plt.plot(ztab1,Rhtab1-1-arrayS[:,9],linewidth=0.1,marker='.',markersize=0.1)
        plt.plot(ztab1,Rhtab1-1,linewidth=0.1,marker='.',markersize=0.1)
        plt.xlabel(r'$\rho/v-1$')
        plt.xlabel(r'$\tau$')
        plt.savefig("pics/point_rho.pdf")
        #plt.show()

    # THETA VELOCITY EVOLUTION
    plt.clf()
    if l10 >1 :
        plt.plot(ztab1,VThtab1,linewidth=0.1,marker='.',markersize=0.1)
    if axiondata:
        plt.plot(ztab2,VThtab2,linewidth=0.1,marker='.',markersize=0.1)
    plt.ylabel(r'$\rho/v-1$')
    plt.xlabel(r'$\tau$')
    plt.title(ups)
    plt.savefig("pics/point_vel.pdf")
    #plt.show()

    # STRING EVOLUTION
    plt.clf()
    if l10 >1 :
        strings = arrayS[:,7]
        fix = [[ztab1[0],strings[0]]]
        i = 0
        for i in range(0, len(ztab1)-1):
            if strings[i] != strings[i+1]:
                fix.append([ztab1[i+1],strings[i+1]])
        stringo = np.asarray(fix)

        co = (sizeL/sizeN)*(3/4)*(1/sizeL)**3
        plt.plot(stringo[1:,0],co*stringo[1:,1]*stringo[1:,0]**2,linewidth=0.5,marker='.',markersize=0.1)
        plt.ylabel("String density [Length/Volume adm U.]")
        plt.xlabel(r'$\tau$')
        plt.title(ups)
        plt.savefig("pics/string.pdf")
        #plt.show()

# ENERGY EVOLUTION
# FORMAT :
# ENERGY  gtx gty gtz Vt Kt  gRx gRy gRz VR KR ...
# ENERGY    1   2   3  4  5    6   7   8  9 10

if os.path.exists('./energy.txt'):
    en = np.genfromtxt('./energy.txt')

    p = 0
    while en[p,0] < ztab1[-1]:
        p = p + 1
    ik = p

    # RESCALING
    sca = en[:,0]**(1-nqcd/2)

    plt.clf()
    plt.semilogy(en[:,0],sca*(en[:,1:4].sum(axis=1))/16.82,   c='b',label=r'$G_\theta$',linewidth=1,marker='.',markersize=0.1)
    plt.semilogy(en[:,0],sca*(en[:,4])/16.82,                 c='k',label=r'$V_\theta$',linewidth=1,marker='.',markersize=0.1)
    plt.semilogy(en[:,0],sca*(en[:,5])/16.82,                 c='r',label=r'$K_\theta$',linewidth=1,marker='.',markersize=0.1)
    plt.semilogy(en[:,0],sca*(en[:,1:6].sum(axis=1))/16.82,   c='k',label=r'$\theta$',linewidth=1,marker='.',markersize=0.1)
    plt.semilogy(en[:ik,0],sca[:ik]*(en[:ik,6:9].sum(axis=1))/16.82,   '--',c='b',label=r'$G_\rho$',linewidth=0.5,marker='.',markersize=0.1)
    plt.semilogy(en[:ik,0],sca[:ik]*(en[:ik,9])/16.82,                 '--',c='k',label=r'$V_\rho$',linewidth=0.5,marker='.',markersize=0.1)
    plt.semilogy(en[:ik,0],sca[:ik]*(en[:ik,10])/16.82,                '--',c='r',label=r'$K_\rho$',linewidth=0.5,marker='.',markersize=0.1)
    plt.semilogy(en[:ik,0],sca[:ik]*(en[:ik,6:11].sum(axis=1))/16.82,  '--',c='k',label=r'$\rho$',linewidth=0.5,marker='.',markersize=0.1)
    plt.ylim([0.01,10000])
    plt.grid(axis='y')
    plt.title(ups)
    plt.ylabel('Energy[misalignment U.]')
    plt.xlabel(r'$\tau$')
    plt.legend(loc='lower left')
    plt.savefig("pics/energy.pdf")
    #plt.show()

    plt.clf()

## MULTI # ANAL
fileMeas = sorted([x for x in [y for y in os.listdir("./m/")] if re.search("axion.m.[0-9]{5}$", x)])

ene = []
enlen = 0
sl = 0
stringdata = []
co = (sizeL/sizeN)*(3/4)*(1/sizeL)**3

for meas in fileMeas:
    an_energy = False
    an_string = False

    fileHdf5 = h5py.File("./m/"+meas, "r")

    for item in list(fileHdf5):
        if item == 'energy':
            #an_energy = True
            an_energy = 'Axion Gr X' in fileHdf5['energy'].attrs
        if item == 'string':
            an_string = True
    #print(fileHdf5, list(fileHdf5), an_energy)
    if an_energy:
        enlen = enlen + 1
        zz  = fileHdf5.attrs[u'z']
        agx = fileHdf5['energy'].attrs[u'Axion Gr X']
        agy = fileHdf5['energy'].attrs[u'Axion Gr Y']
        agz = fileHdf5['energy'].attrs[u'Axion Gr Z']
        ak  = fileHdf5['energy'].attrs[u'Axion Kinetic']
        av  = fileHdf5['energy'].attrs[u'Axion Potential']
        #print(fileHdf5.attrs['Field type'])
        if fileHdf5.attrs[u'Field type'] == b'Saxion':
            sl = sl + 1
            sgx = fileHdf5['energy'].attrs[u'Saxion Gr X']
            sgy = fileHdf5['energy'].attrs[u'Saxion Gr Y']
            sgz = fileHdf5['energy'].attrs[u'Saxion Gr Z']
            sk  = fileHdf5['energy'].attrs[u'Saxion Kinetic']
            sv  = fileHdf5['energy'].attrs[u'Saxion Potential']
            ene.append([zz, agx,agy,agz,av,ak,sgx,sgy,sgz,sv,sk])
        elif fileHdf5.attrs[u'Field type'] == b'Axion':
            ene.append([zz, agx,agy,agz,av,ak,0,0,0,0,0])
    if an_string:
        zz  = fileHdf5.attrs[u'z']
        st = fileHdf5['string'].attrs[u'String number']
        wl = fileHdf5['string'].attrs[u'Wall number']
        stringdata.append([zz, st, wl])

print('en length', enlen, '| saxion length ', sl, 'strings length ', len(stringdata))


# RESCALING
if enlen > 0:
    en = np.array(ene)
    del ene
    sca = en[:,0]**(1-nqcd/2)
    plt.clf()
    if enlen >0:
        plt.semilogy(en[:,0],sca*(en[:,1:4].sum(axis=1))/16.82,   c='b',label=r'$G_\theta$',linewidth=1,marker='.',markersize=0.1)
        plt.semilogy(en[:,0],sca*(en[:,4])/16.82,                 c='k',label=r'$V_\theta$',linewidth=1,marker='.',markersize=0.1)
        plt.semilogy(en[:,0],sca*(en[:,5])/16.82,                 c='r',label=r'$K_\theta$',linewidth=1,marker='.',markersize=0.1)
        plt.semilogy(en[:,0],sca*(en[:,1:6].sum(axis=1))/16.82,   c='k',label=r'$\theta$',linewidth=1,marker='.',markersize=0.1)
    if sl > 0:
        plt.semilogy(en[:sl,0],sca[:sl]*(en[:sl,6:9].sum(axis=1))/16.82,   '--',c='b',label=r'$G_\rho$',linewidth=0.5,marker='.',markersize=0.1)
        plt.semilogy(en[:sl,0],sca[:sl]*(en[:sl,9])/16.82,                 '--',c='k',label=r'$V_\rho$',linewidth=0.5,marker='.',markersize=0.1)
        plt.semilogy(en[:sl,0],sca[:sl]*(en[:sl,10])/16.82,                '--',c='r',label=r'$K_\rho$',linewidth=0.5,marker='.',markersize=0.1)
        plt.semilogy(en[:sl,0],sca[:sl]*(en[:sl,6:11].sum(axis=1))/16.82,  '--',c='k',label=r'$\rho$',linewidth=0.5,marker='.',markersize=0.1)
    plt.grid(axis='y')
    plt.title(ups)
    plt.ylabel('Energy[misalignment U.]')
    plt.xlabel(r'$\tau$')
    plt.legend(loc='lower left')
    plt.savefig("pics/energy2.pdf")
    plt.ylim([0.01,10000])
    plt.savefig("pics/energy22.pdf")
    #plt.show()

    plt.clf()

    # STRING EVOLUTION
if len(stringdata) > 0:
    stringo = np.array(stringdata)
    co = (sizeL/sizeN)*(3/4)*(1/sizeL)**3

    plt.plot(stringo[1:,0],co*stringo[1:,1]*stringo[1:,0]**2,label=r'length/vol',linewidth=0.5,marker='.',markersize=0.1)
    plt.ylabel("String density [Length/Volume adm U.]")
    plt.xlabel(r'$\tau$')
    plt.title(ups)
    plt.legend(loc='lower left',title=r'$\tau_{\rm end}$={%.1f}'%(stringo[-1,0]))
    plt.savefig("pics/string2.pdf")
    plt.clf()

## FINAL ANAL
N3 = sizeN*sizeN*sizeN
print()
analfilename='./m/'+fileMeas[-1]
print('final file analysis for '+ analfilename)
print()
f = h5py.File(analfilename, 'r')
for item in f.attrs.keys():
    print(item + ":", f.attrs[item])

print('contains ' + str(list(f)))

time = f.attrs[u'z']
an_cont = False
an_nspec = False
an_pspec = False


an_cont  = 'bins/contB' in f
an_nspec = 'nSpectrum' in f
an_pspec = 'pSpectrum' in f
if an_cont:
    print('nSpec bin posible')
if an_pspec:
    print('pSpec bin posible')
if an_cont:
    print('contrast bin posible')

if an_cont:
    lis = pa.conbin(analfilename,10)

    plt.clf()
    #plt.loglog(contab,auxtab,       c='b',linewidth=0.3,marker='.',markersize=0.1)
    plt.loglog(lis[:,0],lis[:,1],c='green',label=r'$\tau=%.1f$' % time, linewidth=0.6,marker='.',markersize=0.1)
    plt.ylabel(r'$dP/d\delta$')
    plt.xlabel(r'$\delta$')
    plt.title(ups)
    plt.legend(loc='lower left',title='- -')
    plt.savefig("pics/contrastbin.pdf")
    #plt.show()


# NUMBER SPECTRUM
if an_nspec:
    sizeN = pa.gm(analfilename,'Size')
    nmodes = pa.phasespacedensityBOX(sizeN)
    sizeL = pa.gm(analfilename,'L')
    kmax = len(nmodes)
    klist  = (0.5+np.arange(kmax))*2*math.pi/sizeL
    klist[0]=0


    ctime= pa.gm(analfilename,'ct')

    #Spectrum summed over modes
    occnumber = pa.gm(analfilename,'nsp')
    occnumberk = pa.gm(analfilename,'nspK')
    occnumberg = pa.gm(analfilename,'nspG')
    occnumberv = pa.gm(analfilename,'nspV')

    #plots (k/k1)^3 n_k where n_k is the occupation number
    plt.clf()
    plt.loglog(klist,(klist**3)*occnumberk/nmodes,c='r',linewidth=0.6,marker='.',markersize=0.1,label='K')
    plt.loglog(klist,(klist**3)*occnumberg/nmodes,c='b',linewidth=0.6,marker='.',markersize=0.1,label='G')
    plt.loglog(klist,(klist**3)*occnumberv/nmodes,c='k',linewidth=0.6,marker='.',markersize=0.1,label='V')
    plt.loglog(klist,(klist**3)*occnumber/nmodes,c='k',linewidth=1,label='K+V+G')
    plt.title(ups)
    #plt.ylim([0.00000001,100])
    plt.ylabel(r'$(k/k_1)^3n_k$')
    plt.xlabel(r'comoving {$k [1/R_1 H_1]$}')
    plt.legend(loc='lower left',title=r'$\tau$={%.1f}'%(time))
    plt.savefig("pics/numberspec.pdf")
    #plt.show()

# POWER SPECTRUM
if an_pspec:

    avdens = pa.gm(analfilename,'eA')
    powerSpec = pa.gm(analfilename,'psp')
    # dimensionless variance
    powerSpec = (klist**3)*powerSpec/nmodes

    plt.clf()
    plt.loglog(klist,powerSpec/((avdens**2)*(math.pi**2)),label=r'$\tau$={%.1f}'%(time))
    plt.loglog(klist,np.ones(len(klist)))
    plt.title(ups)
    #plt.ylim([0.00000001,100])
    plt.ylabel(r'$\Delta^2_k$')
    plt.xlabel(r'comoving {$k [1/R_1 H_1]$}')
    plt.legend(loc='lower left',title=r'$\tau$={%.1f}'%(time))
    plt.savefig("pics/powerspectrumP.pdf")
    #plt.show()
