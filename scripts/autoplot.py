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
        plt.plot(ztab1,Thtab1)
    if axiondata:
        plt.plot(ztab2,Thtab2)
    plt.ylim([-3.15,3.15])
    plt.ylabel(r'$\theta$')
    plt.xlabel(r'$\tau$')
    plt.title(ups)
    plt.savefig("pics/point_theta.pdf")
    #plt.show()

    # RHO EVOLUTION
    plt.clf()
    if l10 >1 :
        plt.plot(ztab1,Rhtab1-1-arrayS[:,9],linewidth=0.7)
        plt.plot(ztab1,Rhtab1-1,linewidth=0.1)
        plt.xlabel(r'$\rho/v-1$')
        plt.xlabel(r'$\tau$')
        plt.savefig("pics/point_rho.pdf")
        #plt.show()

    # THETA VELOCITY EVOLUTION
    plt.clf()
    if l10 >1 :
        plt.plot(ztab1,VThtab1,linewidth=0.2,marker='.',markersize=0.1)
    if axiondata:
        plt.plot(ztab2,VThtab2,linewidth=0.2)
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

        co = (sizeL/sizeN)*(3/2)*(1/sizeL)**3
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
    plt.ylim([0.01,1000])
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
co = (sizeL/sizeN)*(3/2)*(1/sizeL)**3

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
    plt.ylim([0.01,1000])
    plt.grid(axis='y')
    plt.title(ups)
    plt.ylabel('Energy[misalignment U.]')
    plt.xlabel(r'$\tau$')
    plt.legend(loc='lower left')
    plt.savefig("pics/energy2.pdf")
    #plt.show()

    plt.clf()

    # STRING EVOLUTION
if len(stringdata) > 0:
    stringo = np.array(stringdata)
    co = (sizeL/sizeN)*(3/2)*(1/sizeL)**3

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
print('final file analysis for '+ fileMeas[-1])
print()
f = h5py.File('./m/'+fileMeas[-1], 'r')
for item in f.attrs.keys():
    print(item + ":", f.attrs[item])

print('contains ' + str(list(f)))

time = f.attrs[u'z']
an_cont = False
an_nspec = False
an_pspec = False

for item in list(f):
    if item == 'bins':
        print('contrast bin posible')
        an_cont = True
    if item == 'nSpectrum':
        print('nSpec bin posible')
        an_nspec = True
    if item == 'pSpectrum':
        print('pSpec bin posible')
        an_pspec = True

if an_cont:
    tc = np.reshape(f['bins/cont'],(10000))
    avdens, maxcon, logmaxcon = tc[0:3]
    bino = tc[3:]
    numbins=10000-3
    #binsizecons = numbins/(N3*math.log(10)*(logmaxcon+5))
    #print(avdens, maxcon, logmaxcon , numbins, binsizecons)
    #contab=10**((logmaxcon+5)*(np.arange(numbins)+0.5)/numbins - 5)
    #auxtab=binsizecons*bino/contab

    # PROCESS IT
    # BINS SELECTED BY  bin = int[(5+log10[d])*numbins/(logmaxcon+5)]
    # DISCARD SMALL BINS AT BEGGINNING
    # JOIN BINS AT THE END
    i0 = 0
    while bino[i0] < 1.99:
            i0 = i0 + 1
    #print(i0)

    bino = bino[i0:]
    #contab = contab[i0:]

    sum = 0
    parsum = 0
    nsubbin = 0
    minimum = 10
    lista=[]
    for bin in range(0,len(bino)):
        parsum += bino[bin]
        nsubbin += 1
        if nsubbin ==1:
                inbin = bin
        if parsum < 10:
            sum += 1
        else:
            enbin = bin
            # rebin and reweight
            # bin corresponds to i0 + Dbin to contrast 10**((logmaxcon+5)*(<bin,bin+1>)/numbins - 5)
            # so one can just compute initial and final and divide
            ## binsizecons = numbins/(N3*math.log(10)*(logmaxcon+5))
            ## contab=10**((logmaxcon+5)*(np.arange(numbins)+0.5)/numbins - 5)
            ## auxtab=binsizecons*bino/contab
            low = 10**((logmaxcon+5)*(i0+inbin)/numbins - 5)
            med = 10**((logmaxcon+5)*(i0+(inbin+enbin+1)/2)/numbins - 5)
            sup = 10**((logmaxcon+5)*(i0+enbin+1)/numbins - 5)
            lista.append([med,parsum/(sup-low)])

            parsum = 0
            nsubbin = 0

    lis = np.array(lista)

    plt.clf()
    #plt.loglog(contab,auxtab,       c='b',linewidth=0.3,marker='.',markersize=0.1)
    plt.loglog(lis[:,0],lis[:,1]/N3,c='green',label=r'$\tau=%.1f$' % time, linewidth=0.6,marker='.',markersize=0.1)
    plt.ylabel(r'$dP/d\delta$')
    plt.xlabel(r'$\delta$')
    plt.title(ups)
    plt.legend(loc='lower left',title='- -')
    plt.savefig("pics/contrastbin.pdf")
    #plt.show()


# NUMBER SPECTRUM
if an_nspec:
    from scipy.optimize import curve_fit
    n2=int(sizeN/2)

    f = h5py.File('./m/'+fileMeas[-1], 'r')
    time = f.attrs[u'z']
    powmax = f['nSpectrum/sK'].size
    ktab = (0.5+np.arange(powmax))*2*math.pi/sizeL

    larvaK = np.reshape(f['nSpectrum/sK'],(powmax))
    larvaG = np.reshape(f['nSpectrum/sG'],(powmax))
    larvaV = np.reshape(f['nSpectrum/sV'],(powmax))

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

    #OCUPATION NUMBER is more interesting D.9 of notes
    nK = larvaK/nmodes2
    nG = larvaG/nmodes2
    nV = larvaV/nmodes2

    lima = int(sizeN/2)
    popt, pcov = curve_fit(funi, np.log10(ktab[:lima]),np.log10(nK+nV+nG)[:lima])
    popt[0]
    koni = 10**popt[0]
    indi = popt[1]
    #print(koni, indi)
    '%.1f k^%.1f'%(koni,indi)

    plt.clf()
    plt.loglog(ktab[:lima],koni*ktab[:lima]**indi,c='gray',linewidth=1,label=r'$%.1f\, k^{%.1f}$' % (koni,indi))
    plt.loglog(ktab[:-2],nK[:-2],c='r',linewidth=0.6,marker='.',markersize=0.1,label='K')
    plt.loglog(ktab[:-2],nG[:-2],c='b',linewidth=0.6,marker='.',markersize=0.1,label='G')
    plt.loglog(ktab[:-2],nV[:-2],c='k',linewidth=0.6,marker='.',markersize=0.1,label='V')
    plt.loglog(ktab[:-2],(nK+nV+nG)[:-2],c='k',linewidth=1,label='K+V+G')
    plt.title(ups)
    #plt.ylim([0.00000001,100])
    plt.ylabel(r'$n_k$')
    plt.xlabel(r'comoving {$k [1/R_1 H_1]$}')
    plt.legend(loc='lower left',title=r'$\tau$={%.1f}'%(time))
    plt.savefig("pics/occnumber.pdf")
    #plt.show()

# POWER SPECTRUM
if an_pspec:
    powmax2 = f['pSpectrum/sP'].size
    ktab2 = (0.5+np.arange(powmax2))*2*math.pi/sizeL
    f['pSpectrum/sP'] , powmax2 , (sizeN/2)*math.sqrt(3)
    larvaP = np.reshape(f['pSpectrum/sP'],(powmax2))
    if powmax2 == powmax:
        av = larvaP/nmodes2
    else :
        pfoca = np.arange(0,powmax2)/n2
        pfoca2 = np.arange(1,powmax2+1)/n2
        pnmodes2 = (n2**3)*(vecvolu(pfoca2)-vecvolu(pfoca))
        av = larvaP/pnmodes2

    plt.clf()
    plt.loglog(ktab2,(ktab2**3)*av/(math.pi**2),label=r'$\tau$={%.1f}'%(time))
    plt.loglog(ktab2,np.ones(powmax2))
    plt.title(ups)
    #plt.ylim([0.00000001,100])
    plt.ylabel(r'$\Delta^2_k$')
    plt.xlabel(r'comoving {$k [1/R_1 H_1]$}')
    plt.legend(loc='lower left',title=r'$\tau$={%.1f}'%(time))
    plt.savefig("pics/powerspectrum.pdf")
    #plt.show()
