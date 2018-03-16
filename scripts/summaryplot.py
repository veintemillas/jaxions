#!/usr/bin/python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import re, os, sys
import h5py
import datetime
from pyaxions import jaxions as pa
import importlib
mark=f"{datetime.datetime.now():%Y-%m-%d}"

# LATEX OUTPUT

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rc('font', family='serif')

# HDF5 DATASETS
#fileMeas = sorted([x for x in [y for y in os.listdir("./m/")] if re.search("axion.m.[0-9]{5}$", x)])
mfiles = pa.findmfiles('./m/')

# SIMULATION DATA
sizeN = pa.gm(mfiles[0],'N')
sizeL = pa.gm(mfiles[0],'L')
nqcd  = pa.gm(mfiles[0],'nqcd')

# SIMULATION NAME?
if sys.argv[-1][-2:] != 'py':
    simname = sys.argv[-1]
else :
    simname = ''
    dirpath = os.getcwd()
    foldername = os.path.basename(dirpath)
    if len(foldername) > 4:
        simname = foldername[4:]

# ID
ups = simname+' : N'+str(sizeN)+' L'+str(sizeL)+' n'+str(nqcd)+' ('+mark+')'
print('ID = '+ups)

print('%s (z=%f,%s) to %s (z=%f,%s)'%(mfiles[0],pa.gm(mfiles[0],'ct'),pa.gm(mfiles[0],'ftype'),mfiles[-1],pa.gm(mfiles[-1],'ct'),pa.gm(mfiles[-1],'ftype')))

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

print('check ... sample.txt')
if os.path.exists('./sample.txt'):
    print('ok')
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

    axiondata = False
    if l10 > 1 :
        arrayS = np.genfromtxt('./sample.txt',skip_footer=l5)
    if l5 > 1 :
        arrayA = np.genfromtxt('./sample.txt',skip_header=l10)
        axiondata = True

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
    print('->pics/point_theta.pdf')
    #plt.show()

    # RHO EVOLUTION
    plt.clf()
    if l10 >1 :
        plt.plot(ztab1,Rhtab1-1-arrayS[:,9],linewidth=0.1,marker='.',markersize=0.1)
        plt.plot(ztab1,Rhtab1-1,linewidth=0.1,marker='.',markersize=0.1)
        plt.xlabel(r'$\rho/v-1$')
        plt.xlabel(r'$\tau$')
        plt.savefig("pics/point_rho.pdf")
        print('->pics/point_rho.pdf')
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
    print('->pics/point_vel.pdf')
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

        co = (sizeL/sizeN)*(3/8)*(1/sizeL)**3
        plt.plot(stringo[1:,0],co*stringo[1:,1]*stringo[1:,0]**2,linewidth=0.5,marker='.',markersize=0.1)
        plt.ylabel("String density [Length/Volume adm U.]")
        plt.xlabel(r'$\tau$')
        plt.title(ups)
        plt.savefig("pics/string.pdf")
        print('->pics/string.pdf')
        #plt.show()

    #plt.show()

    plt.clf()

## MULTI # ANAL

# ENERGY

# ct, K, G, V, T
eAevol = pa.energoA(mfiles)
# ct, K, G, V, T
eSevol = pa.energoS(mfiles)

plt.clf()
if len(eAevol) >0:
    sca = eAevol[:,0]**(1-nqcd/2)
    plt.semilogy(eAevol[:,0],sca*eAevol[:,2]/16.82, c='b',label=r'$G_\theta$',linewidth=1,marker='.',markersize=0.1)
    plt.semilogy(eAevol[:,0],sca*eAevol[:,3]/16.82, c='magenta',label=r'$V_\theta$',linewidth=1,marker='.',markersize=0.1)
    plt.semilogy(eAevol[:,0],sca*eAevol[:,1]/16.82, c='r',label=r'$K_\theta$',linewidth=1,marker='.',markersize=0.1)
    plt.semilogy(eAevol[:,0],sca*eAevol[:,4]/16.82, c='k',label=r'$\theta$',linewidth=1,marker='.',markersize=0.1)
if len(eSevol) >0:
    sca = eSevol[:,0]**(1-nqcd/2)
    plt.semilogy(eSevol[:,0],sca*eSevol[:,2]/16.82,'--',c='b',label=r'$G_\rho$',linewidth=0.5,marker='.',markersize=0.1)
    plt.semilogy(eSevol[:,0],sca*eSevol[:,3]/16.82,'--',c='magenta',label=r'$V_\rho$',linewidth=0.5,marker='.',markersize=0.1)
    plt.semilogy(eSevol[:,0],sca*eSevol[:,1]/16.82,'--',c='r',label=r'$K_\rho$',linewidth=0.5,marker='.',markersize=0.1)
    plt.semilogy(eSevol[:,0],sca*eSevol[:,4]/16.82,'--',c='k',label=r'$\rho$',linewidth=0.5,marker='.',markersize=0.1)
plt.grid(axis='y')
plt.title(ups)
plt.ylabel('Energy[misalignment U.]')
plt.xlabel(r'$\tau$')
plt.legend(loc='lower left')
plt.savefig("pics/energy2.pdf")
print('->pics/energy2.pdf')
plt.ylim([0.01,10000])
plt.savefig("pics/energy22.pdf")
print('->pics/energy22.pdf')

indexi = 2
indexf = (nqcd/2-1)
indexp = indexi+(nqcd/2-1)
plt.clf()
if len(eAevol) >0:
    sca = (eAevol[:,0]**2)/(2.2**indexp + eAevol[:,0]**indexp)
    plt.semilogy(eAevol[:,0],sca*eAevol[:,2]/16.82, c='b',label=r'$G_\theta$',linewidth=1,marker='.',markersize=0.1)
    plt.semilogy(eAevol[:,0],sca*eAevol[:,3]/16.82, c='magenta',label=r'$V_\theta$',linewidth=1,marker='.',markersize=0.1)
    plt.semilogy(eAevol[:,0],sca*eAevol[:,1]/16.82, c='r',label=r'$K_\theta$',linewidth=1,marker='.',markersize=0.1)
    plt.semilogy(eAevol[:,0],sca*eAevol[:,4]/16.82, c='k',label=r'$\theta$',linewidth=1,marker='.',markersize=0.1)
if len(eSevol) >0:
    sca = (eSevol[:,0]**2)/(2.2**indexp + eSevol[:,0]**indexp)
    plt.semilogy(eSevol[:,0],sca*eSevol[:,2]/16.82,'--',c='b',label=r'$G_\rho$',linewidth=0.5,marker='.',markersize=0.1)
    plt.semilogy(eSevol[:,0],sca*eSevol[:,3]/16.82,'--',c='magenta',label=r'$V_\rho$',linewidth=0.5,marker='.',markersize=0.1)
    plt.semilogy(eSevol[:,0],sca*eSevol[:,1]/16.82,'--',c='r',label=r'$K_\rho$',linewidth=0.5,marker='.',markersize=0.1)
    plt.semilogy(eSevol[:,0],sca*eSevol[:,4]/16.82,'--',c='k',label=r'$\rho$',linewidth=0.5,marker='.',markersize=0.1)
plt.grid(axis='y')
plt.title(ups)
plt.ylabel('Energy[scaling to misalignment U.]')
plt.xlabel(r'$\tau$')
plt.legend(loc='lower left')
plt.ylim([0.001,10])
plt.savefig("pics/energy22loglog.pdf")
print('->pics/energy22loglog.pdf')
#plt.show()

plt.clf()

# STRING EVOLUTION

stringDensevol = pa.stringo(mfiles)

if len(stringDensevol) > 0:

    plt.plot(stringDensevol[:,0],stringDensevol[:,1],label=r'length/vol',linewidth=0.5,marker='.',markersize=0.1)
    plt.ylabel("String density [Length/Volume adm U.]")
    plt.xlabel(r'$\tau$')
    plt.title(ups)
    plt.legend(loc='lower left',title=r'$\tau_{\rm end}$={%.1f}'%(stringo[-1,0]))
    plt.savefig("pics/string2.pdf")
    print('->pics/string2.pdf')
    plt.clf()

## FINAL ANAL
N3 = sizeN*sizeN*sizeN
print()

analfilename=mfiles[-1]

if pa.gm(analfilename,'ftype') =='Saxion':
    print('Final %s is Saxion, cancel contrast and spectrum analisys.'%(analfilename))

an_cont = False
an_nspec = False
an_pspec = False


if pa.gm(analfilename,'ftype') =='Axion':
    for meas in reversed(mfiles):
        an_cont  = pa.gm(meas,'bincon?')
        an_nspec = pa.gm(meas,'nsp?')
        an_pspec = pa.gm(meas,'psp?')

        if an_cont and an_nspec and an_pspec:
            analfilename = meas
            print('Final file analysis for '+ analfilename)
            time = pa.gm(analfilename, 'time')
            break

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
    print('->pics/contrastbin.pdf')
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
    print('->pics/numberspec.pdf')
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
    print('->pics/powerspectrumP.pdf')
    #plt.show()
