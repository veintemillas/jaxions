#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import re, os
import h5py
import datetime
import glob
from sympy import integer_nthroot
import pickle
# mark=f"{datetime.datetime.now():%Y-%m-%d}"
# from uuid import getnode as get_mac
# mac = get_mac()
#
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
# plt.rc('font', family='serif')

# ------------------------------------------------------------------------------
#   h5 files
# ------------------------------------------------------------------------------






#   finds all axion.m.XXXXX files under an address

def findmfiles(address='./'):
    if address == '':
        address = './'
    else:
        address += '/'
    list = []
    for filename in glob.iglob(address+'**/axion.m.*', recursive=True):
        try:
            f = h5py.File(filename, 'r')
            list.append(filename)
        except:
            print('Excluded corrupted file in %s: %s'%(address,filename))

          #print(filename)
    return np.array(sorted(list)) ;



#   finds all XXXXX.m.XXXXX files with a exactly specified neme and address

def findnamedmfilesexact(name='./axion'):
    list = []
    for filename in glob.iglob(name+'.m.*', recursive=True):
          list.append(filename)
          #print(filename)
    return sorted(list) ;



#   finds m folders under an address, returns first if more than 1

def findmdir(address='./'):
    # searches for the m directory
    # print('Seach in ', address)
    if address == '':
        address = './'
    else:
        address += '/'
    list = []
    for filename in glob.iglob(address+'/**/m/', recursive=True):
          list.append(filename)
          #print(filename)
    if len(list) ==1:
        ad = list[0]
    elif len(list) == 0:
        print('Error')
        ad = ''
    else :
        print(list)
        print('Error: multiple m directories, returning first')
        ad = list[0]
    return ad;






#   moves 10000 and 100001 files out of m folders down to parent directory

def mv10001(address='./'):
    mdir = findmdir(address)
    odir = mdir[:-2]
    ch = 0
    if os.path.exists(mdir+'axion.m.10000'):
        os.rename(mdir+'axion.m.10000',odir+'axion.m.10000')
        print('moved 10000')
        ch += 1
    if os.path.exists(mdir+'axion.m.10001'):
        os.rename(mdir+'axion.m.10001',odir+'axion.m.10001')
        print('moved 10001')
        ch += 1
    if ch == 0 :
        if not os.path.exists(odir+'axion.m.10000'):
            print('No files found, argument should be a folder containing -r a m/ directory')
    return ;






#   displays attributes of a measurement file

def aximcontent(address='./'):
    f = h5py.File(address, 'r')
    # displays the atribbutes of a file
    print('Attributes of file ',f)
    for item in f.attrs.keys():
        print(item + ":", f.attrs[item])
    print()
    print('[/ic/]')
    for item in f['/ic/'].attrs.keys():
        print("     ",item + ":", f['/ic/'].attrs[item])
    print()
    print('[/potential/]')
    for item in f['/potential/'].attrs.keys():
        print("     ",item + ":", f['/potential/'].attrs[item])
    print()

    # displays the data sets?
    # returns a lists of flags?
    return ;





#   main function to extract data from axion.m.XXXXX files by concept
#

def gml(list,something='z'):
    out=[]
    for f in list:
        try:
            out.append(gm(f,something))
        except:
            print('Corrupted file: %s'%f)

    return np.array(out)

def gm(address,something='summary',printerror=False):

    # the help
    if something == 'help':
        print('---------------------------------------------')
        print('gm help           ')
        print('---------------------------------------------')
        print('ftype       Saxion/Axion      ')
        print('ct/z/time   conformal time    ')
        print('N/Size      Number of lattice points along 1D ')
        print('L           Phyiscal Box Length [ADM u.]     ')
        print('massA       Axion mass    [ADM u.]     ')
        print('massS       Saxion mass   [ADM u.]    ')
        print('msa         Saxion mass*L/N           ')
        print('eA          Energy Axion  [ADM u.]   ')
        print('eS          Energy SAxion [ADM u.]   ')
        print('eGA         Grad En Axion [ADM u.]   ')
        print('eGxA        Grad x En Axion [ADM u.]   ')
        print('eKA         Kin En Axion [ADM u.]   ')
        print('eVA         Pot En Axion [ADM u.]   ')
        print('stringN     String #points ')
        print('stwallN     Walls  #points   ')
        print('stringL     String Length [lattice u.]')
        print('stringNG    String #points weighted by gamma ')
        print('stVel       String velocity')
        print('stVel2      String velocity squared')
        print('stGamma     String gamma')
        print('stDens      String length/Volume [ADM u.] (statistically)')
        print('stDensL     String length/Volume [ADM u.] (directly)')
        print('stDensG     String length/Volume weighted by gamma [ADM u.] (statistically)')
        print('stEDens     String energy density')
        print('stEDensA    Masked axion energy density')
        print('stEDensS    Masked saxion energy density')
        print('stEDensVil  String energy density (Villadoro masking)')
        print('stEDensAVil Masked axion energy density (Villadoro masking)')
        print('stEDensSVil Masked saxion energy density (Villadoro masking)')
        print('stnout      Number of grid points which are not masked')
        print('binconB     binned normalised log10 contrast         ')
        print('binconBmax  maximum log10(contrast)         ')
        print('binconBmin  maximum log10(contrast)         ')
        print('binthetaB   binned normalised theta value        ')
        print('binthetaBmax, binthetaBmin...   ')
        print('bincon?     True/False')
        print('binrho?     True/False')
        print('bintheta?   True/False')
        print('kmax        maximum momentum [int] in the 3D grid ')
        print('         ')
        print('AXION SPECTRUM         ')
        print('nsp         binned number spectrum [total]')
        print('nspK        binned number spectrum [Kinetic energy part]')
        print('nspG        binned number spectrum [Gradient energy part]')
        print('nspV        binned number spectrum [Potential energy part]')
        print('nsp?        True/False')
        print('SAXION SPECTRUM         ')
        print('nspKS        binned number spectrum [Kinetic energy part]')
        print('nspGS        binned number spectrum [Gradient energy part]')
        print('nspVS        binned number spectrum [Potential energy part]')
        print('nspS?       True/False')
        print('         ')
        print('psp         binned power spectrum')
        print('psp?        True/False')
        print('         ')
        print('3Dmape      3D map of axion energy density')
        print('3Dmape?     Do we have energy density 3D map?')
        print('2Dmape      2D slice map of axion energy density')
        print('2Dmape?     Do we have one?')
        print('2DmapP      2D proyection map of density^2')
        print('2DmapP?     Do we have one?')

        print('         ')
        print('mapmC       2D slice map of conformal PQ field')
        print('mapvC       2D slice map of conformal PQ velocity field')
        print('         ')
        print('maptheta    2D slice map of THETA')
        print('mapvheta    2D slice map of THETA_v')
        print('mapEdens    2D slice map of ENERGY in THETA [currentlt only Axion]')
        print('         ')
        return ;

    f = h5py.File(address, 'r')

    #prelim checks
    if (something[:2] == 'at'):
        esp = something[2:]
        gro = esp[:esp.rfind('/')]
        att = esp[esp.rfind('/')+1:]
        try:
            return f[gro].attrs[u'%s'%att]
        except:
            print('what!')
            return 0

    #prelim checks
    ftype = f.attrs.get('Field type').decode()

    if 'R' in f.attrs:
        scaleFactorR = f.attrs[u'R']
    else:
        scaleFactorR = f.attrs[u'z']

    if (something == 'R'):
        return scaleFactorR ;
    # if loop
    if (something == 'ftype'):
        return ftype ;

    if (something == 'ct') or (something == 'z') or (something == 'time'):
        return f.attrs[u'z'] ;
    if (something == 'Size') or (something == 'N') or (something == 'sizeN'):
        return int(f.attrs[u'Size']) ;
    if something == 'L':
        return f.attrs[u'Physical size'] ;
    if something == 'nqcd':
        if 'nQcd' in f:
            nqcd = f.attrs[u'nQcd']
        elif '/potential/' in f:
            if 'nQcd' in f['/potential/'].attrs:
                nqcd = f['/potential/'].attrs[u'nQcd']
            #print('new format!')
        else :
            nqcd = 7.0
        return nqcd ;

    if something == 'shift':
        if '/potential/' in f:
            return f['/potential/'].attrs[u'Shift'] ;
        else:
            return 0.0 ;

    if something == 'indi3':
        if '/potential/' in f:
            return f['/potential/'].attrs[u'Indi3'] ;
        else:
            return -1 ;

    if something == 'delta':
        L = f.attrs[u'Physical size']
        N = f.attrs[u'Size']
        return L/N ;
    if something == 'massA':
        return f.attrs[u'Axion mass'] ;

    if something == 'lambda':
        try:
            return f['/potential/'].attrs[u'LambdaP']
        except:
            typeL = f['/potential/'].attrs['Lambda type']
            e = 2.0
            if typeL == b'z2' :
                z = f.attrs[u'z']
                return f['/potential/'].attrs[u'Lambda']/(z**e) ;
            else :
                return f['/potential/'].attrs[u'Lambda'] ;

    if something == 'msa':
        typeL = f['/potential/'].attrs['Lambda type']
        l = f['/potential/'].attrs[u'Lambda']
        try:
            e = f['/potential/'].attrs[u'Lambda Z2 exponent']
        except:
            e = 2.0
        z = f.attrs[u'z']
        L = f.attrs[u'Physical size']
        N = f.attrs[u'Size']
        if typeL == b'z2' :
            return np.sqrt(2.0*l)*z**(1-e/2)*L/N ;
        else :
            return np.sqrt(2.0*l)*z*L/N ;


    if something == 'massS':
        ll = f['/potential/'].attrs[u'LambdaP']
        return np.sqrt(2*ll) ;

        # change for FRW
    if something == 'logi':
        R = gm(address,'R')
        try:
            ll = f['/potential/'].attrs[u'LambdaP']
            return np.log(np.sqrt(2*ll)*R**2) ;
        except:
            typeL = f['/potential/'].attrs['Lambda type']
            ll = f['/potential/'].attrs[u'Lambda']
            if typeL == b'z2' :
                return np.log(np.sqrt(2*ll)*R) ;
            else:
                return np.log(np.sqrt(2*ll)*R**2) ;



    # initial condition stuff
    if something == 'kc':
        return f['ic'].attrs[u'Critical kappa'] ;
    if something == 'sIter':
        return f['ic'].attrs[u'Smoothing iterations'] ;

    # energies or other stuff
    en_check = 'energy' in f
    if (something[0] == 'e') and en_check :
        ll = len(something)-1
        if ('mask' in something):
            # float to the right of mask with the correct format
            mst = something[something.find('mask'):]
            mmm = '/Redmask_%.2f'%float(something[something.find('mask')+4:])
            maska = ' nMask'
            ll -= len(something[something.find('mask'):])
        else:
            mst = ''
            mmm = ''
            maska =''

        if 'A' in something:
            field = 'Axion'
            ll -= 1
        elif 'S' in something:
            if ftype == 'Saxion':
                field = 'Saxion'
                ll -= 1
            else:
                if printerror :
                     print('[gm] Warning: file contains no Saxion energy!, set to 0.')
                return 0. ;
        else :
            if printerror :
                print('[gm] No Field specified, add both?')
            return 0. ;
        # if ('A' in something) and ('S' in something)
        erequested = 0
        if ll == 0:
            if (something == 'eA'+mst) or (something == 'eS'+mst):
                    erequested += f['energy'+mmm].attrs[field+' Kinetic'+maska] ;
                    erequested += f['energy'+mmm].attrs[field+' Gr Y'+maska] ;
                    erequested += f['energy'+mmm].attrs[field+' Gr X'+maska] ;
                    erequested += f['energy'+mmm].attrs[field+' Gr Z'+maska] ;
                    erequested += f['energy'+mmm].attrs[field+' Potential'+maska] ;
                    return erequested
        if ('K' in something):
            ll -= 1
            erequested += f['energy'+mmm].attrs[field+' Kinetic'+maska] ;
            if ll == 0:
                return erequested
        if ('V' in something):
            ll -= 1
            erequested += f['energy'+mmm].attrs[field+' Potential'+maska] ;
            if ll == 0:
                return erequested
        if ('G' in something) and not (('Gx' in something) or ('Gy' in something) or ('Gz' in something)): # We assume A or S in the call
            ll -= 1
            erequested += f['energy'+mmm].attrs[field+' Gr Y'+maska] ;
            erequested += f['energy'+mmm].attrs[field+' Gr X'+maska] ;
            erequested += f['energy'+mmm].attrs[field+' Gr Z'+maska] ;
            if ll == 0:
                return erequested

        if ('Gx' in something):
            ll -= 2
            erequested += f['energy'+mmm].attrs[field+' Gr X'+maska] ;
            if ll == 0:
                return erequested

        if ('Gy' in something):
            ll -= 2
            erequested += f['energy'+mmm].attrs[field+' Gr Y'+maska] ;
            if ll == 0:
                return erequested

        if ('Gz' in something):
            ll -= 2
            erequested += f['energy'+mmm].attrs[field+' Gr Z'+maska] ;
            if ll == 0:
                return erequested
        if (ll > 0):
            print('[gm] I did not get a part of the request ')
            return erequested

    elif (something[0] == 'e') and not en_check :
        if printerror :
            print('[gm] No energy in the file ',address )
        return 0. ;

    if (something == 'avrho'):
        return f['energy'].attrs['Saxion vev']

    if (something == 'avrhoM'):
        return f['energy'].attrs['Saxion vev nMask']

    if (something == 'nmp'):
        return f['energy'].attrs['Number of masked points']

    # strings
    st_check = ('string' in f)
    if (something[0:2] == 'st') and ftype == 'Axion':
        return 0. ;
    if (something[0:2] == 'st' or something[0:2] == 'xi') and st_check and ftype == 'Saxion':
        if (something == 'stringN'):
            return f['string'].attrs[u'String number'] ;
        if (something == 'stringL'):
            return f['string'].attrs[u'String length'] ;
        if (something == 'stwallN'):
            return f['string'].attrs[u'Wall number'] ;
        if (something == 'stringNG'):
            return f['string'].attrs[u'String number with gamma'] ;
        if (something == 'stVel'):
            return f['string'].attrs[u'String velocity'] ;
        if (something == 'stVel2'):
            return f['string'].attrs[u'String velocity squared'] ;
        if (something == 'stGamma'):
            return f['string'].attrs[u'String gamma'] ;
        if (something == 'stDens') or (something == 'xi'):
            stringN = f['string'].attrs[u'String number']
            L = f.attrs[u'Physical size']
            N = f.attrs[u'Size']
            ct = f.attrs[u'z']
            delta = L/N
            return  (delta/6)*stringN*ct*ct/(L**3) ;
        if (something == 'stnN3'):
            stringN = f['string'].attrs[u'String number']
            N = f.attrs[u'Size']
            return  stringN/N**3 ;
        if (something == 'stDensL'):
            stringL = f['string'].attrs[u'String length']
            L = f.attrs[u'Physical size']
            N = f.attrs[u'Size']
            ct = f.attrs[u'z']
            delta = L/N
            return  (delta/4)*stringL*ct*ct/(L**3) ;
        if (something == 'stDensG'):
            stringNG = f['string'].attrs[u'String number with gamma']
            L = f.attrs[u'Physical size']
            N = f.attrs[u'Size']
            ct = f.attrs[u'z']
            delta = L/N
            return  (delta/6)*stringNG*ct*ct/(L**3) ;

        if (something == 'stEDens'):
            return f['string'].attrs[u'String energy density'] ;
        if (something == 'stEDensA'):
            return f['string'].attrs[u'Masked axion energy density'] ;
        if (something == 'stEDensS'):
            return f['string'].attrs[u'Masked saxion energy density'] ;
        if (something == 'stEDensVil'):
            return f['string'].attrs[u'String energy density (Vil)'] ;
        if (something == 'stEDensAVil'):
            return f['string'].attrs[u'Masked axion energy density (Vil)'] ;
        if (something == 'stEDensSVil'):
            return f['string'].attrs[u'Masked saxion energy density (Vil)'] ;
        if (something == 'stnout'):
            return f['string'].attrs[u'nout'] ;

        if (something[0:5] == 'strme'):
            la = something[6:8]
            es = {
                'st':'String energy density',
                'eA':'Masked axion energy density',
                'eS':'Masked saxion energy density',
                'vs':'String energy density (Vil)',
                'vA':'Masked axion energy density (Vil)',
                'vS':'Masked saxion energy density (Vil)',
                'no':'nout'}[la]
            return f['string/rmask'+something[8:]].attrs[u''+es] ;

        if (something == 'stringCoord') and ('string/coords' in f):
            size = f['string/coords'].size
            # return np.reshape(f['string/coords'],(size,3)) ;
            return f['string/coords']

    elif (something[0:2] == 'st') and not st_check :
        if printerror :
            print('[gm] No string info in the file! Use 0.')
        return 0. ;

    ##########
    # the bins
    ##########

    bin_check = 'bins' in f

    if (something == 'bincon?'):
        return 'bins/contB' in f

    if (something == 'bintheta?'):
        return 'bins/thetaB' in f

    if (something == 'binlt2?'):
        return 'bins/logtheta2B' in f

    if (something[0:2] == 'bi') and not bin_check :
        if printerror :
            print('[gm] Warning: No bins in file. Returning []')
        return ;

    if (something[0:2] == 'bi') and  bin_check :

        #### contrast bin

        binconB_check = False

        if ('bins/contB' in f):
            binconB_check = True
            bincon_string = 'bins/contB'
        elif ('bins/cont' in f) :
            binconB_check = True
            bincon_string = 'bins/cont'

        if (something == 'bincon?'):
            return binconB_check

        if (something[0:4] == 'binc') and not binconB_check :
            if printerror :
                print(""" [gm] Warning: No contrast bin in file (bins/cont or bins/contB). Returning 'None' """)
            return ;
        if (something[0:4] == 'binc') and binconB_check :
            if (something == 'binconB'):
                numBIN = f[bincon_string].attrs[u'Size']
                return np.reshape(f[bincon_string+'/data'],(numBIN)) ;
            if (something == 'binconBmax'):
                return f[bincon_string].attrs[u'Maximum'] ;
            if (something == 'binconBmin'):
                return f[bincon_string].attrs[u'Minimum'] ;

        #### theta bin

        bintheB_check = False

        if ('bins/thetaB' in f):
            bintheB_check = True
            bintheta_string = 'bins/thetaB'
        elif ('bins/theta' in f) :
            bintheB_check = True
            bintheta_string = 'bins/theta'

        if (something == 'bintheta?'):
            return bintheB_check

        if (something[0:4] == 'bint') and not bintheB_check :
            if printerror :
                print(""" [gm] Warning: No bins/thetaB in file. Returning 'None' """)
            return ;
        if (something[0:4] == 'bint') and bintheB_check :
            if (something == 'binthetaB'):
                numBIN = f[bintheta_string].attrs[u'Size']
                return np.reshape(f[bintheta_string+'/data'],(numBIN)) ;
            if (something == 'binthetaBmax'):
                return f[bintheta_string].attrs[u'Maximum'] ;
            if (something == 'binthetaBmin'):
                return f[bintheta_string].attrs[u'Minimum'] ;

        #### logtheta2 bin

        binlt2B_check = False

        if ('bins/logtheta2B' in f):
            binlt2B_check = True

        if (something == 'binlt2?'):
            return binlt2B_check

        if (something[0:4] == 'binl') and not binlt2B_check :
            if printerror :
                print(""" [gm] Warning: No bins/logtheta2 in file. Returning 'None' """)
            return ;
        if (something[0:4] == 'binl') and bintheB_check :
            if (something == 'binlt2B') or (something == 'binlogtheta2B'):
                numBIN = f['bins/logtheta2B'].attrs[u'Size']
                return np.reshape(f['bins/logtheta2B/data'],(numBIN)) ;
            if (something == 'binlt2Bmax') or (something == 'binthetaBmax'):
                return f['bins/logtheta2B'].attrs[u'Maximum'] ;
            if (something == 'binlt2Bmin') or (something == 'binthetaBmin'):
                return f['bins/logtheta2B'].attrs[u'Minimum'] ;


        #### rho bin

        binrhoB_check = False
        if ('bins/rhoB' in f):
            binrhoB_check = True
            binrho_string = 'bins/rhoB'
        elif ('bins/rho' in f) :
            binrhoB_check = True
            binrho_string = 'bins/rho'

        if (something == 'binrho?'):
            return binrhoB_check

        if (something[0:4] == 'binr') and not binrhoB_check :
            if ftype == 'Axion' :
                if printerror :
                    print('[gm] Warning: Axion mode, no rho!')
                return ;
            elif ftype == 'Axion' :
                if printerror :
                    print('[gm] Warning: No bins/thetaB in file. Returning []')
                return ;
        if (something[0:4] == 'binr') and binrhoB_check :
            if (something == 'binrhoB'):
                numBIN = f[binrho_string].attrs[u'Size']
                return np.reshape(f[binrho_string+'/data'],(numBIN)) ;
            if (something == 'binrhoBmax'):
                return f[binrho_string].attrs[u'Maximum'] ;
            if (something == 'binrhoBmin'):
                return f[binrho_string].attrs[u'Minimum'] ;

        #### fsacc bin

        if (something == 'binfaacc') and 'bins/fsacceleration' in f :
            numBIN = f['bins/fsacceleration'].attrs[u'Size']
            return np.reshape(f['bins/fsacceleration/data'],(numBIN)) ;
        if (something == 'binfaaccmax') and 'bins/fsacceleration' in f :
            return f['bins/fsacceleration'].attrs[u'Maximum'] ;
        if (something == 'binfaaccmin') and 'bins/fsacceleration' in f :
            return f['bins/fsacceleration'].attrs[u'Minimum'] ;


    if (something == 'kmax'):
        N = f.attrs[u'Size']
        return math.floor((N//2)*np.sqrt(3)+1) ;

    ##############
    # the spectra
    ##############

    # number spectra

    # nsp_check = ('nSpectrum/sK' in f) or ('nSpectrum/sKVi' in f) or ('nSpectrum/sK_Vi' in f) or ('nSpectrum/sK_Vi2' in f) or ('nSpectrum/sK_Red' in f)
    nsp_check = ('nSpectrum' in f)

    if (something == 'nsp?') :
        return nsp_check
    if (something == 'nspK?') :
        return nsp_check
    if (something[0:3] == 'nsp') and (something[-1] == '?'):
        # print('nSpectrum/s'+something[3:-1])
        return ('nSpectrum/s'+something[3:-1] in f)
    if (something == 'nsp_info'):
        return [a for a in f['nSpectrum']]

    # if (something == 'nspK_Vi?') :
    #     return ('nSpectrum/sK_Vi' in f)
    # if (something == 'nspK_Vi2?') :
    #     return ('nSpectrum/sK_Vi2' in f)
    # if (something == 'nsp_Red?') :
    #     return ('nSpectrum/sK_Red' in f)
    # if (something == 'nspG?') :
    #     return 'nSpectrum/sG' in f
    #
    # if (something == 'nspV?') :
    #     return 'nSpectrum/sV' in f

    if (something[0:3] == 'nsp') and not nsp_check :
        if printerror :
            print(""" [gm] Warning: No nSpec in file. Returning 'None' """)
        return ;
    # note very specific will be obsolete?
    if (something[0:3] == 'nsp') and  nsp_check :
        # if ('nSpectrum/sK' in f):
        #     powmax = f['nSpectrum/sK/data/'].size
        # if ('nSpectrum/sKVi' in f):
        #     powmax = f['nSpectrum/sKVi/data/'].size
        # if ('nSpectrum/sK_Vi' in f):
        #     powmax = f['nSpectrum/sK_Vi/data/'].size
        # if ('nSpectrum/sK_Vi2' in f):
        #     powmax = f['nSpectrum/sK_Vi2/data/'].size
        # if ('nSpectrum/sK_Red' in f):
        #     powmax = f['nSpectrum/sK_Red/data/'].size

        #ktab = (0.5+np.arange(powmax))*2*math.pi/sizeL
        if ftype == 'Axion':
            if (something[:4] == 'nspK'):
                return np.reshape(f['nSpectrum/sK/data/'],(powmax)) ;
            if (something[:4] == 'nspG'):
                return np.reshape(f['nSpectrum/sG/data/'],(powmax)) ;
            if (something[:4] == 'nspV'):
                return np.reshape(f['nSpectrum/sV/data/'],(powmax)) ;
            if (something == 'nsp'):
                spec = np.reshape(f['nSpectrum/sV/data/'],(powmax)) ;
                spec += np.reshape(f['nSpectrum/sG/data/'],(powmax)) ;
                spec += np.reshape(f['nSpectrum/sK/data/'],(powmax)) ;
                return spec ;
        if ftype == 'Saxion':
            if (something[:3] == 'nsp') and ('nSpectrum/s'+something[3:] in f):
                # print('requested '+'nSpectrum/s'+something[3:]+'/data/')
                return np.array(f['nSpectrum/s'+something[3:]+'/data/']) ;

            # if (something == 'nspK') and ('nSpectrum/sK' in f):
            #     return np.reshape(f['nSpectrum/sK/data/'],(powmax)) ;
            # if (something == 'nspG') and ('nSpectrum/sG' in f):
            #     return np.reshape(f['nSpectrum/sG/data/'],(powmax)) ;
            # if (something == 'nspV') and ('nSpectrum/sV' in f):
            #     return np.reshape(f['nSpectrum/sV/data/'],(powmax)) ;

            # obsolete with new definitions, create a function that does this properly
            # TODO
            # if (something == 'nsp'):
            #     spec = np.reshape(f['nSpectrum/sK/data/'],(powmax)) ;
            #     if ('nSpectrum/sG' in f):
            #         spec += np.reshape(f['nSpectrum/sG/data/'],(powmax)) ;
            #     if ('nSpectrum/sV' in f):
            #         spec += np.reshape(f['nSpectrum/sV/data/'],(powmax)) ;
            #     return spec ;

            # if (something == 'nspKVi') and ('nSpectrum/sKVi' in f):
            #     return np.reshape(f['nSpectrum/sKVi/data/'],(powmax)) ;
            # if (something == 'nspGVi') and ('nSpectrum/sGVi' in f):
            #     return np.reshape(f['nSpectrum/sGVi/data/'],(powmax)) ;
            # if (something == 'nspVVi') and ('nSpectrum/sVVi' in f):
            #     return np.reshape(f['nSpectrum/sVVi/data/'],(powmax)) ;

            # if (something == 'nspKVi2') and ('nSpectrum/sKVi2' in f):
            #     return np.reshape(f['nSpectrum/sKVi2/data/'],(powmax)) ;
            # if (something == 'nspGVi2') and ('nSpectrum/sGVi2' in f):
            #     return np.reshape(f['nSpectrum/sGVi2/data/'],(powmax)) ;
            # if (something == 'nspVVi2') and ('nSpectrum/sVVi2' in f):
            #     return np.reshape(f['nSpectrum/sVVi2/data/'],(powmax)) ;

            # if (something == 'nspKRe') and ('nSpectrum/sKRe' in f):
            #     return np.reshape(f['nSpectrum/sKRe/data/'],(powmax)) ;
            # if (something == 'nspKIm') and ('nSpectrum/sKIm' in f):
            #     return np.reshape(f['nSpectrum/sKIm/data/'],(powmax)) ;
            # if (something == 'nspGRe') and ('nSpectrum/sGRe' in f):
            #     return np.reshape(f['nSpectrum/sGRe/data/'],(powmax)) ;
            # if (something == 'nspGIm') and ('nSpectrum/sGIm' in f):
            #     return np.reshape(f['nSpectrum/sGIm/data/'],(powmax)) ;


            # if (something == 'nspK_Red') and ('nSpectrum/sK_Red' in f):
            #     return np.reshape(f['nSpectrum/sK_Red/data/'],(powmax)) ;
            # if (something == 'nspG_Red') and ('nSpectrum/sG_Red' in f):
            #     return np.reshape(f['nSpectrum/sG_Red/data/'],(powmax)) ;
            # if (something == 'nspV_Red') and ('nSpectrum/sV_Red' in f):
            #     return np.reshape(f['nSpectrum/sV_Red/data/'],(powmax)) ;
            #
            # if (something == 'nspK_Vi') and ('nSpectrum/sK_Vi' in f):
            #     return np.reshape(f['nSpectrum/sK_Vi/data/'],(powmax)) ;
            # if (something == 'nspG_Vi') and ('nSpectrum/sG_Vi' in f):
            #     return np.reshape(f['nSpectrum/sG_Vi/data/'],(powmax)) ;
            # if (something == 'nspV_Vi') and ('nSpectrum/sV_Vi' in f):
            #     return np.reshape(f['nSpectrum/sV_Vi/data/'],(powmax)) ;
            #
            # if (something == 'nspK_Vi2') and ('nSpectrum/sK_Vi2' in f):
            #     return np.reshape(f['nSpectrum/sK_Vi2/data/'],(powmax)) ;
            # if (something == 'nspG_Vi2') and ('nSpectrum/sG_Vi2' in f):
            #     return np.reshape(f['nSpectrum/sG_Vi2/data/'],(powmax)) ;
            # if (something == 'nspV_Vi2') and ('nSpectrum/sV_Vi2' in f):
            #     return np.reshape(f['nSpectrum/sV_Vi2/data/'],(powmax)) ;
            #
            # if (something == 'nspK_Re') and ('nSpectrum/sK_Re' in f):
            #     return np.reshape(f['nSpectrum/sK_Re/data/'],(powmax)) ;
            # if (something == 'nspK_Im') and ('nSpectrum/sK_Im' in f):
            #     return np.reshape(f['nSpectrum/sK_Im/data/'],(powmax)) ;
            # if (something == 'nspG_Re') and ('nSpectrum/sG_Re' in f):
            #     return np.reshape(f['nSpectrum/sG_Re/data/'],(powmax)) ;
            # if (something == 'nspG_Im') and ('nSpectrum/sG_Im' in f):
            #     return np.reshape(f['nSpectrum/sG_Im/data/'],(powmax)) ;

    # ssp_check = 'nSpectrum/ssK' in f
    ssp_check = 'nSpectrum/sKS' in f

    if (something == 'nspS?') :
        return ssp_check

    if (something[0:3] == 'nsp') and (something[-1] == 'S')and  ssp_check :
        powmax = f['nSpectrum/sKS/data/'].size
        #ktab = (0.5+np.arange(powmax))*2*math.pi/sizeL
        if (something == 'nspKS'):
            return np.reshape(f['nSpectrum/sKS/data/'],(powmax)) ;
        if (something == 'nspGS'):
            return np.reshape(f['nSpectrum/sGS/data/'],(powmax)) ;
        if (something == 'nspVS'):
            return np.reshape(f['nSpectrum/sVS/data/'],(powmax)) ;
        if (something == 'nspS'):
            spec = np.reshape(f['nSpectrum/sVS/data/'],(powmax)) ;
            spec += np.reshape(f['nSpectrum/sGS/data/'],(powmax)) ;
            spec += np.reshape(f['nSpectrum/sKS/data/'],(powmax)) ;
            return spec ;

    if (something == 'nmodelist') and ('nSpectrum/nmodes' in f):
        return np.array(f['nSpectrum/nmodes/data'])

    if (something == 'aveklist') and ('nSpectrum/averagek' in f):
        return np.array(f['nSpectrum/averagek/data'])

    # mask spectra
    msp_check = 'mSpectrum' in f

    if (something == 'msp?') :
        return msp_check

    if (something == 'msp_info'):
        return [a for a in f['mSpectrum']]

    if (something[0:3] == 'msp') and not msp_check :
        if printerror :
            print('[gm] Warning: No mSpec in file!!! ')
        return ;

    if ftype == 'Saxion':
        if (something[:3] == 'msp') and ('mSpectrum/'+something[3:] in f):
            kmax = gm(address,'kmax')
            arra = np.array(f['mSpectrum/'+something[3:]+'/data/']) ;
            if len(arra) == kmax :
                return arra
            elif len(arra) == kmax*kmax:
                return np.reshape(arra,(kmax,kmax)) ;

    if (something == 'mspW0') and  msp_check and ('mSpectrum/W0' in f) :
        return np.array(f['mSpectrum/W0/data/']) ;
    if (something == 'mspW_Red') and  msp_check and ('mSpectrum/W_Red' in f):
        return np.array(f['mSpectrum/W_Red/data/']) ;
    if (something == 'mspW_Vi') and  msp_check and ('mSpectrum/W_Vi' in f):
        return np.array(f['mSpectrum/W_Vi/data/']) ;

    if (something == 'mspM_Red') and  msp_check and ('mSpectrum/M_Red' in f):
        # powmax = f['mSpectrum/W/data/'].size
        # if (something == 'msp'):
        kmax = gm(address,'kmax')
        return np.reshape(np.array(f['mSpectrum/M_Red/data/']),(kmax,kmax)) ;

    if (something == 'mspM_Vi') and  msp_check and ('mSpectrum/M_Vi' in f):
        # powmax = f['mSpectrum/W/data/'].size
        # if (something == 'msp'):
        kmax = gm(address,'kmax')
        return np.reshape(np.array(f['mSpectrum/M_Vi/data/']),(kmax,kmax)) ;
    if (something == 'mspM_Vi2') and  msp_check and ('mSpectrum/M_Vi2' in f):
        # powmax = f['mSpectrum/W/data/'].size
        # if (something == 'msp'):
        kmax = gm(address,'kmax')
        return np.reshape(np.array(f['mSpectrum/M_Vi2/data/']),(kmax,kmax)) ;

    # power spectra
    psp_check = 'pSpectrum' in f

    if (something == 'psp?') :
        return psp_check

    if (something == 'pspMA?') :
        return 'pSpectrum/sPmasked' in f

    if (something == 'pspMA2?') :
        return 'pSpectrum/sPmasked2' in f

    if (something[0:3] == 'psp') and not psp_check :
        if printerror :
            print('[gm] Warning: No pSpec in file. ')
        return ;
    if (something[0:3] == 'psp') and  psp_check :
        if (something == 'psp') and 'pSpectrum/sP' in f:
            return np.array(f['pSpectrum/sP/data/']) ;

        if (something == 'pspMA') and 'pSpectrum/sPmasked' in f:
            return np.array(f['pSpectrum/sPmasked/data/']) ;

        if (something == 'pspMA2') and 'pSpectrum/sPmasked2' in f:
            return np.array(f['pSpectrum/sPmasked2/data/']) ;

    # maps

    if (something[0:5] == 'slice'):
        esp = something[5:]
        map = esp[:esp.rfind('/')]
        typ = esp[esp.rfind('/')+1:]
        mal = np.array(f[map][typ].value)
        N = f.attrs[u'Size']
        if len(mal) == N*N*2:
            return mal.reshape(N,N,2)
        elif len(mal) == N*N:
            return mal.reshape(N,N)

    map_check = 'map' in f

    if (something == 'map?'):
        return map_check ;

    if (something[0:3] == 'map') and not map_check :
        if printerror :
            print('[gm] Warning: No map in file!. ')
        return ;
    if (something[0:3] == 'map') and  map_check :
        mapad = 'map'
        if (something[0:4] == 'mapp') and  map_check :
            mapad = 'mapp'
        N = f.attrs[u'Size']
        ct = f.attrs[u'z']
        if (something == mapad+'mC') and (ftype == 'Saxion'):
            # return np.reshape(f['map']['m'].value.reshape(N,N,2)) ;
            return f[mapad]['m'].value.reshape(N,N,2) ;
        if (something == mapad+'mC') and (ftype == 'Axion'):
            return ;
        if (something == mapad+'vC') and (ftype == 'Saxion'):
            return f[mapad]['v'].value.reshape(N,N,2) ;
        if (something == mapad+'vC') and (ftype == 'Axion'):
            return ;
        if (something == mapad+'theta') and (ftype == 'Saxion'):
            temp = np.array(f[mapad]['m'].value.reshape(N,N,2))
            temp = np.arctan2(temp[:,:,1], temp[:,:,0])
            return temp ;
        if (something == mapad+'theta') and (ftype == 'Axion'):
            temp = np.array(f[mapad]['m'].value.reshape(N,N))
            return temp/scaleFactorR ;
        if (something == mapad+'vheta') and (ftype == 'Axion'):
            temp = np.array(f[mapad]['v'].value.reshape(N,N))
            return temp ;
        if (something == mapad+'rho') and (ftype == 'Saxion'):
            temp = np.array(f[mapad]['m'].value.reshape(N,N,2))
            # te = f.attrs[u'z']
            return np.sqrt(temp[:,:,0]**2 + temp[:,:,1]**2)/scaleFactorR

        if (something == 'mapE'):
            if 'map/E' in f:
                return np.array(f[mapad]['E'].value.reshape(N,N)) ;


        if (something == 'mapEdens') and (ftype == 'Axion'):
            theta = np.array(f[mapad]['m'].value.reshape(N,N))/scaleFactorR
            massA2 = f.attrs[u'Axion mass']
            massA2 *= massA2
            mapa = massA2*2*np.sin(theta/2)**2
            kine = np.array(f['map']['v'].value.reshape(N,N))
            mapa += ((kine - theta/ct)**2)/(2*ct*ct)
            return mapa ;

    # 3D density maps
    if something == '3Dmape?':
        if ('energy/density' in f) or ('energy/redensity' in f):
            return True
        else :
            return False

    if something == '3Dmape':
    # three options:
    # 1-legacy ... (I do not remember)
    # 2-energy/density/theta/
    # 3-energy/redensity/data
    # assume it is either one or the other, favour redensity in case there is only one
        if 'energy/redensity' in f:
            redN3 = f['energy']['redensity'].size
            redN = integer_nthroot(redN3, 3)[0]
            if printerror:
                print('reduced ',redN)
            return f['energy/redensity'].value.reshape(redN,redN,redN)
        if 'energy/density/theta' in f:
            redN = f['energy/density'].attrs[u'Size']
            redZ = f['energy/density'].attrs[u'Depth']
            if printerror:
                print('Reduced ',redN)
            return f['energy/density/theta'].value.reshape(redN,redN,redZ)

    if something == '3Dmapefull':
            if 'energy/density/theta' in f:
                redN = f['energy/density'].attrs[u'Size']
                redZ = f['energy/density'].attrs[u'Depth']
                if printerror:
                    print('Giving you the fullest N=',redN,redZ)

                return f['energy/density/theta'].value.reshape(redN,redN,redZ)

    if something == '2Dmape?':
        if ('map/E' in f) :
            return True
        else :
            return False

    if something == '2Dmape':
        if ('map/E' in f) :
            sizeN = f.attrs[u'Size']
            return f['map']['E'].value.reshape(sizeN,sizeN)

    if something == '2DmapP?':
        if ('map/P' in f) :
            return True
        else :
            return False

    if something == '2DmapP':
        if ('map/P' in f) :
            sizeN = f.attrs[u'Size']
            return f['map']['P'].value.reshape(sizeN,sizeN)


    # the irrelevants
    if something == 'Depth':
        return f.attrs[u'Depth'] ;
    if something == 'zi':
        return f.attrs[u'zInitial'] ;
    if something == 'zf':
        return f.attrs[u'zFinal'] ;

    if (something == 'rhovev'):
        if ssp_check:
            BV0 = f['nSpectrum/sVS/data/'][0]
            V = f.attrs[u'Physical size']**3
            ms = gm(address,'massS')
            return np.sqrt(BV0*2/ms/V)/scaleFactorR

    if something == 'summary':
        nqcd = gm(address,'nqcd')
        ct = f.attrs[u'z']
        print('---------------------Summary---------------------')
        print('file: %s (%s - mode)'%(address,ftype))
        print('-------------------------------------------------')
        print('N=%d    L=%.2f    ct=%.5f'%(f.attrs[u'Size'],f.attrs[u'Physical size'],f.attrs[u'z']))
        print('massA= %.5e, (nqcd = %.2f, ct^n/2=%.5e)'%(f.attrs[u'Axion mass'],gm(address,'nqcd'),ct**(nqcd/2)))
        print()
        if map_check:
            print('2Dmap', end=' - ')
        if en_check:
            print('Energy', end=' - ')
        if ('energy/density' in f) or ('energy/redensity' in f):
            print('Energy 3D', end=' - ')
        if st_check:
            print('String', end=' - ')
        if 'string/data' in f :
            print('strings 3D', end=' - ')
        print() ; print()
        print('Partic Spectra:',end=' ')
        if 'nSpectrum/sK' in f:
            print('Axion K', end=' - ')
        if 'nSpectrum/sG' in f:
            print('Axion G', end=' - ')
        if 'nSpectrum/sV' in f:
            print('Axion V', end=' - ')
        if 'nSpectrum/sKS' in f:
            print('Saxion K', end=' - ')
        if 'nSpectrum/sGS' in f:
            print('Saxion G', end=' - ')
        if 'nSpectrum/sVS' in f:
            print('Saxion V', end=' - ')
        print()
        print('Energy Spectra:',end=' ')
        if 'pSpectrum/sP' in f:
            print('Axion ', end=' - ')
        if 'pSpectrum/sPS' in f:
            print('Saxion', end=' ')
        print()
        print('Binned data   :',end=' ')

        if ('bins/contB' in f) :
            print('E-contrast', end=' - ')
        if ('bins/thetaB' in f) or ('bins/contB' in f):
            print('theta', end=' - ')
        if ('bins/logtheta2B' in f):
            print('log theta^2', end=' - ')
        if ('bins/rhoB' in f) or ('bins/rho' in f):
            print('rho', end=' ')
        print()
        return ;

    print('Argument %s not recognised/found!'%(something))
    return ;







# ------------------------------------------------------------------------------
#   utilities
# ------------------------------------------------------------------------------






def meas2human(inte):
    sa=[]
    for n in range(0,40):
        s=int(2**n)
        if (inte & s):
            if (inv_measdic[s] != "MEAS_EMPTY"):
                sa.append(s)
                print(s, inv_measdic[s])
    return sa






measdic = { "MEAS_NOTHING" : 0,
"MEAS_BINTHETA"     : 1,
"MEAS_BINRHO"       : 2,
"MEAS_BINLOGTHETA2" : 4,
"MEAS_BINDELTA"     : 8,
"MEAS_EMPTY"        : 16,
"MEAS_STRING"	    : 32,
"MEAS_STRINGMAP"    : 64,
"MEAS_STRINGCOO"    : 128,
"MEAS_ENERGY"       : 256,
"MEAS_ENERGY3DMAP"  : 512,
"MEAS_REDENE3DMAP"  : 1024,
"MEAS_2DMAP"        : 2048,
"MEAS_3DMAP"        : 4096,
"MEAS_MASK"         : 8192,
"MEAS_PSP_A"        : 16384,
"MEAS_PSP_S"        : 32768,
"MEAS_NSP_A"        : 65536,
"MEAS_NSP_S"        : 131072,
"MEAS_NNSPEC"       : 262144,
          }
inv_measdic = {v: k for k, v in measdic.items()}








# build a measurement list
class mli:
    def __init__(self,msa=1.0,L=6.0,N=1024):
        self.mtab = [] ;
        self.ctab = [] ;
        self.me = 0;
        self.msa = msa;
        self.L = L;
        self.N = 1024;
        self.ctend = 1000;
        self.outa = []
        self.outb = []

    def dic (self):
        return measdic
    def nmt (self):
        self.me = 0
    def me_add (self, meas):
        self.me |= meas
    def me_adds (self, meass):
        self.me |= measdic[fildic(meass)]
        print("adds %s (%d) > me %d"%(fildic(meass), measdic[fildic(meass)],self.me))

    #caca
    def nmt_rem (self,meas):
        self.me -= meas2human(meas)
    def nmt_prt (self):
        meas2human(self.me)
    # def nmt_prt (self,meas):
    #     pa.meas2human(meas)

    def addset (self,measN, zi, zf, meastype=0, scale='lin'):
        if scale=='lin':
            self.ctab.append(np.linspace(zi,zf,measN+1))
        if scale=='logi':
            temp = np.linspace(zi,zf,measN+1)
            # logi = log (msa ct/a)
            # ct = a exp(temp)*a/msa
            self.ctab.append(np.exp(temp)*self.L/self.N/self.msa)
        self.mtab.append(meastype | self.me)
        print("Set with me %s created"%(meastype | self.me))
    def give(self,name="./measfile.dat"):
        outa = [a for a in self.ctab[0]]
        outb = [self.mtab[0] for a in self.ctab[0]]
#         print(outa)
        print("Printing sets with measures: ",self.mtab)
        outa = []
        outb = []

        for imt in range(len(self.mtab)):
            outa += [t for t in list(self.ctab[imt])]
            outb += [self.mtab[imt] for t in list(self.ctab[imt])]

        outa=np.array(outa)
        outb=np.array(outb)
        self.outa = sorted(list(set(outa)))
        # print(set(outa))
        # print(self.outa)

        for ct in self.outa:
            mask = (outa == ct)
            if sum(mask)>1:
                ii=0
                print("%f merged "%(ct),end="")
                for ca in outb[mask]:
                    ii |= ca
                    print("%d "%(ca),end="")
                print(" into %d "%(ii))
                self.outb.append(ii)
            else :
                ii = outb[mask][0]
                self.outb.append(ii)
            # print("%f %d"%(ct,ii))

        cap = []
        file = open(name,"w")
        print(self.outa)
        for i in range(len(self.outa)):
            if self.outa[i] <= self.ctend :
                print("%f %d"%(self.outa[i],self.outb[i]))
                file.write("%f %d\n"%(self.outa[i],self.outb[i]))
                cap.append([self.outa[i],self.outb[i]])
        file.close()


        return cap







def fildic(meas):
    if (meas in ['MEAS_BINTHETA','BINTHETA','bin theta','bintheta','thetabin','tbin']):
        return 'MEAS_BINTHETA'
    if (meas in ['MEAS_BINRHO','BINRHO','bin rho','binrho','rhobin','rbin']):
        return 'MEAS_BINRHO'
    if (meas in ['MEAS_BINLOGTHETA2','BINLOGTHETA2','bin logtheta2','bin logtheta','binlogtheta','ltbin']):
        return 'MEAS_BINLOGTHETA2'
    if (meas in ['MEAS_BINDELTA','BINDELTA','bin delta','bindelta','bin contrast','delta bin','conbin','deltabin']):
        return 'MEAS_BINDELTA'
    if (meas in ['MEAS_STRING','STRING','string','str','string density','string number','string plaquetes']):
        return 'MEAS_STRING'
    if (meas in ['MEAS_STRINGMAP','STRINGMAP','string map','str map','string 3D map']):
        return 'MEAS_STRINGMAP'
    if (meas in ['MEAS_STRINGCOO','STRINGCOO','string coordinates','strco','string coordinate list']):
        return 'MEAS_STRINGCOO'
    if (meas in ['MEAS_ENERGY','ENERGY','energy']):
        return 'MEAS_ENERGY'
    if (meas in ['MEAS_ENERGY3DMAP','ENERGY3DMAP','energy 3D map']):
        return 'MEAS_ENERGY3DMAP'
    if (meas in ['reduced energy 3D map','reduced energy map','redensity']):
        return 'MEAS_REDENE3DMAP'
    if (meas in ['MEAS_2DMAP','2DMAP','map 2D','plot2D','slice']):
        return 'MEAS_2DMAP'
    if (meas in ['MEAS_3DMAP','3DMAP','map 3D','plot3D','configuration 3D','full configuration 3D']):
        return 'MEAS_3DMAP'
    if (meas in ['MEAS_MASK','MASK','spectrum mask','mask']):
        return 'MEAS_MASK'
    if (meas in ['MEAS_PSP_A','PSP_A','energy power spectrum','energy power spectrum axion','psp','pspa','power spectrum']):
        return 'MEAS_PSP_A'
    if (meas in ['MEAS_NSP_A','NSP_A','axion spectrum','nspa']):
        return 'MEAS_NSP_A'
    if (meas in ['MEAS_NSP_S','NSP_S','Saxion spectrum','nspS']):
        return 'MEAS_NSP_S'
    if (meas in ['MEAS_NNSPEC','nmodelist','nmodes']):
        return 'MEAS_NNSPEC'










# ------------------------------------------------------------------------------
#   n modes for normalising power spectra (move to other file?)
# ------------------------------------------------------------------------------

from math import exp, log10, fabs, atan, log, atan2






#   volume of a box size 1 contained inside a sphere of radius rR

def volu( rR ):

    if rR <= 1.0:
        return (4*math.pi/3)*rR**3 ;

    elif 1.0 < rR <= math.sqrt(2.):
        return (2*math.pi/3)*(9*rR**2-4*rR**3-3) ;

    elif math.sqrt(2.) < rR < math.sqrt(3.):
        a2 = rR**2-2
        a = math.sqrt(a2)
        b = 8*a - 4*(3*rR**2 -1)*(atan(a)-atan(1/a))
        return b - (8/3)*(rR**3)*atan2(a*(6*rR + 4*rR**3 -2*rR**5),6*rR**4-2-12*rR**2) ;

    elif  math.sqrt(3) < rR:
        return 8. ;






#   Feturns an approximation of the number of modes
#   that we have binned as a function of |k|
#   using the previous volume function

def phasespacedensityBOXapprox ( sizeN ):

    n2 = int(sizeN//2)
    powmax = math.floor((sizeN//2)*np.sqrt(3)+1) ;
    foca = np.arange(0,powmax)/n2
    foca2 = np.arange(1,powmax+1)/n2
    vecvolu=np.vectorize(volu)
    return (n2**3)*(vecvolu(foca2)-vecvolu(foca)) ;






#   This would return the exact number of modes but it is expensive

def phasespacedensityBOXexact ( sizeN ):
    n2 = int(sizeN//2)
    powmax = math.floor((sizeN//2)*np.sqrt(3)+1) ;
    bins = np.zeros(powmax,dtype=int)
    for i in range(-n2+1,n2+1):
        for j in range(-n2+1,n2+1):
            for k in range(-n2+1,n2+1):
                mod = math.floor(math.sqrt(i*i+j*j+k*k))
                #print(i, j, k, mod)
                bins[mod] += 1
    return bins ;






#   this counts the modes starting from the highest

def phasespacedensityBOXexactHIGH ( sizeN, sizen ):
    from itertools import chain
    n2 = int(sizeN//2)
    powmax = math.floor((sizeN//2)*np.sqrt(3)+1) ;
    concatenated = chain(range(-n2+1,-n2+sizen+1), range(n2-sizen,n2+1))
    lis = list(concatenated);
    bins = np.zeros(powmax,dtype=int)
    n = 0
    for i in range(0,len(lis)):
        x = lis[i]
        for j in range(0,len(lis)):
            y = lis[j]
            for k in range(0,len(lis)):
                z = lis[k]
                mod = math.floor(math.sqrt(x*x+y*y+z*z))
                bins[mod] += 1
    return bins ;






#   My 1% approximation to the number of modes binned
#   using the exact values when the number of modes is small

def phasespacedensityBOX_old ( sizeN, res=0.01 ):
    approx = phasespacedensityBOXapprox ( sizeN ) ;
    if (approx.max()*res**2 < 0.1):
        print(exact)
        return phasespacedensityBOXexact ( sizeN )

    # 1% error requires 100^2 modes
    sli = 0
    for i in range(0,len(approx)):
        if approx[i] > int(1/res**2):
            break
        sli=sli+1

    # print(sli,approx.max()*res**2 )
    exact = phasespacedensityBOXexact ( 2*sli )
    for i in range(0,sli):
        approx[i] = exact[i]

    exact = phasespacedensityBOXexactHIGH ( sizeN, 2*sli ) ;
    for i in range(len(exact)-sli,len(exact)):
        approx[i] = exact[i]

    return approx ;

def phasespacedensityBOX ( sizeN):
    approx = phasespacedensityBOXapprox ( sizeN ) ;

    sli = 0
    for i in range(0,len(approx)):
        if approx[i] > 10000:
            sli = i
            break

    dira = os.path.dirname(__file__)
    # print(sli,approx.max()*res**2
    dump = pickle.load( open( dira+'/data/512mod.p', "rb" ) )
    ss = min(sizeN//2,256)
    # print(ss,sizeN//2,256)
    approx[:ss] = dump[:ss]
    # exact = phasespacedensityBOXexact ( 2*sli )
    # for i in range(0,sli):
    #     approx[i] = exact[i]

    exact = phasespacedensityBOXexactHIGH ( sizeN, 2*sli ) ;
    for i in range(len(exact)-sli,len(exact)):
        approx[i] = exact[i]

    return approx ;






#   returns a list of modes

def modelist ( sizeN ):
    n2 = int(sizeN//2)
    powmax = math.floor((sizeN//2)*np.sqrt(3)+1) ;
    lis = []
    for i in range(-n2+1,n2+1):
        for j in range(-n2+1,n2+1):
            for k in range(-n2+1,n2+1):
                mod = math.sqrt(i*i+j*j+k*k)
                lis.append(mod)
    return sorted(set(lis)) ;






# ------------------------------------------------------------------------------
#   contrast bins
# ------------------------------------------------------------------------------






#   returns a list of logarithmic bins and bin heights from a axion.m.XXXXX file
#   and a mimumum number of X points per bin by grouping
#   variable size bins

def conbin(file, X=10):
    return logbin(gm(file,'binconB'), gm(file,'binconBmin'), gm(file,'binconBmax'), gm(file,'Size'), X) ;

def lt2bin(file, X=10):
    return logbin(gm(file,'binlt2B'), gm(file,'binlt2Bmin'), gm(file,'binlt2Bmax'), gm(file,'Size'), X) ;




#   returns a list of logarithmic bins and bin heights from a list
#   a minimum log, maximum log, N (N3 is the original number of points)
#   and a minimum X of points in the bin

# normalise contbin with variable width bins of minimum X points
def logbin(logbins, mincon, maxcon, N, X):
    numBIN = len(logbins)
    N3 = N*N*N
    bw = (maxcon-mincon)/numBIN
    bino = logbins*N3*bw
    numbins = numBIN

    # removes first points until first bin with X points
    i0 = 0
    # while bino[i0] < X:
    #         i0 = i0 + 1
    # bino = bino[i0:]

    sum = 0
    parsum = 0
    nsubbin = 0
    minimum = 10
    lista=[]

    weigthedaverage = 0.0

    # JOIN BINS ALL ALONG to have a minimum of X points
    for bin in range(0,len(bino)):
        # adds bin value to partial bin
        parsum += bino[bin]
        # adds bin value times position (corrected for bias)
        weigthedaverage += bino[bin]*10**(mincon + bw*(i0+ bin + 0.5))
        # number of bins added to parsum
        nsubbin += 1
        # adds bin value to partial bin

        if nsubbin == 1:
                # records starting bin
                inbin = bin
        if parsum < X:
            # if parsum if smaller than X we will continue
            # adding bins
            sum += 1
        else:
            enbin = bin

            low = 10**(mincon + bw*(i0+inbin))
            # med = 10**(mincon + bw*(i0+(inbin+enbin+1)/2))
            ave = weigthedaverage/parsum
            sup = 10**(mincon + bw*(i0+enbin+1))
            lista.append([ave,parsum/(N3*(sup-low))])

            parsum = 0
            nsubbin = 0
            weigthedaverage = 0.0

    return np.array(lista)         ;






#   builds the cumulative distribution







# ------------------------------------------------------------------------------
#   contrast bins
# ------------------------------------------------------------------------------






#   returns a list of logarithmic bins and bin heights from a axion.m.XXXXX file
#   and a mimumum number of X points per bin by grouping
#   variable size bins

def thetabin(file, X=10):
    return linbin(gm(file,'binthetaB'), gm(file,'binthetaBmin'), gm(file,'binthetaBmax'), gm(file,'Size'), X) ;

def rhobin(file, X=10):
    return linbin(gm(file,'binrhoB'), gm(file,'binrhoBmin'), gm(file,'binrhoBmax'), gm(file,'Size'), X) ;




#   returns a list of logarithmic bins and bin heights from a list
#   a minimum log, maximum log, N (N3 is the original number of points)
#   and a minimum X of points in the bin

# normalise contbin with variable width bins of minimum X points
def linbin(bins, min, max, N, X):
    numbins = len(bins)
    N3 = N*N*N
    bw = (max-min)/numbins
    # PDF
    bino = bins*N3*bw

    # removes first points until first bin with X points
    i0 = 0
    # while bino[i0] < X:
    #         i0 = i0 + 1
    # bino = bino[i0:]

    sum = 0
    parsum = 0
    nsubbin = 0
    minimum = 10
    lista=[]

    weigthedaverage = 0.0

    # JOIN BINS ALL ALONG to have a minimum of X points
    for bin in range(0,len(bino)):
        # adds bin value to partial bin
        parsum += bino[bin]

        # adds bin value times position (corrected for bias)
        weigthedaverage += bino[bin]*(i0 + bin + 0.5)

        # number of bins added to parsum
        nsubbin += 1

        if nsubbin == 1:
                # records starting bin lower boundary
                inbinB = bin
        if (parsum < X) and (bin < len(bino)-1):
            # if parsum if smaller than X we will continue
            # adding bins
            sum += 1
        else:
            # records ending upper boundary
            enbinB = bin + 1

            low = min + bw*(i0+inbinB)
            # med = min + bw*(i0+(inbinB+enbinB)/2)
            avposition = min + bw*weigthedaverage/parsum
            sup = min + bw*(i0+enbinB)
            lista.append([avposition,parsum/(N3*(sup-low))])
            # if bins are desired change to this output
            #lista.append([avposition,parsum/(N3*(sup-low)),low,sup])

            parsum = 0
            nsubbin = 0
            weigthedaverage = 0.0

    return np.array(lista)         ;






#   builds the cumulative distribution







# ------------------------------------------------------------------------------
#   normalises power spectra
# ------------------------------------------------------------------------------






#   takes a binned |FT|^2 of the DENSITY
#   divides by number of modes and normalises it
#   to be the logarithmic variance
#   note that our code normalises the |FT|^2 of the DENSITY
#   with a factor L^3/2*N^6 (that we also use for the nSpectrum)
#   the 1/2 is the 1/2 of kinetic, gradient and mass term ;-)
#   Since we want to multiply by k^3/2pi^2 to make the dimensionless variance,
#   we only use k^3/pi^2 for the 1/2 is already there since the code

def normalisePspectrum(psp, nmodes, avdens, N, L):
    kmax   = len(psp)
    klist  = (0.5+np.arange(kmax))*2*math.pi/L
    norma  = 1/((math.pi**2)*(avdens**2))
    return norma*(klist**3)*psp/nmodes;






# ------------------------------------------------------------------------------
#   higher level wrappers
# ------------------------------------------------------------------------------






#   outputs the string evolution

def stringo(mfiles):
    stringo = []
    for f in mfiles:
        if gm(f,'ftype') == 'Saxion':
            stringo.append([gm(f,'ct'),gm(f,'stDens')])
    return np.array(stringo) ;






#   outputs the Wall evolution

def wallo(mfiles):
    wallo = []
    for f in mfiles:
        if gm(f,'ftype') == 'Saxion':
            wallo.append([gm(f,'ct'),gm(f,'stwallN')*gm(f,'ct')*gm(f,'delta')**2/gm(f,'L')**3])
    return np.array(wallo) ;






#   outputs the energy evolution

def energo(mfiles):
    eevol = []
    for f in mfiles:
        eevol.append([gm(f,'ct'),gm(f,'eA'),gm(f,'eS')])
    return np.array(eevol) ;






#   outputs the energy evolution of axion components

def energoA(mfiles):
    eevol = []
    for f in mfiles:
        eevol.append([gm(f,'ct'),gm(f,'eKA'),gm(f,'eGA'),gm(f,'eVA'),gm(f,'eA')])
    return np.array(eevol) ;






#   outputs the energy evolution of Saxion components

def energoS(mfiles):
    eevol = []
    for f in mfiles:
        eevol.append([gm(f,'ct'),gm(f,'eKS'),gm(f,'eGS'),gm(f,'eVS'),gm(f,'eS')])
    return np.array(eevol) ;






# ------------------------------------------------------------------------------
#   ANALYSIS TOOLS simulation class + averager
# ------------------------------------------------------------------------------







class simgen:
    def __init__(self, N):
        self.N = N
        self.nmax = math.floor((self.N//2)*np.sqrt(3)+1)
        self.nmodes = phasespacedensityBOX(self.N)




class simu:
    def __init__(self, dirname):
        self.dirname = dirname
        self.mfiles = findmfiles(self.dirname)

        self.N = gm(self.mfiles[0],'Size')
        self.nqcd = gm(self.mfiles[0],'nqcd')
        self.msa = gm(self.mfiles[0],'msa')
        self.L = gm(self.mfiles[0],'L')
        self.zI = gm(self.mfiles[0],'zi')
        self.zF = gm(self.mfiles[0],'zf')
        self.zi = gm(self.mfiles[0],'z')
        self.zf = gm(self.mfiles[-1],'z')

        self.finished = False
        self.wkb = False
        self.preprop = False

        if self.zf > self.zF :
            self.finished = True
        if self.zi < self.zI :
            self.preprop = True

        # danger flag
        self.safe = True

        # mode list
        self.mora = simgen(self.N)
        self.klist=(6.28318/self.L)*(np.arange(self.mora.nmax)+0.5)

        ################
        # power spectrum
        # dim variance
        ################

        self.mfilesp = []

        # flag in case there is no spectrum?
        self.anypsp = False

        for mf in self.mfiles:
            if gm(mf,'psp?') :
                self.mfilesp.append(mf)
                self.anypsp = True

        # last spectrum
        self.psp = None
        if self.anypsp :
            for mf in reversed(self.mfilesp):
                self.lastfilep = mf
                self.psp = gm(self.lastfilep,'psp')
                if self.psp is not None:
    #                 print(self.dirname,self.lastfilep,self.N,len(self.psp),len(self.klist))
                    self.avdens = gm(self.lastfilep,'eA')
                    self.norma = 1/((math.pi**2)*(self.avdens**2))
                    self.pspt = gm(self.lastfilep,'ct')
                    self.psp = (self.norma)*self.psp*(self.klist**3)/(self.mora.nmodes)
    #                 print(self.lastfilep)
                    break

        # checks

        self.spok = False

        if self.psp is not None:
            self.spok = True
            if self.psp.max() > 500. :
                self.spok = False


        #################
        # number spectrum
        #################

        self.mfilesn = []

        # flag in case there is no spectrum?
        self.anynsp = False

        for mf in self.mfiles:
            if gm(mf,'nsp?') and gm(mf,'ftype') =='Axion':
                self.mfilesn.append(mf)
                self.anynsp = True

        # last spectrum
        self.nsp = None
        self.nspK = None
        self.nspG = None
        self.nspV = None

        if self.anynsp :
            for mf in reversed(self.mfilesn):
                self.lastfilen = mf
                self.nsp = gm(self.lastfilep,'nsp')

        if self.nsp is not None:
            self.nsp = (self.klist**3)*self.nsp/(self.mora.nmodes)
            self.nspK = (self.klist**3)*gm(self.lastfilen,'nspK')/(self.mora.nmodes)
            self.nspG = (self.klist**3)*gm(self.lastfilen,'nspG')/(self.mora.nmodes)
            self.nspV = (self.klist**3)*gm(self.lastfilen,'nspV')/(self.mora.nmodes)
            if self.nsp.max() > 1000. :
                self.spok = False

        #################
        # CONT bins
        #################

        self.mfilesc = []

        # flag in case there is no contrast bin at all?
        self.anycon = False

        for mf in self.mfiles:
            if gm(mf, 'bincon?') :
                self.mfilesc.append(mf)
                self.anycon = True


    def listpspct (self, time):
        timetab=[]
        for mf in self.mfilesp:
            timetab.append(gm(mf,'ct'))
        timetab=np.array(timetab)
        pos=np.abs(timetab-time).argmin()

        tmppsp = gm(self.mfilesp[pos],'psp')
        if tmppsp is not None:
            tmpavdens = gm(self.mfilesp[pos],'eA')
            tmpnorma = 1/((math.pi**2)*(tmpavdens**2))
            tmppspt = gm(self.mfilesp[pos],'ct')
            tmppsp = (tmpnorma)*tmppsp*(self.klist**3)/(self.mora.nmodes)
            return tmppspt, self.klist, tmppsp

    def listnspct (self, time):
        timetab=[]
        for mf in self.mfilesn:
            timetab.append(gm(mf,'ct'))
        timetab=np.array(timetab)
        pos=np.abs(timetab-time).argmin()

        tmpnsp = gm(self.mfilesn[pos],'nsp')
        if tmpnsp is not None:
            tmpnspt = gm(self.mfilesn[pos],'ct')
            tmpnsp = tmpnsp*(self.klist**3)/(self.mora.nmodes)
            return tmpnspt, self.klist, tmpnsp

    def listcont (self, time):
        timetab=[]
        for mf in self.mfilesc:
            timetab.append(gm(mf,'ct'))
        timetab=np.array(timetab)
        pos=np.abs(timetab-time).argmin()

        tmpcon = conbin(self.mfilesc[pos],100)
        if tmpcon is not None:
            tmpt = gm(self.mfilesc[pos],'ct')
            return tmpt, tmpcon[:,0], tmpcon[:,1]

    def addpspplot (self):
        if self.spok == True :
            return plt.loglog(self.klist,self.psp,label='t=%.1f N=%d L=%.1f'%(self.pspt,self.N,self.L))

    def addpspplott (self, time):
        timetab=[]
        for mf in self.mfiles:
            timetab.append(gm(mf,'ct'))

        timetab=np.array(timetab)
        pos=np.abs(timetab-time).argmin()
        self.tmppsp = gm(self.mfiles[pos],'psp')
        if self.tmppsp is not None:
            self.tmpavdens = gm(self.mfiles[pos],'eA')
            self.tmpnorma = 1/((math.pi**2)*(self.tmpavdens**2))
            self.tmppspt = gm(self.mfiles[pos],'ct')
            self.tmppsp = (self.tmpnorma)*self.tmppsp*(self.klist**3)/(self.mora.nmodes)
            return plt.loglog(self.klist,self.tmppsp,label='t=%.1f N=%d L=%.1f'%(self.tmppspt,self.N,self.L))





# stores different files for a given observable and presents interpolated averages and variance
class averager:
    def __init__(self,name):
        self.name = name
        self.xlist = []
        self.ylist = []
        self.xmislist = []
        self.xmisloglist = []
        self.xmaslist = []
        self.xgrid = []
        self.yave = []
        self.ystd = []
        self.number = 0
        self.pbase = []
        self.npoints = 0
        self.labels = []

    def adds(self,x, y,lab=''):
        self.xlist.append(x)
        self.ylist.append(y)
        self.xmislist.append(x.min())
        self.xmisloglist.append(np.min(x[x>0.0]))
        self.xmaslist.append(x.max())
        self.number += 1
        self.labels.append(lab)

    # this calculates averages in log
    # would be nice to have it for linear, in that case, one would like to have mislog and mis
    # at the moment, cannot handle 0.0 in xlist[i]
    # if a point x=0.0 is present, like in the powerspectra, they have to be dropped in the call
    def calculateave(self,n_points):
        self.npoints=n_points
        mis = min(self.xmisloglist)*1.01
        mas = max(self.xmaslist)*0.99
        print(mis,mas,n_points,'points in log space')
        self.xgrid = np.logspace(np.log10(mis),np.log10(mas),n_points)
        xbase = np.log10(self.xgrid)
        ybase = np.zeros(n_points)
        y2base = np.zeros(n_points)
        self.pbase = np.zeros(n_points)
        for i in range(0,self.number) :
            templist = np.array(self.xlist[i])
#             inte = interp1d(np.log10(self.xlist[i]),np.log10(self.ylist[i]),fill_value=(1,1))
#             mask = (self.xmisloglist[i] < self.xgrid) * (self.xgrid < self.xmaslist[i])
#             pimp = np.power(10.,inte(xbase))
            mask = (self.xmisloglist[i] < self.xgrid) * (self.xgrid < self.xmaslist[i])
            pimp = np.power(10.,np.interp(xbase,np.log10(self.xlist[i]),np.log10(self.ylist[i]),left=1,right=1))
            ybase += pimp*mask
            y2base += (pimp**2)*mask
            self.pbase += mask
        self.yave = ybase/self.pbase
        self.ystd = np.sqrt((y2base/self.pbase)-(self.yave)**2)








# ------------------------------------------------------------------------------
#   cuties
# ------------------------------------------------------------------------------


def plotbin(f):
    plt.figure(figsize=(10,10))
    plt.suptitle('%s t=%.2f (%s)'%(f,gm(f,'time'),gm(f,'ftype')))
    buf = thetabin(f,1000)
    plt.subplot(221,title='theta')
    plt.semilogy(buf[:,0],buf[:,1],'orange',linewidth=0.6,marker='.',markersize=0.1)
    # plt.semilogy(bins,np.exp(-np.abs(bins/0.02))+.6e-8/(0.001+bins**2))

    plt.subplot(222,title='log theta^2')
    buf = lt2bin(f,100)
    plt.loglog(buf[:,0],buf[:,1],'r',linewidth=0.6,marker='.',markersize=0.1)

    plt.subplot(223,title='contrast')
    buf = conbin(f,100)
    plt.loglog(buf[:,0],buf[:,1],'g',linewidth=0.6,marker='.',markersize=0.1)

    plt.subplot(224,title='rho')
    if gm(f,'ftype')=='Saxion':
        buf = rhobin(f,100)
        plt.semilogy(buf[:,0],buf[:,1],linewidth=0.6,marker='.',markersize=0.1) ;
        plt.axvline(x=1,linewidth=0.2);    plt.axvline(x=1+gm(f,'shift'),linewidth=0.5)
    plt.show()
    return;


# ------------------------------------------------------------------------------
#   load and manage sample.txt files
# ------------------------------------------------------------------------------






#   finds the sample.txt file and returns arrayS and arrayA
#   can be problematic if any is empty

def loadsample(address='./'):
    mdir = findmdir(address)
    odir = mdir[:-2]
    fina = odir+'./sample.txt'
    if os.path.exists(fina):

        with open(fina) as f:
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
            #print('1 - lines/SAX/AX ',len(lines),l10,l5)
            #if l5 > 0 :
            arrayA = np.genfromtxt(fina,skip_header=l10)
            #    arrayS = np.empty([])
            #if l10 > 0 :
            arrayS = np.genfromtxt(fina,skip_footer=l5)
            #    arrayA = arrayS = np.empty([])
            #print('2 - lines/SAX/AX ',len(lines),len(arrayS),len(arrayA))
        #axiondata = len(arrayA) >0

#         if l10 >1 :
#             ztab1 = arrayS[:,0]
#             Thtab1 = np.arctan2(arrayS[:,4],arrayS[:,3])
#             Rhtab1 = np.sqrt(arrayS[:,3]**2 + arrayS[:,4]**2)/ztab1
#             VThtab1 = Thtab1 + (arrayS[:,3]*arrayS[:,6]-arrayS[:,4]*arrayS[:,5])/(ztab1*Rhtab1**2)
#         if axiondata:
#             ztab2 = arrayA[:,0]
#             Thtab2 = arrayA[:,2]/ztab2
#             VThtab2 = arrayA[:,3]
    # if (l10 == 0) and (l5 == 0):
    #     print('[loadsample] no output!')
    #     return ;
    # if (l10 > 0) and (l5 == 0):
    #     print('[loadsample] return only arrayS')
    #     return arrayS ;
    # if (l10 == 0) and (l5 > 0):
    #     print('[loadsample] return only arrayA')
    #     return arrayA ;
    return arrayS, arrayA;






#   Loads a sample.txt file and returns tables with data

def axev(address='./'):
    arrayS, arrayA = loadsample(address)
    lS = len(arrayS)
    lA = len(arrayA)
    ou = 0
    if lS >1 :
        ztab1 = arrayS[:,0]
        Thtab1 = np.arctan2(arrayS[:,4],arrayS[:,3])
        Rhtab1 = np.sqrt(arrayS[:,3]**2 + arrayS[:,4]**2)/ztab1
        # note that this is unshifted velocity!
        VThtab1 = Thtab1 + (arrayS[:,3]*arrayS[:,6]-arrayS[:,4]*arrayS[:,5])/(ztab1*Rhtab1**2)
        #
        strings = arrayS[:,7]
        fix = [[ztab1[0],strings[0]]]
        i = 0
        for i in range(0, len(ztab1)-1):
            if strings[i] != strings[i+1]:
                fix.append([ztab1[i+1],strings[i+1]])
        stringo = np.asarray(fix)

        ou += 1
    if len(arrayA) >0 :
        ztab2 = arrayA[:,0]
        Thtab2 = arrayA[:,2]/ztab2
        VThtab2 = arrayA[:,3]
        ou += 2

    if   ou == 3 :
        print('Saxion + Axion (ztab1, Thtab1, Rhtab1, VThtab1, stringo, ztab2, Thtab2, VThtab2)')
        return ztab1, Thtab1, Rhtab1, VThtab1, stringo, ztab2, Thtab2, VThtab2 ;
    elif ou == 1 :
        print('Saxion  (ztab1, Thtab1, Rhtab1, VThtab1, stringo)')
        return ztab1, Thtab1, Rhtab1, VThtab1, stringo ;
    elif ou == 2 :
        print('Axion (ztab2, Thtab2, VThtab2)')
        return ztab2, Thtab2, VThtab2 ;
    else :
        return ;






def axevS(address='./sample.txt'):
    with open(address) as f:
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
    arrayA = np.genfromtxt(address,skip_header=l10)
    arrayS = np.genfromtxt(address,skip_footer=l5)

    lS = len(arrayS)
    lA = len(arrayA)
    ou = 0
    if lS >1 :
        ztab1 = arrayS[:,0]
        Thtab1 = np.arctan2(arrayS[:,4],arrayS[:,3])
        Rhtab1 = np.sqrt(arrayS[:,3]**2 + arrayS[:,4]**2)/ztab1
        # note that this is unshifted velocity!
        VThtab1 = Thtab1 + (arrayS[:,3]*arrayS[:,6]-arrayS[:,4]*arrayS[:,5])/(ztab1*Rhtab1**2)
        #
        strings = arrayS[:,7]
        fix = [[ztab1[0],strings[0]]]
        i = 0
        for i in range(0, len(ztab1)-1):
            if strings[i] != strings[i+1]:
                fix.append([ztab1[i+1],strings[i+1]])
        stringo = np.asarray(fix)

        ou += 1
    if len(arrayA) >0 :
        ztab2 = arrayA[:,0]
        Thtab2 = arrayA[:,2]/ztab2
        VThtab2 = arrayA[:,3]
        ou += 2

    if   ou == 3 :
        print('Saxion + Axion (ztab1, Thtab1, Rhtab1, VThtab1, stringo, ztab2, Thtab2, VThtab2)')
        return ztab1, Thtab1, Rhtab1, VThtab1, stringo, ztab2, Thtab2, VThtab2 ;
    elif ou == 1 :
        print('Saxion  (ztab1, Thtab1, Rhtab1, VThtab1, stringo)')
        return ztab1, Thtab1, Rhtab1, VThtab1, stringo ;
    elif ou == 2 :
        print('Axion (ztab2, Thtab2, VThtab2)')
        return ztab2, Thtab2, VThtab2 ;
    else :
        return ;

def xit(logi):
    return (249.48 + 38.8431*logi + 1086.06* logi*logi)/(21775.3 + 3665.11*logi)

def xitt(logi):
    return 0.21738*np.log(1 + 0.1564394*np.exp(logi))


#   qt plot!

# def axevplot(arrayS, arrayA):
#     lS = len(arrayS)
#     lA = len(arrayA)
#
#     if l10 >1 :
#         ztab1 = arrayS[:,0]
#     #UNSHIFTED
#         Thtab1 = np.arctan2(arrayS[:,4],arrayS[:,3])
#         Rhtab1 = arrayS[:,3]**2 + arrayS[:,4]**2
#         #theta_z
#         VThtab1 = (arrayS[:,3]*arrayS[:,6]-arrayS[:,4]*arrayS[:,5])/(Rhtab1)
#         #ctheta_z
#         #VThtab1 = ztab1*(arrayS[:,3]*arrayS[:,6]-arrayS[:,4]*arrayS[:,5])/(Rhtab1) + Thtab1
#         Rhtab1 = np.sqrt(Rhtab1)/ztab1
#
#     #SHIFTED
#         arrayS[:,3] = arrayS[:,3]-ztab1*arrayS[:,9]
#         Thtab1_shift = np.arctan2(arrayS[:,4],arrayS[:,3])
#         Rhtab1_shift = arrayS[:,3]**2 + arrayS[:,4]**2
#         #theta_z
#         VThtab1_shift = (arrayS[:,3]*arrayS[:,6]-arrayS[:,4]*arrayS[:,5])/(Rhtab1_shift)
#         #ctheta_z
#         #VThtab1_shift = ztab1*(arrayS[:,3]*arrayS[:,6]-arrayS[:,4]*arrayS[:,5])/(Rhtab1_shift) + Thtab1_shift
#         Rhtab1_shift = np.sqrt(Rhtab1_shift)/ztab1
#
#     #STRINGS
#         strings = arrayS[:,7]
#         fix = [[ztab1[0],strings[0]]]
#         i = 0
#         for i in range(0, len(ztab1)-1):
#             if strings[i] != strings[i+1]:
#                 fix.append([ztab1[i+1],strings[i+1]])
#         stringo = np.asarray(fix)
#
#         co = (sizeL/sizeN)*(3/2)*(1/sizeL)**3
#     if len(arrayA) >0 :
#         ztab2 = arrayA[:,0]
#         Thtab2 = arrayA[:,2]/ztab2
#         #theta_z
#         VThtab2 = (arrayA[:,3]-Thtab2)/ztab2
#         #ctheta_z
#         #VThtab2 = arrayA[:,3]
#
#     #PLOT
#     from pyqtgraph.Qt import QtGui, QtCore
#     import pyqtgraph as pg
#
#     #QtGui.QApplication.setGraphicsSystem('raster')
#     app = QtGui.QApplication([])
#     #mw = QtGui.QMainWindow()
#     #mw.resize(800,800)
#
#     win = pg.GraphicsWindow(title="Evolution idx=0")
#     win.resize(1000,600)
#     win.setWindowTitle('jaxions evolution')
#
#     # Enable antialiasing for prettier plots
#     pg.setConfigOptions(antialias=True)
#
#     p1 = win.addPlot(title=r'theta evolution')
#
#     # p1.PlotItem.('left',r'$\theta$')
#     if l10 >1 :
#         p1.plot(ztab1,Thtab1,pen=(100,100,100))
#         p1.plot(ztab1,Thtab1_shift,pen=(255,255,255))
#     if axiondata:
#         p1.plot(ztab2,Thtab2,pen=(255,255,0))
#     p1.setLabel('left',text='theta')
#     p1.setLabel('bottom',text='time')
#
#
#     p2 = win.addPlot(title=r'theta_t evolution')
#
#     # p1.PlotItem.('left',r'$\theta$')
#     if l10 >1 :
#         p2.plot(ztab1,VThtab1,pen=(100,100,100))
#         p2.plot(ztab1,VThtab1_shift,pen=(255,255,255))
#     if axiondata:
#         p2.plot(ztab2,VThtab2,pen=(255,255,0))
#     p2.setLabel('left',text='theta_t')
#     p2.setLabel('bottom',text='time')
#
#
#     if l10 >1 :
#         win.nextRow()
#
#         p3 = win.addPlot(title=r'rho evolution')
#         p3.plot(ztab1,Rhtab1,pen=(200,0,0),name='unshifted')
#         p3.plot(ztab1,Rhtab1-arrayS[:,9],pen=(100,100,100),name='unshifted')
#         p3.plot(ztab1,Rhtab1_shift,pen=(255,255,255),name='shifted')
#
#         p3.setLabel('left',text='rho/v')
#         p3.setLabel('bottom',text='time')
#
#         p4 = win.addPlot(title=r'string evolution')
#
#         # p1.PlotItem.('left',r'$\theta$')
#
#         p4.plot(stringo[1:,0],co*stringo[1:,1]*stringo[1:,0]**2,pen=(255,255,255),symbolBrush=(153,255,204))
#         p4.setLabel('left',text='Length/Volume')
#         p4.setLabel('bottom',text='time')
#
#
#     ## Start Qt event loop unless running in interactive mode or using pyside.
#     if __name__ == '__main__':
#         import sys
#         if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#             QtGui.QApplication.instance().exec_()
#     return ;
