#!/usr/bin/python3

import numpy as np
import math
from pyaxions import jaxions as pa
from numpy.linalg import inv


#   builds the (masked) axion kinetic spectrum with the correction matrix
#   needs nm = nmodelist
#   options for spmask:
#     spmask = 'nomask' -> Fields unmasked
#     spmask = 'Red' -> Red-Gauss (default)
#     spmask = 'Vi' -> Masked with rho/v
#     spmask = 'Vi2' -> Masked with (rho/v)^2

def nspcor(mfile, nm, spmask='Red'):
    if spmask == 'nomask':
        return pa.gm(mfile,'nspK')
    elif spmask == 'Red':
        s0 = pa.gm(mfile,'nspK_Red')
        m = pa.gm(mfile,'mspM_Red')
        s1 = (pa.gm(mfile,'L')**3)*np.dot(inv(m),s0/nm)
        return s1
    elif spmask == 'Vi':
        s0 = pa.gm(f,'nspK_Vi')
        m = pa.gm(f,'mspM_Vi')
        s1 = (pa.gm(mfile,'L')**3)*np.dot(inv(m),s0/nm)
        return s1
    elif spmask == 'Vi2':
        s0 = pa.gm(f,'nspK_Vi2')
        m = pa.gm(f,'mspM_Vi2')
        s1 = (pa.gm(mfile,'L')**3)*np.dot(inv(m),s0/nm)
        return s1
    else:
        print('Wrong option for spmask!')






#   builds the (masked) axion kinetic spectrum with the correction matrix and outputs the time evolution

class nspevol:
    def __init__(self, mfiles, spmask='Red'):
        self.sizeN = pa.gm(mfiles[0],'sizeN')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.nm = pa.gm(mfiles[0],'nmodelist')
        self.avek = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm)*(2*math.pi/self.sizeL)
        # identify modes less than N/2
        self.k_below = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm) <= self.sizeN/2
        self.ttab = []
        self.logtab = []
        self.nsp = []
        self.nspcor = [] # corrected spectrum
        for f in mfiles:
            if pa.gm(f,'nsp?'):
                t = pa.gm(f,'time')
                self.ttab.append(t)
                if spmask == 'nomask':
                    self.nsp.append(pa.gm(f,'nspK'))
                elif spmask == 'Red':
                    s0 = pa.gm(f,'nspK_Red')
                    m = pa.gm(f,'mspM_Red')
                    s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                    self.nsp.append(s0)
                    self.nspcor.append(s1)
                elif spmask == 'Vi':
                    s0 = pa.gm(f,'nspK_Vi')
                    m = pa.gm(f,'mspM_Vi')
                    s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                    self.nsp.append(s0)
                    self.nspcor.append(s1)
                elif spmask == 'Vi2':
                    s0 = pa.gm(f,'nspK_Vi2')
                    m = pa.gm(f,'mspM_Vi2')
                    s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                    self.nsp.append(s0)
                    self.nspcor.append(s1)
                else:
                    print('Wrong option for spmask!')
                print('\rbuilt up to log = %.2f'%np.log(t*self.msa*self.sizeN/self.sizeL),end="")
        print("")
        self.ttab = np.array(self.ttab)
        self.logtab = np.log(self.ttab*self.msa*self.sizeN/self.sizeL)
        self.nsp = np.array(self.nsp)
        self.nspcor = np.array(self.nspcor)







#   builds the (masked) axion energy spectrum with the correction matrix and outputs the time evolution
#   NOTE: The energy density is evaluated just by muptiplying the kinetic energy by 2.

class espevol:
    def __init__(self, mfiles, spmask='Red'):
        self.sizeN = pa.gm(mfiles[0],'sizeN')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.nm = pa.gm(mfiles[0],'nmodelist')
        self.avek = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm)*(2*math.pi/self.sizeL)
        # identify modes less than N/2
        self.k_below = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm) <= self.sizeN/2
        self.ttab = []
        self.logtab = []
        self.esp = []
        self.espcor = [] # corrected spectrum
        for f in mfiles:
            if pa.gm(f,'nsp?'):
                t = pa.gm(f,'time')
                self.ttab.append(t)
                if spmask == 'nomask':
                    e0 = (self.avek**2)*pa.gm(f,'nspK')/(t*(math.pi**2)*self.nm)
                    self.esp.append(e0)
                elif spmask == 'Red':
                    s0 = pa.gm(f,'nspK_Red')
                    m = pa.gm(f,'mspM_Red')
                    s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                    e0 = (self.avek**2)*s0/(t*(math.pi**2)*self.nm)
                    e1 = (self.avek**2)*s1/(t*(math.pi**2)*self.nm)
                    self.esp.append(e0)
                    self.espcor.append(e1)
                elif spmask == 'Vi':
                    s0 = pa.gm(f,'nspK_Vi')
                    m = pa.gm(f,'mspM_Vi')
                    s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                    e0 = (self.avek**2)*s0/(t*(math.pi**2)*self.nm)
                    e1 = (self.avek**2)*s1/(t*(math.pi**2)*self.nm)
                    self.esp.append(e0)
                    self.espcor.append(e1)
                elif spmask == 'Vi2':
                    s0 = pa.gm(f,'nspK_Vi2')
                    m = pa.gm(f,'mspM_Vi2')
                    s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                    e0 = (self.avek**2)*s0/(t*(math.pi**2)*self.nm)
                    e1 = (self.avek**2)*s1/(t*(math.pi**2)*self.nm)
                    self.esp.append(e0)
                    self.espcor.append(e1)
                else:
                    print('Wrong option for spmask!')
                print('\rbuilt up to log = %.2f'%np.log(t*self.msa*self.sizeN/self.sizeL),end="")
        print("")
        self.ttab = np.array(self.ttab)
        self.logtab = np.log(self.ttab*self.msa*self.sizeN/self.sizeL)
        self.esp = np.array(self.esp)
        self.espcor = np.array(self.espcor)
