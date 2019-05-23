#!/usr/bin/python3

import numpy as np
import math
import pickle
from pyaxions import jaxions as pa
from numpy.linalg import inv
from scipy.optimize import curve_fit






# ------------------------------------------------------------------------------
#   Analytical fit
# ------------------------------------------------------------------------------






#   fitting function as cubic polynomial
def func(x, a0, a1, a2, a3):
    return a0 + a1*x + a2*(x**2) + a3*(x**3)

#   derivative of the fitting function
def dfunc(x, a0, a1, a2, a3):
    return a1 + 2*a2*x + 3*a3*(x**2)






#   perform analytical fit for each mode
#   options for spmask:
#     spmask = 'nomask' -> Fields unmasked
#     spmask = 'Red' -> Red-Gauss (default)
#     spmask = 'Vi' -> Masked with rho/v
#     spmask = 'Vi2' -> Masked with (rho/v)^2

class fitP:
    def __init__(self, mfiles, spmask='Red', lltype='Z2'):
        self.sizeN = pa.gm(mfiles[0],'Size')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda')
        self.nm = pa.gm(mfiles[0],'nmodelist')
        self.avek = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm)*(2*math.pi/self.sizeL)
        # identify modes less than N/2
        self.k_below = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm) <= self.sizeN/2
        # create lists of the evolution of axion number spectrum (kinetic part)
        ttab = []
        nsptab = []
        for meas in mfiles:
            if pa.gm(meas,'nsp?'):
                t = pa.gm(meas,'time')
                if spmask == 'nomask':
                    print('nomask option is not supported now...')
                elif spmask == 'Red':
                    binK = pa.gm(meas,'nspK_Red')
                elif spmask == 'Vi':
                    binK = pa.gm(meas,'nspK_Vi')
                elif spmask == 'Vi2':
                    binK = pa.gm(meas,'nspK_Vi2')
                else:
                    print('Wrong option for spmask!')
                nsp = (self.avek**2)*binK/(2*t*(math.pi**2)*self.nm)
                ttab.append(t)
                nsptab.append(nsp)
        self.t = np.array(ttab)
        self.nsp = np.array(nsptab)
        if lltype == 'Z2':
            self.log = np.log(self.t*self.sizeN*self.msa/self.sizeL)
        elif lltype == 'fixed':
            self.log = np.log(math.sqrt(2.*self.LL)*self.t**2)
        # cutoff time (chosen as log(ms/H) = 4)
        istart = np.abs(self.log - 4.).argmin()
        self.param = []
        self.paramv = []
        self.listihc = []
        self.dataP = []
        self.datalog = []
        # transpose
        nspT = np.transpose(self.nsp)
        iterkmax = len(self.avek[self.k_below])
        for ik in range(iterkmax):
            print('\rfit: k = %.2f, %d/%d'%(self.avek[ik],ik+1,iterkmax),end="")
            ihc = np.abs(self.avek[ik]*self.t - 2*math.pi).argmin() # save the time index corresponding to the horizon crossing
            xdata = self.log[istart:]
            ydata = self.t[istart:]*nspT[ik,istart:]
            Nparam = 4 # number of parameters for the fitting function
            if len(xdata) >= Nparam and not ik == 0:
                popt, pcov = curve_fit(func, xdata, ydata, maxfev = 20000)
            else:
                popt = [np.nan]*(Nparam)
                pcov = [np.nan]*(Nparam)
            self.param.append(popt)
            self.paramv.append(pcov)
            self.dataP.append(ydata)
            self.datalog.append(xdata)
            self.listihc.append(ihc)
        print("")
        self.dataP = np.array(self.dataP)
        self.datalog = np.array(self.datalog)






# ------------------------------------------------------------------------------
#   Instantaneous spectrum
# ------------------------------------------------------------------------------

#   options for spmask:
#     spmask = 'nomask' -> Fields unmasked
#     spmask = 'Red' -> Red-Gauss (default)
#     spmask = 'Vi' -> Masked with rho/v
#     spmask = 'Vi2' -> Masked with (rho/v)^2

#   calculate instantaneous spectrum based on analytical fit
class inspA:
    def __init__(self, mfiles, spmask='Red', lltype='Z2'):
        fitp = fitP(mfiles,spmask)
        self.lltype = lltype
        self.sizeN = fitp.sizeN
        self.sizeL = fitp.sizeL
        self.msa = fitp.msa
        self.LL = fitp.LL
        self.nm = fitp.nm
        self.avek = fitp.avek
        self.k_below = fitp.k_below
        self.F = [] # instantaneous spectrum F
        self.Fnorm = [] # normalization factor of F
        self.t = [] # time
        self.x = [] # x-axis (k/RH)
        istart = np.abs(fitp.log - 4.).argmin()
        iterkmax = len(self.avek[self.k_below])
        for id in range(len(fitp.t)):
            t = fitp.t[id]
            log = fitp.log[id]
            print('\rcalc F: %d/%d, log = %.2f'%(id+1,len(fitp.log),log),end="")
            if id >= istart:
                Fbinbuf = []
                x = []
                for ik in range(iterkmax):
                    # calculate only modes inside the horizon
                    if id >= fitp.listihc[ik]:
                        l = fitp.param[ik]
                        Fval = dfunc(log,*l)/(t**5)
                        if not np.isnan(Fval):
                            Fbinbuf.append(Fval)
                            x.append(self.avek[ik]*t)
                Fbinbuf = np.array(Fbinbuf)
                # normalize
                dx = np.gradient(x)
                Fdx = Fbinbuf*dx
                self.F.append(Fbinbuf/Fdx.sum())
                self.Fnorm.append(Fdx.sum())
                self.x.append(np.array(x))
                self.t.append(t)
        print("")
        self.F = np.array(self.F)
        self.Fnorm = np.array(self.Fnorm)
        self.x = np.array(self.x) # x = k/RH
        self.t = np.array(self.t) # time
        if lltype == 'Z2':
            self.log = np.log(self.t*self.sizeN*self.msa/self.sizeL)
        elif lltype == 'fixed':
            self.log = np.log(math.sqrt(2.*self.LL)*self.t**2)






#   calculate instantaneous spectrum based on backward difference
class inspB:
    def __init__(self, mfiles, spmask='Red', lltype='Z2'):
        self.lltype = lltype
        self.sizeN = pa.gm(mfiles[0],'sizeN')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda')
        self.nm = pa.gm(mfiles[0],'nmodelist')
        self.avek = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm)*(2*math.pi/self.sizeL)
        # identify modes less than N/2
        self.k_below = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm) <= self.sizeN/2
        self.F = [] # instantaneous spectrum F
        self.Fnorm = [] # normalization factor of F
        self.t = [] # time
        self.x = [] # x-axis (k/RH)
        for meas in mfiles:
            if meas == mfiles[0]:
                tp = pa.gm(meas,'time')
                if spmask == 'nomask':
                    binK = pa.gm(meas,'nspK')
                elif spmask == 'Red':
                    binK = pa.gm(meas,'nspK_Red')
                elif spmask == 'Vi':
                    binK = pa.gm(meas,'nspK_Vi')
                elif spmask == 'Vi2':
                    binK = pa.gm(meas,'nspK_Vi2')
                else:
                    print('Wrong option for spmask!')
                spp = (self.avek**2)*binK/((math.pi**2)*self.nm)
            elif pa.gm(meas,'nsp?'):
                t = pa.gm(meas,'time')
                dt = t - tp
                if spmask == 'nomask':
                    binK = pa.gm(meas,'nspK')
                elif spmask == 'Red':
                    binK = pa.gm(meas,'nspK_Red')
                elif spmask == 'Vi':
                    binK = pa.gm(meas,'nspK_Vi')
                elif spmask == 'Vi2':
                    binK = pa.gm(meas,'nspK_Vi2')
                else:
                    print('Wrong option for spmask!')
                sp = (self.avek**2)*binK/((math.pi**2)*self.nm)
                t = tp + dt/2
                x = self.avek*t
                diff = (sp - spp)/((t**4)*dt)
                # normalize
                dx = np.gradient(x)
                Fdx = (diff*dx)[self.k_below]
                self.F.append(diff/Fdx.sum())
                self.Fnorm.append(Fdx.sum())
                self.t.append(t)
                self.x.append(x)
                tp = pa.gm(meas,'time')
                spp = sp
        self.F = np.array(self.F)
        self.Fnorm = np.array(self.Fnorm)
        self.x = np.array(self.x) # x = k/RH
        self.t = np.array(self.t) # time
        if lltype == 'Z2':
            self.log = np.log(self.t*self.sizeN*self.msa/self.sizeL)
        elif lltype == 'fixed':
            self.log = np.log(math.sqrt(2.*self.LL)*self.t**2)






#   calculate instantaneous spectrum based on central difference
class inspC:
    def __init__(self, mfiles, spmask='Red', lltype='Z2'):
        self.lltype = lltype
        self.sizeN = pa.gm(mfiles[0],'sizeN')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda')
        self.nm = pa.gm(mfiles[0],'nmodelist')
        self.avek = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm)*(2*math.pi/self.sizeL)
        # identify modes less than N/2
        self.k_below = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm) <= self.sizeN/2
        self.F = [] # instantaneous spectrum F
        self.Fnorm = [] # normalization factor of F
        self.t = [] # time
        self.x = [] # x-axis (k/RH)
        msplist = [mf for mf in mfiles if pa.gm(mf,'nsp?')]
        for id in range(len(msplist)):
            if (id != 0) and (id != len(msplist)-1):
                t1 = pa.gm(msplist[id-1],'time')
                t2 = pa.gm(msplist[id+1],'time')
                if spmask == 'nomask':
                    binK1 = pa.gm(msplist[id-1],'nspK')
                    binK2 = pa.gm(msplist[id+1],'nspK')
                elif spmask == 'Red':
                    binK1 = pa.gm(msplist[id-1],'nspK_Red')
                    binK2 = pa.gm(msplist[id+1],'nspK_Red')
                elif spmask == 'Vi':
                    binK1 = pa.gm(msplist[id-1],'nspK_Vi')
                    binK2 = pa.gm(msplist[id+1],'nspK_Vi')
                elif spmask == 'Vi2':
                    binK1 = pa.gm(msplist[id-1],'nspK_Vi2')
                    binK2 = pa.gm(msplist[id+1],'nspK_Vi2')
                else:
                    print('Wrong option for spmask!')
                sp1 = (self.avek**2)*binK1/((math.pi**2)*self.nm)
                sp2 = (self.avek**2)*binK2/((math.pi**2)*self.nm)
                t = pa.gm(msplist[id],'time')
                x = self.avek*t
                dt = t2 - t1
                diff = (sp2 - sp1)/((t**4)*dt)
                # normalize
                dx = np.gradient(x)
                Fdx = (diff*dx)[self.k_below]
                self.F.append(diff/Fdx.sum())
                self.Fnorm.append(Fdx.sum())
                self.t.append(t)
                self.x.append(x)
        self.F = np.array(self.F)
        self.Fnorm = np.array(self.Fnorm)
        self.x = np.array(self.x) # x = k/RH
        self.t = np.array(self.t) # time
        if lltype == 'Z2':
            self.log = np.log(self.t*self.sizeN*self.msa/self.sizeL)
        elif lltype == 'fixed':
            self.log = np.log(math.sqrt(2.*self.LL)*self.t**2)





#   take ensemble average of the instantaneous spectra
#   assuming a list of inspA/inspB/inspC class objects
class inspave:
    def __init__(self, insplist):
        Nreal = len(insplist) # number of realizations
        self.lltype = insplist[0].lltype
        self.sizeN = insplist[0].sizeN
        self.sizeL = insplist[0].sizeL
        self.msa = insplist[0].msa
        self.LL = insplist[0].LL
        self.nm = insplist[0].nm
        self.avek = insplist[0].avek
        self.k_below = insplist[0].k_below
        self.x = insplist[0].x
        self.t = insplist[0].t
        self.log = insplist[0].log
        self.F = []
        self.dF = []
        self.Fnorm = []
        for id in range(len(insplist[0].t)):
            F = [0]*(len(self.x[id]))
            Fsq = [0]*(len(self.x[id]))
            Fnorm = 0
            for insp in insplist:
                F += insp.F[id]
                Fsq += np.square(insp.F[id])
                Fnorm += insp.Fnorm[id]
            F = F/Nreal
            Fsq = Fsq/Nreal - F*F
            Fnorm = Fnorm/Nreal
            self.F.append(F)
            self.dF.append(np.sqrt(Fsq))
            self.Fnorm.append(Fnorm)
            print('\r%d/%d, log = %.2f'%(id+1,len(insplist[0].t),insplist[0].log[id]),end="")
        print("")
        self.F = np.array(self.F)
        self.dF = np.array(self.dF)
        self.Fnorm = np.array(self.Fnorm)






#   rebin data of F(x) such that bins are homogeneous in log(x)
#   fixed data points (nbin) within given interval [cmin,cmax*(ms/H)]
#   assuming input as an inspave class object
class rebinF:
    def __init__(self, inspave, nbin, cmin, cmax):
        self.lltype = inspave.lltype
        self.sizeN = inspave.sizeN
        self.sizeL = inspave.sizeL
        self.msa = inspave.msa
        self.LL = inspave.LL
        self.t = inspave.t
        self.log = inspave.log
        self.xbin = []
        self.inspmbin = []
        self.nmbin = []
        self.xwr = []
        self.xlim = []
        for id in range(len(inspave.t)):
            print('\r%d/%d, log = %.2f'%(id+1,len(inspave.t),inspave.log[id]),end="")
            if self.lltype == 'Z2':
                msoverH = inspave.t[id]*inspave.sizeN*inspave.msa/inspave.sizeL
            elif self.lltype == 'fixed':
                msoverH = math.sqrt(2.*self.LL)*(inspave.t[id]**2)
            x = inspave.x[id]
            inspmtab = inspave.F[id]
            xmin = cmin
            xmax = msoverH*cmax
            x_within_range = ((x > xmin) & (x < xmax))
            xlim = x[x_within_range]
            inspmlim = inspmtab[x_within_range]
            # do not rebin if number of data points is less than nbin
            if len(xlim) < nbin:
                #print(r' number of data points (%d) is less than nbin (%d)! (log = %.2f)'%(len(xlim),nbin,inspave.log[id]))
                xbin = xlim
                inspmbin = inspmlim
                nmbin = [1 for i in range(len(xlim))]
            else:
                # prepare for rebin
                lnx = np.log(xlim)
                minlnx = np.min(lnx)
                maxlnx = np.max(lnx)
                dlnx = (maxlnx - minlnx)/nbin
                xbinbuf = []
                inspmbinbuf = []
                nmbinbuf = []
                # rebin
                for i in range(nbin):
                    lnxstart = minlnx+dlnx*i
                    lnxend = minlnx+dlnx*(i+1)
                    if i == 0:
                        x_in_bin = ((lnx >= lnxstart) & (lnx <= lnxend))
                    else:
                        x_in_bin = ((lnx > lnxstart) & (lnx <= lnxend))
                    xave = np.mean(xlim[x_in_bin])
                    mave = np.mean(inspmlim[x_in_bin])
                    if not np.isnan(xave):
                        xbinbuf.append(xave)
                        inspmbinbuf.append(mave)
                        nmbinbuf.append(len(xlim[x_in_bin]))
                    else:
                        lnxlast = lnxend # save the last boundary of empty bin
                # if the actual number of bins is less than the specified value of nbin,
                # do not use a homogeneous log bin for smaller k, and rebin higher k
                # until the number of bin becomes nbin.
                if not len(xbinbuf) == nbin:
                    #iloop = 0
                    while len(xbinbuf) < nbin:
                        #print('%d-th while loop lnxlast = %f, datalength = %d'%(iloop+1,lnxlast,len(xbinbuf)))
                        #iloop = iloop + 1
                        lnxleft = np.array([ele for ele in lnx if ele <= lnxlast])
                        xbinbuf = np.exp(lnxleft)
                        inspmbinbuf = inspmlim[:len(xbinbuf)]
                        nmbinbuf = [1 for ind in range(len(xbinbuf))]
                        naux = len(xbinbuf)
                        lnxlastre = lnxlast
                        dlnxre = (maxlnx - lnxlastre)/(nbin-naux)
                        for i in range(nbin-naux):
                            lnxstart = lnxlastre+dlnxre*i
                            lnxend = lnxlastre+dlnxre*(i+1)
                            x_in_bin = ((lnx > lnxstart) & (lnx <= lnxend))
                            xave = np.mean(xlim[x_in_bin])
                            mave = np.mean(inspmlim[x_in_bin])
                            if not np.isnan(xave):
                                xbinbuf = np.append(xbinbuf,xave)
                                inspmbinbuf = np.append(inspmbinbuf,mave)
                                nmbinbuf = np.append(nmbinbuf,len(xlim[x_in_bin]))
                            else:
                                lnxlast = lnxend # save the last boundary of empty bin
                    #print("homogeneous bin was not possible! (%d/%d, log = %.2f)"%(id+1,len(inspave.t),inspave.log[id]))
                    #print("no rebin for x < %f (%d points) and rebin for x > %f (%d points)"%(math.exp(lnxlast),naux,math.exp(lnxlast),nbin-naux))
                xbin = np.array(xbinbuf)
                inspmbin = np.array(inspmbinbuf)
                nmbin = np.array(nmbinbuf)
                # end of rebin
            self.xbin.append(xbin)
            self.inspmbin.append(inspmbin)
            self.nmbin.append(nmbin)
            self.xwr.append(x_within_range)
            self.xlim.append(xlim)
            # end of id loop
        print("")
        self.xbin = np.array(self.xbin)
        self.inspmbin = np.array(self.inspmbin)
        self.nmbin = np.array(self.nmbin)
        self.xlim = np.array(self.xlim)






#   save the data of instantaneous spectra as pickle files
#   assuming input as an inspave class object
def saveF(inspave, name='./F'):
    yname = name + '_y.pickle'
    dyname = name + '_dy.pickle'
    xname = name + '_x.pickle'
    logname = name + '_log.pickle'
    nname = name + '_Fnorm.pickle'
    with open(yname,'wb') as wy:
        pickle.dump(inspave.F, wy)
    with open(dyname,'wb') as wdy:
        pickle.dump(inspave.dF, wdy)
    with open(xname,'wb') as wx:
        pickle.dump(inspave.x, wx)
    with open(logname,'wb') as wlog:
        pickle.dump(inspave.log, wlog)
    with open(nname,'wb') as wn:
        pickle.dump(inspave.Fnorm, wn)






#   read the data of instantaneous spectra
class readF:
    def __init__(self, name='./F'):
        yname = name + '_y.pickle'
        dyname = name + '_dy.pickle'
        xname = name + '_x.pickle'
        logname = name + '_log.pickle'
        nname = name + '_Fnorm.pickle'
        with open(yname,'rb') as ry:
            self.F = pickle.load(ry)
        with open(dyname,'rb') as rdy:
            self.dF = pickle.load(rdy)
        with open(xname,'rb') as rx:
            self.x = pickle.load(rx)
        with open(logname,'rb') as rlog:
            self.log = pickle.load(rlog)
        with open(nname,'rb') as rn:
            self.Fnorm = pickle.load(rn)






# ------------------------------------------------------------------------------
#   Spectra with the correction matrix
# ------------------------------------------------------------------------------

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
    def __init__(self, mfiles, spmask='Red', lltype='Z2', cor='nocorrection'):
        self.sizeN = pa.gm(mfiles[0],'sizeN')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda')
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
                    self.nsp.append(s0)
                    if cor == 'correction':
                        m = pa.gm(f,'mspM_Red')
                        s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                        self.nspcor.append(s1)
                elif spmask == 'Vi':
                    s0 = pa.gm(f,'nspK_Vi')
                    self.nsp.append(s0)
                    if cor == 'correction':
                        m = pa.gm(f,'mspM_Vi')
                        s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                        self.nspcor.append(s1)
                elif spmask == 'Vi2':
                    s0 = pa.gm(f,'nspK_Vi2')
                    self.nsp.append(s0)
                    if cor == 'correction':
                        m = pa.gm(f,'mspM_Vi2')
                        s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                        self.nspcor.append(s1)
                else:
                    print('Wrong option for spmask!')
                if lltype == 'Z2':
                    logi = math.log(t*self.sizeN*self.msa/self.sizeL)
                elif lltype == 'fixed':
                    logi = math.log(math.sqrt(2.*self.LL)*t**2)
                print('\rbuilt up to log = %.2f'%logi,end="")
        print("")
        self.ttab = np.array(self.ttab)
        if lltype == 'Z2':
            self.logtab = np.log(self.ttab*self.sizeN*self.msa/self.sizeL)
        elif lltype == 'fixed':
            self.logtab = np.log(math.sqrt(2.*self.LL)*self.ttab**2)
        self.nsp = np.array(self.nsp)
        self.nspcor = np.array(self.nspcor)







#   builds the (masked) axion energy spectrum with the correction matrix and outputs the time evolution
#   NOTE: The energy density is evaluated just by muptiplying the kinetic energy by 2.

class espevol:
    def __init__(self, mfiles, spmask='Red', lltype='Z2', cor='nocorrection'):
        self.sizeN = pa.gm(mfiles[0],'sizeN')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda')
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
                    e0 = (self.avek**2)*s0/(t*(math.pi**2)*self.nm)
                    self.esp.append(e0)
                    if cor == 'correction':
                        m = pa.gm(f,'mspM_Red')
                        s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                        e1 = (self.avek**2)*s1/(t*(math.pi**2)*self.nm)
                        self.espcor.append(e1)
                elif spmask == 'Vi':
                    s0 = pa.gm(f,'nspK_Vi')
                    e0 = (self.avek**2)*s0/(t*(math.pi**2)*self.nm)
                    self.esp.append(e0)
                    if cor == 'correction':
                        m = pa.gm(f,'mspM_Vi')
                        s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                        e1 = (self.avek**2)*s1/(t*(math.pi**2)*self.nm)
                        self.espcor.append(e1)
                elif spmask == 'Vi2':
                    s0 = pa.gm(f,'nspK_Vi2')
                    e0 = (self.avek**2)*s0/(t*(math.pi**2)*self.nm)
                    self.esp.append(e0)
                    if cor == 'correction':
                        m = pa.gm(f,'mspM_Vi2')
                        s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                        e1 = (self.avek**2)*s1/(t*(math.pi**2)*self.nm)
                        self.espcor.append(e1)
                else:
                    print('Wrong option for spmask!')
                if lltype == 'Z2':
                    logi = math.log(t*self.sizeN*self.msa/self.sizeL)
                elif lltype == 'fixed':
                    logi = math.log(math.sqrt(2.*self.LL)*t**2)
                print('\rbuilt up to log = %.2f'%logi,end="")
        print("")
        self.ttab = np.array(self.ttab)
        if lltype == 'Z2':
            self.logtab = np.log(self.ttab*self.sizeN*self.msa/self.sizeL)
        elif lltype == 'fixed':
            self.logtab = np.log(math.sqrt(2.*self.LL)*self.ttab**2)
        self.esp = np.array(self.esp)
        self.espcor = np.array(self.espcor)
