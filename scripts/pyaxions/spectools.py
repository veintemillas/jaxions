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
#   Calculate q
# ------------------------------------------------------------------------------

# class Setq:
#   calculate chi^2 and estimate q for a given time step specified by id
#   use log bin in k (rebinned)
#   fixed data points (nbin) within given interval (xmin,xmax)
#
# Arguments:
#   inspx           x-axis of the instantaneous spectrum
#   inspy           y-axis of the instantaneous spectrum
#   insplog         array for log(m/H)
#   id              time index (index of insp array)
#   xmin            lower limit of the interval used to evaluate q
#   xmax            upper limit of the interval used to evaluate q
#   nbin            number of bins (default 30)
#   qstart          lower limit of q to scan (default 0)
#   qend            upper limit of q to scan (default 1.5)
#   qpoints         number of points to scan (default 5000)
#
# Output:
#   self.chi2q      chi^2 as a function of q
#   self.chi2qc     chi^2 as a function of q (using conservative estimate of sigma based on maximum distance from mean)
#   self.chi2min    minimum value of chi^2
#   self.chi2minc   minimum value of chi^2 (using conservative estimate of sigma based on maximum distance from mean)
#   self.qbest      best fit value of q
#   self.qbestid    index of qbest in qarr = np.linspace(qstart,qend,qpoints)
#   self.qupper     "1 sigma" upper limit of q
#   self.qupperc    "1 sigma" upper limit of q (using conservative estimate of sigma based on maximum distance from mean)
#   self.qlower     "1 sigma" lower limit of q
#   self.qlowerc    "1 sigma" lower limit of q (using conservative estimate of sigma based on maximum distance from mean)
#   self.xbin       x-axis for rebinned instantaneous spectrum F(x) and B(x) = x^q*F(x) where x = k/RH
#   self.Bqbin      array for B(x) = x^q*F(x) with shape [i,j] where i is index of q and j is index of x
#   self.Bfid       1d array for B(x) = x^qfid*F(x) corresponding to a fiducial value of q
#   self.Bsigma     "sigma" to define confidence interval based on average of (Bfid-Bfidmean)^2
#   self.Bsigmac    conservative estimate of "sigma" to define confidence interval based on maximum value of (Bfid-Bfidmean)^2
#   self.inspmbin   rebinned instantaneous spectrum F(x)
#   self.nmbin      number of modes in each bin (currently not used)
#   self.xlim       x within the range specified by (xmin,xmax)
#   self.xwr        flags to identify xlim
#
class Setq:
    def __init__(self, inspx, inspy, insplog, id, xmin, xmax, nbin=30, qstart=0, qend=1.5, qpoints=5000):
        qarr = np.linspace(qstart,qend,qpoints)
        x = inspx[id]
        inspmtab = inspy[id]
        x_within_range = ((x > xmin) & (x < xmax))
        xlim = x[x_within_range]
        inspmlim = inspmtab[x_within_range]
        # do not calculate chi^2 if there are not enough data points
        if len(xlim) < 2:
            #print(r' cannot optimize since number of data points is less than 2! (log = %.2f)'%insplog[id])
            chi2q = [np.inf for q in qarr]
            chi2qc = [np.inf for q in qarr]
            chi2min = np.inf
            chi2minc = np.inf
            qbest = np.nan
            qbestid = np.nan
            qupper = np.nan
            qupperc = np.nan
            qlower = np.nan
            qlowerc = np.nan
            xbin = xlim
            Bqbin = [np.nan for ind in range(len(xlim))]
            Bqfidmean = np.nan
            Bqfidsigma = np.nan
            Bqfidsigmac = np.nan
            inspmbin = inspmlim
            nmbin = [1 for i in range(len(xlim))]
        else:
            # do not rebin if number of data points is less than nbin
            if len(xlim) < nbin:
                #print(r' number of data points (%d) is less than nbin (%d)! (log = %.2f)'%(len(xlim),nbin,insplog[id]))
                xbin = xlim
                inspmbin = inspmlim
                nmbin = [1 for i in range(len(xlim))]
                nbin = len(xlim)
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
                # if actual number of bins is less than specified value of nbin,
                # do not use homogeneous log bin for smaller k, and rebin higher k
                # until number of bin becomes nbin.
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
                    #print("homogeneous bin was not possible! (%d/%d, log = %.2f)"%(id+1,len(insplog),insplog[id]))
                    #print("no rebin for x < %f (%d points) and rebin for x > %f (%d points)"%(math.exp(lnxlast),naux,math.exp(lnxlast),nbin-naux))
                xbin = np.array(xbinbuf)
                inspmbin = np.array(inspmbinbuf)
                nmbin = np.array(nmbinbuf)
                # end of rebin
            # next calculate q
            # Bq is normalized at the centar of specified interval
            xcenter = math.sqrt(xmax*xmin)
            ixc = np.abs(xbin - xcenter).argmin()
            # find a fiducial value of q
            chi2aux = []
            for q in qarr:
                Bqnorm = ((xbin**q)*inspmbin)[ixc]
                Bq = (xbin**q)*inspmbin/Bqnorm
                Bqmean = Bq.sum()/nbin
                Bqsq = np.square(Bq-Bqmean)
                chi2aux.append(Bqsq.sum())
            chi2aux = np.array(chi2aux)
            qfidid = chi2aux.argmin()
            qfid = np.nan if math.isnan(min(chi2aux)) else qarr[qfidid]
            Bqfidnorm = ((xbin**qfid)*inspmbin)[ixc]
            Bqfid = (xbin**qfid)*inspmbin/Bqfidnorm
            Bqfidmean = Bqfid.sum()/nbin
            Bqfidsigma = np.sqrt(np.sum(np.square(Bqfid-Bqfidmean))/nbin)
            Bqfidsigmac = np.max(np.abs(Bqfid-Bqfidmean)) # conservative estimate of sigma based on maximum distance from mean
            # build chi^2 and identify confidence interval
            chi2q = []
            chi2qc = []
            Bqbin = []
            for q in qarr:
                Bqnorm = ((xbin**q)*inspmbin)[ixc]
                Bq = (xbin**q)*inspmbin/Bqnorm
                Bqsq = np.square(Bq-Bqfidmean)/((Bqfidsigma**2)*nbin)
                Bqsqc = np.square(Bq-Bqfidmean)/((Bqfidsigmac**2)*nbin)
                chi2q.append(Bqsq.sum())
                chi2qc.append(Bqsqc.sum())
                Bqbin.append(Bq)
            chi2q = np.array(chi2q)
            chi2qc = np.array(chi2qc)
            chi2min = min(chi2q)
            chi2minc = min(chi2qc)
            qbestid = chi2q.argmin()
            qbest = np.nan if math.isnan(min(chi2q)) else qarr[qbestid]
            # identify error of q
            q_CL  = (chi2q - chi2min) <= 1.
            q_CLc  = (chi2qc - chi2minc) <= 1.
            q1sigma = qarr[q_CL]
            q1sigmac = qarr[q_CLc]
            qupper = np.nanmax(q1sigma)
            qupperc = np.nanmax(q1sigmac)
            qlower = np.nanmin(q1sigma)
            qlowerc = np.nanmin(q1sigmac)
            Bqbin = np.array(Bqbin)
        # end of the case len(xlim) >= 2
        self.chi2q = chi2q
        self.chi2qc = chi2qc
        self.chi2min = chi2min
        self.chi2minc = chi2minc
        self.qbest = qbest
        self.qbestid = qbestid
        self.qupper = qupper
        self.qupperc = qupperc
        self.qlower = qlower
        self.qlowerc = qlowerc
        self.xbin = xbin
        self.Bqbin = Bqbin
        self.Bfid = Bqfidmean
        self.Bsigma = Bqfidsigma
        self.Bsigmac = Bqfidsigmac
        self.inspmbin = inspmbin
        self.nmbin = nmbin
        self.xwr = x_within_range
        self.xlim = xlim






# class Scanq:
#   estimate q at every time step
#   x range is taken as (cxmin,cxmax*(m/H))
#
# Arguments:
#   inspx           x-axis of the instantaneous spectrum
#   inspy           y-axis of the instantaneous spectrum
#   insplog         array for log(m/H)
#   nbin            number of bins (default 30)
#   cxmin           lower limit of the interval (default 30)
#   cxmax           upper limit of the interval specified as cxmax*(m/H) (default 1/6)
#   qstart          lower limit of q to scan (default 0)
#   qend            upper limit of q to scan (default 1.5)
#   qpoints         number of points to scan (default 5000)
#
# Output:
#   self.chi2q      chi^2 as a function of q
#   self.chi2qc     chi^2 as a function of q (using conservative estimate of sigma based on maximum distance from mean)
#   self.chi2min    minimum value of chi^2
#   self.chi2minc   minimum value of chi^2 (using conservative estimate of sigma based on maximum distance from mean)
#   self.qbest      best fit value of q
#   self.qbestid    index of qbest in qarr = np.linspace(qstart,qend,qpoints)
#   self.qupper     "1 sigma" upper limit of q
#   self.qupperc    "1 sigma" upper limit of q (using conservative estimate of sigma based on maximum distance from mean)
#   self.qlower     "1 sigma" lower limit of q
#   self.qlowerc    "1 sigma" lower limit of q (using conservative estimate of sigma based on maximum distance from mean)
#   self.xbin       x-axis for rebinned instantaneous spectrum F(x) and B(x) = x^q*F(x) where x = k/RH
#   self.Bqbin      array for B(x) = x^q*F(x) with shape [id,i,j] where id is index of time step, i is index of q, and j is index of x
#   self.Bfid       1d array for B(x) = x^qfid*F(x) corresponding to a fiducial value of q
#   self.Bsigma     "sigma" to define confidence interval based on average of (Bfid-Bfidmean)^2
#   self.Bsigmac    conservative estimate of "sigma" to define confidence interval based on maximum value of (Bfid-Bfidmean)^2
#   self.inspmbin   rebinned instantaneous spectrum F(x)
#   self.nmbin      number of modes in each bin (currently not used)
#   self.qarr       array for q
#   self.logtab     array for log(m/H)
#
class Scanq:
    def __init__(self, inspx, inspy, insplog, nbin=30, cxmin=30., cxmax=1/6., qstart=0, qend=1.5, qpoints=5000):
        qarr = np.linspace(qstart,qend,qpoints)
        self.chi2q = []
        self.chi2qc = []
        self.chi2min = []
        self.chi2minc = []
        self.qbest = []
        self.qbestid = []
        self.qupper = []
        self.qupperc = []
        self.qlower = []
        self.qlowerc = []
        self.xbin = []
        self.Bqbin = []
        self.Bfid = []
        self.Bsigma = []
        self.Bsigmac = []
        self.inspmbin = []
        self.nmbin = []
        for id in range(len(insplog)):
            print('\r%d/%d, log = %.2f'%(id+1,len(insplog),insplog[id]),end="")
            msoverH = math.exp(insplog[id])
            xmin = cxmin
            xmax = cxmax*msoverH
            sqt = Setq(inspx,inspy,insplog,id,xmin,xmax,nbin,qstart,qend,qpoints)
            self.chi2q.append(sqt.chi2q)
            self.chi2qc.append(sqt.chi2qc)
            self.chi2min.append(sqt.chi2min)
            self.chi2minc.append(sqt.chi2minc)
            self.qbest.append(sqt.qbest)
            self.qbestid.append(sqt.qbestid)
            self.qupper.append(sqt.qupper)
            self.qupperc.append(sqt.qupperc)
            self.qlower.append(sqt.qlower)
            self.qlowerc.append(sqt.qlowerc)
            self.xbin.append(sqt.xbin)
            self.Bqbin.append(sqt.Bqbin)
            self.Bfid.append(sqt.Bfid)
            self.Bsigma.append(sqt.Bsigma)
            self.Bsigmac.append(sqt.Bsigmac)
            self.inspmbin.append(sqt.inspmbin)
            self.nmbin.append(sqt.nmbin)
        print("")
        self.chi2min = np.array(self.chi2min)
        self.chi2minc = np.array(self.chi2minc)
        self.qbest = np.array(self.qbest)
        self.qupper = np.array(self.qupper)
        self.qupperc = np.array(self.qupperc)
        self.qlower = np.array(self.qlower)
        self.qlowerc = np.array(self.qlowerc)
        self.Bfid = np.array(self.Bfid)
        self.Bsigma = np.array(self.Bsigma)
        self.Bsigmac = np.array(self.Bsigmac)
        self.qarr = qarr
        self.logtab = insplog
        self.cxmaxopt = cxmax





# class Scanqopt:
#   estimate q at every time step
#   x range is taken as (cxmin,cxmax*(m/H))
#   optimize xmax such that the error of q takes the smallest value :
#       scan over cxmax from cxmaxstart to cxmaxend
#       and use the value of cxmax that leads to the smallest value of sigma_q
#
# Arguments:
#   inspx           x-axis of the instantaneous spectrum
#   inspy           y-axis of the instantaneous spectrum
#   insplog         array for log(m/H)
#   nbin            number of bins (default 30)
#   cxmin           lower limit of the interval (default 30)
#   cxmaxstart      lower limit of cxmax to scan (default 0.15)
#   cxmaxend        upper limit of cxmax to scan (default 0.5)
#   cxmaxpoints     number of points to scan over cxmax (default 200)
#   qstart          lower limit of q to scan (default 0)
#   qend            upper limit of q to scan (default 1.5)
#   qpoints         number of points to scan (default 5000)
#
# Output:
#   self.chi2q      chi^2 as a function of q
#   self.chi2qc     chi^2 as a function of q (using conservative estimate of sigma based on maximum distance from mean)
#   self.chi2min    minimum value of chi^2
#   self.chi2minc   minimum value of chi^2 (using conservative estimate of sigma based on maximum distance from mean)
#   self.qbest      best fit value of q
#   self.qbestid    index of qbest in qarr = np.linspace(qstart,qend,qpoints)
#   self.qupper     "1 sigma" upper limit of q
#   self.qupperc    "1 sigma" upper limit of q (using conservative estimate of sigma based on maximum distance from mean)
#   self.qlower     "1 sigma" lower limit of q
#   self.qlowerc    "1 sigma" lower limit of q (using conservative estimate of sigma based on maximum distance from mean)
#   self.xbin       x-axis for rebinned instantaneous spectrum F(x) and B(x) = x^q*F(x) where x = k/RH
#   self.Bqbin      array for B(x) = x^q*F(x) with shape [id,i,j] where id is index of time step, i is index of q, and j is index of x
#   self.Bfid       1d array for B(x) = x^qfid*F(x) corresponding to a fiducial value of q
#   self.Bsigma     "sigma" to define confidence interval based on average of (Bfid-Bfidmean)^2
#   self.Bsigmac    conservative estimate of "sigma" to define confidence interval based on maximum value of (Bfid-Bfidmean)^2
#   self.inspmbin   rebinned instantaneous spectrum F(x)
#   self.nmbin      number of modes in each bin (currently not used)
#   self.qarr       array for q
#   self.logtab     array for log(m/H)
#   self.cxmaxopt   array for optimized values of cxmax
#
class Scanqopt:
    def __init__(self, inspx, inspy, insplog, nbin=30, cxmin=30., cxmaxstart=0.15, cxmaxend=0.5, cxmaxpoints=200, qstart=0, qend=1.5, qpoints=5000):
        qarr = np.linspace(qstart,qend,qpoints)
        self.chi2q = []
        self.chi2qc = []
        self.chi2min = []
        self.chi2minc = []
        self.qbest = []
        self.qbestid = []
        self.qupper = []
        self.qupperc = []
        self.qlower = []
        self.qlowerc = []
        self.xbin = []
        self.Bqbin = []
        self.Bfid = []
        self.Bsigma = []
        self.Bsigmac = []
        self.inspmbin = []
        self.nmbin = []
        self.cxmaxopt = []
        for id in range(len(insplog)):
            print('\r%d/%d, log = %.2f'%(id+1,len(insplog),insplog[id]))
            msoverH = math.exp(insplog[id])
            xmin = cxmin
            sqt = Setq(inspx,inspy,insplog,id,xmin,cxmaxstart*msoverH,nbin,qstart,qend,qpoints)
            sigmaq = max(sqt.qbest - sqt.qlower,sqt.qupper - sqt.qbest)
            copt = cxmaxstart
            for c in np.linspace(cxmaxstart,cxmaxend,cxmaxpoints)[1:]:
                print('\rcxmax = %.3f'%c,end="")
                xmax = c*msoverH
                sqtt = Setq(inspx,inspy,insplog,id,xmin,xmax,nbin,qstart,qend,qpoints)
                sigmaqt = max(sqtt.qbest - sqtt.qlower,sqtt.qupper - sqtt.qbest)
                if sigmaqt < sigmaq:
                    sqt = sqtt
                    sigmaq = sigmaqt
                    copt = c
            print("")
            self.chi2q.append(sqt.chi2q)
            self.chi2qc.append(sqt.chi2qc)
            self.chi2min.append(sqt.chi2min)
            self.chi2minc.append(sqt.chi2minc)
            self.qbest.append(sqt.qbest)
            self.qbestid.append(sqt.qbestid)
            self.qupper.append(sqt.qupper)
            self.qupperc.append(sqt.qupperc)
            self.qlower.append(sqt.qlower)
            self.qlowerc.append(sqt.qlowerc)
            self.xbin.append(sqt.xbin)
            self.Bqbin.append(sqt.Bqbin)
            self.Bfid.append(sqt.Bfid)
            self.Bsigma.append(sqt.Bsigma)
            self.Bsigmac.append(sqt.Bsigmac)
            self.inspmbin.append(sqt.inspmbin)
            self.nmbin.append(sqt.nmbin)
            self.cxmaxopt.append(copt)
        #print("")
        self.chi2min = np.array(self.chi2min)
        self.chi2minc = np.array(self.chi2minc)
        self.qbest = np.array(self.qbest)
        self.qupper = np.array(self.qupper)
        self.qupperc = np.array(self.qupperc)
        self.qlower = np.array(self.qlower)
        self.qlowerc = np.array(self.qlowerc)
        self.Bfid = np.array(self.Bfid)
        self.Bsigma = np.array(self.Bsigma)
        self.Bsigmac = np.array(self.Bsigmac)
        self.qarr = qarr
        self.logtab = insplog
        self.cxmaxopt = np.array(self.cxmaxopt)





#   save the data of q as pickle files
#   assuming input as an Scanqopt class object
def saveq(scanqopt, name='./qopt'):
    qname = name + '_q.pickle'
    quppername = name + '_dqu.pickle'
    qlowername = name + '_dql.pickle'
    quppercname = name + '_dquc.pickle'
    qlowercname = name + '_dqlc.pickle'
    logname = name + '_log.pickle'
    cxmaxoptname = name + '_cxmax.pickle'
    with open(qname,'wb') as wq:
        pickle.dump(scanqopt.qbest, wq)
    with open(quppername,'wb') as wqu:
        pickle.dump(scanqopt.qupper, wqu)
    with open(qlowername,'wb') as wql:
        pickle.dump(scanqopt.qlower, wql)
    with open(quppercname,'wb') as wquc:
        pickle.dump(scanqopt.qupperc, wquc)
    with open(qlowercname,'wb') as wqlc:
        pickle.dump(scanqopt.qlowerc, wqlc)
    with open(logname,'wb') as wl:
        pickle.dump(scanqopt.logtab, wl)
    with open(cxmaxoptname,'wb') as wc:
        pickle.dump(scanqopt.cxmaxopt, wc)






#   read the data of q
class readq:
    def __init__(self, name='./qopt'):
        qname = name + '_q.pickle'
        quppername = name + '_dqu.pickle'
        qlowername = name + '_dql.pickle'
        quppercname = name + '_dquc.pickle'
        qlowercname = name + '_dqlc.pickle'
        logname = name + '_log.pickle'
        cxmaxoptname = name + '_cxmax.pickle'
        with open(qname,'rb') as rq:
            self.qbest = pickle.load(rq)
        with open(quppername,'rb') as rqu:
            self.qupper = pickle.load(rqu)
        with open(qlowername,'rb') as rql:
            self.qlower = pickle.load(rql)
        with open(quppercname,'rb') as rquc:
            self.qupperc = pickle.load(rquc)
        with open(qlowercname,'rb') as rqlc:
            self.qlowerc = pickle.load(rqlc)
        with open(logname,'rb') as rl:
            self.logtab = pickle.load(rl)
        with open(cxmaxoptname,'rb') as rc:
            self.cxmaxopt = pickle.load(rc)






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
