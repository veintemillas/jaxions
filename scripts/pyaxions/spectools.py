#!/usr/bin/python3

import numpy as np
import math
import pickle
from pyaxions import jaxions as pa
from numpy.linalg import inv
from scipy.optimize import curve_fit






def sdata(data, name, dataname):
    pname = name + '_' + dataname + '.pickle'
    with open(pname,'wb') as w:
        pickle.dump(data, w)

def rdata(name, dataname):
    pname = name + '_' + dataname + '.pickle'
    with open(pname,'rb') as r:
        return pickle.load(r)





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
#   options for rmask:
#     rmask = '%.2f' -> label from rmasktable (default 2.00)
#     rmask = 'nolabel' -> just try to read nK_Red without rmasklabel (for old data)

class fitP:
    def __init__(self, mfiles, spmask='Red', rmask='2.00', lltype='Z2'):
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
                    if rmask == 'nolabel':
                        binK = pa.gm(meas,'nspK_Red')
                    else:
                        binK = pa.gm(meas,'nspK_Red'+'_'+rmask)
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
#   options for rmask:
#     rmask = '%.2f' -> label from rmasktable (default 2.00)
#     rmask = 'nolabel' -> just try to read nK_Red without rmasklabel (for old data)

#   calculate instantaneous spectrum based on analytical fit
class inspA:
    def __init__(self, mfiles, spmask='Red', rmask='2.00', lltype='Z2'):
        fitp = fitP(mfiles,spmask,rmask)
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





#   calculate instantaneous spectrum based on analytical fit
#   time steps specified in the arguments
class inspAt:
    def __init__(self, mfiles, logi, logf, nlog, spmask='Red', rmask='2.00', lltype='Z2'):
        fitp = fitP(mfiles,spmask,rmask)
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
        self.x = [] # x-axis (k/RH)
        self.log = np.linspace(logi,logf,nlog)
        if lltype == 'Z2':
            self.t = np.exp(self.log)*self.sizeL/(self.sizeN*self.msa)
        elif lltype == 'fixed':
            self.t = np.sqrt(np.exp(self.log)/np.sqrt(2.*self.LL))
        istart = np.abs(self.log - 4.).argmin()
        iterkmax = len(self.avek[self.k_below])
        for id in range(len(self.log)):
            log = self.log[id]
            t = self.t[id]
            print('\rcalc F: %d/%d, log = %.2f'%(id+1,len(self.log),log),end="")
            if id >= istart:
                Fbinbuf = []
                x = []
                for ik in range(iterkmax):
                    # calculate only modes inside the horizon
                    ihc = np.abs(self.avek[ik]*self.t - 2*math.pi).argmin() # time index corresponding to the horizon crossing
                    if id >= ihc:
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
        print("")
        self.F = np.array(self.F)
        self.Fnorm = np.array(self.Fnorm)
        self.x = np.array(self.x) # x = k/RH







#   calculate instantaneous spectrum based on backward difference
class inspB:
    def __init__(self, mfiles, spmask='Red', rmask='2.00', lltype='Z2'):
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
                    if rmask == 'nolabel':
                        binK = pa.gm(meas,'nspK_Red')
                    else:
                        binK = pa.gm(meas,'nspK_Red'+'_'+rmask)
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
                    if rmask == 'nolabel':
                        binK = pa.gm(meas,'nspK_Red')
                    else:
                        binK = pa.gm(meas,'nspK_Red'+'_'+rmask)
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
    def __init__(self, mfiles, spmask='Red', rmask='2.00', lltype='Z2'):
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
                    if rmask == 'nolabel':
                        binK1 = pa.gm(msplist[id-1],'nspK_Red')
                        binK2 = pa.gm(msplist[id+1],'nspK_Red')
                    else:
                        binK1 = pa.gm(msplist[id-1],'nspK_Red'+'_'+rmask)
                        binK2 = pa.gm(msplist[id+1],'nspK_Red'+'_'+rmask)
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
    sdata(inspave.F,name,'y')
    sdata(inspave.dF,name,'dy')
    sdata(inspave.x,name,'x')
    sdata(inspave.log,name,'log')
    sdata(inspave.Fnorm,name,'Fnorm')






#   read the data of instantaneous spectra
class readF:
    def __init__(self, name='./F'):
        self.F = rdata(name,'y')
        self.dF = rdata(name,'dy')
        self.x = rdata(name,'x')
        self.log = rdata(name,'log')
        self.Fnorm = rdata(name,'Fnorm')






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
#
# Output:
#   self.chi2min    minimum value of chi^2
#   self.chi2minn    minimum value of chi^2 (usinf sigman defined below)
#   self.chi2minc   minimum value of chi^2 (using conservative estimate of sigma based on maximum distance from mean)
#   self.qbest      best fit value of q
#   self.mbest      best fit value of m (normalization of the model)
#   self.sigmaq     "1 sigma" confidence interval of q
#   self.sigmaqn     "1 sigma" confidence interval of q (using sigman defined below)
#   self.sigmaqc    "1 sigma" confidence interval of q (using conservative estimate of sigma based on maximum distance from mean)
#   self.xbin       x-axis for rebinned instantaneous spectrum F(x) where x = k/RH
#   self.Fbin       y-axis for rebinned instantaneous spectrum F(x)
#   self.sigma      "sigma" to define confidence interval based on the residuals of different bins
#   self.sigman     sigma = residuals/sqrt(n_bin) This might underestimate the error and depend on n_bin
#   self.sigmac     conservative estimate of "sigma" to define confidence interval based on the maximum value of residuals of different bins
#
#   self.alpha      Delta chi^2(q) can be reconstructed by using alpha, beta, gamma:
#   self.beta       Delta chi^2(q) = (alpha*q^2 + 2*beta*q + gamma)/sigma^2 - chi^2_min
#   self.gamma
#
#   self.nmbin      number of modes in each bin (currently not used)
#   self.xlim       x within the range specified by (xmin,xmax)
#   self.xwr        flags to identify xlim
#
class Setq:
    def __init__(self, inspx, inspy, insplog, id, xmin, xmax, nbin=30):
        Deltachisq = 1. # value of Deltachi^2 to define confidence interval
        x = inspx[id]
        inspmtab = inspy[id]
        x_within_range = ((x > xmin) & (x < xmax))
        xlim = x[x_within_range]
        inspmlim = inspmtab[x_within_range]
        # do not calculate chi^2 if there are not enough data points
        if len(xlim) < 4:
            #print(r' cannot optimize since number of data points is less than 4! (log = %.2f)'%insplog[id])
            chi2min = np.inf
            chi2minn = np.inf
            chi2minc = np.inf
            qbest = np.nan
            mbest = np.nan
            sigmaq = np.nan
            sigmaqn = np.nan
            sigmaqc = np.nan
            xbin = xlim
            Fbin = inspmlim
            sigma = np.nan
            sigman = np.nan
            sigmac = np.nan
            alpha = np.nan
            beta = np.nan
            gamma = np.nan
            nmbin = [1 for i in range(len(xlim))]
        else:
            # do not rebin if number of data points is less than nbin
            if len(xlim) < nbin:
                #print(r' number of data points (%d) is less than nbin (%d)! (log = %.2f)'%(len(xlim),nbin,insplog[id]))
                xbin = xlim
                Fbin = inspmlim
                nmbin = [1 for i in range(len(xlim))]
                nbin = len(xlim)
            else:
                # prepare for rebin
                lnx = np.log(xlim)
                minlnx = np.min(lnx)
                maxlnx = np.max(lnx)
                dlnx = (maxlnx - minlnx)/nbin
                xbinbuf = []
                Fbinbuf = []
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
                        Fbinbuf.append(mave)
                        nmbinbuf.append(len(xlim[x_in_bin]))
                    else:
                        lnxlast = lnxend # save the last boundary of empty bin
                # if actual number of bins is less than specified value of nbin,
                # do not use homogeneous log bin for lower k, and rebin higher k
                # until number of bin becomes nbin.
                if not len(xbinbuf) == nbin:
                    #iloop = 0
                    while len(xbinbuf) < nbin:
                        #print('%d-th while loop lnxlast = %f, datalength = %d'%(iloop+1,lnxlast,len(xbinbuf)))
                        #iloop = iloop + 1
                        lnxleft = np.array([ele for ele in lnx if ele <= lnxlast])
                        xbinbuf = np.exp(lnxleft)
                        Fbinbuf = inspmlim[:len(xbinbuf)]
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
                                Fbinbuf = np.append(Fbinbuf,mave)
                                nmbinbuf = np.append(nmbinbuf,len(xlim[x_in_bin]))
                            else:
                                lnxlast = lnxend # save the last boundary of empty bin
                    #print("homogeneous bin was not possible! (%d/%d, log = %.2f)"%(id+1,len(insplog),insplog[id]))
                    #print("no rebin for x < %f (%d points) and rebin for x > %f (%d points)"%(math.exp(lnxlast),naux,math.exp(lnxlast),nbin-naux))
                xbin = np.array(xbinbuf)
                Fbin = np.array(Fbinbuf)
                nmbin = np.array(nmbinbuf)
            # end of rebin
            # next calculate q
            Su = nbin
            Sl = np.sum(np.log(xbin))
            Sll = np.sum(np.log(xbin)**2)
            SL = np.sum(np.log(Fbin))
            SLL = np.sum(np.log(Fbin)**2)
            SlL = np.sum(np.log(xbin)*np.log(Fbin))
            alpha = Sll*(Su*Sll/(Sl**2)-1)
            beta = (Sll/Sl)*(Su*SlL/Sl-SL)
            gamma = SLL - 2.*SL*SlL/Sl + Su*((SlL/Sl)**2)
            qbest = (Sl*SL-SlL*Su)/(Sll*Su-Sl**2)
            mbest = (Sll*SL-SlL*Sl)/(Sll*Su-Sl**2)
            vecone = np.ones(len(xbin))
            sigmasq = np.sum(np.square(np.log(Fbin)+qbest*np.log(xbin)-mbest*vecone))
            sigmasqn = np.sum(np.square(np.log(Fbin)+qbest*np.log(xbin)-mbest*vecone))/(nbin)
            sigmasqc = np.max(np.square(np.log(Fbin)+qbest*np.log(xbin)-mbest*vecone)) # conservative estimate of sigma based on maximum distance from best fit
            sigma = math.sqrt(sigmasq)
            sigman = math.sqrt(sigmasqn)
            sigmac = math.sqrt(sigmasqc)
            chi2min = np.sum(np.square(np.log(Fbin)+qbest*np.log(xbin)-mbest*vecone))/sigmasq
            chi2minn = np.sum(np.square(np.log(Fbin)+qbest*np.log(xbin)-mbest*vecone))/sigmasqn
            chi2minc = np.sum(np.square(np.log(Fbin)+qbest*np.log(xbin)-mbest*vecone))/sigmasqc
            sigmaq = math.sqrt(beta**2-alpha*(gamma-sigmasq*(chi2min+Deltachisq)))/alpha
            sigmaqn = math.sqrt(beta**2-alpha*(gamma-sigmasqn*(chi2minn+Deltachisq)))/alpha
            sigmaqc = math.sqrt(beta**2-alpha*(gamma-sigmasqc*(chi2minc+Deltachisq)))/alpha
        # end of the case len(xlim) >= 4
        self.chi2min = chi2min
        self.chi2minn = chi2minn
        self.chi2minc = chi2minc
        self.qbest = qbest
        self.mbest = mbest
        self.sigmaq = sigmaq
        self.sigmaqn = sigmaqn
        self.sigmaqc = sigmaqc
        self.xbin = xbin
        self.Fbin = Fbin
        self.sigma = sigma
        self.sigman = sigman
        self.sigmac = sigmac
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
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
#
# Output:
#   self.chi2min    minimum value of chi^2
#   self.chi2minn    minimum value of chi^2 (usinf sigman defined below)
#   self.chi2minc   minimum value of chi^2 (using conservative estimate of sigma based on maximum distance from mean)
#   self.qbest      best fit value of q
#   self.mbest      best fit value of m (normalization of the model)
#   self.sigmaq     "1 sigma" confidence interval of q
#   self.sigmaqn     "1 sigma" confidence interval of q (using sigman defined below)
#   self.sigmaqc    "1 sigma" confidence interval of q (using conservative estimate of sigma based on maximum distance from mean)
#   self.xbin       x-axis for rebinned instantaneous spectrum F(x) where x = k/RH
#   self.Fbin       y-axis for rebinned instantaneous spectrum F(x)
#   self.sigma      "sigma" to define confidence interval based on the mean of residuals of different bins
#   self.sigman     sigma = residuals/sqrt(n_bin) This might underestimate the error and depend on n_bin
#   self.sigmac     conservative estimate of "sigma" to define confidence interval based on the maximum value of residuals of different bins
#
#   self.alpha      Delta chi^2(q) can be reconstructed by using alpha, beta, gamma:
#   self.beta       Delta chi^2(q) = (alpha*q^2 + 2*beta*q + gamma)/sigma^2 - chi^2_min
#   self.gamma
#
#   self.nmbin      number of modes in each bin (currently not used)
#   self.log     　　array for log(m/H)
#
class Scanq:
    def __init__(self, inspx, inspy, insplog, nbin=30, cxmin=30., cxmax=1/6.):
        self.chi2min = []
        self.chi2minn = []
        self.chi2minc = []
        self.qbest = []
        self.mbest = []
        self.sigmaq = []
        self.sigmaqn = []
        self.sigmaqc = []
        self.xbin = []
        self.Fbin = []
        self.sigma = []
        self.sigman = []
        self.sigmac = []
        self.alpha = []
        self.beta = []
        self.gamma = []
        self.nmbin = []
        for id in range(len(insplog)):
            print('\r%d/%d, log = %.2f'%(id+1,len(insplog),insplog[id]),end="")
            msoverH = math.exp(insplog[id])
            xmin = cxmin
            xmax = cxmax*msoverH
            sqt = Setq(inspx,inspy,insplog,id,xmin,xmax,nbin)
            self.chi2min.append(sqt.chi2min)
            self.chi2minn.append(sqt.chi2minn)
            self.chi2minc.append(sqt.chi2minc)
            self.qbest.append(sqt.qbest)
            self.mbest.append(sqt.mbest)
            self.sigmaq.append(sqt.sigmaq)
            self.sigmaqn.append(sqt.sigmaqn)
            self.sigmaqc.append(sqt.sigmaqc)
            self.xbin.append(sqt.xbin)
            self.Fbin.append(sqt.Fbin)
            self.sigma.append(sqt.sigma)
            self.sigman.append(sqt.sigman)
            self.sigmac.append(sqt.sigmac)
            self.alpha.append(sqt.alpha)
            self.beta.append(sqt.beta)
            self.gamma.append(sqt.gamma)
            self.nmbin.append(sqt.nmbin)
        print("")
        self.chi2min = np.array(self.chi2min)
        self.chi2minn = np.array(self.chi2minn)
        self.chi2minc = np.array(self.chi2minc)
        self.qbest = np.array(self.qbest)
        self.mbest = np.array(self.mbest)
        self.sigmaq = np.array(self.sigmaq)
        self.sigmaqn = np.array(self.sigmaqn)
        self.sigmaqc = np.array(self.sigmaqc)
        self.sigma = np.array(self.sigma)
        self.sigman = np.array(self.sigman)
        self.sigmac = np.array(self.sigmac)
        self.alpha = np.array(self.alpha)
        self.beta = np.array(self.beta)
        self.gamma = np.array(self.gamma)
        self.log = insplog
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
#
# Output:
#   self.chi2min    minimum value of chi^2
#   self.chi2minn    minimum value of chi^2 (usinf sigman defined below)
#   self.chi2minc   minimum value of chi^2 (using conservative estimate of sigma based on maximum distance from mean)
#   self.qbest      best fit value of q
#   self.mbest      best fit value of m (normalization of the model)
#   self.sigmaq     "1 sigma" confidence interval of q
#   self.sigmaqn     "1 sigma" confidence interval of q (using sigman defined below)
#   self.sigmaqc    "1 sigma" confidence interval of q (using conservative estimate of sigma based on maximum distance from mean)
#   self.xbin       x-axis for rebinned instantaneous spectrum F(x) where x = k/RH
#   self.Fbin       y-axis for rebinned instantaneous spectrum F(x)
#   self.sigma      "sigma" to define confidence interval based on the mean of residuals of different bins
#   self.sigman     sigma = residuals/sqrt(n_bin) This might underestimate the error and depend on n_bin
#   self.sigmac     conservative estimate of "sigma" to define confidence interval based on the maximum value of residuals of different bins
#
#   self.alpha      Delta chi^2(q) can be reconstructed by using alpha, beta, gamma:
#   self.beta       Delta chi^2(q) = (alpha*q^2 + 2*beta*q + gamma)/sigma^2 - chi^2_min
#   self.gamma
#
#   self.nmbin      number of modes in each bin (currently not used)
#   self.log        array for log(m/H)
#   self.cxmaxopt   array for optimized values of cxmax
#
class Scanqopt:
    def __init__(self, inspx, inspy, insplog, nbin=30, cxmin=30., cxmaxstart=0.15, cxmaxend=0.5, cxmaxpoints=200):
        self.chi2min = []
        self.chi2minn = []
        self.chi2minc = []
        self.qbest = []
        self.mbest = []
        self.sigmaq = []
        self.sigmaqn = []
        self.sigmaqc = []
        self.xbin = []
        self.Fbin = []
        self.sigma = []
        self.sigman = []
        self.sigmac = []
        self.alpha = []
        self.beta = []
        self.gamma = []
        self.nmbin = []
        self.cxmaxopt = []
        for id in range(len(insplog)):
            print('\r%d/%d, log = %.2f'%(id+1,len(insplog),insplog[id]),end="")
            msoverH = math.exp(insplog[id])
            xmin = cxmin
            sqt = Setq(inspx,inspy,insplog,id,xmin,cxmaxstart*msoverH,nbin)
            sigmaq = sqt.sigmaq
            copt = cxmaxstart
            for c in np.linspace(cxmaxstart,cxmaxend,cxmaxpoints)[1:]:
                #print('\rcxmax = %.3f'%c)
                xmax = c*msoverH
                sqtt = Setq(inspx,inspy,insplog,id,xmin,xmax,nbin)
                sigmaqt = sqtt.sigmaq
                if sigmaqt < sigmaq:
                    sqt = sqtt
                    sigmaq = sigmaqt
                    copt = c
            #print("")
            self.chi2min.append(sqt.chi2min)
            self.chi2minn.append(sqt.chi2minn)
            self.chi2minc.append(sqt.chi2minc)
            self.qbest.append(sqt.qbest)
            self.mbest.append(sqt.mbest)
            self.sigmaq.append(sqt.sigmaq)
            self.sigmaqn.append(sqt.sigmaqn)
            self.sigmaqc.append(sqt.sigmaqc)
            self.xbin.append(sqt.xbin)
            self.Fbin.append(sqt.Fbin)
            self.sigma.append(sqt.sigma)
            self.sigman.append(sqt.sigman)
            self.sigmac.append(sqt.sigmac)
            self.alpha.append(sqt.alpha)
            self.beta.append(sqt.beta)
            self.gamma.append(sqt.gamma)
            self.nmbin.append(sqt.nmbin)
            self.cxmaxopt.append(copt)
        print("")
        self.chi2min = np.array(self.chi2min)
        self.chi2minn = np.array(self.chi2minn)
        self.chi2minc = np.array(self.chi2minc)
        self.qbest = np.array(self.qbest)
        self.mbest = np.array(self.mbest)
        self.sigmaq = np.array(self.sigmaq)
        self.sigmaqn = np.array(self.sigmaqn)
        self.sigmaqc = np.array(self.sigmaqc)
        self.sigma = np.array(self.sigma)
        self.sigman = np.array(self.sigman)
        self.sigmac = np.array(self.sigmac)
        self.alpha = np.array(self.alpha)
        self.beta = np.array(self.beta)
        self.gamma = np.array(self.gamma)
        self.log = insplog
        self.cxmaxopt = np.array(self.cxmaxopt)





#   save the data of q as pickle files
#   assuming input as an Scanqopt class object
def saveq(scanqopt, name='./qopt'):
    sdata(scanqopt.qbest,name,'q')
    sdata(scanqopt.mbest,name,'m')
    sdata(scanqopt.sigmaq,name,'sigmaq')
    sdata(scanqopt.sigmaqn,name,'sigmaqn')
    sdata(scanqopt.sigmaqc,name,'sigmaqc')
    sdata(scanqopt.log,name,'log')
    sdata(scanqopt.cxmaxopt,name,'cxmax')






#   read the data of q
class readq:
    def __init__(self, name='./qopt'):
        self.qbest = rdata(name,'q')
        self.mbest = rdata(name,'m')
        self.sigmaq = rdata(name,'sigmaq')
        self.sigmaqn = rdata(name,'sigmaqn')
        self.sigmaqc = rdata(name,'sigmaqc')
        self.log = rdata(name,'log')
        self.cxmaxopt = rdata(name,'cxmax')






# ------------------------------------------------------------------------------
#   String density
# ------------------------------------------------------------------------------

#   evolution of string density parameter
class strevol:
    def __init__(self, mfiles, lltype='Z2', diff='nodiff', sigma=1/4., thresh=0.0001):
        self.sizeN = pa.gm(mfiles[0],'Size')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda')
        self.t = []
        self.xi = []
        self.dxidl = []
        self.dxidt = []
        for ff in mfiles:
            self.t.append(pa.gm(ff,'ct'))
            self.xi.append(pa.gm(ff,'stDens'))
        self.t = np.array(self.t)
        self.xi = np.array(self.xi)
        if lltype == 'Z2':
            self.log = np.log(self.t*self.sizeN*self.msa/self.sizeL)
        elif lltype == 'fixed':
            self.log = np.log(math.sqrt(2.*self.LL)*self.t**2)
        # calculate derivative of xi and smooth the result with Gaussian function
        if diff == 'diff':
            xref = sigma*np.sqrt(2.0*np.log(1/thresh))
            dxdl = np.gradient(self.xi)/np.gradient(self.log)
            dxdt = np.gradient(self.xi)/np.gradient(self.t)
            for i in range (len(self.t)):
                xc = self.log[i]
                dx = self.log - xc
                above_thresh = np.abs(dx) < xref
                gaussian = np.exp(-(dx) ** 2 / (2 * sigma ** 2))
                gaussian2 = gaussian[above_thresh]
                gaussian2 = gaussian2/gaussian2.sum()
                dxdl2 = dxdl[above_thresh]
                dxdt2 = dxdt[above_thresh]
                smoothed_dxdl = sum(dxdl2 * gaussian2)
                smoothed_dxdt = sum(dxdt2 * gaussian2)
                self.dxidl.append(smoothed_dxdl)
                self.dxidt.append(smoothed_dxdt)
        self.dxidl = np.array(self.dxidl)
        self.dxidt = np.array(self.dxidt)







#   take ensemble average of the string density parameter
#   assuming a list of strevol class objects
class strave:
    def __init__(self, strevollist):
        Nreal = len(strevollist) # number of realizations
        self.sizeN = strevollist[0].sizeN
        self.sizeL = strevollist[0].sizeL
        self.msa = strevollist[0].msa
        self.LL = strevollist[0].LL
        self.t = strevollist[0].t
        self.log = strevollist[0].log
        xi = [0]*len(strevollist[0].xi)
        xisq = [0]*len(strevollist[0].xi)
        dxdl = [0]*len(strevollist[0].dxidl)
        dxdlsq = [0]*len(strevollist[0].dxidl)
        dxdt = [0]*len(strevollist[0].dxidt)
        dxdtsq = [0]*len(strevollist[0].dxidt)
        for sl in strevollist:
            xi += sl.xi
            xisq += sl.xi*sl.xi
            dxdl += sl.dxidl
            dxdlsq += sl.dxidl*sl.dxidl
            dxdt += sl.dxidt
            dxdtsq += sl.dxidt*sl.dxidt
        self.xi = xi/Nreal
        self.xierr = np.sqrt(xisq/Nreal - self.xi*self.xi)
        self.dxidl = dxdl/Nreal
        self.dxidlerr = np.sqrt(dxdlsq/Nreal - self.dxidl*self.dxidl)
        self.dxidt = dxdt/Nreal
        self.dxidterr = np.sqrt(dxdtsq/Nreal - self.dxidt*self.dxidt)






#   save the data of string density parameter as pickle files
#   assuming input as an strave class object
def savestr(strave, name='./str'):
    sdata(strave.xi,name,'xi')
    sdata(strave.xierr,name,'xierr')
    sdata(strave.dxidl,name,'dxidl')
    sdata(strave.dxidlerr,name,'dxidlerr')
    sdata(strave.dxidt,name,'dxidt')
    sdata(strave.dxidterr,name,'dxidterr')
    sdata(strave.t,name,'t')
    sdata(strave.log,name,'log')






#   read the data of string density parameter
class readstr:
    def __init__(self, name='./str'):
        self.xi = rdata(name,'xi')
        self.xierr = rdata(name,'xierr')
        self.dxidl = rdata(name,'dxidl')
        self.dxidlerr = rdata(name,'dxidlerr')
        self.dxidt = rdata(name,'dxidt')
        self.dxidterr = rdata(name,'dxidterr')
        self.t = rdata(name,'t')
        self.log = rdata(name,'log')







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
#   options for rmask:
#     rmask = '%.2f' -> label from rmasktable (default 2.00)
#     rmask = 'nolabel' -> just try to read nK_Red without rmasklabel (for old data)

def nspcore(mfile, nm, spmasklabel='Red_2.00'):
    s0 = pa.gm(mfile,'nsp'+spmasklabel)
    #    print('Attempted to read nsp%s'%(spmasklabel))
    try :
        m = pa.gm(mfile,'mspM_'+spmasklabel[2:])
        pasa = m.shape == (len(s0),len(s0))
    except:
        return s0

    s1 = (pa.gm(mfile,'L')**3)*np.dot(inv(m),s0/nm)
    return s1

def nspcor(mfile, nm, spmask='Red', rmask='2.00'):
    if spmask == 'nomask':
        return pa.gm(mfile,'nspK')
    elif spmask == 'Red':
        if rmask == 'nolabel':
            s0 = pa.gm(mfile,'nspK_Red')
            m = pa.gm(mfile,'mspM_Red')
        else:
            s0 = pa.gm(mfile,'nspK_Red'+'_'+rmask)
            m = pa.gm(mfile,'mspM_Red'+'_'+rmask)
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
    def __init__(self, mfiles, spmask='Red', rmask='2.00', lltype='Z2', cor='nocorrection'):
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
                    if rmask == 'nolabel':
                        s0 = pa.gm(f,'nspK_Red')
                    else:
                        s0 = pa.gm(f,'nspK_Red'+'_'+rmask)
                    self.nsp.append(s0)
                    if cor == 'correction':
                        if rmask == 'nolabel':
                            m = pa.gm(f,'mspM_Red')
                        else:
                            m = pa.gm(f,'mspM_Red'+'_'+rmask)
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


class nspevol2:
    def __init__(self, mfiles, spmasklabel='Red_2.00', cor='nocorrection'):
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
                self.ttab.append(pa.gm(f,'time'))
                logi = pa.gm(f,'logi')
                self.logtab.append(logi)
                s0 = pa.gm(f,'nspK_'+spmasklabel)
                self.nsp.append(s0)
                if cor == 'correction':
                    m = pa.gm(f,'mspM_'+spmasklabel)
                    s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                    self.nspcor.append(s1)
                print('\rbuilt up to log = %.2f'%logi,end="")
        print("")
        self.ttab = np.array(self.ttab)
        self.logtab = np.array(self.logtab)
        self.nsp = np.array(self.nsp)
        self.nspcor = np.array(self.nspcor)






#   builds the (masked) axion energy spectrum with the correction matrix and outputs the time evolution
#   NOTE: The energy density is evaluated just by muptiplying the kinetic energy by 2.

class espevol:
    def __init__(self, mfiles, spmask='Red', rmask='2.00', lltype='Z2', cor='nocorrection'):
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
                    if rmask == 'nolabel':
                        s0 = pa.gm(f,'nspK_Red')
                    else:
                        s0 = pa.gm(f,'nspK_Red'+'_'+rmask)
                    e0 = (self.avek**2)*s0/(t*(math.pi**2)*self.nm)
                    self.esp.append(e0)
                    if cor == 'correction':
                        if rmask == 'nolabel':
                            m = pa.gm(f,'mspM_Red')
                        else:
                            m = pa.gm(f,'mspM_Red'+'_'+rmask)
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


class espevol2:
    def __init__(self, mfiles, spmasklabel='Red_2.00', cor='nocorrection'):
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
                logi = pa.gm(f,'logi')
                self.ttab.append(t)
                self.logtab.append(logi)
                s0 = pa.gm(f,'nspK_'+spmasklabel)
                e0 = (self.avek**2)*s0/(t*(math.pi**2)*self.nm)
                self.esp.append(e0)
                if cor == 'correction':
                    m = pa.gm(f,'mspM_'+spmasklabel)
                    s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                    e1 = (self.avek**2)*s1/(t*(math.pi**2)*self.nm)
                    self.espcor.append(e1)
                print('\rbuilt up to log = %.2f'%logi,end="")
        print("")
        self.ttab = np.array(self.ttab)
        self.logtab = np.array(self.logtab)
        self.esp = np.array(self.esp)
        self.espcor = np.array(self.espcor)






#   take ensemble average of the (masked) axion energy spectrum
#   assuming a list of espevol class objects
class espave:
    def __init__(self, esplist, cor='nocorrection'):
        Nreal = len(esplist) # number of realizations
        self.sizeN = esplist[0].sizeN
        self.sizeL = esplist[0].sizeL
        self.msa = esplist[0].msa
        self.LL = esplist[0].LL
        self.nm = esplist[0].nm
        self.avek = esplist[0].avek
        self.k_below = esplist[0].k_below
        self.t = esplist[0].ttab
        self.log = esplist[0].logtab
        self.esp = []
        self.desp = []
        self.espcor = []
        self.despcor = []
        for id in range(len(self.t)):
            esp = [0]*(len(self.avek))
            espsq = [0]*(len(self.avek))
            if cor == 'correction':
                espcor = [0]*(len(self.avek))
                espcorsq = [0]*(len(self.avek))
            for el in esplist:
                esp += el.esp[id]
                espsq += np.square(el.esp[id])
                if cor == 'correction':
                    espcor += el.espcor[id]
                    espcorsq += np.square(el.espcor[id])
            esp = esp/Nreal
            espsq = espsq/Nreal - esp*esp
            if cor == 'correction':
                espcor = espcor/Nreal
                espcorsq = espcorsq/Nreal - espcor*espcor
            self.esp.append(esp)
            self.desp.append(np.sqrt(espsq))
            if cor == 'correction':
                self.espcor.append(espcor)
                self.despcor.append(np.sqrt(espsqcor))
            print('\r%d/%d, log = %.2f'%(id+1,len(self.t),self.log[id]),end="")
        print("")
        self.esp = np.array(self.esp)
        self.desp = np.array(self.desp)
        self.espcor = np.array(self.espcor)
        self.despcor = np.array(self.despcor)






#   save the data of axion energy spectra as pickle files
#   assuming input as an espave class object
def saveesp(espave, name='./esp'):
    sdata(espave.esp,name,'e')
    sdata(espave.desp,name,'de')
    sdata(espave.espcor,name,'ec')
    sdata(espave.despcor,name,'dec')
    sdata(espave.t,name,'t')
    sdata(espave.log,name,'log')
    sdata(espave.nm,name,'nm')
    sdata(espave.avek,name,'k')
    sdata(espave.k_below,name,'kb')






#   read the data of axion energy spectra
class readesp:
    def __init__(self, name='./esp'):
        self.esp = rdata(name,'e')
        self.desp = rdata(name,'de')
        self.espcor = rdata(name,'ec')
        self.despcor = rdata(name,'dec')
        self.t = rdata(name,'t')
        self.log = rdata(name,'log')
        self.nm = rdata(name,'nm')
        self.avek = rdata(name,'k')
        self.k_below = rdata(name,'kb')





class combiq:
    def __init__(self, mfiles):
        self.sizeN = pa.gm(mfiles[0],'sizeN')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.nm = pa.gm(mfiles[0],'nmodelist')
        self.k = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm)*(2*np.pi/self.sizeL)
        self.lk   = np.log(self.k)
        self.ct = pa.gml(mfiles,'ct')
        self.lct  = np.log(self.ct)
        self.logi  = pa.gml(mfiles,'logi')
        self.cmassS  = np.sqrt(2*pa.gml(mfiles,'lambda')*pa.gml(mfiles,'ct')**2)

        self.nmax = len(self.nm)
        self.ntab = np.arange(0,self.nmax+1)
        self.nv   = 4*np.pi*(self.ntab[1:]**3-self.ntab[:-1]**3)/3
        self.corr = self.nv/self.nm

        self.nsp_tab  = []
        self.lnsp_tab = []

        self.nsp = 0
        self.sp = {}

        self.F      = 1
        self.nspI   = 1
        self.lF   = 1

        self.name_tab = []
        self.order    = 0

        self.nsp_rebin  = 1
        self.lnsp_rebin = 1
        self.F_rebin    = 1
        self.lF_rebin   = 1
        self.k_rebin    = 0
        self.nspI_rebin = 0
        self.lk_rebin   = 0
        self.lk_rebin_n   = 0

        self.qtab   = 1
        self.stab   = 1
        self.qsigma = 1
        self.qfit   = 1
        self.qlogi  = 1

        self.qtab_rebin   = 1
        self.stab_rebin   = 1
        self.qsigma_rebin = 1
        self.qfit_rebin   = 1
        self.qlogi_rebin  = 1

        self.xi_tab = []
        self.xi_jk  = []
        self.xi  = 0
        self.exi = 0
        self.rc1 = 0
        self.rc2 = 0
        self.rc3 = 0

    def addsimu(self,mfiles2,setlisttoadd=['nspK'],mask='_Red',setname=''):
        if setname=='':
            setname = str(self.order)
        tempct = pa.gml(mfiles2,'ct')

        # old legacy
        # tempspe = pa.gml(mfiles2,'nspK_Red')
        # self.nsp_tab.append(tempspe)
        # self.lnsp_tab.append(np.log(tempspe))
        # self.xi_tab.append(pa.gml(mfiles2,'stDens'))
        # # self.order = self.order+1
        # # self.name_tab.append(setname)
        # print("New set %s added"%setname)
        # print("len(nsp_tab)=%d "%len(self.nsp_tab))

        # new dic
        for set in setlisttoadd:
            # if mask != '':
            #     sptype = set+'_'+mask+'_tab'
            # else :
            #     sptype = set+'_tab'
            if set[:3] == 'nsp':
                settab = set+mask+'_tab'
                setcal = set+mask
            else :
                settab = set+'_tab'
                setcal = set

            if not settab in self.sp:
                self.sp[settab] = []
            tempspe = pa.gml(mfiles2,setcal)
            self.sp[settab].append(tempspe)

#     def rebin(self,bindet):
        # combines lk's, lnsp's

    def average(self,setlisttoav=['nspK'],mask='_Red',setname=''):
        #legacy
        # self.nsp = 0
        # self.xi = 0
        # for se in range(len(self.nsp_tab)):
        #     self.nsp += self.nsp_tab[se]
        #     self.xi += self.xi_tab[se]
        # self.nsp = self.nsp/len(self.nsp_tab)
        # self.lnsp = np.log(self.nsp)
        # self.xi = self.xi/len(self.nsp_tab)
        #
        # der = np.gradient(self.xi,self.ct)
        # self.rc1 = -(der/self.xi/self.ct**2)*self.ct**2/2
        # self.rc2 = -(1/self.logi*1/self.ct**2)*self.ct**2/2
        # self.rc3 = (1/self.logi*0.5*der/self.xi/self.ct)*self.ct**2/2

        #new dic

        # sonthing like this can select tabs
        # for l in lis:
        #     if l[:3] == 'nsp' and l[-4:] == '_tab':
        #         print(l)
        for set in setlisttoav:
            if set[:3] == 'nsp':
                settab = set+mask+'_tab'
                setcal = set+mask
            else :
                settab = set+'_tab'
                setcal = set

            if not settab in self.sp:
                print(settab,' not found, skipping its average')
            else :
                self.sp[setcal] = 0
                for se in range(len(self.sp[settab])):
                    self.sp[setcal] += self.sp[settab][se]
                self.sp[setcal] /= len(self.sp[settab])

                self.sp[setcal+'_sigma'] = 0
                for se in range(len(self.sp[settab])):
                    self.sp[setcal+'_sigma'] += (self.sp[settab][se] - self.sp[setcal])
                self.sp[setcal+'_sigma'] /= len(self.sp[settab])
                self.sp[setcal+'_sigma'] = np.sqrt(self.sp[setcal+'_sigma'])/len(self.sp[settab])

            if set == 'xi':
                xi = self.sp[setcal]
                der = np.gradient(xi,self.ct)
                # -xi_t/xi 2H
                self.sp[setcal+'_rc1'] = -(der/xi/self.ct**2)*self.ct**2/2
                # -mu_0/m_eff (H) /2H
                self.sp[setcal+'_rc2'] = -(1/self.logi*1/self.ct**2)*self.ct**2/2
                # mu_0/m_eff (xi_t/2xi ) /2H
                self.sp[setcal+'_rc3'] = (1/self.logi*0.5*der/xi/self.ct)*self.ct**2/2

#        eNsp = 0
#        eXi  = 0
#
##        nTaus = self.nsp_tab.shape[0]
##        nMoms = self.nsp_tab.shape[1]
#
#        self.nsp_jk  = np.zeros((self.order))
#        self.lnsp_jk = np.zeros((self.order))
#        self.xi_jk   = np.zeros((self.order))
#
#        for nMeas in range(len(self.nsp_tab)):
#            tNsp = 0
#            tXi  = 0
#            for se in range(len(self.nsp_tab)):
#                if se == nMeas:
#                    continue
#                tNsp += self.nsp_tab[se]
#                tXi  += self.xi_tab[se]
#            tNsp = (tNsp*(self.order-1))/self.order
#            tXi  = (tXi *(self.order-1))/self.order
#
#            print(tNsp.shape)
#            self.nsp_jk [nMeas] = tNsp
#            self.lnsp_jk[nMeas] = np.log(tNsp)
#            self.xi_jk  [nMeas] = tXi
#
#        self.eNsp  = np.cov(self.nsp_jk,  rowvar=False, bias=True)*(self.order-1)
#        self.elNsp = np.cov(self.lnsp_jk, rowvar=False, bias=True)*(self.order-1)
#        self.eXi   = np.cov(self.xi_jk,   rowvar=False, bias=True)*(self.order-1)

    def rebin(self,setlisttorebin=['nspK_Red'],logbinsperdecade=10):

        lkmin = self.lk[1]
        lkmax = self.lk[-1]
        nvin = logbinsperdecade*(lkmax-lkmin)/np.log(10.)
        bins = np.linspace(lkmin,lkmax,int(nvin))
        lkk = self.lk[1:]
        his0 = np.histogram(lkk,bins=bins)
        his = np.histogram(lkk,weights=lkk,bins=bins)
        mask = his0[0] > 0
        self.lk_rebin = his[0][mask]/his0[0][mask]
        self.k_rebin  = np.exp(self.lk_rebin)
        rSS=[]
        rSS2=[]
        self.lk_rebin_n = his0[0][mask]

        #legacy
        # for t in range(len(self.ct)):
        #     lsp = self.lnsp[t][1:]
        #
        #     hiss= np.histogram(lkk,weights=lsp,bins=bins)
        #
        #     rSS.append(hiss[0][mask]/his0[0][mask])
        #
        # self.lnsp_rebin= np.array(rSS)
        # self.nsp_rebin= np.exp(self.lnsp_rebin)
        for setname in setlisttorebin:
            for t in range(len(self.ct)):
                lsp = np.log(self.sp[setname][t][1:])
                hiss= np.histogram(lkk,weights=lsp,bins=bins)
                lsp_ave = hiss[0][mask]/his0[0][mask]
                rSS.append(lsp_ave)

                hiss2= np.histogram(lkk,weights=lsp**2,bins=bins)
                lsp2_ave = hiss2[0][mask]/his0[0][mask]
                rSS2.append( np.sqrt(np.abs(lsp2_ave - lsp_ave**2))/his0[0][mask] )
            self.sp[setname+'_lrebin'] = np.array(rSS)
            self.sp[setname+'_lrebin_sigma'] = np.array(rSS2)
            self.sp[setname+'_rebin'] = np.exp(np.array(rSS))

    def logbin(self,setlisttorebin=['nspK_Red'],logbinsperdecade=10):
        #
        lkmin = self.lk[1]
        lkmax = self.lk[-1]
        nvin = logbinsperdecade*(lkmax-lkmin)/np.log(10.)
        bins = np.linspace(lkmin,lkmax,int(nvin))
        lkk = self.lk[1:]
        his0 = np.histogram(lkk,bins=bins)
        his = np.histogram(lkk,weights=lkk,bins=bins)
        mask = his0[0] > 0
        self.lk_rebin = his[0][mask]/his0[0][mask]
        self.k_rebin  = np.exp(self.lk_rebin)
        rSS=[]
        rSS2=[]
        self.lk_rebin_n = his0[0][mask]

        for setname in setlisttorebin:
            for t in range(len(self.ct)):
                lsp = np.log(self.sp[setname][t][1:])
                hiss= np.histogram(lkk,weights=lsp,bins=bins)
                lsp_ave = hiss[0][mask]/his0[0][mask]
                rSS.append(lsp_ave)

                hiss2= np.histogram(lkk,weights=lsp**2,bins=bins)
                lsp2_ave = hiss2[0][mask]/his0[0][mask]
                rSS2.append( np.sqrt(np.abs(lsp2_ave - lsp_ave**2))/his0[0][mask] )
            self.sp[setname+'_lrebin'] = np.array(rSS)
            self.sp[setname+'_lrebin_sigma'] = np.array(rSS2)
            self.sp[setname+'_rebin'] = np.exp(np.array(rSS))

    # It could take an extra array of points instead of self.ct
    def computeF(self,array='nspK_Red',Ng=4,poliorder=1):
        self.average() # do I need this?
        # if array == 'nspK':
            # spe = self.lnsp
        if not array in self.sp:
            print('No available set!, try average first or input data!')
            return 0

        spe = np.log(self.sp[array])
        if '_rebin' in array:
            kkk = self.lk_rebin
        else:
            kkk = self.lk

        # kkk = self.lk
        # elif array == 'nsp_rebin':
        #     spe = self.lnsp_rebin
        #     kkk = self.lk_rebin


        # spectrum
        sout = []
        # derivative with respect to ... time or conformal time
        dout = []
        mout = []
        for ct0 in self.ct:
            cuve = np.argsort((self.ct-ct0)**2)
            x = self.lct[cuve][:Ng]
            lis = []
            der = []
            mas = []
            for kc in range(len(kkk)):
                y = spe[cuve,kc][:Ng]
                # fit y = x pp[0] + pp[1]
                p = np.polyfit(x,y,poliorder)
                pp = np.poly1d(p)
                # evaluate y at the function, not the data point
                va = np.exp(pp(np.log(ct0)))
                lis.append(va)
                # evaluate the derivative as ds/dt = (s/t) (d log s / d log t)
                # version: conformal time
                pp2 = np.polyder(pp)
                logder = pp2(np.log(ct0))
                der.append((va/ct0)*logder)
                # version: usual time = ctime^2
                # der.append((va/ct0**2)*pp[0]/2)
                mas.append(pp[-1])
            sout.append(lis)
            dout.append(der)
            mout.append(mas)

        # legacy
        # if array == 'nsp':
        #     self.F = np.array(dout)
        #     self.nspI = np.array(sout)
        #     self.lF = np.array(mout)
        # elif array == 'nsp_rebin':
        #     self.F_rebin = np.array(dout)
        #     self.nspI_rebin = np.array(sout)
        #     self.lF_rebin = np.array(mout)

        self.sp[array+'_F'] = np.array(dout)
        self.sp[array+'_I'] = np.array(sout)
        self.sp[array+'_lF'] = np.array(mout)

    # computes the exponent of the spectrum as a function of time
    def buildqq(self,array='nspK_Red_F',xmin=30,xxmax=1/4,qtab=np.linspace(0.2,1.5,1000)):
        # legacy
        # if array == 'F':
        #     spe = self.F
        #     kkk = self.k
        # elif array == 'F_rebin':
        #     spe = self.F_rebin
        #     kkk = self.k_rebin

        spe = self.sp[array]

        if '_rebin' in array:
            kkk = self.k_rebin
        else:
            kkk = self.k

        tout = []
        qout = []
        sigma = []
        logi = []
        tabout = []
        sout = []
        tabout_full = []


        for t in range(len(self.ct)):
            ct0 = self.ct[t]
            mask = (kkk*ct0 > xmin ) * (kkk < xxmax * self.cmassS[t] ) * (spe[t] > 0)
#             print('%s t=%d=%f %d'%(array,t,ct0,mask.sum()))
            if mask.sum() < 2:
                continue
            ta = np.log(spe[t][mask])
            ka = np.log(kkk[mask])
            ma = ka**0 # allows reweigthing
#             print('ta',ta)
#             print('ka',ka)

            a1 = (ka/ma).sum()/(1/ma).sum()
            b1 = (ta/ma).sum()/(1/ma).sum()
            a2 = (ka/ma).sum()/(ka**2/ma).sum()
            b2 = (ta*ka/ma).sum()/(ka**2/ma).sum()
            q_min = (b1*a2-b2)/(1-a1*a2)

        #   compute a sensible value of sigma
            s_min = ((ta + ka*q_min)/ma).sum()/((1/ma).sum())
            pre = (s_min - ka*q_min - ta)**2/ma

#             print('s =%f q=%f'%(s_min,q_min))
            prechi2_0 = pre.sum()
            ksig2 = pre.mean()
            ksig = np.sqrt(ksig2)
            chi0 = prechi2_0/ksig2

#             print('prechi2 %f ksig2'%(s_min,q_min))
            def cacique(q):
                ss = ((ta + ka*q)/ma).sum()/((1/ma).sum())
                pre = (ss - ka*q - ta)**2/ma
                return pre.sum()

            sil = np.array([cacique(q) for q in qtab])
            chirel=sil/ksig2 - chi0
            inter =  chirel < 1
            if inter.sum() <1:
                print('problem!')
                print('chirel min %f max %f'%(chirel.min(),chirel.max()))
                continue
            CL2 = qtab[inter][-1] - q_min
            CL1 = q_min - qtab[inter][0]
            qout.append(q_min)
            sout.append(s_min)

            sigma.append([CL1,CL2])
            tout.append(ct0)
            logi.append(self.logi[t])
            tabout.append([ka,ta])

        # legacy
        # if array == 'F':
        #     self.qtab = np.array(qout)
        #     self.stab = np.array(sout)
        #     self.qsig = np.array(sigma)
        #     self.qfit = np.array(tabout)
        #     self.qlogi = np.array(logi)
        # elif array == 'F_rebin':
        #     self.qtab_rebin = np.array(qout)
        #     self.stab_rebin = np.array(sout)
        #     self.qsig_rebin = np.array(sigma)
        #     self.qfit_rebin = np.array(tabout)
        #     self.qlogi_rebin = np.array(logi)
        # newps
        self.sp[array+'_qtab'] = np.array(qout)
        self.sp[array+'_stab'] = np.array(sout)
        self.sp[array+'_qsig'] = np.array(sigma)
        self.sp[array+'_qfit'] = np.array(tabout)
        self.sp[array+'_qlogi'] = np.array(logi)
