#!/usr/bin/python3

import numpy as np
import math
import pickle
from pyaxions import jaxions as pa
from numpy.linalg import inv
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d






def sdata(data, name, dataname):
    pname = name + '_' + dataname + '.pickle'
    with open(pname,'wb') as w:
        pickle.dump(data, w)

def rdata(name, dataname):
    pname = name + '_' + dataname + '.pickle'
    with open(pname,'rb') as r:
        return pickle.load(r)

def smo(K0,sigma):
    # If data was corrected by m matrix, it can be negative occationally
    lK0 = np.log10(np.absolute(K0))
    lK = lK0
    lK[1:] = gaussian_filter1d(lK0[1:],sigma,mode='nearest',cval=0.0)
    return 10**lK




# ------------------------------------------------------------------------------
#   Analytical fit
# ------------------------------------------------------------------------------




#   fitting function for subhorizon regime
def fsubh(x, a0, a1, a2):
    return a0 + a1*x + a2*(x**2)
def dfsubh(x, a0, a1, a2):
    return a1 + 2*a2*x

#   fitting function for superhorizon regime
def fsuph(x, a0, a1, a2, a3):
    return a0 + a1*x - np.log(1+(a3*np.exp(x))**a2)
def dfsuph(x, a0, a1, a2, a3):
    return a1 - a2 + a2/(1 + (a3*np.exp(x))**a2)

def logistic(x, x0, sigma):
    return (1-np.tanh((x-x0)/sigma))/2




class fitP:
    def __init__(self, P, log, t, k, chc=1., logstart=4., verbose=True):
        self.paramsup = []
        self.paramsub = []
        self.paramvsup = []
        self.paramvsub = []
        self.listihc = []
        self.listfitsup = []
        self.listfitsub = []
        PT = np.transpose(P)
        bo = ((-np.inf,0,0,0),(np.inf,np.inf,np.inf,np.inf))
        iterkmax = len(k)
        for ik in range(iterkmax):
            if verbose:
                print('\rfit P:  k = %.2f [%d/%d]'%(k[ik],ik+1,iterkmax),end="")
            ihc = np.abs(k[ik]*t - chc*2*math.pi).argmin() # time index corresponding to the horizon crossing
            masksup = np.where((log <= log[ihc]) & (log >= logstart))
            masksub = np.where(log >= max(logstart,log[ihc]))
            xdatasup = np.log(k[ik]*t[masksup[0]])
            ydatasup = np.log(PT[ik,masksup[0]])
            xdatasub = np.log(k[ik]*t[masksub[0]])
            ydatasub = np.log(PT[ik,masksub[0]])
            Npsup = 4 # number of parameters for the fitting function
            Npsub = 3
            if len(xdatasup) >= Npsup and not ik == 0:
                psup, pvsup = curve_fit(fsuph, xdatasup, ydatasup, bounds=bo, maxfev = 20000)
                self.listfitsup.append(True)
            else:
                psup = [np.nan]*(Npsup)
                pvsup = [np.nan]*(Npsup)
                self.listfitsup.append(False)
            if len(xdatasub) >= Npsub and not ik == 0:
                psub, pvsub = curve_fit(fsubh, xdatasub, ydatasub, maxfev = 20000)
                self.listfitsub.append(True)
            else:
                psub = [np.nan]*(Npsub)
                pvsub = [np.nan]*(Npsub)
                self.listfitsub.append(False)
            self.paramsup.append(psup)
            self.paramvsup.append(pvsup)
            self.paramsub.append(psub)
            self.paramvsub.append(pvsub)
            self.listihc.append(ihc)
        if verbose:
            print("")





#   Old functions are left below, just in case


#   fitting function as cubic polynomial
def func(x, a0, a1, a2, a3):
    return a0 + a1*x + a2*(x**2) + a3*(x**3)

#   derivative of the fitting function
def dfunc(x, a0, a1, a2, a3):
    return a1 + 2*a2*x + 3*a3*(x**2)




class fitP_old:
    def __init__(self, P, log, t, k, logstart=4., verbose=True):
        mask = np.where(log >= logstart)
        self.param = []
        self.paramv = []
        self.listihc = []
        self.dataP = []
        PT = np.transpose(P)
        iterkmax = len(k)
        for ik in range(iterkmax):
            if verbose:
                print('\rfit: k = %.2f, %d/%d'%(k[ik],ik+1,iterkmax),end="")
            tmask = t[mask[0]]
            ihc = np.abs(k[ik]*tmask - 2*math.pi).argmin() # save the time index corresponding to the horizon crossing
            xdata = log[mask[0]]
            ydata = PT[ik,mask[0]]
            Nparam = 4 # number of parameters for the fitting function
            if len(xdata) >= Nparam and not ik == 0:
                popt, pcov = curve_fit(func, xdata, ydata, maxfev = 20000)
            else:
                popt = [np.nan]*(Nparam)
                pcov = [np.nan]*(Nparam)
            self.param.append(popt)
            self.paramv.append(pcov)
            self.dataP.append(ydata)
            self.listihc.append(ihc)
        if verbose:
            print("")
        self.dataP = np.array(self.dataP)
        self.log = log[mask[0]]
        self.t = t[mask[0]]


#   obsolate?
#
#   perform analytical fit for each mode
#   options for spmask:
#     spmask = 'nomask' -> Fields unmasked
#     spmask = 'Red' -> Red-Gauss (default)
#     spmask = 'Vi' -> Masked with rho/v
#     spmask = 'Vi2' -> Masked with (rho/v)^2
#   options for rmask:
#     rmask = '%.2f' -> label from rmasktable (default 2.00)
#     rmask = 'nolabel' -> just try to read nK_Red without rmasklabel (for old data)
# class fitP:
#     def __init__(self, mfiles, spmask='Red', rmask='2.00', logstart=4.):
#         self.sizeN = pa.gm(mfiles[0],'Size')
#         self.sizeL = pa.gm(mfiles[0],'L')
#         self.msa = pa.gm(mfiles[0],'msa')
#         self.LL = pa.gm(mfiles[0],'lambda0')
#         self.nm = pa.gm(mfiles[0],'nmodelist')
#         self.avek = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm)*(2*math.pi/self.sizeL)
#         # identify modes less than N/2
#         self.k_below = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm) <= self.sizeN/2
#         self.lz2e = pa.gm(mfiles[0],'lz2e')
#         # create lists of the evolution of axion number spectrum (kinetic part)
#         ttab = []
#         logtab = []
#         Ptab = []
#         for meas in mfiles:
#             if pa.gm(meas,'nsp?'):
#                 t = pa.gm(meas,'time')
#                 log = pa.gm(meas,'logi')
#                 if spmask == 'nomask':
#                     print('nomask option is not supported now...')
#                 elif spmask == 'Red':
#                     if rmask == 'nolabel':
#                         binK = pa.gm(meas,'nspK_Red')
#                     else:
#                         binK = pa.gm(meas,'nspK_Red'+'_'+rmask)
#                 elif spmask == 'Vi':
#                     binK = pa.gm(meas,'nspK_Vi')
#                 elif spmask == 'Vi2':
#                     binK = pa.gm(meas,'nspK_Vi2')
#                 else:
#                     print('Wrong option for spmask!')
#                 # P = k^3 N(k)/(2 pi^2) = R^4 drho_a/dk
#                 P = (self.avek**2)*binK/((math.pi**2)*self.nm)
#                 ttab.append(t)
#                 logtab.append(log)
#                 Ptab.append(P)
#         self.t = np.array(ttab)
#         self.log = np.array(logtab)
#         Ptab = np.array(Ptab)
#         # cutoff time (chosen as log(ms/H) = logstart (default 4))
#         istart = np.abs(self.log - logstart).argmin()
#         self.param = []
#         self.paramv = []
#         self.listihc = []
#         self.dataP = []
#         self.datalog = []
#         # transpose
#         PT = np.transpose(Ptab)
#         iterkmax = len(self.avek[self.k_below])
#         for ik in range(iterkmax):
#             print('\rfit: k = %.2f, %d/%d'%(self.avek[ik],ik+1,iterkmax),end="")
#             ihc = np.abs(self.avek[ik]*self.t - 2*math.pi).argmin() # save the time index corresponding to the horizon crossing
#             xdata = self.log[istart:]
#             ydata = PT[ik,istart:]
#             Nparam = 4 # number of parameters for the fitting function
#             if len(xdata) >= Nparam and not ik == 0:
#                 popt, pcov = curve_fit(func, xdata, ydata, maxfev = 20000)
#             else:
#                 popt = [np.nan]*(Nparam)
#                 pcov = [np.nan]*(Nparam)
#             self.param.append(popt)
#             self.paramv.append(pcov)
#             self.dataP.append(ydata)
#             self.datalog.append(xdata)
#             self.listihc.append(ihc)
#         print("")
#         self.dataP = np.array(self.dataP)
#         self.datalog = np.array(self.datalog)





#   obsolate?
# class fitP2:
#     def __init__(self, mfiles, spmasklabel='Red_2.00', cor='nocorrection', logstart=4., verbose=True):
#         self.sizeN = pa.gm(mfiles[0],'Size')
#         self.sizeL = pa.gm(mfiles[0],'L')
#         self.msa = pa.gm(mfiles[0],'msa')
#         self.LL = pa.gm(mfiles[0],'lambda0')
#         self.nm = pa.gm(mfiles[0],'nmodelist')
#         self.avek = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm)*(2*math.pi/self.sizeL)
#         # identify modes less than N/2
#         self.k_below = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm) <= self.sizeN/2
#         self.lz2e = pa.gm(mfiles[0],'lz2e')
#
#         # create lists of the evolution of axion number spectrum (kinetic part)
#         mfnsp = mfiles[pa.gml(mfiles,'nsp?')]
#         ttab = []
#         logtab = []
#         mtab = []
#         Ptab = []
#         istart = np.abs(pa.gml(mfiles,'logi') - logstart).argmin()
#         for meas in mfnsp:
#             if np.where(mfiles==meas) >= istart:
#                 t = pa.gm(meas,'time')
#                 log = pa.gm(meas,'logi')
#                 s0 = pa.gm(meas,'nspK_'+spmasklabel)
#                 if cor == 'correction':
#                     m = pa.gm(meas,'mspM_'+spmasklabel)
#                     binK = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
#                 else:
#                     binK = s0
#                 # P = k^3 N(k)/(2 pi^2) = R^4 drho_a/dk
#                 P = (self.avek**2)*binK/((math.pi**2)*self.nm)
#                 ttab.append(t)
#                 logtab.append(log)
#                 mtab.append(meas)
#                 Ptab.append(P)
#         self.t = np.array(ttab)
#         self.log = np.array(logtab)
#         self.m = mtab
#         Ptab = np.array(Ptab)
#
#         # fitting
#         self.param = []
#         self.paramv = []
#         self.listihc = []
#         self.dataP = []
#         PT = np.transpose(Ptab)
#         iterkmax = len(self.avek[self.k_below])
#         for ik in range(iterkmax):
#             if verbose:
#                 print('\rfit: k = %.2f, %d/%d'%(self.avek[ik],ik+1,iterkmax),end="")
#             ihc = np.abs(self.avek[ik]*self.t - 2*math.pi).argmin() # save the time index corresponding to the horizon crossing
#             xdata = self.log
#             ydata = PT[ik,:]
#             Nparam = 4 # number of parameters for the fitting function
#             if len(xdata) >= Nparam and not ik == 0:
#                 popt, pcov = curve_fit(func, xdata, ydata, maxfev = 20000)
#             else:
#                 popt = [np.nan]*(Nparam)
#                 pcov = [np.nan]*(Nparam)
#             self.param.append(popt)
#             self.paramv.append(pcov)
#             self.dataP.append(ydata)
#             self.listihc.append(ihc)
#         if verbose:
#             print("")
#         self.dataP = np.array(self.dataP)




#   assuming P extrepolated to rmask->0
#class fitPext:
#    def __init__(self, Pext, verbose=True):
#        self.sizeN = Pext.sizeN
#        self.sizeL = Pext.sizeL
#        self.msa = Pext.msa
#        self.LL = Pext.LL
#        self.nm = Pext.nm
#        self.avek = Pext.avek
#        self.k_below = Pext.k_below
#        self.lz2e = Pext.lz2e
#
#        self.t = Pext.t
#        self.log = Pext.log
#
#        # create lists of the evolution of axion number spectrum (kinetic part)
#        Ptab = []
#        for id in range(len(self.log)):
#            P = Pext.param[id][:,1]+Pext.param[id][:,2]+Pext.param[id][:,4]
#            Ptab.append(P)
#        Ptab = np.array(Ptab)
#
#        # fitting
#        self.param = []
#        self.paramv = []
#        self.listihc = []
#        self.dataP = []
#        PT = np.transpose(Ptab)
#        iterkmax = len(self.avek[self.k_below])
#        for ik in range(iterkmax):
#            if verbose:
#                print('\rfit: k = %.2f, %d/%d'%(self.avek[ik],ik+1,iterkmax),end="")
#            ihc = np.abs(self.avek[ik]*self.t - 2*math.pi).argmin() # save the time index corresponding to the horizon crossing
#            xdata = self.log
#            ydata = PT[ik,:]
#            Nparam = 4 # number of parameters for the fitting function
#            if len(xdata) >= Nparam and not ik == 0:
#                popt, pcov = curve_fit(func, xdata, ydata, maxfev = 20000)
#            else:
#                popt = [np.nan]*(Nparam)
#                pcov = [np.nan]*(Nparam)
#            self.param.append(popt)
#            self.paramv.append(pcov)
#            self.dataP.append(ydata)
#            self.listihc.append(ihc)
#        if verbose:
#            print("")
#        self.dataP = np.array(self.dataP)






# ------------------------------------------------------------------------------
#   Instantaneous spectrum
# ------------------------------------------------------------------------------




#   calculate instantaneous spectrum based on analytical fit
class calcF:
    def __init__(self, P, log, t, k, k_below, **kwargs):
        if 'sigma' in kwargs:
            sigma = kwargs['sigma']
        else:
            sigma = 0.1
        if 'chc' in kwargs:
            chc = kwargs['chc']
        else:
            chc = 1.
        if 'logstart' in kwargs:
            logstart = kwargs['logstart']
        else:
            logstart = 4.
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = True
        self.F = [] # instantaneous spectrum F
        self.Fnorm = [] # normalization factor of F
        self.t = [] # time
        self.log = []
        self.x = [] # x-axis (k/RH)
        mask = np.where(log >= logstart)
        logm = log[mask[0]]
        tm = t[mask[0]]
        fitp = fitP(P,log,t,k,chc,logstart,verbose)
        for id in range(len(tm)):
            tt = tm[id]
            logt = logm[id]
            if verbose:
                print('\rcalc F: log = %.2f [%d/%d]'%(logt,id+1,len(tm)),end="")
            Fbin = []
            xbin = []
            Fbinaux = []
            xbinaux = []
            for ik in range(1,len(k)):
                lsup = fitp.paramsup[ik]
                lsub = fitp.paramsub[ik]
                xx = k[ik]*tt
                lxx = math.log(xx)
                lxhc = math.log(k[ik]*t[fitp.listihc[ik]])
                try:
                    if (fitp.listfitsup[ik] and fitp.listfitsub[ik]):
                        Fval = (logistic(lxx,lxhc,sigma)*math.exp(fsuph(lxx,*lsup))*dfsuph(lxx,*lsup)+(1-logistic(lxx,lxhc,sigma))*math.exp(fsubh(lxx,*lsub))*dfsubh(lxx,*lsub))/(xx*(tt**4))
                    elif fitp.listfitsup[ik]:
                        Fval = math.exp(fsuph(lxx,*lsup))*dfsuph(lxx,*lsup)/(xx*(tt**4))
                    elif fitp.listfitsub[ik]:
                        Fval = math.exp(fsubh(lxx,*lsub))*dfsubh(lxx,*lsub)/(xx*(tt**4))
                    else:
                        Fval = np.nan
                except:
                    Fval = np.nan
                if not (np.isnan(Fval) or (Fval < 0)):
                    # Fval can have huge negative value when two fit functions are interpolated
                    # (behavior of the fit function outside their domain may not be well controlled).
                    # We remove such a point for the calculation of the normalization factor.
                    Fbinaux.append(Fval)
                    xbinaux.append(xx)
                Fbin.append(Fval)
                xbin.append(xx)
            Fbinaux = np.array(Fbinaux)
            xbinaux = np.array(xbinaux)
            # normalize
            dx = np.gradient(xbinaux)
            # normalization factor is calculated by using only modes below the Nyquist frequency
            x_below = xbinaux <= np.amax(k[k_below])*tt
            Fdx = (Fbinaux*dx)[x_below]
            self.F.append(np.array(Fbin)/Fdx.sum())
            self.Fnorm.append(Fdx.sum())
            self.x.append(xbin)
            self.t.append(tt)
            self.log.append(logt)
        if verbose:
            print("")
        self.F = np.array(self.F)
        self.Fnorm = np.array(self.Fnorm)
        self.x = np.array(self.x) # x = k/RH
        self.t = np.array(self.t) # time
        self.log = np.array(self.log)




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
    def __init__(self, mfiles, spmask='Red', rmask='2.00', logstart=4.):
        fitp = fitP_old(mfiles,spmask,rmask,logstart)
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
        self.log = []
        self.x = [] # x-axis (k/RH)
        istart = np.abs(fitp.log - logstart).argmin()
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
                self.log.append(log)
        print("")
        self.F = np.array(self.F)
        self.Fnorm = np.array(self.Fnorm)
        self.x = np.array(self.x) # x = k/RH
        self.t = np.array(self.t) # time
        self.log = np.array(self.log)





#   calculate instantaneous spectrum based on analytical fit
#   time steps specified in the arguments
class inspAt:
    def __init__(self, mfiles, logi, logf, nlog, spmask='Red', rmask='2.00', logstart=4.):
        fitp = fitP_old(mfiles,spmask,rmask,logstart)
        self.lz2e = fitp.lz2e
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
        self.t = np.power(np.exp(self.log)/math.sqrt(2.*self.LL),2./(4.-self.lz2e))
        istart = np.abs(self.log - logstart).argmin()
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
    def __init__(self, mfiles, spmask='Red', rmask='2.00'):
        self.lz2e = pa.gm(mfiles[0],'lz2e')
        self.sizeN = pa.gm(mfiles[0],'sizeN')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda0')
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
        self.log = np.log(math.sqrt(2.*self.LL)*np.power(self.t,2.-self.lz2e/2.))






#   calculate instantaneous spectrum based on central difference
class inspC:
    def __init__(self, mfiles, spmask='Red', rmask='2.00'):
        self.sizeN = pa.gm(mfiles[0],'sizeN')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda0')
        self.nm = pa.gm(mfiles[0],'nmodelist')
        self.avek = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm)*(2*math.pi/self.sizeL)
        # identify modes less than N/2
        self.k_below = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm) <= self.sizeN/2
        self.F = [] # instantaneous spectrum F
        self.Fnorm = [] # normalization factor of F
        self.t = [] # time
        self.log = []
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
                log = pa.gm(msplist[id],'logi')
                x = self.avek*t
                dt = t2 - t1
                diff = (sp2 - sp1)/((t**4)*dt)
                # normalize
                dx = np.gradient(x)
                Fdx = (diff*dx)[self.k_below]
                self.F.append(diff/Fdx.sum())
                self.Fnorm.append(Fdx.sum())
                self.t.append(t)
                self.log.append(log)
                self.x.append(x)
        self.F = np.array(self.F)
        self.Fnorm = np.array(self.Fnorm)
        self.x = np.array(self.x) # x = k/RH
        self.t = np.array(self.t) # time
        self.log = np.array(self.log)




#   calculate instantaneous spectrum
#   options for difftype:
#     difftype = 'A' -> analytical fit
#     difftype = 'B' -> backward difference
#     difftype = 'C' -> central difference
class insp:
    def __init__(self, P, log, t, k, k_below, **kwargs):
        if 'difftype' in kwargs:
            difftype = kwargs['difftype']
        else :
            difftype = 'A'
        if 'logstart' in kwargs:
            logstart = kwargs['logstart']
        else :
            logstart = 4
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else :
            verbose = True
        if 'LL' in kwargs:
            LL = kwargs['LL']
        else :
            LL = 25000.
        if 'lz2e' in kwargs:
            lz2e = kwargs['lz2e']
        else :
            lz2e = 2.
        self.F = [] # instantaneous spectrum F
        self.Fnorm = [] # normalization factor of F
        self.t = [] # time
        self.log = []
        self.x = [] # x-axis (k/RH)
        if difftype == 'A':
            fitp = fitP_old(P,log,t,k,logstart,verbose)
            knyq = np.amax(k[k_below])
            for id in range(len(fitp.t)):
                tt = fitp.t[id]
                logt = fitp.log[id]
                if verbose:
                    print('\rcalc F: %d/%d, log = %.2f'%(id+1,len(fitp.t),logt),end="")
                Fbinbuf = []
                x = []
                for ik in range(len(k)):
                    # calculate only modes inside the horizon
                    if id >= fitp.listihc[ik]:
                        l = fitp.param[ik]
                        Fval = dfunc(logt,*l)/(tt**5)
                        if not np.isnan(Fval):
                            Fbinbuf.append(Fval)
                            x.append(k[ik]*tt)
                Fbinbuf = np.array(Fbinbuf)
                # normalize
                dx = np.gradient(x)
                # normalization factor is calculated by using only modes below the Nyquist frequency
                x_below = np.array(x) <= knyq*tt
                Fdx = (Fbinbuf*dx)[x_below]
                self.F.append(Fbinbuf/Fdx.sum())
                self.Fnorm.append(Fdx.sum())
                self.x.append(np.array(x))
                self.t.append(tt)
                self.log.append(logt)
            if verbose:
                print("")
        elif difftype == 'B':
            for id in range(len(log)):
                if verbose:
                    print('\rcalc F: %d/%d, log = %.2f'%(id+1,len(log),log[id]),end="")
                if id != 0:
                    t1 = t[id-1]
                    t2 = t[id]
                    sp1 = P[id-1]
                    sp2 = P[id]
                    dt = t2 - t1
                    tt = t1 + dt/2
                    x = k*tt
                    diff = (sp2 - sp1)/((tt**4)*dt)
                    # normalize
                    dx = np.gradient(x)
                    Fdx = (diff*dx)[k_below]
                    self.F.append(diff/Fdx.sum())
                    self.Fnorm.append(Fdx.sum())
                    self.t.append(tt)
                    self.x.append(x)
                    self.log.append(math.log(math.sqrt(2.*LL)*math.power(tt,2.-lz2e/2.)))
            if verbose:
                print("")
        elif difftype == 'C':
            for id in range(len(log)):
                if verbose:
                    print('\rcalc F: %d/%d, log = %.2f'%(id+1,len(log),log[id]),end="")
                if (id != 0) and (id != len(log)-1):
                    t1 = t[id-1]
                    t2 = t[id+1]
                    sp1 = P[id-1]
                    sp2 = P[id+1]
                    tt = t[id]
                    logt = log[id]
                    x = k*tt
                    dt = t2 - t1
                    diff = (sp2 - sp1)/((tt**4)*dt)
                    # normalize
                    dx = np.gradient(x)
                    Fdx = (diff*dx)[k_below]
                    self.F.append(diff/Fdx.sum())
                    self.Fnorm.append(Fdx.sum())
                    self.t.append(tt)
                    self.log.append(logt)
                    self.x.append(x)
        else:
            print("wrong difftype option!")
        self.F = np.array(self.F)
        self.Fnorm = np.array(self.Fnorm)
        self.x = np.array(self.x) # x = k/RH
        self.t = np.array(self.t) # time
        self.log = np.array(self.log)




#   obsolate?
#
#   calculate instantaneous spectrum
#   options for difftype:
#     difftype = 'A' -> analytical fit
#     difftype = 'B' -> backward difference
#     difftype = 'C' -> central difference
#   options for correction:
#     cor = 'correction' -> spectrum corrected by the matrix
#   options for indices:
#     indices = list of integers
#     if specified, instantaneous spectrum is calculaetd only for time slices selected by the list
# class insp:
#     def __init__(self, mfiles, difftype='C', spmasklabel='Red_2.00', cor='nocorrection', indices=[], logstart=4., verbose=True):
#         self.sizeN = pa.gm(mfiles[0],'sizeN')
#         self.sizeL = pa.gm(mfiles[0],'L')
#         self.msa = pa.gm(mfiles[0],'msa')
#         self.LL = pa.gm(mfiles[0],'lambda0')
#         self.nm = pa.gm(mfiles[0],'nmodelist')
#         self.avek = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm)*(2*math.pi/self.sizeL)
#         self.k_below = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm) <= self.sizeN/2
#         self.lz2e = pa.gm(mfiles[0],'lz2e')
#         self.F = [] # instantaneous spectrum F
#         self.Fnorm = [] # normalization factor of F
#         self.t = [] # time
#         self.log = []
#         self.x = [] # x-axis (k/RH)
#         if indices == []:
#             indices = range(len(mfiles))
#         msplist = [mf for mf in mfiles if pa.gm(mf,'nsp?')]
#         mfilesm = mfiles[indices]
#         msplistm = [mf for mf in mfilesm if pa.gm(mf,'nsp?')]
#         if difftype == 'A':
#             fitp = fitP2(mfiles,spmasklabel,cor,logstart,verbose)
#             iterkmax = len(self.avek[self.k_below])
#             for id in range(len(fitp.t)):
#                 if fitp.m[id] in msplistm:
#                     t = fitp.t[id]
#                     log = fitp.log[id]
#                     if verbose:
#                         print('\rcalc F: %d/%d, log = %.2f'%(msplistm.index(fitp.m[id])+1,len(msplistm),log),end="")
#                     Fbinbuf = []
#                     x = []
#                     for ik in range(iterkmax):
#                         # calculate only modes inside the horizon
#                         if id >= fitp.listihc[ik]:
#                             l = fitp.param[ik]
#                             Fval = dfunc(log,*l)/(t**5)
#                             if not np.isnan(Fval):
#                                 Fbinbuf.append(Fval)
#                                 x.append(self.avek[ik]*t)
#                     Fbinbuf = np.array(Fbinbuf)
#                     # normalize
#                     dx = np.gradient(x)
#                     Fdx = Fbinbuf*dx
#                     self.F.append(Fbinbuf/Fdx.sum())
#                     self.Fnorm.append(Fdx.sum())
#                     self.x.append(np.array(x))
#                     self.t.append(t)
#                     self.log.append(log)
#             if verbose:
#                 print("")
#         elif difftype == 'B':
#             for id in range(len(msplist)):
#                 if msplist[id] in msplistm:
#                     if verbose:
#                         print('\rcalc F: %d/%d, log = %.2f'%(msplistm.index(msplist[id])+1,len(msplistm),pa.gm(msplist[id],'logi')),end="")
#                     if id != 0:
#                         t1 = pa.gm(msplist[id-1],'time')
#                         t2 = pa.gm(msplist[id],'time')
#                         s01 = pa.gm(msplist[id-1],'nspK_'+spmasklabel)
#                         s02 = pa.gm(msplist[id],'nspK_'+spmasklabel)
#                         if cor == 'correction':
#                             m1 = pa.gm(msplist[id-1],'mspM_'+spmasklabel)
#                             m2 = pa.gm(msplist[id],'mspM_'+spmasklabel)
#                             binK1 = (self.sizeL**3)*np.dot(inv(m1),s01/self.nm)
#                             binK2 = (self.sizeL**3)*np.dot(inv(m2),s02/self.nm)
#                         else:
#                             binK1 = s01
#                             binK2 = s02
#                         sp1 = (self.avek**2)*binK1/((math.pi**2)*self.nm)
#                         sp2 = (self.avek**2)*binK2/((math.pi**2)*self.nm)
#                         dt = t2 - t1
#                         t = t1 + dt/2
#                         x = self.avek*t
#                         diff = (sp2 - sp1)/((t**4)*dt)
#                         # normalize
#                         dx = np.gradient(x)
#                         Fdx = (diff*dx)[self.k_below]
#                         self.F.append(diff/Fdx.sum())
#                         self.Fnorm.append(Fdx.sum())
#                         self.t.append(t)
#                         self.x.append(x)
#                         self.log.append(math.log(math.sqrt(2.*self.LL)*math.power(t,2.-self.lz2e/2.)))
#             if verbose:
#                 print("")
#         elif difftype == 'C':
#             for id in range(len(msplist)):
#                 if msplist[id] in msplistm:
#                     if verbose:
#                         print('\rcalc F: %d/%d, log = %.2f'%(msplistm.index(msplist[id])+1,len(msplistm),pa.gm(msplist[id],'logi')),end="")
#                     if (id != 0) and (id != len(msplist)-1):
#                         t1 = pa.gm(msplist[id-1],'time')
#                         t2 = pa.gm(msplist[id+1],'time')
#                         s01 = pa.gm(msplist[id-1],'nspK_'+spmasklabel)
#                         s02 = pa.gm(msplist[id+1],'nspK_'+spmasklabel)
#                         if cor == 'correction':
#                             m1 = pa.gm(msplist[id-1],'mspM_'+spmasklabel)
#                             m2 = pa.gm(msplist[id+1],'mspM_'+spmasklabel)
#                             binK1 = (self.sizeL**3)*np.dot(inv(m1),s01/self.nm)
#                             binK2 = (self.sizeL**3)*np.dot(inv(m2),s02/self.nm)
#                         else:
#                             binK1 = s01
#                             binK2 = s02
#                         sp1 = (self.avek**2)*binK1/((math.pi**2)*self.nm)
#                         sp2 = (self.avek**2)*binK2/((math.pi**2)*self.nm)
#                         t = pa.gm(msplist[id],'time')
#                         log = pa.gm(msplist[id],'logi')
#                         x = self.avek*t
#                         dt = t2 - t1
#                         diff = (sp2 - sp1)/((t**4)*dt)
#                         # normalize
#                         dx = np.gradient(x)
#                         Fdx = (diff*dx)[self.k_below]
#                         self.F.append(diff/Fdx.sum())
#                         self.Fnorm.append(Fdx.sum())
#                         self.t.append(t)
#                         self.log.append(log)
#                         self.x.append(x)
#             if verbose:
#                 print("")
#         else:
#             print("wrong difftype option!")
#         self.F = np.array(self.F)
#         self.Fnorm = np.array(self.Fnorm)
#         self.x = np.array(self.x) # x = k/RH
#         self.t = np.array(self.t) # time
#         self.log = np.array(self.log)





#   calculate instantaneous spectrum from P extrepolated to rmask->0
#   options for difftype:
#     difftype = 'A' -> analytical fit
#     difftype = 'B' -> backward difference
#     difftype = 'C' -> central difference
#   options for indices:
#     indices = list of integers
#     if specified, instantaneous spectrum is calculaetd only for time slices selected by the list
#
#class inspext:
#    def __init__(self, Pext, difftype='C', indices=[], verbose=True):
#        self.sizeN = Pext.sizeN
#        self.sizeL = Pext.sizeL
#        self.msa = Pext.msa
#        self.LL = Pext.LL
#        self.nm = Pext.nm
#        self.avek = Pext.avek
#        self.k_below = Pext.k_below
#        self.lz2e = Pext.lz2e
#        self.F = [] # instantaneous spectrum F
#        self.Fnorm = [] # normalization factor of F
#        self.t = [] # time
#        self.log = []
#        self.x = [] # x-axis (k/RH)
#        if indices == []:
#            indices = range(len(Pext.log))
#        if difftype == 'A':
#            fitp = fitPext(Pext,verbose)
#            iterkmax = len(self.avek[self.k_below])
#            for id in range(len(fitp.t)):
#                if id in indices:
#                    t = fitp.t[id]
#                    log = fitp.log[id]
#                    if verbose:
#                        print('\rcalc F: %d/%d, log = %.2f'%(indices.index(id)+1,len(indices),log),end="")
#                    Fbinbuf = []
#                    x = []
#                    for ik in range(iterkmax):
#                        # calculate only modes inside the horizon
#                        if id >= fitp.listihc[ik]:
#                            l = fitp.param[ik]
#                            Fval = dfunc(log,*l)/(t**5)
#                            if not np.isnan(Fval):
#                                Fbinbuf.append(Fval)
#                                x.append(self.avek[ik]*t)
#                    Fbinbuf = np.array(Fbinbuf)
#                    # normalize
#                    dx = np.gradient(x)
#                    Fdx = Fbinbuf*dx
#                    self.F.append(Fbinbuf/Fdx.sum())
#                    self.Fnorm.append(Fdx.sum())
#                    self.x.append(np.array(x))
#                    self.t.append(t)
#                    self.log.append(log)
#            if verbose:
#                print("")
#        elif difftype == 'B':
#            for id in range(len(Pext.log)):
#                if id in indices:
#                    if verbose:
#                        print('\rcalc F: %d/%d, log = %.2f'%(indices.index(id)+1,len(indices),Pext.log[id]),end="")
#                    if id != 0:
#                        t1 = Pext.t[id-1]
#                        t2 = Pext.t[id]
#                        sp1 = Pext.param[id-1][:,1]+Pext.param[id-1][:,2]+Pext.param[id-1][:,4]
#                        sp2 = Pext.param[id][:,1]+Pext.param[id][:,2]+Pext.param[id][:,4]
#                        dt = t2 - t1
#                        t = tp + dt/2
#                        x = self.avek*t
#                        diff = (sp2 - sp1)/((t**4)*dt)
#                        # normalize
#                        dx = np.gradient(x)
#                        Fdx = (diff*dx)[self.k_below]
#                        self.F.append(diff/Fdx.sum())
#                        self.Fnorm.append(Fdx.sum())
#                        self.t.append(t)
#                        self.x.append(x)
#                        self.log.append(math.log(math.sqrt(2.*self.LL)*math.power(t,2.-self.lz2e/2.)))
#            if verbose:
#                print("")
#        elif difftype == 'C':
#            for id in range(len(Pext.log)):
#                if id in indices:
#                    if verbose:
#                        print('\rcalc F: %d/%d, log = %.2f'%(indices.index(id)+1,len(indices),Pext.log[id]),end="")
#                    if (id != 0) and (id != len(Pext.log)-1):
#                        t1 = Pext.t[id-1]
#                        t2 = Pext.t[id+1]
#                        sp1 = Pext.param[id-1][:,1]+Pext.param[id-1][:,2]+Pext.param[id-1][:,4]
#                        sp2 = Pext.param[id+1][:,1]+Pext.param[id+1][:,2]+Pext.param[id+1][:,4]
#                        t = Pext.t[id]
#                        log = Pext.log[id]
#                        x = self.avek*t
#                        dt = t2 - t1
#                        diff = (sp2 - sp1)/((t**4)*dt)
#                        # normalize
#                        dx = np.gradient(x)
#                        Fdx = (diff*dx)[self.k_below]
#                        self.F.append(diff/Fdx.sum())
#                        self.Fnorm.append(Fdx.sum())
#                        self.t.append(t)
#                        self.log.append(log)
#                        self.x.append(x)
#            if verbose:
#                print("")
#        else:
#            print("wrong difftype option!")
#        self.F = np.array(self.F)
#        self.Fnorm = np.array(self.Fnorm)
#        self.x = np.array(self.x) # x = k/RH
#        self.t = np.array(self.t) # time
#        self.log = np.array(self.log)





#   take ensemble average of the instantaneous spectra
#   assuming a list of inspA/inspB/inspC class objects
class inspave:
    def __init__(self, insplist):
        Nreal = len(insplist) # number of realizations
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
            msoverH = math.exp(inspave.log[id])
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
    sdata(inspave.t,name,'t')
    sdata(inspave.Fnorm,name,'Fnorm')






#   read the data of instantaneous spectra
class readF:
    def __init__(self, name='./F'):
        self.F = rdata(name,'y')
        self.dF = rdata(name,'dy')
        self.x = rdata(name,'x')
        self.log = rdata(name,'log')
        self.t = rdata(name,'t')
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
#   typesigma       option to estimate sigma^2 in the denominator of chi^2
#                        0 : sigma = residuals of different bins
#                        1 : sigma = residuals/sqrt(n_bin)
#                        2 : conservative estimate of "sigma" to define confidence interval based on the maximum value of residuals of different bins
#
# Output:
#   self.chi2min    minimum value of chi^2
#   self.qbest      best fit value of q
#   self.mbest      best fit value of m (normalization of the model)
#   self.sigmaq     "1 sigma" confidence interval of q
#   self.sigmam     "1 sigma" confidence interval of m
#   self.xbin       x-axis for rebinned instantaneous spectrum F(x) where x = k/RH
#   self.Fbin       y-axis for rebinned instantaneous spectrum F(x)
#   self.sigma      "sigma" to define confidence interval based on the residuals of different bins
#
#   self.alphaq      Delta chi^2(q) can be reconstructed by using alpha, beta, gamma:
#   self.betaq       Delta chi^2(q) = (alpha*q^2 + 2*beta*q + gamma)/sigma^2 - chi^2_min
#   self.gammaq
#
#   self.alpham      Delta chi^2(m) can be reconstructed by using alpha, beta, gamma:
#   self.betam       Delta chi^2(m) = (alpha*m^2 + 2*beta*m + gamma)/sigma^2 - chi^2_min
#   self.gammam
#
#   self.nmbin      number of modes in each bin (currently not used)
#   self.xlim       x within the range specified by (xmin,xmax)
#   self.xwr        flags to identify xlim
#
class Setq:
    def __init__(self, inspx, inspy, insplog, id, xmin, xmax, **kwargs):
        if 'nbin' in kwargs:
            nbin = kwargs['nbin']
        else:
            nbin = 30
        if 'typesigma' in kwargs:
            typesigma = kwargs['typesigma']
        else:
            typesigma = 1
        if 'norebin' in kwargs:
            norebin = kwargs['norebin']
        else:
            norebin = False
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
            sigmam = np.nan
            xbin = xlim
            Fbin = inspmlim
            sigma = np.nan
            sigman = np.nan
            sigmac = np.nan
            alphaq = np.nan
            betaq = np.nan
            gammaq = np.nan
            alpham = np.nan
            betam = np.nan
            gammam = np.nan
            nmbin = [1 for i in range(len(xlim))]
        else:
            # do not rebin if number of data points is less than nbin
            if len(xlim) < nbin:
                #print(r' number of data points (%d) is less than nbin (%d)! (log = %.2f)'%(len(xlim),nbin,insplog[id]))
                xbin = xlim
                Fbin = inspmlim
                nmbin = [1 for i in range(len(xlim))]
                nbin = len(xlim)
            # do not rebin for norebin option
            elif norebin==True:
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
            alphaq = Sll - Sl*Sl/Su
            betaq = SlL - Sl*SL/Su
            gammaq = SLL - SL*SL/Su
            alpham = Su - Sl*Sl/Sll
            betam = SlL*Sl/Sll - SL
            gammam = SLL - SlL*SlL/Sll
            qbest = (Sl*SL-SlL*Su)/(Sll*Su-Sl**2)
            mbest = (Sll*SL-SlL*Sl)/(Sll*Su-Sl**2)
            vecone = np.ones(len(xbin))
            if typesigma==0:
                sigmasq = np.sum(np.square(np.log(Fbin)+qbest*np.log(xbin)-mbest*vecone))
            elif typesigma==1:
                sigmasq = np.sum(np.square(np.log(Fbin)+qbest*np.log(xbin)-mbest*vecone))/(nbin)
            elif typesigma==2:
                sigmasq = np.max(np.square(np.log(Fbin)+qbest*np.log(xbin)-mbest*vecone)) # conservative estimate of sigma based on maximum distance from best fit
            else:
                print("wrong typesigma option!")
            sigma = math.sqrt(sigmasq)
            chi2min = np.sum(np.square(np.log(Fbin)+qbest*np.log(xbin)-mbest*vecone))/sigmasq
            sigmaq = math.sqrt(betaq**2-alphaq*(gammaq-sigmasq*(chi2min+Deltachisq)))/alphaq
            sigmam = math.sqrt(betam**2-alpham*(gammam-sigmasq*(chi2min+Deltachisq)))/alpham
        # end of the case len(xlim) >= 4
        self.chi2min = chi2min
        self.qbest = qbest
        self.mbest = mbest
        self.sigmaq = sigmaq
        self.sigmam = sigmam
        self.xbin = xbin
        self.Fbin = Fbin
        self.sigma = sigma
        self.alphaq = alphaq
        self.betaq = betaq
        self.gammaq = gammaq
        self.alpham = alpham
        self.betam = betam
        self.gammam = gammam
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
#   typesigma       option to estimate sigma^2 in the denominator of chi^2
#                        0 : sigma = residuals of different bins
#                        1 : sigma = residuals/sqrt(n_bin)
#                        2 : conservative estimate of "sigma" to define confidence interval based on the maximum value of residuals of different bins
#
# Output:
#   self.chi2min    minimum value of chi^2
#   self.qbest      best fit value of q
#   self.mbest      best fit value of m (normalization of the model)
#   self.sigmaq     "1 sigma" confidence interval of q
#   self.sigmam     "1 sigma" confidence interval of m
#   self.xbin       x-axis for rebinned instantaneous spectrum F(x) where x = k/RH
#   self.Fbin       y-axis for rebinned instantaneous spectrum F(x)
#   self.sigma      "sigma" to define confidence interval based on the mean of residuals of different bins
#
#   self.alphaq      Delta chi^2(q) can be reconstructed by using alpha, beta, gamma:
#   self.betaq       Delta chi^2(q) = (alpha*q^2 + 2*beta*q + gamma)/sigma^2 - chi^2_min
#   self.gammaq
#
#   self.alpham      Delta chi^2(m) can be reconstructed by using alpha, beta, gamma:
#   self.betam       Delta chi^2(m) = (alpha*m^2 + 2*beta*m + gamma)/sigma^2 - chi^2_min
#   self.gammam
#
#   self.nmbin      number of modes in each bin (currently not used)
#   self.log     　　array for log(m/H)
#
class Scanq:
    def __init__(self, inspx, inspy, insplog, cxmin=30., cxmax=1/6., **kwargs):
        if 'nbin' in kwargs:
            nb = kwargs['nbin']
        else:
            nb = 30
        if 'typesigma' in kwargs:
            types = kwargs['typesigma']
        else:
            types = 1
        if 'norebin' in kwargs:
            noreb = kwargs['norebin']
        else:
            noreb = False
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = True
        self.chi2min = []
        self.qbest = []
        self.mbest = []
        self.sigmaq = []
        self.sigmam = []
        self.xbin = []
        self.Fbin = []
        self.sigma = []
        self.alphaq = []
        self.betaq = []
        self.gammaq = []
        self.alpham = []
        self.betam = []
        self.gammam = []
        self.nmbin = []
        for id in range(len(insplog)):
            if verbose==True:
                print('\r%d/%d, log = %.2f'%(id+1,len(insplog),insplog[id]),end="")
            msoverH = math.exp(insplog[id])
            xmin = cxmin
            xmax = cxmax*msoverH
            sqt = Setq(inspx,inspy,insplog,id,xmin,xmax,nbin=nb,typesigma=types,norebin=noreb)
            self.chi2min.append(sqt.chi2min)
            self.qbest.append(sqt.qbest)
            self.mbest.append(sqt.mbest)
            self.sigmaq.append(sqt.sigmaq)
            self.sigmam.append(sqt.sigmam)
            self.xbin.append(sqt.xbin)
            self.Fbin.append(sqt.Fbin)
            self.sigma.append(sqt.sigma)
            self.alphaq.append(sqt.alphaq)
            self.betaq.append(sqt.betaq)
            self.gammaq.append(sqt.gammaq)
            self.alpham.append(sqt.alpham)
            self.betam.append(sqt.betam)
            self.gammam.append(sqt.gammam)
            self.nmbin.append(sqt.nmbin)
        if verbose==True:
            print("")
        self.chi2min = np.array(self.chi2min)
        self.qbest = np.array(self.qbest)
        self.mbest = np.array(self.mbest)
        self.sigmaq = np.array(self.sigmaq)
        self.sigmam = np.array(self.sigmam)
        self.sigma = np.array(self.sigma)
        self.alphaq = np.array(self.alphaq)
        self.betaq = np.array(self.betaq)
        self.gammaq = np.array(self.gammaq)
        self.alpham = np.array(self.alpham)
        self.betam = np.array(self.betam)
        self.gammam = np.array(self.gammam)
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
#   typesigma       option to estimate sigma^2 in the denominator of chi^2
#                        0 : sigma = residuals of different bins
#                        1 : sigma = residuals/sqrt(n_bin)
#                        2 : conservative estimate of "sigma" to define confidence interval based on the maximum value of residuals of different bins
#
# Output:
#   self.chi2min    minimum value of chi^2
#   self.qbest      best fit value of q
#   self.mbest      best fit value of m (normalization of the model)
#   self.sigmaq     "1 sigma" confidence interval of q
#   self.sigmam     "1 sigma" confidence interval of m
#   self.xbin       x-axis for rebinned instantaneous spectrum F(x) where x = k/RH
#   self.Fbin       y-axis for rebinned instantaneous spectrum F(x)
#   self.sigma      "sigma" to define confidence interval based on the mean of residuals of different bins
#
#   self.alphaq      Delta chi^2(q) can be reconstructed by using alpha, beta, gamma:
#   self.betaq       Delta chi^2(q) = (alpha*q^2 + 2*beta*q + gamma)/sigma^2 - chi^2_min
#   self.gammaq
#
#   self.alpham      Delta chi^2(m) can be reconstructed by using alpha, beta, gamma:
#   self.betam       Delta chi^2(m) = (alpha*m^2 + 2*beta*m + gamma)/sigma^2 - chi^2_min
#   self.gammam
#
#   self.nmbin      number of modes in each bin (currently not used)
#   self.log        array for log(m/H)
#   self.cxmaxopt   array for optimized values of cxmax
#
class Scanqopt:
    def __init__(self, inspx, inspy, insplog, cxmin=30., cxmaxstart=0.15, cxmaxend=0.5, cxmaxpoints=200, **kwargs):
        if 'nbin' in kwargs:
            nb = kwargs['nbin']
        else:
            nb = 30
        if 'typesigma' in kwargs:
            types = kwargs['typesigma']
        else:
            types = 1
        if 'norebin' in kwargs:
            noreb = kwargs['norebin']
        else:
            noreb = False
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = True
        self.chi2min = []
        self.qbest = []
        self.mbest = []
        self.sigmaq = []
        self.sigmam = []
        self.xbin = []
        self.Fbin = []
        self.sigma = []
        self.alphaq = []
        self.betaq = []
        self.gammaq = []
        self.alpham = []
        self.betam = []
        self.gammam = []
        self.nmbin = []
        self.cxmaxopt = []
        for id in range(len(insplog)):
            if verbose==True:
                print('\r%d/%d, log = %.2f'%(id+1,len(insplog),insplog[id]),end="")
            msoverH = math.exp(insplog[id])
            xmin = cxmin
            sqt = Setq(inspx,inspy,insplog,id,xmin,cxmaxstart*msoverH,nbin=nb,typesigma=types,norebin=noreb)
            sigmaq = sqt.sigmaq
            copt = cxmaxstart
            for c in np.linspace(cxmaxstart,cxmaxend,cxmaxpoints)[1:]:
                #print('\rcxmax = %.3f'%c)
                xmax = c*msoverH
                sqtt = Setq(inspx,inspy,insplog,id,xmin,xmax,nbin=nb,typesigma=types,norebin=noreb)
                sigmaqt = sqtt.sigmaq
                if sigmaqt < sigmaq:
                    sqt = sqtt
                    sigmaq = sigmaqt
                    copt = c
            #print("")
            self.chi2min.append(sqt.chi2min)
            self.qbest.append(sqt.qbest)
            self.mbest.append(sqt.mbest)
            self.sigmaq.append(sqt.sigmaq)
            self.sigmam.append(sqt.sigmam)
            self.xbin.append(sqt.xbin)
            self.Fbin.append(sqt.Fbin)
            self.sigma.append(sqt.sigma)
            self.alphaq.append(sqt.alphaq)
            self.betaq.append(sqt.betaq)
            self.gammaq.append(sqt.gammaq)
            self.alpham.append(sqt.alpham)
            self.betam.append(sqt.betam)
            self.gammam.append(sqt.gammam)
            self.nmbin.append(sqt.nmbin)
            self.cxmaxopt.append(copt)
        if verbose==True:
            print("")
        self.chi2min = np.array(self.chi2min)
        self.qbest = np.array(self.qbest)
        self.mbest = np.array(self.mbest)
        self.sigmaq = np.array(self.sigmaq)
        self.sigmam = np.array(self.sigmam)
        self.sigma = np.array(self.sigma)
        self.alphaq = np.array(self.alphaq)
        self.betaq = np.array(self.betaq)
        self.gammaq = np.array(self.gammaq)
        self.alpham = np.array(self.alpham)
        self.betam = np.array(self.betam)
        self.gammam = np.array(self.gammam)
        self.log = insplog
        self.cxmaxopt = np.array(self.cxmaxopt)





#   save the data of q as pickle files
#   assuming input as an Scanqopt class object
def saveq(scanqopt, name='./qopt'):
    sdata(scanqopt.qbest,name,'q')
    sdata(scanqopt.mbest,name,'m')
    sdata(scanqopt.sigmaq,name,'sigmaq')
    sdata(scanqopt.sigmam,name,'sigmam')
    sdata(scanqopt.log,name,'log')
    sdata(scanqopt.cxmaxopt,name,'cxmax')






#   read the data of q
class readq:
    def __init__(self, name='./qopt'):
        self.qbest = rdata(name,'q')
        self.mbest = rdata(name,'m')
        self.sigmaq = rdata(name,'sigmaq')
        self.sigmam = rdata(name,'sigmam')
        self.log = rdata(name,'log')
        self.cxmaxopt = rdata(name,'cxmax')





#   take ensemble average of q
#   assuming input as list of Scanq class object
def aveq(qlist):
    Ntime = len(qlist[0].log)
    Nreal = len(qlist)
    q = [0]*(Ntime)
    qsq = [0]*(Ntime)
    for ir in range(Nreal):
        q += qlist[ir].qbest
        qsq += np.square(qlist[ir].qbest)
    q = q/Nreal
    qsq = (qsq - Nreal*q*q)/(Nreal-1)
    sigmaq = np.sqrt(qsq)
    log = qlist[0].log
    return [q,sigmaq,sigmaq/math.sqrt(Nreal),log]





# ------------------------------------------------------------------------------
#   String density
# ------------------------------------------------------------------------------

#   evolution of string density parameter
class strevol:
    def __init__(self, mfiles, diff='nodiff', sigma=1/4., thresh=0.0001):
        self.sizeN = pa.gm(mfiles[0],'Size')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda0')
        self.t = pa.gml(mfiles,'ct')
        self.log = pa.gml(mfiles,'logi')
        self.xi = pa.gml(mfiles,'stDens')
        self.dxidl = []
        self.dxidt = []
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
        try:
            self.sizeN = strevollist[0].sizeN
            self.sizeL = strevollist[0].sizeL
            self.msa = strevollist[0].msa
            self.LL = strevollist[0].LL
        except:
            pass
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
#   Energy
# ------------------------------------------------------------------------------

class energy:
    def __init__(self, mfiles, rmask='0'):
        if rmask=='0':
            self.t = pa.gml(mfiles,'ct')
            self.log = pa.gml(mfiles,'logi')
            self.eA = pa.gml(mfiles,'eA')
            self.eAK = pa.gml(mfiles,'eAK')
            self.eAG = pa.gml(mfiles,'eAG')
            self.eS = pa.gml(mfiles,'eS')
            self.eSK = pa.gml(mfiles,'eSK')
            self.eSG = pa.gml(mfiles,'eSG')
            self.eSV = pa.gml(mfiles,'eSV')
            self.avrho = pa.gml(mfiles,'avrho')
            # physical energy densities (rho_a,rho_s)
            self.rA = 2.*self.eAK/self.t**2/self.avrho**2
            self.rS = self.eS/self.t**2
        else:
            Nt = pa.gm(mfiles[0],'sizeN')**3
            mask = pa.gml(mfiles,'nsp?')
            self.t = pa.gml(mfiles[mask],'ct')
            self.log = pa.gml(mfiles[mask],'logi')
            self.eA = pa.gml(mfiles[mask],'eA')
            self.eAK = pa.gml(mfiles[mask],'eAK')
            self.eAG = pa.gml(mfiles[mask],'eAG')
            self.eS = pa.gml(mfiles[mask],'eS')
            self.eSK = pa.gml(mfiles[mask],'eSK')
            self.eSG = pa.gml(mfiles[mask],'eSG')
            self.eSV = pa.gml(mfiles[mask],'eSV')
            self.avrho = pa.gml(mfiles[mask],'avrho')
            self.eAM = pa.gml(mfiles[mask],'eAmask'+rmask)
            self.eAKM = pa.gml(mfiles[mask],'eAKmask'+rmask)
            self.eAGM = pa.gml(mfiles[mask],'eAGmask'+rmask)
            self.eSM = pa.gml(mfiles[mask],'eSmask'+rmask)
            self.eSKM = pa.gml(mfiles[mask],'eSKmask'+rmask)
            self.eSGM = pa.gml(mfiles[mask],'eSGmask'+rmask)
            self.eSVM = pa.gml(mfiles[mask],'eSVmask'+rmask)
            self.avrhoM = pa.gml(mfiles[mask],'eavrhoMmask'+rmask)
            self.nmp = pa.gml(mfiles[mask],'enmpmask'+rmask)
            self.avrhoout = (self.avrho*Nt-self.avrhoM*self.nmp)/(Nt-self.nmp)
            # physical energy densities (rho_a,rho_s,rho_str)
            self.rA = 2.*(self.eAK*Nt-self.eAKM*self.nmp)/(Nt-self.nmp)/self.t**2/self.avrhoout**2
            self.rS = (self.eS*Nt-self.eSM*self.nmp)/(Nt-self.nmp)/self.t**2
            self.rstr = (self.eA+self.eS)/self.t**2 - self.rA - self.rS




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
    def __init__(self, mfiles, spmask='Red', rmask='2.00', cor='nocorrection'):
        self.sizeN = pa.gm(mfiles[0],'sizeN')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda0')
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
                logi = pa.gm(f,'logi')
                self.logtab.append(logi)
                print('\rbuilt up to log = %.2f'%logi,end="")
        print("")
        self.ttab = np.array(self.ttab)
        self.logtab = np.array(self.logtab)
        self.nsp = np.array(self.nsp)
        self.nspcor = np.array(self.nspcor)




class nspevol2:
    def __init__(self, mfiles, spmasklabel='Red_2.00', cor='nocorrection'):
        self.sizeN = pa.gm(mfiles[0],'sizeN')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda0')
        self.nm = pa.gm(mfiles[0],'nmodelist')
        self.avek = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm)*(2*math.pi/self.sizeL)
        # identify modes less than N/2
        self.k_below = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm) <= self.sizeN/2
        self.ttab = []
        self.logtab = []
        self.nsp = []
        self.nspcor = [] # corrected spectrum
        mfnsp = [mf for mf in mfiles if pa.gm(mf,'nsp?')]
        for f in mfnsp:
            self.ttab.append(pa.gm(f,'time'))
            logi = pa.gm(f,'logi')
            self.logtab.append(logi)
            s0 = pa.gm(f,'nspK_'+spmasklabel)
            self.nsp.append(s0)
            if cor == 'correction':
                m = pa.gm(f,'mspM_'+spmasklabel)
                s1 = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                self.nspcor.append(s1)
            print('\rbuilt up to log = %.2f [%d/%d]'%(logi,mfnsp.index(f)+1,len(mfnsp)),end="")
        print("")
        self.ttab = np.array(self.ttab)
        self.logtab = np.array(self.logtab)
        self.nsp = np.array(self.nsp)
        self.nspcor = np.array(self.nspcor)






#   builds the (masked) axion energy spectrum with the correction matrix and outputs the time evolution
#   NOTE: The energy density is evaluated just by muptiplying the kinetic energy by 2.

class espevol:
    def __init__(self, mfiles, spmask='Red', rmask='2.00', cor='nocorrection'):
        self.sizeN = pa.gm(mfiles[0],'sizeN')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda0')
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
                logi = pa.gm(f,'logi')
                self.logtab.append(logi)
                print('\rbuilt up to log = %.2f'%logi,end="")
        print("")
        self.ttab = np.array(self.ttab)
        self.logtab = np.array(self.logtab)
        self.esp = np.array(self.esp)
        self.espcor = np.array(self.espcor)


class espevol2:
    def __init__(self, mfiles, spmasklabel='Red_2.00', cor='nocorrection'):
        self.sizeN = pa.gm(mfiles[0],'sizeN')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda0')
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





# ------------------------------------------------------------------------------
#   Extrapolation to rmask -> 0
# ------------------------------------------------------------------------------




#   fitting function
def frmask(r, a0, a1, a2, a3, a4, a5):
    return np.log(a0/(1+r**6) + a1 +  a2/np.sqrt(1+a3*r**2) + a4*np.sinc(a5*r))




#   analytical fit of the axion kinetic spectrum as a function of rmask
class Pext:
    def __init__(self, mfiles, rmasktable, verbose=True, rcrit=0, cor='correction', logstart=4.):
        self.sizeN = pa.gm(mfiles[0],'Size')
        self.sizeL = pa.gm(mfiles[0],'L')
        self.msa = pa.gm(mfiles[0],'msa')
        self.LL = pa.gm(mfiles[0],'lambda0')
        self.nm = pa.gm(mfiles[0],'nmodelist')
        self.avek = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm)*(2*math.pi/self.sizeL)
        # identify modes less than N/2
        self.k_below = np.sqrt(pa.gm(mfiles[0],'aveklist')/self.nm) <= self.sizeN/2
        self.lz2e = pa.gm(mfiles[0],'lz2e')

        self.rmasktable = rmasktable

        # create lists of the evolution of axion number spectrum (kinetic part)
        # shape of P array: [log,rmasklabel,k]
        mfnsp = mfiles[pa.gml(mfiles,'nsp?')]
        ttab = []
        logtab = []
        Parr = []
        istart = np.abs(pa.gml(mfiles,'logi') - logstart).argmin()
        if verbose:
            print('creating data lists:')
        for meas in mfnsp:
            if np.where(mfiles==meas) >= istart:
                t = pa.gm(meas,'time')
                log = pa.gm(meas,'logi')
                Psubarr = []
                if verbose:
                    print('log = %.2f'%log)
                for rm in rmasktable:
                    if verbose:
                        print('\rrmask = '+rm,end="")
                    if rm == '0':
                        binK = pa.gm(meas,'nspK_'+rm)
                    else:
                        s0 = pa.gm(meas,'nspK_Red_'+rm)
                        if cor == 'correction':
                            m = pa.gm(meas,'mspM_Red_'+rm)
                            binK = (self.sizeL**3)*np.dot(inv(m),s0/self.nm)
                        else:
                            binK = s0
                    # P = k^3 N(k)/(2 pi^2) = R^4 drho_a/dk
                    P = (self.avek**2)*binK/((math.pi**2)*self.nm)
                    Psubarr.append(P)
                if verbose:
                    print("")
                ttab.append(t)
                logtab.append(log)
                Parr.append(np.array(Psubarr))
        self.t = np.array(ttab)
        self.log = np.array(logtab)
        self.Parr = Parr

        self.rmasklist = []
        for rm in rmasktable:
            self.rmasklist.append(float(rm))
        self.rmasklist = np.array(self.rmasklist)

        # fitting
        msR = self.msa*self.sizeN/self.sizeL
        idrmask_fit = np.where(self.rmasklist >= rcrit*self.msa)
        self.param = []
        if verbose:
            print('fitting the data:')
        for il in np.arange(len(self.log)):
            print('log = %.2f'%self.log[il])
            parambuf = []
            for ik in np.arange(len(self.avek)):
                if verbose:
                    print('\rk = %.2f, %d/%d'%(self.avek[ik],ik+1,len(self.avek)),end="")
                xdata = self.rmasklist[idrmask_fit[0]]/self.msa
                ydata = np.log(self.Parr[il][idrmask_fit[0],ik])
                uosc = self.msa*self.avek[ik]/msR
                pinit=[1,1,1,0.15,1e-2,0.62*uosc]
                pbounds=([0,0,0,0.01,0,0.5*uosc],[np.inf,np.inf,np.inf,0.2,np.inf,0.7*uosc])
                try:
                    popt, pcov = curve_fit(frmask, xdata, ydata, p0=pinit, maxfev = 20000, bounds=pbounds)
                except:
                    popt = [np.nan]*6
                    pcov = [np.nan]*6
                parambuf.append(popt)
            if verbose:
                print("")
            self.param.append(np.array(parambuf))

    def fs(self,r,ik,il):
        return np.exp(frmask(r,self.param[il][ik,0],0,0,0,0,0))

    def fk(self,r,ik,il):
        return np.exp(frmask(r,0,self.param[il][ik,1],0,0,0,0))

    def fw(self,r,ik,il):
        return np.exp(frmask(r,0,0,self.param[il][ik,2],self.param[il][ik,3],0,0))

    def fsinc(self,r,ik,il):
        return np.exp(frmask(r,0,0,0,0,self.param[il][ik,4],self.param[il][ik,5]))

    def xdata(self):
        return self.rmasklist/self.msa

    def ydata(self,ik,il):
        return self.Parr[il][:,ik]




#   save the data of P[rmask] as pickle files
#   assuming input as an Pext class object
def savePext(Pext, name='./Pext'):
    sdata(Pext.sizeN,name,'sizeN')
    sdata(Pext.sizeL,name,'sizeL')
    sdata(Pext.msa,name,'msa')
    sdata(Pext.LL,name,'LL')
    sdata(Pext.nm,name,'nm')
    sdata(Pext.avek,name,'avek')
    sdata(Pext.k_below,name,'k_below')
    sdata(Pext.lz2e,name,'lz2e')
    sdata(Pext.rmasktable,name,'rmasktable')
    sdata(Pext.t,name,'t')
    sdata(Pext.log,name,'log')
    sdata(Pext.Parr,name,'Parr')
    sdata(Pext.rmasklist,name,'rmasklist')
    sdata(Pext.param,name,'param')




#   read the data of P[rmask]
class readPext:
    def __init__(self, name='./Pext'):
        self.sizeN = rdata(name,'sizeN')
        self.sizeL = rdata(name,'sizeL')
        self.msa = rdata(name,'msa')
        self.LL = rdata(name,'LL')
        self.nm = rdata(name,'nm')
        self.avek = rdata(name,'avek')
        self.k_below = rdata(name,'k_below')
        self.lz2e = rdata(name,'lz2e')
        self.rmasktable = rdata(name,'rmasktable')
        self.t = rdata(name,'t')
        self.log = rdata(name,'log')
        self.Parr = rdata(name,'Parr')
        self.rmasklist = rdata(name,'rmasklist')
        self.param = rdata(name,'param')

    def fs(self,r,ik,il):
        return np.exp(frmask(r,self.param[il][ik,0],0,0,0,0,0))

    def fk(self,r,ik,il):
        return np.exp(frmask(r,0,self.param[il][ik,1],0,0,0,0))

    def fw(self,r,ik,il):
        return np.exp(frmask(r,0,0,self.param[il][ik,2],self.param[il][ik,3],0,0))

    def fsinc(self,r,ik,il):
        return np.exp(frmask(r,0,0,0,0,self.param[il][ik,4],self.param[il][ik,5]))

    def xdata(self):
        return self.rmasklist/self.msa

    def ydata(self,ik,il):
        return self.Parr[il][:,ik]






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
