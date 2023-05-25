#!/usr/bin/python3

import numpy as np
import math
import pickle
from pyaxions import jaxions as pa
from scipy.optimize import curve_fit
from scipy import fftpack


def sdata(data, name, dataname):
    pname = name + '_' + dataname + '.pickle'
    with open(pname,'wb') as w:
        pickle.dump(data, w)

def rdata(name, dataname):
    pname = name + '_' + dataname + '.pickle'
    with open(pname,'rb') as r:
        return pickle.load(r)


# ------------------------------------------------------------------------------
#   energy and number spectrum
# ------------------------------------------------------------------------------

class espevol:
    def __init__(self, mfiles, esplabel='espK_0'):
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
        mfnsp = [mf for mf in mfiles if pa.gm(mf,'esp?')]
        for f in mfnsp:
            self.ttab.append(pa.gm(f,'time'))
            logi = pa.gm(f,'logi')
            self.logtab.append(logi)
            sK = pa.gm(f,esplabel)
            self.esp.append(sK)
            #print('\rbuilt up to log = %.2f [%d/%d]'%(logi,mfnsp.index(f)+1,len(mfnsp)),end="")
        #print("")
        self.ttab = np.array(self.ttab)
        self.logtab = np.array(self.logtab)
        self.esp = np.array(self.esp)

def saveesp(espe, name='./esp', esplabel='espK_0', esponly=False):
    sdata(espe.esp,name,esplabel)
    if not esponly:
        sdata(espe.nm,name,'nm')
        sdata(espe.avek,name,'k')
        sdata(espe.k_below,name,'k_below')
        sdata(espe.ttab,name,'t')
        sdata(espe.logtab,name,'log')

class readS:
    def __init__(self,dataname='./Sdata/S',esplabel='espK_0'):
        self.dataname = dataname
        self.nm = rdata(dataname,'nm')
        self.k = rdata(dataname,'k')
        self.k_below = rdata(dataname,'k_below')
        self.t = rdata(dataname,'t')
        self.log = rdata(dataname,'log')
        self.esp = rdata(dataname,esplabel)
        
        self.Narr = []
        self.Earr = []
        for id in range(len(self.log)):
            self.Narr.append(((self.k**2)*self.esp[id]/self.nm)/((math.pi**2)*self.t[id]))
            self.Earr.append((self.k**3)*self.esp[id]/self.nm/(math.pi**2))
        self.Narr = np.array(self.Narr)
        self.Earr = np.array(self.Earr)
    
    # print number spectrum N = drho/d(k/R)/(H*f_a^2) at a given log
    def N(self,log):
        it = np.abs(self.log - log).argmin()
        return ((self.k**2)*self.esp[it]/self.nm)/((math.pi**2)*self.t[it])
        
    # print energy spectrum E = drho/d(logk)/(H*f_a)^2 at a given log
    def E(self,log):
        it = np.abs(self.log - log).argmin()
        return (self.k**3)*self.esp[it]/self.nm/(math.pi**2)

    # print mode evolution
    def Nevol(self,ik):
        return ((self.k[ik]**2)*self.esp[:,ik]/self.nm[ik])/((math.pi**2)*self.t)
    
    def Eevol(self,ik):
        return (self.k[ik]**3)*self.esp[:,ik]/self.nm[ik]/(math.pi**2)


# ------------------------------------------------------------------------------
#   analytical fit of the mode evolution
# ------------------------------------------------------------------------------

# fit functions
def f2(lx, a0, a1, a2, a3, a4):
    return a0 + a1*lx - np.log(1 + (a2*np.exp(lx))**(a3 + a4*lx))
def f3(lx, a0, a1, a2, a3, a4, a5):
    return a0 + a1*lx - np.log(1 + (a2*np.exp(lx))**(a3 + a4*lx + a5*(lx**2)))
def f4(lx, a0, a1, a2, a3, a4, a5, a6):
    return a0 + a1*lx - np.log(1 + (a2*np.exp(lx))**(a3 + a4*lx + a5*(lx**2) + a6*(lx**3)))
def f5(lx, a0, a1, a2, a3, a4, a5, a6, a7):
    return a0 + a1*lx - np.log(1 + (a2*np.exp(lx))**(a3 + a4*lx + a5*(lx**2) + a6*(lx**3) + a7*(lx**4)))

# use large log approximation for modes that were alredy inside the horizon at the beginning
def f2a(lx, a0, a1, a2):
    return a0 + a1*lx + a2*(lx**2)
def f3a(lx, a0, a1, a2, a3):
    return a0 + a1*lx + a2*(lx**2) + a3*(lx**3)
def f4a(lx, a0, a1, a2, a3, a4):
    return a0 + a1*lx + a2*(lx**2) + a3*(lx**3) + a4*(lx**4)
def f5a(lx, a0, a1, a2, a3, a4, a5):
    return a0 + a1*lx + a2*(lx**2) + a3*(lx**3) + a4*(lx**4) + a5*(lx**5)

# derivatives
def df2(lx, a0, a1, a2, a3, a4):
    ex = (a2*np.exp(lx))**(a3 + a4*lx)
    numer = a4*(np.log(a2)+lx) + a3 + a4*lx
    return a1 - numer*(ex/(1+ex))

def df3(lx, a0, a1, a2, a3, a4, a5):
    ex = (a2*np.exp(lx))**(a3 + a4*lx + a5*(lx**2))
    numer = (a4 + 2*a5*lx)*(np.log(a2)+lx) + a3 + a4*lx + a5*(lx**2)
    return a1 - numer*(ex/(1+ex))

def df4(lx, a0, a1, a2, a3, a4, a5, a6):
    ex = (a2*np.exp(lx))**(a3 + a4*lx + a5*(lx**2) + a6*(lx**3))
    numer = (a4 + 2*a5*lx + 3*a6*(lx**2))*(np.log(a2)+lx) + a3 + a4*lx + a5*(lx**2) + a6*(lx**3)
    return a1 - numer*(ex/(1+ex))

def df5(lx, a0, a1, a2, a3, a4, a5, a6, a7):
    ex = (a2*np.exp(lx))**(a3 + a4*lx + a5*(lx**2) + a6*(lx**3) + a7*(lx**4))
    numer = (a4 + 2*a5*lx + 3*a6*(lx**2) + 4*a7*(lx**3))*(np.log(a2)+lx) + a3 + a4*lx + a5*(lx**2) + a6*(lx**3) + a7*(lx**4)
    return a1 - numer*(ex/(1+ex))

def df2a(lx, a0, a1, a2):
    return a1 + 2*a2*lx
def df3a(lx, a0, a1, a2, a3):
    return a1 + 2*a2*lx + 3*a3*(lx**2)
def df4a(lx, a0, a1, a2, a3, a4):
    return a1 + 2*a2*lx + 3*a3*(lx**2) + 4*a4*(lx**3)
def df5a(lx, a0, a1, a2, a3, a4, a5):
    return a1 + 2*a2*lx + 3*a3*(lx**2) + 4*a4*(lx**3) + 5*a5*(lx**4)
    
def ftrend(lx, order, *args):
    if order == 2:
        return f2(lx,*args)
    elif order == 3:
        return f3(lx,*args)
    elif order == 4:
        return f4(lx,*args)
    elif order == 5:
        return f5(lx,*args)
    
def ftrenda(lx, order, *args):
    if order == 2:
        return f2a(lx,*args)
    elif order == 3:
        return f3a(lx,*args)
    elif order == 4:
        return f4a(lx,*args)
    elif order == 5:
        return f5a(lx,*args)

def dftrend(lx, order, *args):
    if order == 2:
        return df2(lx,*args)
    elif order == 3:
        return df3(lx,*args)
    elif order == 4:
        return df4(lx,*args)
    elif order == 5:
        return df5(lx,*args)
    
def dftrenda(lx, order, *args):
    if order == 2:
        return df2a(lx,*args)
    elif order == 3:
        return df3a(lx,*args)
    elif order == 4:
        return df4a(lx,*args)
    elif order == 5:
        return df5a(lx,*args)

def filterf(p, xh, x, y):
    if math.exp(x[0]) < xh:
        Np = p+3
        try:
            if p==2:
                bo = ((-np.inf,4.,0.2,3.0,-np.inf),(np.inf,6.,np.inf,4.0,np.inf))
                par, parv = curve_fit(f2, x, y, bounds=bo, maxfev = 20000)
            elif p==3:
                bo = ((-np.inf,4.,0.2,3.0,-np.inf,-2.),(np.inf,6.,np.inf,4.0,np.inf,2.))
                par, parv = curve_fit(f3, x, y, bounds=bo, maxfev = 20000)
            elif p==4:
                bo = ((-np.inf,4.,0.2,3.0,-np.inf,-2.,-1.),(np.inf,6.,np.inf,4.0,np.inf,2.,1.))
                par, parv = curve_fit(f4, x, y, bounds=bo, maxfev = 20000)
            elif p==5:
                bo = ((-np.inf,4.,0.2,3.0,-np.inf,-2.,-1.,-1.),(np.inf,6.,np.inf,4.0,np.inf,2.,1.,1.))
                par, parv = curve_fit(f5, x, y, bounds=bo, maxfev = 20000)
            flag = True
        except:
            par = [np.nan]*(Np)
            parv = [np.nan]*(Np)
            flag = False
    else:
        Np = p+1
        try:
            if p==2:
                par, parv = curve_fit(f2a, x, y, maxfev = 20000)
            elif p==3:
                par, parv = curve_fit(f3a, x, y, maxfev = 20000)
            elif p==4:
                par, parv = curve_fit(f4a, x, y, maxfev = 20000)
            elif p==5:
                par, parv = curve_fit(f5a, x, y, maxfev = 20000)
            flag = True
        except:
            par = [np.nan]*(Np)
            parv = [np.nan]*(Np)
            flag = False
    return par, parv, flag

# For a given value of lambda_physical, calculate red-shift exponent for the saxion spectrum
def saxionZ(k, t, LL, lz2e=0):
    mr = np.sqrt(2.*LL)/t**(.5*lz2e)
    r = k/mr/t
    return 3 + (r**2 + 0.5*lz2e)/(1 + r**2)
    
# Perform analytical fit
# Switch to a simplified fit function when x[0] becomes larger than xh (default xh = -1)
# If xh = -1 (or some negative value), do not use this simplification (no discontinuity but time-consuming)
# If saxionmass = True, it takes account of the non-trivial redshift for saxion spectrum
class fitS:
    def __init__(self, data, log, t, k, **kwargs):
        if 'p' in kwargs:
            p = kwargs['p']
        else:
            p = 3
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = 1
        if 'logstart' in kwargs:
            logstart = kwargs['logstart']
        else:
            logstart = 4.
        if 'xh' in kwargs:
            xh = kwargs['xh']
        else:
            xh = -1
        if 'saxionmass' in kwargs:
            saxionmass = kwargs['saxionmass']
        else:
            saxionmass = False
        if 'LL' in kwargs:
            LL = kwargs['LL']
        else:
            LL = 1600.0
        if 'lz2e' in kwargs:
            lz2e = kwargs['lz2e']
        else:
            lz2e = 0
        if p not in [2,3,4,5]:
            print("order of polynomial (p) not supported")
            return None
        self.param = []
        self.paramv = []
        self.listfit = []
        mask = np.where(log >= logstart)
        # identify ik at which x[0] becomes larger than xh
        if xh < 0:
            xh = t[-1]*k[-1]
        if saxionmass:
            xh = 0 # For saxion modes, use simple polynomial functions from the beginning
        tt = t[mask[0]]
        x0 = tt[0]*k
        his = np.abs(x0-xh).argmin()
        self.ikhistart = his
        if x0[his] < xh:
            self.ikhistart = his+1
        for ik in range(len(k)):
            if verbose == 1:
                print('\rfit:  k = %.2f [%d/%d]'%(k[ik],ik+1,len(k)),end="",flush=True)
            elif verbose == 2:
                print('fit:  k = %.2f [%d/%d]'%(k[ik],ik+1,len(k)),flush=True)
            xdata = np.log(k[ik]*tt)
            if saxionmass:
                zz = saxionZ(k[ik],tt,LL,lz2e)
                ydata = np.log(data[mask[0],ik]*(tt**(zz-4)))
            else:
                ydata = np.log(data[mask[0],ik])
            if not ik == 0:
                par, parv, flag = filterf(p,xh,xdata,ydata)
                self.listfit.append(flag)
            else:
                par = [np.nan]*(p+3)
                parv = [np.nan]*(p+3)
                self.listfit.append(False)
            self.param.append(par)
            self.paramv.append(parv)
        if verbose:
            print("")

# save parameters in fitS class object (fS)
def saveParam(fS, name='./P'):
    sdata(fS.param,name,'param')
    sdata(fS.listfit,name,'listfit')
    sdata(fS.ikhistart,name,'ikhistart')
    
# read parameters
class readParam:
    def __init__(self, name='./P'):
        self.param = rdata(name,'param')
        self.listfit = rdata(name,'listfit')
        self.ikhistart = rdata(name,'ikhistart')


# ------------------------------------------------------------------------------
#   instantaneous spectrum
# ------------------------------------------------------------------------------

# subtract linear trend
def subtraction(res, w):
    a = (res[-1]-res[0])/(w[-1]-w[0])
    b = (res[0]*w[-1]-res[-1]*w[0])/(w[-1]-w[0])
    lin = a*w+b
    res2 = res - a*w-b
    return res2, lin, a, b
    
# return alias frequency for signal frequency (f) and sample frequency (fs)
def falias(f,fs):
    n = np.floor(f/fs)
    return min(np.abs(f - n*fs),np.abs(f - (n+1)*fs))

def checkfreq(fs, fcrit, ftol):
    n = np.floor(fcrit/fs)
    fc1 = ftol + n*fs
    fc2 = (n+1)*fs - ftol
    fc3 = ftol + (n+1)*fs
    if fcrit < fc1:
        return fc1
    elif fc1 <= fcrit < fc2:
        return fc2
    else:
        return fc3
        
def filtergauss(k, res, t, nmeas, fcutmin=1., antialiasing=True, sigma=0.25, lambdatype='z2', LL=1600., cftol=0.5):
    fs = nmeas/(t[-1]-t[0])       # sample frequency
    fn = k/np.pi                  # frequency of 2k axion oscillation
    ftol = min(cftol*fs/2.,fs/2.) # factor cftol times Nyquist crequency
    if ftol < 0:
        ftol = 0
    if antialiasing:
        if lambdatype == 'fixed':
            fcrit = t[-1]*np.sqrt(0.5*LL)/np.pi  # critical frequency above which modes do not exhibit the saxion mass crossing
            fc = checkfreq(fs,fcrit,ftol)
            if fn < fc:
                fa = falias(fn,fs)
            else:
                fa = ftol
        else:
            fa = falias(fn,fs)
    else:
        fa = fn
    wa = 2*np.pi*max(fa,fcutmin)
    res_sub, lin, a, b = subtraction(res, t*k)
    sig_dst = fftpack.dst(res_sub,norm='ortho',type=1)
    w_dst = np.pi*np.linspace(1,res_sub.size,res_sub.size)/(t[-1]-t[0])
    sig_dst_filtered = sig_dst*np.exp(-(w_dst/(wa*sigma))**2/2)
    filtered_dst = fftpack.idst(sig_dst_filtered,norm='ortho',type=1)
    return filtered_dst + lin
    
    
class calcF:
    def __init__(self, data, log, t, k, k_below, **kwargs):
        if 'p' in kwargs:
            po = kwargs['p']
        else:
            po = 2
        if 'logstart' in kwargs:
            logst = kwargs['logstart']
        else:
            logst = 4.
        if 'verbose' in kwargs:
            verb = kwargs['verbose']
        else:
            verb = True
        if 'usedata' in kwargs:
            usedata = kwargs['usedata']
        else:
            usedata = False
        if 'fitp' in kwargs:
            fitin = kwargs['fitp']
        else:
            fitin = []
        if 'xh' in kwargs:
            xhi = kwargs['xh']
        else:
            xhi = -1
        if 'saxionmass' in kwargs:
            saxionmassi = kwargs['saxionmass']
        else:
            saxionmassi = False
        if 'LL' in kwargs:
            LLi = kwargs['LL']
        else:
            LLi = 1600.0
        if 'lz2e' in kwargs:
            lz2ei = kwargs['lz2e']
        else:
            lz2ei = 0
        if 'antialiasing' in kwargs:
            antialiasing = kwargs['antialiasing']
        else:
            antialiasing = False
        if 'sigma' in kwargs:
            sigma = kwargs['sigma']
        else:
            sigma = 0.25
        if 'nmeas' in kwargs:
            nmeas = kwargs['nmeas']
        else:
            nmeas = 250
        if 'fcutmin' in kwargs:
            fcutmin = kwargs['fcutmin']
        else:
            fcutmin = 3.0
        if 'lambdatype' in kwargs:
            lambdatype = kwargs['lambdatype']
        else:
            lambdatype = 'z2'
        if 'cftol' in kwargs:
            cftol = kwargs['cftol']
        else:
            cftol = 0.5
        mask = np.where(log >= logst)
        logm = log[mask[0]]
        tm = t[mask[0]]
        if usedata:
            fitp = fitin
        else:
            fitp = fitS(data,log,t,k,p=po,verbose=verb,logstart=logst,xh=xhi,saxionmass=saxionmassi,LL=LLi,lz2e=lz2ei)
        Farr = [] # instantaneous spectrum F
        xarr = []
        Farr_aux = []
        xarr_aux = []
        Farr_fit = [] # instantaneous spectrum F (fit only)
        for ik in range(1,len(k)):
            #if verbose:
            #    print('\rcalc F (differentiate): k = %.2f [%d/%d]'%(k[ik],ik+1,len(k)),end="")
            par = fitp.param[ik]
            xx = k[ik]*tm
            lxx = np.log(xx)
            if fitp.listfit[ik]:
                if ik < fitp.ikhistart:
                    if saxionmassi:
                        zz = saxionZ(k[ik],tm,LLi,lz2ei)
                        Fk_fit = np.exp(ftrend(lxx,po,*par))*dftrend(lxx,po,*par)/xx/(tm**(zz-4))
                        res = data[mask[0],ik]*(tm**(zz-4)) - np.exp(ftrend(lxx,po,*par))
                    else:
                        Fk_fit = np.exp(ftrend(lxx,po,*par))*dftrend(lxx,po,*par)/xx
                        res = data[mask[0],ik] - np.exp(ftrend(lxx,po,*par))
                    fres = filtergauss(k[ik],res,tm,nmeas,fcutmin,antialiasing,sigma,lambdatype,LLi,cftol)
                    if saxionmassi:
                        Fk = Fk_fit + np.gradient(fres,xx[1]-xx[0],edge_order=2)/(tm**(zz-4))
                    else:
                        Fk = Fk_fit + np.gradient(fres,xx[1]-xx[0],edge_order=2)
                else:
                    if saxionmassi:
                        zz = saxionZ(k[ik],tm,LLi,lz2ei)
                        Fk_fit = np.exp(ftrenda(lxx,po,*par))*dftrenda(lxx,po,*par)/xx/(tm**(zz-4))
                        res = data[mask[0],ik]*(tm**(zz-4)) - np.exp(ftrenda(lxx,po,*par))
                    else:
                        Fk_fit = np.exp(ftrenda(lxx,po,*par))*dftrenda(lxx,po,*par)/xx
                        res = data[mask[0],ik] - np.exp(ftrenda(lxx,po,*par))
                    fres = filtergauss(k[ik],res,tm,nmeas,fcutmin,antialiasing,sigma,lambdatype,LLi,cftol)
                    if saxionmassi:
                        Fk = Fk_fit + np.gradient(fres,xx[1]-xx[0],edge_order=2)/(tm**(zz-4))
                    else:
                        Fk = Fk_fit + np.gradient(fres,xx[1]-xx[0],edge_order=2)
            else:
                Fk_fit = [np.nan]*len(tm)
                Fk = [np.nan]*len(tm)
            if not np.isnan(np.sum(Fk)):
                Farr_aux.append(Fk)
                xarr_aux.append(xx)
            Farr.append(Fk)
            xarr.append(xx)
            Farr_fit.append(Fk_fit)
        #if verbose:
        #    print("")
        Farr = np.transpose(np.array(Farr))
        xarr = np.transpose(np.array(xarr))
        Farr_aux = np.transpose(np.array(Farr_aux))
        xarr_aux = np.transpose(np.array(xarr_aux))
        Farr_fit = np.transpose(np.array(Farr_fit))
        # normalization factor
        Fnorm = []
        for id in range(len(tm)):
            dx = np.gradient(xarr_aux[id,:])
            # normalization factor is calculated by using only modes below the Nyquist frequency
            x_below = xarr_aux[id,:] <= np.amax(k[k_below])*tm[id]
            Fdx = (Farr_aux[id,:]*dx)[x_below]
            Fnorm.append(Fdx.sum())
        self.F = Farr # instantaneous spectrum F
        self.x = xarr
        self.F_fit = Farr_fit # instantaneous spectrum F (fit only)
        self.Fnorm = np.array(Fnorm) # normalization factor of F
        self.t = tm
        self.log = logm
        

class calcFdiff:
    def __init__(self, data, log, t, k, k_below, difftype='C', **kwargs):
        if 'saxionmass' in kwargs:
            saxionmassi = kwargs['saxionmass']
        else:
            saxionmassi = False
        if 'LL' in kwargs:
            LLi = kwargs['LL']
        else:
            LLi = 1600.0
        if 'lz2e' in kwargs:
            lz2ei = kwargs['lz2e']
        else:
            lz2ei = 0
        Farr = [] # instantaneous spectrum F
        xarr = []
        Fnorm = []
        tarr = []
        logarr = []
        for id in range(len(log)):
            try:
                if (difftype == 'B' or difftype == 'backward') and id != 0:
                    t1 = t[id-1]
                    t2 = t[id]
                    if saxionmassi:
                        z1 = saxionZ(k[1:],t1,LLi,lz2ei)
                        z2 = saxionZ(k[1:],t2,LLi,lz2ei)
                        sp1 = data[id-1,1:]*t1**(z1-4)
                        sp2 = data[id,1:]*t2**(z2-4)
                    else:
                        sp1 = data[id-1,1:]
                        sp2 = data[id,1:]
                        tt = (t1 + t2)/2
                elif (difftype == 'C' or difftype == 'central') and id != 0:
                    t1 = t[id-1]
                    t2 = t[id+1]
                    if saxionmassi:
                        z1 = saxionZ(k[1:],t1,LLi,lz2ei)
                        z2 = saxionZ(k[1:],t2,LLi,lz2ei)
                        sp1 = data[id-1,1:]*t1**(z1-4)
                        sp2 = data[id+1,1:]*t2**(z2-4)
                    else:
                        sp1 = data[id-1,1:]
                        sp2 = data[id+1,1:]
                    tt = t[id]
                dt = t2 - t1
                if saxionmassi:
                    zz = saxionZ(k[1:],tt,LLi,lz2ei)
                    diff = (sp2 - sp1)/(k[1:]*dt)/tt**(zz-4)
                else:
                    diff = (sp2 - sp1)/(k[1:]*dt)
                Farr.append(diff)
                xarr.append(k[1:]*tt)
                tarr.append(tt)
                logarr.append(log[id])
                # normalization factor
                dx = np.gradient(k[1:]*tt)
                # normalization factor is calculated by using only modes below the Nyquist frequency
                x_below = k[1:]*tt <= np.amax(k[k_below])*tt
                Fdx = (diff*dx)[x_below]
                Fnorm.append(Fdx.sum())
            except:
                pass
        self.F = np.array(Farr)
        self.x = np.array(xarr)
        self.Fnorm = np.array(Fnorm)
        self.t = np.array(tarr)
        self.log = np.array(logarr)
        
        
# ------------------------------------------------------------------------------
#   spectral index
# ------------------------------------------------------------------------------

# class setq:
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
class setq:
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
        if 'discardnegative' in kwargs:
            discneg = kwargs['discardnegative']
        else:
            discneg = False
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
            flag_exception = False 
            if discneg==True:
                xbin = xbin[Fbin > 0]
                Fbin = Fbin[Fbin > 0]
                nbin = len(Fbin)
                if nbin < 4:
                    print(r'number of data points becomes less than 4 after discarding bins with negative F! (log = %.2f)'%insplog[id])
                    flag_exception = True
            Su = nbin
            Sl = np.sum(np.log(xbin))
            Sll = np.sum(np.log(xbin)**2)
            SL = np.sum(np.log(Fbin))
            SLL = np.sum(np.log(Fbin)**2)
            SlL = np.sum(np.log(xbin)*np.log(Fbin))
            if flag_exception:
                # if discneg==True and nbin < 4 we do not trust the result
                alphaq = np.nan
                betaq = np.nan
                gammaq = np.nan
                alpham = np.nan
                betam = np.nan
                gammam = np.nan
                qbest = np.nan
                mbest = np.nan
            else:
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
            if flag_exception:
                chi2min = np.nan
                sigma = np.nan
                sigmaq = np.nan
                sigmam = np.nan
            else:
                chi2min = np.sum(np.square(np.log(Fbin)+qbest*np.log(xbin)-mbest*vecone))/sigmasq
                sigma = math.sqrt(sigmasq)
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


# class scanq:
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
class scanq:
    def __init__(self, inspx, inspy, insplog, cxmin=50., cxmax=1/4., **kwargs):
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
        if 'discardnegative' in kwargs:
            discneg = kwargs['discardnegative']
        else:
            discneg = False
        if 'skipnan' in kwargs:
            sknan = kwargs['skipnan']
        else:
            sknan = False
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
        self.log = []
        for id in range(len(insplog)):
            if verbose==True:
                print('\r%d/%d, log = %.2f'%(id+1,len(insplog),insplog[id]),end="")
            msoverH = math.exp(insplog[id])
            xmin = cxmin
            xmax = cxmax*msoverH
            sqt = setq(inspx,inspy,insplog,id,xmin,xmax,nbin=nb,typesigma=types,norebin=noreb,discardnegative=discneg)
            if np.isnan(sqt.qbest) and sknan:
                pass
            else:
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
                self.log.append(insplog[id])
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
        self.log = np.array(self.log)
        self.cxmaxopt = cxmax


#   take ensemble average of q
#   assuming input as list of scanq class object
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
#   energy radiation rate
# ------------------------------------------------------------------------------

def calcGamma(energy, t, log, **kwargs):
    if 'p' in kwargs:
        p = kwargs['p']
    else:
        p = 2
    if 'logstart' in kwargs:
        logstart = kwargs['logstart']
    else:
        logstart = 4.
    if 'sigma' in kwargs:
        sigma = kwargs['sigma']
    else:
        sigma = 0.25
    if p not in [2,3,4,5]:
        print("order of polynomial (p) not supported")
        return None

    li = np.abs(log - logstart).argmin()
    enem = energy[li:]
    tm = t[li:]
    
    if 'z' in kwargs:
        z = kwargs['z']
        zm = z[li:]
    else:
        zm = np.full(len(tm),4.)

    freq = np.pi*np.linspace(1,enem.size,enem.size)/(tm[-1]-tm[0])
    freqN = np.pi*enem.size/(tm[-1]-tm[0])
    
    if p==2:
        param, paramv = curve_fit(f2a, np.log(tm), np.log(enem*(tm**zm)))
    elif p==3:
        param, paramv = curve_fit(f3a, np.log(tm), np.log(enem*(tm**zm)))
    elif p==4:
        param, paramv = curve_fit(f4a, np.log(tm), np.log(enem*(tm**zm)))
    elif p==5:
        param, paramv = curve_fit(f5a, np.log(tm), np.log(enem*(tm**zm)))
        
    res = enem*(tm**zm) - np.exp(ftrenda(np.log(tm),p,*param))
    
    # subtract linear trend
    a = (res[-1]-res[0])/(tm[-1]-tm[0])
    b = (res[0]*tm[-1]-res[-1]*tm[0])/(tm[-1]-tm[0])
    res = res - a*tm - b
    
    # DST
    dst = fftpack.dst(res,norm='ortho',type=1)
    dst_filtered = dst*np.exp(-(freq/(freqN*sigma))**2/2)
    res_filtered_dst = fftpack.idst(dst_filtered,norm='ortho',type=1)
    
    # This is Gamma/(f_a^2 H^3)
    gamma = (tm**(5-zm))*(np.exp(ftrenda(np.log(tm),p,*param))*dftrenda(np.log(tm),p,*param)/tm + a + np.gradient(res_filtered_dst)/np.gradient(tm))
    
    # Just a finite difference, for comparison
    gamma_diff = (tm**(5-zm))*(np.gradient(enem*(tm**zm))/np.gradient(tm))
    
    return gamma, gamma_diff, tm


# ------------------------------------------------------------------------------
#   string length
# ------------------------------------------------------------------------------

def lvrest(eK, eG, eV, t, Lbox):
    
    # total energy, Lagrangian, and equation of state
    E = (t**2)*(Lbox**3)*(eK + eG + eV)
    L = (t**2)*(Lbox**3)*(eK - eG - eV)
    w = (eK - eG/3. - eV)/(eK + eG + eV)
    
    # tentative values, need to be confirmed and generalized
    mu = 2.*0.892
    fv = 0.368
    
    # rest-frame length, velocities, and xi parameter
    lr = (E+fv*L)/(1.-fv)/mu
    vLsq = (E+L)/(E+fv*L)
    vwsq = (1.+3.*w+2.*fv)/(2.+fv*(1.+3.*w))
    vssq = 2.*eK/(eK+eG)
    xir = lr*t*t/4./Lbox**3
    
    return xir, np.sqrt(vLsq), np.sqrt(vwsq), np.sqrt(vssq), lr
    
