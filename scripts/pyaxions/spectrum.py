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


# Perform analytical fit
# Switch to a simplified fit function when x[0] becomes larger than xh (default xh = -1)
# If xh = -1 (or some negative value), do not use this simplification (no discontinuity but time-consuming)
class fitP:
    def __init__(self, P, log, t, k, **kwargs):
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
        tt = t[mask[0]]
        x0 = tt[0]*k
        his = np.abs(x0-xh).argmin()
        self.ikhistart = his
        if x0[his] < xh:
            self.ikhistart = his+1
        for ik in range(len(k)):
            if verbose == 1:
                print('\rfit P:  k = %.2f [%d/%d]'%(k[ik],ik+1,len(k)),end="",flush=True)
            elif verbose == 2:
                print('fit P:  k = %.2f [%d/%d]'%(k[ik],ik+1,len(k)),flush=True)
            xdata = np.log(k[ik]*t[mask[0]])
            ydata = np.log(P[mask[0],ik])
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

# save parameters
def savePP(fP, name='./PP'):
    sdata(fP.param,name,'param')
    sdata(fP.listfit,name,'listfit')
    sdata(fP.ikhistart,name,'ikhistart')
    
# read parameters
class readPP:
    def __init__(self, name='./PP'):
        self.param = rdata(name,'param')
        self.listfit = rdata(name,'listfit')
        self.ikhistart = rdata(name,'ikhistart')


# ------------------------------------------------------------------------------
#   instantaneous spectrum
# ------------------------------------------------------------------------------

def filterDST(k, sigma, res, t):
    y  = res
    T = t[-1]-t[0]
    # The Discrete Sine Transform of the signal
    sig_dst = fftpack.dst(y,norm='ortho',type=1)
    # The corresponding frequencies
    freq_dst = np.pi*np.linspace(1,y.size,y.size)/T
    # filter
    # high-pass Gauss
    sig_dst_filtered = sig_dst*np.exp(-(freq_dst/(k*sigma))**2/2)
    filtered_dst = fftpack.idst(sig_dst_filtered,norm='ortho',type=1)
    return filtered_dst, sig_dst, sig_dst_filtered, freq_dst
    
    
class calcF:
    def __init__(self, P, log, t, k, k_below, **kwargs):
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
            fitpin = kwargs['fitp']
        else:
            fitpin = []
        if 'xh' in kwargs:
            xhi = kwargs['xh']
        else:
            xhi = -1
        if 'sigma' in kwargs:
            sigma = kwargs['sigma']
        else:
            sigma = 0.5
        self.F = [] # instantaneous spectrum F
        self.Fnorm = [] # normalization factor of F
        self.F_fit = [] # instantaneous spectrum F (fit only)
        self.Fnorm_fit = [] # normalization factor of F (fit only)
        mask = np.where(log >= logst)
        logm = log[mask[0]]
        tm = t[mask[0]]
        if usedata:
            fitp = fitpin
        else:
            fitp = fitP(P,log,t,k,p=po,verbose=verb,logstart=logst,xh=xhi)
        Farr = []
        xarr = []
        Farr_aux = []
        xarr_aux = []
        Farr_fit = []
        Farr_fit_aux = []
        xarr_fit_aux = []
        # for test
        #print('kcut = %f'%(cmin/tm[-1]))
        # test
        for ik in range(1,len(k)):
            #if verbose:
            #    print('\rcalc F (differentiate): k = %.2f [%d/%d]'%(k[ik],ik+1,len(k)),end="")
            par = fitp.param[ik]
            xx = k[ik]*tm
            lxx = np.log(xx)
            if fitp.listfit[ik]:
                if ik < fitp.ikhistart:
                    Fk_fit = np.exp(ftrend(lxx,po,*par))*dftrend(lxx,po,*par)/xx
                    res = P[mask[0],ik] - np.exp(ftrend(lxx,po,*par))
                    # subtract linear trend
                    a = (res[-1]-res[0])/(xx[-1]-xx[0])
                    b = (res[0]*xx[-1]-res[-1]*xx[0])/(xx[-1]-xx[0])
                    res = res - a*xx-b
                    # DST
                    fres, dst, dst_fil, freq = filterDST(k[ik],sigma,res,tm)
                    Fk = Fk_fit + a + np.gradient(fres,xx[1]-xx[0],edge_order=2)
                else:
                    Fk_fit = np.exp(ftrenda(lxx,po,*par))*dftrenda(lxx,po,*par)/xx
                    Fk = Fk_fit
            else:
                Fk_fit = [np.nan]*len(tm)
                Fk = [np.nan]*len(tm)
            if not np.isnan(np.sum(Fk)):
                Farr_aux.append(Fk)
                xarr_aux.append(xx)
            if not np.isnan(np.sum(Fk_fit)):
                Farr_fit_aux.append(Fk_fit)
                xarr_fit_aux.append(xx)
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
        Farr_fit_aux = np.transpose(np.array(Farr_fit_aux))
        xarr_fit_aux = np.transpose(np.array(xarr_fit_aux))
        # normalize
        for id in range(len(tm)):
            dx = np.gradient(xarr_aux[id,:])
            dx_fit = np.gradient(xarr_fit_aux[id,:])
            # normalization factor is calculated by using only modes below the Nyquist frequency
            x_below = xarr_aux[id,:] <= np.amax(k[k_below])*tm[id]
            x_below_fit = xarr_fit_aux[id,:] <= np.amax(k[k_below])*tm[id]
            Fdx = (Farr_aux[id,:]*dx)[x_below]
            Fdx_fit = (Farr_fit_aux[id,:]*dx_fit)[x_below_fit]
            self.F.append(Farr[id,:]/Fdx.sum())
            self.F_fit.append(Farr_fit[id,:]/Fdx_fit.sum())
            self.Fnorm.append(Fdx.sum())
            self.Fnorm_fit.append(Fdx_fit.sum())
        self.F = np.array(self.F)
        self.Fnorm = np.array(self.Fnorm)
        self.F_fit = np.array(self.F_fit)
        self.Fnorm_fit = np.array(self.Fnorm_fit)
        self.x = xarr
        self.t = tm
        self.log = logm
        

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
