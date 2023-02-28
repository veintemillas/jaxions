# JAXIONS MODULE TO CALCULATE THE SPECTRUM SHAPE FROM F, q, e
# returns drho/dk in some units
import numpy as np
import scipy.integrate

def fAH1(fA):
    ma = 57e-6 * (1e20/fA)
    H1 = 3.45e-9 * (ma/50e-6)**0.338
    return fA/H1

def F(x,q,e):
    return (x)**3/(1+x**((3+q)/e))**(e)

def no(y,q,x1):
    if q == 1:
        return np.log(y/x1)
    else :
        return ((y)**(1-q) - x1**((1-q)))/(1-q)

def xi(fAH1,ct):
    return 0.24*np.log(fAH1*ct**2)

def II(x,q,e):
    return F(x,q,e)*8*np.pi**2

def drhodk():
    scipy.integrate.quad(II, 0, 1,args=())

def calcu(q,e,x0,fA,x1=100,test=False, klist = np.logspace(-1,4,200)):
    """
    Computes d rho / d k
    from a F ~ (x)**3/(1+x**((3+q)/e))**(e)
    
    """
    fah1 = fAH1(fA)
    # compute table of norma
    # integrate numerically until x = 100 
    # analytically from x = 100 to cut-off
    # int dx 1/x^q = ((y)^(1-q) - 100^((1-q)))/(1-q)
    # the integral depends on q,e and y, but only the UV depends on y
    n0 = scipy.integrate.quad(F, 0, x1,args=(q,e))
    #     print(n0)
    #     y = m_r/H = fA/H1 R^2
    #     yl = np.logspace(1,np.log(fah1),1000)
    #     plt.loglog(yl,n0[0]+no(yl,q,x1))
    #     plt.show()

    def FN(x,x0,x1):
        y = fah1
        return F(x/x0,q,e)/(n0[0]+no(y,q,x1))/x0
    def I(R,k,x0):
        y = fah1*R**2
        return 8*np.pi*0.24*np.log(y)**2 *F(R*k/x0,q,e)/(n0[0]+no(y,q,x1))/x0

    if test:
        norm0 = n0[0]+no(fah1,q,x1)
        n0_1 = scipy.integrate.quad(F, 0, 2*x1,args=(q,e))[0]
        n1_1 = no(fah1,q,2*x1)
        print(n0[0],no(fah1,q,x1),n0_1,n1_1)      
        print(norm0,n0_1+n1_1)
        # this fails for UV dominated integrals... convergence!
        print(scipy.integrate.quad(FN, 0, np.inf,args=(x0,x1)))
    #     R=np.logspace(-3,0,100)
    #     plt.loglog(R,I(R,1))
    #     plt.loglog(R,I(R,10))
    #     plt.loglog(R,I(R,0.1))

    # integrate from the cut-off given by k'=m_r/x1, k/R' = m_r/x1, R' = 100 k / m_r 
    r = []
    for k in klist:
        i = scipy.integrate.quad(I, x1*k/fah1, 1,args=(k,x0))
#         print(i)
        r.append([k,i[0],i[1]])
    return np.array(r)

# let us count the axion number density
def sumn(drdk):
    """
    Calculates axion number from d rho / d k
    assuming R=1, vaxions ADM units, etc...
    
    """
    kl = drdk[:,0]
    lk = np.log(kl)
    dlk = lk[1:]-lk[:-1]
    ii  = 0.5*(drdk[1:,1]+drdk[:-1,1])
    return (dlk*ii).sum() , (dlk*ii)[-1]
