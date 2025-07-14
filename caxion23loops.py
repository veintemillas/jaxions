import numpy as np
import matplotlib.pyplot as plt
from pyaxions import jaxions as pa

from scipy.integrate import quad
from scipy.interpolate import CubicSpline

import importlib

from IPython.display import clear_output
import timeit

import pickle

import h5py

import subprocess


def simu(R,msa,N,Ng=2,plota=False,rescale=1,n_save=200):
    '''simu(R,msa,N,Ng=2,plota=False,rescale=1,n_save=200) 
    Prepares files for a NxN caxion3d simulation and runs then.
    ICs are prepared in python as a static loop of radious R
    where theta is calcualted from KR equivalent B-field 
    and rho according to the distance from the string and msa
    dx = 1, msa = equivalent to saxion mass
    The program currently works only in Minkowsky, FRW is easy to implement
    Ng is the number of neighbours in the laplacian
    n_save is hoe many measurements are desired before collapse (approx(
    rescale does not work always, keep it to 1
    plota gives some extra info'''
    if plota:
        print('Simulation')
    N_create = N//rescale
    R_create = R/rescale
    msa_create = msa*rescale
    # creates directly a simulation
    # if rescale > 1 we create it smaller to later interpolate
    if plota:
        print('Creation with N,R,msa = ', N_create,R_create,msa_create)
    theta = thetaics(N_create,N_create,R_create)
    phi   = phiics(theta,msa_create)

    # create caxion3d generic ICs file
    
    JAXI = generic_jax(msa_create,N_create,Ng=Ng,dump=0)
    create_jax(JAXI)

    if plota:
        print('Copy into file', N_create,R_create,msa_create)
    copyics(phi,filename='out/m/axion.00000',n=0)

    if plota:
        print('Run jaxions', N,R,msa)
    # run jaxions!
    dump = int(N*np.sqrt(12)/n_save)
    JAXI = generic_jax(msa,N,Ng=Ng,dump=dump)
    run_jax(JAXI)
    
    # pack everything 
    
    result = subprocess.run('mv axion.log.* out', shell=True, capture_output=True, text=True)
    # print("STDOUT:", result.stdout,result.stderr)
    result = subprocess.run('mv log-c*.txt out', shell=True, capture_output=True, text=True)
    # print("STDOUT:", result.stdout,result.stderr)
    result = subprocess.run('mv create.sh run.sh out', shell=True, capture_output=True, text=True)
    # print("STDOUT:", result.stdout,result.stderr)
    name = namea(N,msa,Ng)
    result = subprocess.run('rm -r %s'%name, shell=True, capture_output=True, text=True)
    result = subprocess.run('mv out %s'%name, shell=True, capture_output=True, text=True)
    # print("STDOUT:", result.stdout,result.stderr)

def namea(N,msa,Ng):
    return 'data/out%d-%d-%d'%(N,1000*msa,Ng)

def generic_jax(msa,N,Ng=2,dump=100):
    GRID=" --size %d --depth 1 --zgrid 1"%N
    SIMU=" --device gpu --measCPU  --steps 20000000 --wDz 1.0 --lap %d"%Ng
    PHYS=" --vqcd0 --mink --notheta --msa %f --lsize %d  --zf %d "%(msa,N,N)
    INCO=" --ctype smooth --zi 0.1 --sIter 0 --nncore "
    # you might want to add plot --p2Dmap
    OUTP=" --dump %d --meas 128 --p2DmapYZ  --nologmpi --verbose 0 "%dump
    return GRID+SIMU+PHYS+INCO+OUTP     

def create_jax(JAXI):
    #!rm axion.log.*
    subprocess.run('rm axion.log.*', shell=True, capture_output=True, text=True)
    # create any ICS
    with open ('create.sh', 'w') as rsh:
        rsh.write('''\
        #! /bin/bash
        mpirun -np 1 caxion3d %s --steps 0 --p3D 1 > log-create.txt
        '''%(JAXI))
    # !chmod u+x create.sh
    # !./create.sh
    subprocess.run('chmod u+x create.sh', shell=True, capture_output=True, text=True)
    subprocess.run('./create.sh', shell=True, capture_output=True, text=True)
    return JAXI
    
def run_jax(JAXI):    
    # run sim
    subprocess.run('rm out/m/axion.m.*', shell=True, capture_output=True, text=True)
    with open ('run.sh', 'w') as rsh:
        rsh.write('''\
        #! /bin/bash
        mpirun -np 1 caxion3d %s --index 0 > log-con.txt 
        '''%(JAXI))
    # !chmod u+x run.sh
    # !./run.sh
    subprocess.run('chmod u+x run.sh', shell=True, capture_output=True, text=True)
    subprocess.run('./run.sh', shell=True, capture_output=True, text=True)

def copyics(phi,filename='out/m/axion.00000',n=0):
    Nz,Nx,c = phi.shape
    print('saving jaxions file',phi.shape)
    f1 = h5py.File(filename, 'r+')     # open the file
    data = f1['/m']       # load the data
    # paste field
    # save
    data[...] = np.reshape(phi,Nz*Nx*2)
    # velocity field (MINKOWSKY!)
    vata = f1['/v']       # load the data
    vata[...] = np.reshape(phi*n,Nz*Nx*2)
    f1.close()     

# Auxiliary 
def buildr(mf):
    N = pa.gm(mf[0],'N')
    n = np.arange(N)
    t = pa.gml(mf,'ct')
    r = np.zeros_like(t)
    g = np.zeros_like(t)
    for it in range(len(mf)):
        li = np.reshape(pa.gm(mf[it],'da/chunk/m/'),(N,2))[:,0]
        r[it],g[it] = findmer(li)
    return t,r,g

def findmer(li,ftype='re_complex'):
    """ 
    # finds the coordinate where 
    # a real part of a complex field changes sign
    # a real field changes by pi 
    # in a 1D array """
    R = -1
    m = -1
    if ftype=='re_complex':        
        for i in range(len(li)-1):
            if ((li[i+1]>0) & (li[i]<0)):
                m = i
                R = m-li[m]/ (li[m+1]-li[m])
                break
    if ftype=='theta':
        for i in range(len(li)-1):
            if np.abs(li[i+1]-li[i])>3.1415:
                m = i
                R = m+0.5
                break
    if m>0:
        fr=3
        nmin = max(0,m-fr)
        b,c = np.polyfit(np.arange(nmin,m+fr)-R,li[nmin:m+fr],1)
        slope = b
        return R,slope
    return 0, 1

# Functions for initial conditions
def rhof(r):
    # rho profile for a straight string
    # checked to a good approximation
    # r in units of 1/ms
    r2 = r*r
    r4 = r2*r2
    return (0.43*r +0.164*r2+0.036*r4)/(1+0.039*r+0.2*r2+0.036*r4)

def phiics(theta,msa):
    """ 
    phiics(theta,msa)
    # takes a theta loop IC field and returns 
    # phi = rho exp(i theta)
    # with the rho field adjusted to minimise the EOM
    # currently, curvature is not taken into account
    """
    R,s = findmer(theta[0,:],ftype='theta')
    nz,nrh=theta.shape
    rh = np.arange(nrh)
    z  = np.arange(nz)
    RH,Z = np.meshgrid(rh,z)
    phi = np.zeros((nz,nrh,2))
    rho = rhof(msa * np.sqrt((RH-R)**2+Z**2))
    phi[:,:,0]=rho*np.cos(theta)
    phi[:,:,1]=rho*np.sin(theta)
    return phi
    
def thetaics(nrh,nz,R,plota=False,readIC=True):
    """ thetaics(nrh,nz,R,plota=False,readIC=True)
    # creates a nrh x nz map with the theta field 
    # of a loop of radious R 
    # By integrating static B-field along z
    # code units dx=1, v=1
    plota = true plots the field
    readIC = False forces creation of ICS, otherwise, they are uploaded if existing
    
    """
    
    name = 'theta_%dx%dR%d.pkl'%(nrh,nz,R)
    b_create = True
    # check if ICs are already present
    if readIC:
        try:
            with open('aux/'+name, "rb") as f:
                print('read from file')
                theta = pickle.load(f)
                b_create = False
        except IOError:
            print('ICs not found we will create them')
    if b_create:
        # coordinates
        rh = np.arange(nrh)
        z = np.arange(nz)
        # fields
        theta = np.zeros((nz,nrh))
        Bzfie = np.zeros((nz,nrh))
        
        # set initial conditions at z=0
        theta[0,(rh<=R)] = np.pi
        
        RH,Z = np.meshgrid(rh,z)
        
        # read interpolatio table
        with open('aux/tableszetat1t2_4.pkl', 'rb') as f:
            dica = pickle.load(f)
            
        zeta,t1,t2 = dica['zeta'],dica['t1'],dica['t2']
        f1 = CubicSpline(zeta, t1)
        f2 = CubicSpline(zeta, t2)
        
        def I1f(z):
            return (3*np.pi/4*z + 2**1.5*z**3/(1-z**2)) * f1(z)
        def I2f(z):
            return (np.pi + 2**1.5*z**2/(1-z**2)) * f2(z)
    
        def Bz(RH,Z,R):
            ZETA = 2*R*RH/(R**2 + RH**2 + Z**2)
            return (R*RH*I1f(ZETA)-R**2*I2f(ZETA))/(R**2+RH**2+Z**2)**(3/2)
        # precalculation B
        Bzfie = Bz(RH,Z,R)
    
        def filltheta(theta,Bzfie,z,rh,R,Z,threshold=10,calculate=True):
            check = np.zeros((len(rh)))
            for i in range(nrh):
                for j in range(1,nz-1):
                    if (rh[i]-R)**2+z[j]**2 > threshold**2:
                        # Far from the string we sum B fields
                        theta[j,i] = theta[j-1,i]+Bzfie[j-1:j+1,i].mean()*(Z[j,i]-Z[j-1,i])
                    else:
                        # Close to the string 
                        if calculate:
                            # calculate numerically the integrals
                            def Bzr(zi,Ri):
                                return Bz(rh[i],zi,Ri)
                            res,err = quad(Bzr, z[j-1], z[j],args=(R))
                            theta[j,i] = theta[j-1,i]+res
                            check[i] += 1
                        else:
                            # or use the simple analytical expression
                            theta[j,i] = np.arctan2(z[j],(rh[i]-R))
        
            return check

        # this finally calculates theta
        threshold=10
        check = filltheta(theta,Bzfie,z=z,rh=rh,R=R,Z=Z,threshold=threshold,calculate=True)

        #saves the ICs for future use
        file = open('aux/'+name,'wb')
        pickle.dump(theta, file)
        file.close()

    if plota:
        fig,ax=plt.subplots(1,2,figsize=(20,20))
        i = ax[0].imshow(theta,cmap=pa.thetacmap,origin='lower',vmax=np.pi,vmin=-np.pi)
        pa.colorbar(i)
        i = ax[1].imshow(theta,origin='lower')
        pa.colorbar(i)
        ax[1].set_xlim(R-2*R/10,R+2*R/10)
        ax[1].set_ylim(0,2*R/10)

    return theta

# Build interpolation tables and f1,f2 tables
# save them for further use
def buildf1f2(ninterp=10000):
    name = 'tableszetat1t2_%d.pkl'%np.log10(ninterp)
    
    def I1i(x,zet):
        return np.cos(x)/(1-zet*np.cos(x))**(3/2)
    def I2i(x,zet):
        return 1/(1-zet*np.cos(x))**(3/2)
    zeta = np.linspace(0,1,ninterp)[:-1]

    I1t = zeta*0
    I2t = zeta*0
    for i in range(len(zeta)):
        # call quad to integrate f from -2 to 2
        res1, err1 = quad(I1i, 0, np.pi,args=(zeta[i]))
        I1t[i] = res1
        res2, err2 = quad(I2i, 0, np.pi,args=(zeta[i]))
        I2t[i] = res2
    
    t1,t2 = np.ones(len(zeta)+1),np.ones(len(zeta)+1)
    zc = zeta[1:]
    t1[1:-1] = (I1t[1:]/(3*np.pi/4*zc + 2**1.5*zc**3/(1-zc**2)))
    t2[1:-1] = (I2t[1:]/(np.pi*zc/zc + 2**1.5*zc**2/(1-zc**2)))
    zeta = np.linspace(0,1,ninterp)
    # f1 = CubicSpline(zeta, t1)
    # f2 = CubicSpline(zeta, t2)
    dica = {'zeta':zeta,'t1':t1,'t2':t2}
    
    file = open('aux/'+name,'wb')
    pickle.dump(dica, file)
    file.close()
    print(name,' saved')
    return zeta, t1, t2