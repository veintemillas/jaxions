# JAXIONS MODULE TO PREPARE A string.dat FILE FOR SIMULATIONS WITH THE "String" INITIAL CONDITIONS
import numpy as np
import matplotlib.pyplot as plt
from pyaxions import jaxions as pa

def randomstrings(N, LEN, NSTRINGS=1, ITER=3, XCF=0.5, YCF=0.5, ZCF=0.5, PATH = './'):

    """
    randomstrings(N,L,NSTRINGS,ITER,XCF,YCF,ZCF,PATH)

    1) Generates Nstrings random string loops
    2) Stores their (x,y,z)-coordinates and an additional list marking the endpoint of every string (with a 1) to avoid connecting disconnected loops
    3) Saves the generated configuration in "string.dat". This file will be read and processed at the beginning of the jaxions simulation


    N is the number of grid points (must be the same as for the planned simulation!)
    LEN is the length of the respective loop (for multiple strings the length will be randomly set to values of order of +- 50% of the given LEN)
    NSTRINGS is the number of strings that will be generated
    ITER  is the number of iteratations for the random scattering of the coordinates (Idea: The larger the number, the more complicated the form of the loop)

    XCF specifies the center of the loop on the x-axis (ranges from 0 to 1)
    YCF specifies the center of the loop on the y-axis (ranges from 0 to 1)
    ZCF specifies the center of the loop on the z-axis (ranges from 0 to 1)

    For multiple strings, the center coordinates are chosen randomly and these values are not considered!

    PATH is a string containing the path to the folder where you want to store the string.dat file
    """

    xx, yy, zz = np.array([]), np.array([]), np.array([])
    eps = np.array([]) #For marking the endpoints

    #Generate Nstrings random string loops
    for string in range(NSTRINGS):

        if NSTRINGS == 1:
            #use the respectiv input values
            xc,yc,zc = N*XCF, N*YCF, N*ZCF
            LL = LEN
        else:
            xc,yc,zc = N*np.random.uniform(0.2,0.8), N*np.random.uniform(0.2,0.8), N*np.random.uniform(0.2,0.8) #randomly distribute the strings in the volume (not too close to the boundary)
            LL = np.random.randint(LEN - 0.5*LEN, LEN + 0.5*LEN + 1 )  #upper limit is exclusiv, thats why we add one

        s = np.linspace(0, 2*np.pi, LL)

        r = np.random.random(3)*2*np.pi
        a = np.random.random(3)
        x = a[0]*LL*np.cos(s+r[0])
        y = a[1]*LL*np.cos(s+r[1])
        z = a[2]*LL*np.cos(s+r[2])

        for i in range(2,ITER):
            r = np.random.random(3)*2*np.pi
            a = np.random.random(3)
            x += a[0]*(LL/i)*np.cos(i*s+r[0])
            y += a[1]*(LL/i)*np.cos(i*s+r[1])
            z += a[2]*(LL/i)*np.cos(i*s+r[2])

        #Normalisation
        pv = (np.sqrt((x[1:]-x[:-1])**2+(y[1:]-y[:-1])**2+(z[1:]-z[:-1])**2).sum())
        #print(pv)
        x = x/pv*LL + xc
        y = y/pv*LL + yc
        z = z/pv*LL + zc
        m = (x >= N) + (y >= N) +(x >= N)
        if m.sum()>0:
            print('fail')

        #mark endpoint of the string -> important to avoid contributions to the calculation of the axion field from sections that are not from the same string
        ep = np.zeros(len(x))
        ep[-1] = 1

        xx = np.append(xx,x).flatten()
        yy = np.append(yy,y).flatten()
        zz = np.append(zz,z).flatten()
        eps = np.append(eps,ep).flatten()

    coords = np.column_stack((xx, yy, zz, eps))
    np.savetxt(PATH + 'string.dat', coords, delimiter=' ', fmt='%.2f %.2f %.2f %i')

    return xx,yy,zz

def onestring(N,R,n=4,m='l',ar=0,xcf=0.5,ycf=0.5,zcf=0.5,dz=-0.5):
    """

    Creates a string.dat file with the coordinates of a :
        string loop (m='l')
        polyhedra of n vertices (m='s')
        knot (m='k')
    centered at N*xcf, N*ycf, N*zcf+dz

    N  : number of grid points to be used in the simulation, which helps in
         specifying the number of poins ~ O(1) per grid cube
    R  : Radius
    ar : angle of rotation around z axis, if desired

    returns x,y,z,ep (arrays of coordinates and endpoints)
    """
    xc,yc,zc=N*xcf,N*ycf,N*zcf+dz
    # loop
    if m == 'l':
        s = np.linspace(0.000001,2*np.pi,int(2*np.pi*R))
        x = R*np.cos(s)
        y = R*np.sin(s)
        z = 0*s
    elif m == 's':
        v  = 2*np.pi*np.linspace(0,n,n+1)/n
        vx = R*np.cos(v)
        vy = R*np.sin(v)
        pv = int((np.sqrt((vx[1:]-vx[:-1])**2+(vy[1:]-vy[:-1])**2).sum())/n)
        tn = pv*n
        x = np.zeros(tn)
        y = np.zeros(tn)
        f = np.arange(0,pv)/pv
        for base in range(n):
            x[base*pv:(base+1)*pv] = vx[base] + (vx[base+1]-vx[base])*f
            y[base*pv:(base+1)*pv] = vy[base] + (vy[base+1]-vy[base])*f
        z = 0*x
    elif m == 'k':
        t = np.linspace(0.000001,2*np.pi,2*int(2*np.pi*R))
        x = R/3*(np.sin(t)+2*np.sin(2*t))
        y = R/3*(np.cos(t)-2*np.cos(2*t))
        z = R/3*(-np.sin(3*t))
    if ar != 0:
        xr = x*np.cos(ar) + y*np.sin(ar)
        y  = -x*np.sin(ar) + y*np.cos(ar)
        x  = xr
    x = x + xc
    y = y + yc
    z = z + zc
    ep = np.zeros(len(x),dtype=int)
    ep[-1] = 1
    xy = np.column_stack((x, y, z, ep))
    np.savetxt('./string.dat', xy, delimiter=' ', fmt='%.2f %.2f %.2f %d')   # X is an array
    return x,y,z,ep

#Implementation of the string IC for their simulations for the paper "Radiation of Goldstone bosons from cosmic strings" (PRD Vol. 35, Nr. 4, 1987) by Vilenkin and Vachaspati
e1 = np.array([1, 0, 0])  # Unit vector x1
e2 = np.array([0, 1, 0])  # Unit vector x2
e3 = np.array([0, 0, 1])  # Unit vector x3

def a(zeta, alpha):
    result = np.empty((len(zeta), 3))
    for i in range(len(zeta)):
        result[i] = (1 / alpha) * (e1 * np.sin(alpha * zeta[i]) + e3 * np.cos(alpha * zeta[i]))
    return result

def b(zeta, beta, psi):
    result = np.empty((len(zeta), 3))
    for i in range(len(zeta)):
        result[i] = (1 / beta) * ((e1 * np.cos(psi) + e2 * np.sin(psi)) * np.sin(beta * zeta[i]) + e3 * np.cos(beta * zeta[i]))
    return result

def vv_string(N, R, alpha, beta, psi, t, xcf=0.5, ycf=0.5, zcf=0.5, dz = -0.5, PATH = './'):
    """
    vv_string(N, R, ALPHA, BETA, PSI, T, XCF, YCF, ZCF)

    1) Generates a string IC as in "Radiation of Goldstone bosons from cosmic strings" (PRD Vol. 35, Nr. 4, 1987) by Vilenkin and Vachaspati
    2) Stores their (x,y,z)-coordinates and an additional list marking the endpoint of every string (with a 1) to avoid connecting disconnected loops
    3) Saves the generated configuration in "string.dat". This file will be read and processed at the beginning of the jaxions simulation


    N is the number of grid points (must be the same as for the planned simulation!)
    R is the string radius
    ALPHA and BETA are constants (alpha = N1/R, beta = N2/R, with N1 and N2 relatively prime integers)
    PSI is another constant, that controls the rotation of the string around the z-axis (from 0 to 2pi)

    XCF specifies the center of the loop on the x-axis (ranges from 0 to 1)
    YCF specifies the center of the loop on the y-axis (ranges from 0 to 1)
    ZCF specifies the center of the loop on the z-axis (ranges from 0 to 1)
    DZ is a shift in the z-coordinate

    PATH is a string containing the path to the folder where you want to store the string.dat file

    Check the paper for details about the choice of parameters etc.
    """
    xc, yc, zc = N * xcf, N * ycf, N * zcf + dz

    zeta = np.linspace(0, 2 * np.pi * R, int(2 * np.pi * R))

    a_zeta = a(zeta-t, alpha)
    b_zeta = b(zeta+t, beta, psi)

    x = []
    y = []
    z = []

    for i in range(len(zeta)):
        x_i = 0.5 * (a_zeta[i] + b_zeta[i])
        x_i = x_i + np.array([xc, yc, zc])
        x.append(x_i[0])
        y.append(x_i[1])
        z.append(x_i[2])

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    ep = np.zeros(len(zeta), dtype=int)
    ep[-1] = 1

    data = np.column_stack((x, y, z, ep))
    np.savetxt(PATH + './string.dat', data, delimiter=' ', fmt='%.2f %.2f %.2f %d')

    return x, y, z, ep

def arctan3(r,i,a):
    c = np.cos(a)
    s = np.sin(a)
    return np.arctan2(i*c+r*s,r*c-i*s)

def fu(f,iz=0,iy=0,ix=0,a=0):
    """

    Creates slices plots of axion configuration in file f containing the point
    ix, iy, iz
    allowing a rotation of theta of size a

    """
    R  = pa.gm(f,'R')
    N  = pa.gm(f,'N')
    m  = np.reshape(pa.gm(f,'da/m/data'),(N,N,N,2))
    t  = arctan3(m[iz,:,:,0],m[iz,:,:,1],a)
    r  = np.sqrt(m[iz,:,:,0]**2+m[iz,:,:,1]**2)/R
    fig,ax=plt.subplots(3,3,figsize=(20,20))
    i = ax[0,0].imshow(t,vmin=-np.pi,vmax=np.pi,cmap=pa.thetacmap,origin='lower')
    pa.colorbar(i)
    i = ax[0,1].imshow(r,vmin=0,vmax=1,cmap='Greys',origin='lower')
    pa.colorbar(i)
    ax[0,2].plot(t[iy,:]) ; ax[0,0].set_xlabel('x');ax[0,0].set_ylabel('y') ; ax[0,1].set_xlabel('x')
    t  = arctan3(m[:,iy,:,0],m[:,iy,:,1],a)
    r  = np.sqrt(m[:,iy,:,0]**2+m[:,iy,:,1]**2)/R
    i = ax[1,0].imshow(t,vmin=-np.pi,vmax=np.pi,cmap=pa.thetacmap,origin='lower')
    pa.colorbar(i)
    i = ax[1,1].imshow(r,vmin=0,vmax=1,cmap='plasma',origin='lower')
    pa.colorbar(i)
    ax[1,2].plot(t[iz,:]) ; ax[1,0].set_xlabel('x');ax[1,0].set_ylabel('z') ; ax[1,1].set_xlabel('x')
    t  = arctan3(m[:,:,ix,0],m[:,:,ix,1],a)
    r  = np.sqrt(m[:,:,ix,0]**2+m[:,:,ix,1]**2)/R
    i = ax[2,0].imshow(t,vmin=-np.pi,vmax=np.pi,cmap=pa.thetacmap,origin='lower')
    pa.colorbar(i)
    i = ax[2,1].imshow(r,vmin=0,vmax=1,cmap='plasma',origin='lower')
    pa.colorbar(i)
    ax[2,2].plot(t[iz,:]) ; ax[2,0].set_xlabel('y');ax[2,0].set_ylabel('z') ; ax[2,1].set_xlabel('y')
