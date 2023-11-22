# JAXIONS MODULE TO PREPARE A string.dat FILE FOR SIMULATIONS WITH THE "String" INITIAL CONDITIONS
import numpy as np
import matplotlib.pyplot as plt
from pyaxions import jaxions as pa

import numpy as np

import numpy as np

def randomstrings(N=256, LEN=2 * np.pi * 256 // 4, SEED = None, NSTRINGS=1, ITER=3, KINKS=0, XCF=0.5, YCF=0.5, ZCF=0.5, PATH='./'):
    """
    randomstrings(N, LEN, SEED, NSTRINGS, ITER, KINKS, XCF, YCF, ZCF, PATH)

    Generates NSTRINGS random string loops with a random number of 90-degree angle kinks.
    Stores their (x, y, z)-coordinates and marks the endpoints of every string.
    Saves the configuration in "string_with_random_kinks.dat."

    N is the number of grid points (must match the simulation).
    LEN is the length of the respective loop.
    SEED is an abitrary integer that can be used to reproduce the random parameters
    NSTRINGS is the number of strings that will be generated.
    ITER is the number of iterations for the random scattering of the coordinates.
    (** Currently removed **) KINKS is the (maximum) number of kinks for each string (randomized between 0 and NUM_KINKS, if multiple strings).
    XCF specifies the center of the loop on the x-axis (ranges from 0 to 1).
    YCF specifies the center of the loop on the y-axis (ranges from 0 to 1).
    ZCF specifies the center of the loop on the z-axis (ranges from 0 to 1).
    PATH is a string containing the path to the folder where you want to store the string_with_random_kinks.dat file.
    """

    # Set a random seed for reproducibility
    if SEED is not None:
        seed = SEED
        np.random.seed(seed)
    else:
        seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed)
    print('If you want to reproduce the exact same loop shape, use SEED=%d'%seed)

    xx, yy, zz = np.array([]), np.array([]), np.array([])
    eps = np.array([])  # For marking the endpoints

    # Generate NSTRINGS random string loops
    for string in range(NSTRINGS):
        if NSTRINGS == 1:
            xc, yc, zc = N * XCF, N * YCF, N * ZCF
            LL = int(LEN)
        else:
            xc, yc, zc = N * np.random.uniform(0.2, 0.8), N * np.random.uniform(0.2, 0.8), N * np.random.uniform(0.2,0.8)  # Randomly distribute the strings
            LL = np.random.randint(LEN - 0.5 * LEN, LEN + 0.5 * LEN + 1)  # Random length

        s = np.linspace(0, 2 * np.pi, LL)

        r = np.random.random(3) * 2 * np.pi
        a = np.random.random(3)
        x = a[0] * LL * np.cos(s + r[0])
        y = a[1] * LL * np.cos(s + r[1])
        z = a[2] * LL * np.cos(s + r[2])

        for i in range(2, ITER):
            r = np.random.random(3) * 2 * np.pi
            a = np.random.random(3)
            x += a[0] * (LL / i) * np.cos(i * s + r[0])
            y += a[1] * (LL / i) * np.cos(i * s + r[1])
            z += a[2] * (LL / i) * np.cos(i * s + r[2])

        # Removed kinks for now, need to discuss how to create loops with kinks appropriately -> collisions probably
        #Introduce a random number of 90-degree angle kinks (between 0 and NUM_KINKS, for multiple strings)
        #if NSTRINGS == 1:
        #    num_kinks = KINKS
        #else:
        #    num_kinks = np.random.randint(0, KINKS + 1)

        #kink_indices = np.sort(np.random.choice(LL, num_kinks, replace=False))
        #kink_angle = np.pi / 2  # 90-degree angle, hardcoded for now

        #for kink_index in kink_indices:
        #    x[kink_index] += LL * np.cos(kink_angle)
        #    y[kink_index] += LL * np.cos(kink_angle)
        #    z[kink_index] += LL * np.cos(kink_angle)

        # Normalize
        pv = (np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2 + (z[1:] - z[:-1]) ** 2).sum())
        x = x / pv * LL + xc
        y = y / pv * LL + yc
        z = z / pv * LL + zc
        m = (x >= N) + (y >= N) + (x >= N)
        if m.sum() > 0:
            print('fail')

        # Mark endpoint of the string
        ep = np.zeros(len(x))
        ep[-1] = 1

        xx = np.append(xx, x).flatten()
        yy = np.append(yy, y).flatten()
        zz = np.append(zz, z).flatten()
        eps = np.append(eps, ep).flatten()

    coords = np.column_stack((xx, yy, zz, eps))

    # Save seed and other relevant parameters in the output file
    with open(PATH + 'string.dat', 'w') as file:
        file.write(f"# Seed: {SEED}\n")
        file.write(f"# N: {N}\n")
        file.write(f"# LEN: {LEN}\n")
        file.write(f"# NSTRINGS: {NSTRINGS}\n")
        file.write(f"# ITER: {ITER}\n")
        file.write(f"# KINKS: {KINKS}\n")
        file.write(f"# XCF: {XCF}\n")
        file.write(f"# YCF: {YCF}\n")
        file.write(f"# ZCF: {ZCF}\n")
        np.savetxt(file, coords, delimiter=' ', fmt='%.2f %.2f %.2f %i')

    return xx, yy, zz


def onestring(N = 256,R =256//4, NPOLY=4, SHAPE='l', AR=0, XCF=0.5, YCF=0.5, ZCF=0.5, DZ=-0.5):
    """

    Creates a string.dat file with the coordinates of a :
        string loop (M='l')
        polyhedra of n vertices (M='s')
        knot (M='k')
    centered at N*XCF, N*YCF, N*ZCF+DZ

    N  : number of grid points to be used in the simulation, which helps in
         specifying the number of poins ~ O(1) per grid cube
    R  : Radius
    AR : angle of rotation around z axis, if desired

    returns x,y,z
    """
    xc,yc,zc = N*XCF,N*YCF,N*ZCF+DZ
    # loop
    if SHAPE == 'l':
        s = np.linspace(0.000001,2*np.pi,int(2*np.pi*R))
        x = R*np.cos(s)
        y = R*np.sin(s)
        z = 0*s
    elif SHAPE == 's':
        v  = 2*np.pi*np.linspace(0,NPOLY,NPOLY+1)/NPOLY
        vx = R*np.cos(v)
        vy = R*np.sin(v)
        pv = int((np.sqrt((vx[1:]-vx[:-1])**2+(vy[1:]-vy[:-1])**2).sum())/NPOLY)
        tn = pv*NPOLY
        x = np.zeros(tn)
        y = np.zeros(tn)
        f = np.arange(0,pv)/pv
        for base in range(n):
            x[base*pv:(base+1)*pv] = vx[base] + (vx[base+1]-vx[base])*f
            y[base*pv:(base+1)*pv] = vy[base] + (vy[base+1]-vy[base])*f
        z = 0*x
    elif SHAPE == 'k':
        t = np.linspace(0.000001,2*np.pi,2*int(2*np.pi*R))
        x = R/3*(np.sin(t)+2*np.sin(2*t))
        y = R/3*(np.cos(t)-2*np.cos(2*t))
        z = R/3*(-np.sin(3*t))
    if AR != 0:
        xr = x*np.cos(AR) + y*np.sin(AR)
        y  = -x*np.sin(AR) + y*np.cos(AR)
        x  = xr
    x = x + xc
    y = y + yc
    z = z + zc

    ep = np.zeros(len(x),dtype=int)
    ep[-1] = 1

    xy = np.column_stack((x, y, z, ep))
    np.savetxt('./string.dat', xy, delimiter=' ', fmt='%.2f %.2f %.2f %d')

    # Save input parameters in the output file
    with open(PATH + 'string.dat', 'w') as file:
        file.write(f"# N: {N}\n")
        file.write(f"# R: {R}\n")
        file.write(f"# NPOLY: {n}\n")
        file.write(f"# M: {m}\n")
        file.write(f"# AR: {AR}\n")
        file.write(f"# XCF: {XCF}\n")
        file.write(f"# YCF: {YCF}\n")
        file.write(f"# ZCF: {ZCF}\n")
        file.write(f"# DZ: {DZ}\n")
        np.savetxt(file, coords, delimiter=' ', fmt='%.2f %.2f %.2f %i')

    return x,y,z


#Burden solutions

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

def burden(N=256, R=256//4, ALPHA=1.0/64, BETA=1.0/64, PSI=np.pi/2, T=0.0, XCF=0.5, YCF=0.5, ZCF=0.5, DZ = -0.5, PATH = './'):
    """
    burden(N, R, ALPHA, BETA, PSI, T, XCF, YCF, ZCF)

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
    xc, yc, zc = N * XCF, N * YCF, N * ZCF + DZ

    zeta = np.linspace(0, 2 * np.pi * R, int(2 * np.pi * R))

    a_zeta = a(zeta-T, ALPHA)
    b_zeta = b(zeta+T, BETA, PSI)

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

    # Save input parameters in the output file
    with open(PATH + 'string.dat', 'w') as file:
        file.write(f"# N: {N}\n")
        file.write(f"# R: {R}\n")
        file.write(f"# ALPHA: {ALPHA}\n")
        file.write(f"# BETA: {BETA}\n")
        file.write(f"# PSI: {PSI}\n")
        file.write(f"# T: {T}\n")
        file.write(f"# XCF: {XCF}\n")
        file.write(f"# YCF: {YCF}\n")
        file.write(f"# ZCF: {ZCF}\n")
        file.write(f"# DZ: {DZ}\n")
        np.savetxt(file, coords, delimiter=' ', fmt='%.2f %.2f %.2f %i')

    return x, y, z

def longstring(N=256, AUX=1, A=0.5, D=10, D1 = 256/4, D2 = 3*256/4, DIST=20, ORIENTATION='z', PATH = './'):
    """
    longstring(N, AUX, A, D, D1, D2, DIST, ORIENTATION, PATH)

    *** Work in progress ***

    1) Generates ICs with different longstrings, at the moment one sinusodially displace string and AUX straigth strings (to avoid boundary effects)
    2) Stores their (x,y,z)-coordinates and an additional list marking the endpoint of every string (with a 1) to avoid connecting disconnected loops
    3) Saves the generated configuration in "string.dat". This file will be read and processed at the beginning of the jaxions simulation


    N is the number of grid points (must be the same as for the planned simulation!)
    AUX is the number of straight strings (atm 1 or 3, work in progress ..)
    A allows you to shift the string coordinates
    D controls the amplitude of the sinusodially displaced string
    D1 and D2 controls shifts of the additional strings for the AUX=3 case
    DIST controls the distance between the strings
    ORIENTATION allows you to choose the orientation of the strings

    PATH is a string containing the path to the folder where you want to store the string.dat file
    """
    if AUX==1:
        x0, y0, x1, y1 = N/2-DIST, N/2, N/2+DIST, N/2  # Center points for x and y

        #sinusodially displaced string
        if ORIENTATION == 'z':
            z = np.linspace(0, N, N+1)
            x = A + x0 + D * np.sin(2 * np.pi * z / N)
            y = A + y0 + z * 0  # Set y values to point in the z direction (0)
        elif ORIENTATION == 'x':
            x = np.linspace(0, N, N+1)
            y = A + y0 + D * np.sin(2 * np.pi * x / N)
            z = A + x0 + x * 0  # Set z values to point in the x direction (0)
        elif ORIENTATION == 'y':
            y = np.linspace(0, N, N+1)
            x = A + x0 + D * np.sin(2 * np.pi * y / N)
            z = A + y0 + y * 0  # Set z values to point in the y direction (0)
        else:
            raise ValueError("Invalid orientation. Use 'x', 'y', or 'z'.")

        ep = x * 0
        ep[-1] = 1

        #straight string
        if ORIENTATION == 'z':
            z2 = np.linspace(N, 0, N+1)
            x2 = A + x1 + z2 * 0
            y2 = A + y1 + z2 * 0  # Set y values to point in the z direction (0)
        elif ORIENTATION == 'x':
            x2 = np.linspace(N, 0, N+1)
            y2 = A + y1 + x2 * 0
            z2 = A + x1 + x2 * 0  # Set z values to point in the x direction (0)
        elif ORIENTATION == 'y':
            y2 = np.linspace(N, 0, N+1)
            x2 = A + x1 + y2 * 0
            z2 = A + y1 + y2 * 0  # Set z values to point in the y direction (0)

        ep2 = x2 * 0
        ep2[-1] = 1

        x = np.concatenate((x, x2))
        y = np.concatenate((y, y2))
        z = np.concatenate((z, z2))
        e = np.concatenate((ep, ep2))

        coords = np.column_stack((x,y,z,e))

    elif AUX==3:

        #sinusodially displaced string
        if ORIENTATION == 'z':
            z = np.linspace(0, N, N+1)
            x = A + D1 + D * np.sin(2*np.pi*z/N)
            y = A + D1 + z*0  # Set y values to point in the z direction (0)
        elif ORIENTATION == 'x':
            x = np.linspace(0, N, N+1)
            y = A + D1 + D * np.sin(2 * np.pi * x / N)
            z = A + D1 + x * 0  # Set z values to point in the x direction (0)
        elif ORIENTATION == 'y':
            y = np.linspace(0, N, N+1)
            x = A + D1 + D * np.sin(2 * np.pi * y / N)
            z = A + D1 + y * 0  # Set z values to point in the y direction (0)
        else:
            raise ValueError("Invalid orientation. Use 'x', 'y', or 'z'.")

        #endpoints
        e = z*0
        e[-1] = 1

        #straight string
        if ORIENTATION == 'z':
            z2 = np.linspace(N, 0, N+1)
            x2 = A + D1 + z2*0
            y2 = A + D2 + z2*0  # Set y values to point in the z direction (0)

            ep2 = z2 * 0
            ep2[-1] = 1

            x = np.concatenate((x,x2))
            y = np.concatenate((y,y2))
            z = np.concatenate((z,z2))
            e = np.concatenate((e,e2))

            z2 = np.linspace(N,0,N+1)
            x2 = A + D2 + z2*0
            y2 = A + D1 + z2*0

            e2 = z2*0
            e2[-1] = 1

            x = np.concatenate((x,x2))
            y = np.concatenate((y,y2))
            z = np.concatenate((z,z2))
            e = np.concatenate((e,e2))

            z2 = np.linspace(0,N,N+1)
            x2 = A + D2 + z2*0
            y2 = A + D2 + z2*0
            e2 = z2*0
            e2[-1] = 1

            x = np.concatenate((x,x2))
            y = np.concatenate((y,y2))
            z = np.concatenate((z,z2))
            e = np.concatenate((e,e2))

        elif ORIENTATION == 'x':
            x2 = np.linspace(N, 0, N+1)
            y2 = A + D1 + x2 * 0
            z2 = A + D2 + x2 * 0  # Set z values to point in the x direction (0)

            ep2 = x2 * 0
            ep2[-1] = 1

            x = np.concatenate((x,x2))
            y = np.concatenate((y,y2))
            z = np.concatenate((z,z2))
            e = np.concatenate((e,e2))

            x2 = np.linspace(N, 0, N+1)
            y2 = A + D2 + x2 * 0
            z2 = A + D1 + x2 * 0  # Set z values to point in the x direction (0)

            ep2 = x2 * 0
            ep2[-1] = 1

            x = np.concatenate((x,x2))
            y = np.concatenate((y,y2))
            z = np.concatenate((z,z2))
            e = np.concatenate((e,e2))
            x2 = np.linspace(N, 0, N+1)
            y2 = A + D2 + x2 * 0
            z2 = A + D2 + x2 * 0  # Set z values to point in the x direction (0)

            ep2 = x2 * 0
            ep2[-1] = 1

            x = np.concatenate((x,x2))
            y = np.concatenate((y,y2))
            z = np.concatenate((z,z2))
            e = np.concatenate((e,e2))

        elif ORIENTATION == 'y':
            y2 = np.linspace(N, 0, N+1)
            x2 = A + D1 + y2 * 0
            z2 = A + D2 + y2 * 0  # Set z values to point in the y direction (0)

            ep2 = y2 * 0
            ep2[-1] = 1

            x = np.concatenate((x,x2))
            y = np.concatenate((y,y2))
            z = np.concatenate((z,z2))
            e = np.concatenate((e,e2))

            y2 = np.linspace(N, 0, N+1)
            x2 = A + D2 + y2 * 0
            z2 = A + D1 + y2 * 0  # Set z values to point in the y direction (0)

            ep2 = y2 * 0
            ep2[-1] = 1

            x = np.concatenate((x,x2))
            y = np.concatenate((y,y2))
            z = np.concatenate((z,z2))
            e = np.concatenate((e,e2))

            y2 = np.linspace(N, 0, N+1)
            x2 = A + D2 + y2 * 0
            z2 = A + D2 + y2 * 0  # Set z values to point in the y direction (0)

            ep2 = y2 * 0
            ep2[-1] = 1

            x = np.concatenate((x,x2))
            y = np.concatenate((y,y2))
            z = np.concatenate((z,z2))
            e = np.concatenate((e,e2))

    else:
        raise ValueError("So far, there is only AUX=1 and AUX=3 available.")

    coords = np.column_stack((x, y, z, e))
    np.savetxt('string.dat', coords, delimiter=' ', fmt='%.2f %.2f %.2f %d')

    with open(PATH + 'string.dat', 'w') as file:
        file.write(f"# N: {N}\n")
        file.write(f"# AUX: {AUX}\n")
        file.write(f"# A: {A}\n")
        file.write(f"# D: {D}\n")
        file.write(f"# D1: {D1}\n")
        file.write(f"# D2: {D2}\n")
        file.write(f"# DIST: {DIST}\n")
        file.write(f"# ORIENTATION: {ORIENTATION}\n")
        np.savetxt(file, coords, delimiter=' ', fmt='%.2f %.2f %.2f %i')

    return x, y, z


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
