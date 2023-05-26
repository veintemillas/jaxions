# JAXIONS MODULE TO PREPARE A string.dat FILE FOR SIMULATIONS WITH THE "String" INITIAL CONDITIONS
import numpy as np

def randomstrings(N, LEN, NSTRINGS=1, ITER=3, XCF=0.5, YCF=0.5, ZCF=0.5, PATH = './'):

    """
    randomstrings(N,L,NSTRINGS,ITER,XCF,YCF,ZCF,PATH)

    1) Generates Nstrings random string loops
    2) Stores their (x,y,z)-coordinates and an additional list marking the endpoint of every string (with a 1) to avoid wrong contributions from the sections connceting two distinct strings
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
