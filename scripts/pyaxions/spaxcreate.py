#!/usr/bin/python3

import matplotlib
import numpy as np
from scipy import interpolate


def genspec(N, L, R, mA, X, klist, v02, mode='v'):
    """
    Creates the IC for jAxions spax option in 'initialspectrum.txt'

    :par N  : integer     [#points/dimension]
    :par L  : float       [box length in L1 units]
    :par R  : float       [scale factor]
    :par mA : float       [axion mass in H1 units]
    :par X  : float       [number density of axions/misalignment]
    :par kl : float array [array of wavenumbers in k1 units]
    :par v2 : float array [array of |FT|^2 in arbitrary units]
    :par m  : string      ['m'/'v' for v2 = |FT{theta}|^2/|FT{theta'}|^2 ]

    returns k, m, v
    array k               [array of wavenumbers that jAxions will interpret]
    array m               [array of |FT{theta}| for jAxions]
    array v               [array of |FT{theta'}| for jAxions]

    creates 'initialspectrum.txt'

    Typical usage :

    N = 256
    L = 6
    R = 1.0
    nTopSus = 7.0
    mA = R**nTopSus
    X = 0.5

    k1 = np.linspace(0,1000,5000)
    x0, q, ms = 1/1, 0.5, 80
    m2 = 1/(1+(k1/x0)**(4+q))*1/(1+(k1/ms)**(8))

    k, m, v = spaxcreate.genspec(N,L,R,mA, X, k1, m2, 'm')

    # plot the spectrum
    # plt.semilogy(k1,np.sqrt(m2))
    # plt.xscale('linear')
    # plt.xlim(0,10)
    """
    print('jAxions SPAX generator')
    print('----------------------')
    print('mode '+mode+' selected')
    f = interpolate.interp1d(klist, v02)
    k0 = 2*np.pi/L
    k = k0*np.arange(2*N)
    if ( k[0] < klist[0]) :
        print('Error: k[0] < kl_input[0]   [%f < %f]; reduce k1 lower limit! '%(k[0],klist[0]))
        return 0,0,0
    if ( k[-1] > klist[-1]) :
        print('Error: k[-1] > kl_input[-1] [%f > %f]; increase k1 upper limit! '%(k[-1],klist[-1]))
        return 0,0,0

    v03 = f(k)
    w  = np.sqrt(k**2 + mA*mA*R*R)
    # compute n as if v03 would be the |FFT(psi')|^2/N*6 in our code
    # phase space x occupation number  [the volume factor is from the FFT normalisation]
    if mode =='v':
        occnumber = (k0*k*k/(2*np.pi**2))*((v03)/(2*w)*L**3)
    if mode =='m':
        # alternative if |FFT(psi|^2 is passed
        occnumber = (k0*k*k/(2*np.pi**2))*((v03*w/2)*L**3)
    # number in ADM units
    suma = occnumber.sum()

    # Kinetic energy has to be half of the misalignment (16)
    # Gradient/Potential energy will be the other half
    v03 = v03*(X*8)/suma

    if mode == 'v':
        v = np.sqrt(v03)
        m = v/w
        m[0] = 0
    if mode == 'm':
        m = np.sqrt(v03)
        v = m*w

    xy = np.column_stack((m, v))
    np.savetxt('initialspectrum.txt', xy, delimiter=' ', fmt='%.8e %.8e')   # X is an array
    print('File initialspectrum.txt created')
    print('Done')
    return k, m, v
