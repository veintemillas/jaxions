import numpy as np
import matplotlib.pyplot as plt
from pyaxions import jaxions as pa

from scipy.integrate import quad
from scipy.interpolate import CubicSpline

import importlib, os, pickle, h5py, subprocess, timeit

from IPython.display import clear_output


# ---------------------------------------------------------------------------
# Top-level simulation driver
# ---------------------------------------------------------------------------

def simu(R, msa, N, Ng=2, Np=1, omp=1, plota=False, rescale=1, n_save=200,
         gpu=False, verb=0, options='', outdir=None):
    '''simu(R, msa, N, Ng=2, Np=1, omp=1, plota=False, rescale=1, n_save=200,
            gpu=False, verb=0, options='', outdir=None)

    Prepares files for a NxN caxion3d simulation and runs them.
    ICs are prepared in python as a static loop of radius R
    where theta is calculated from KR equivalent B-field
    and rho according to the distance from the string and msa.
    dx = 1, msa = equivalent to saxion mass.
    The program currently works only in Minkowski; FRW is easy to implement.

    Ng       : number of neighbours in the Laplacian
    Np       : number of MPI ranks
    omp      : number of OMP threads
    gpu      : use GPU device
    verb     : jaxions verbosity level
    options  : extra jaxion commands, e.g. ' --p2DmapYZ'
    outdir   : output directory name.
               If None (default), saved to data/outN-1000*msa-Ng in cwd.
               If given, saved to that name directly in cwd.
    n_save   : approximate number of measurements before collapse
    rescale  : create ICs at 1/rescale resolution, then run at full N
               (keep at 1 for now to avoid problems)
    '''
    if plota:
        print('Simulation')

    N_create   = N // rescale
    R_create   = R / rescale
    msa_create = msa * rescale

    if plota:
        print('Creation with N, R, msa =', N_create, R_create, msa_create)

    theta = thetaics(N_create, N_create, R_create, plota=plota, readIC=True)
    phi   = phiics(theta, msa_create)

    # Build and run ICs-only step (creates the HDF5 skeleton)
    JAXI, GRID, _ = generic_jax(msa_create, N_create, R=R_create, Ng=1, Np=Np,
                                 gpu=False, verb=verb, dump=1, options=options)
    create_jax(GRID + JAXI, Np=Np, omp=omp)

    if plota:
        print('Copy into file', N_create, R_create, msa_create)
    copyics(np.transpose(phi, (1, 0, 2)), filename='out/m/axion.00000', n=0)

    if plota:
        print('Run jaxions', N, R, msa)

    dump = int(N * np.sqrt(12) / n_save)
    JAXI, GRID, _ = generic_jax(msa, N, R=R, Ng=Ng, Np=Np, gpu=gpu, verb=verb,
                                 dump=dump, options=options)
    run_jax(GRID + JAXI + ' --index 0 ', Np=Np, omp=omp)

    # Pack output files
    subprocess.run('mv axion.log.* out',      shell=True, capture_output=True, text=True)
    subprocess.run('mv log-c*.txt out',       shell=True, capture_output=True, text=True)
    subprocess.run('mv create.sh run.sh out', shell=True, capture_output=True, text=True)

    # Resolve destination directory
    if outdir is None:
        dest = namea(N, msa, Ng)           # e.g. data/out128-500-2
        os.makedirs('data', exist_ok=True)  # ensure data/ exists
    else:
        dest = outdir                       # placed directly in cwd

    # Safe move: refuse to clobber anything that isn't a simulation directory
    if os.path.exists(dest):
        if os.path.isdir(dest):
            subprocess.run('rm -r %s' % dest, shell=True, capture_output=True, text=True)
        else:
            raise RuntimeError(
                'Destination %s exists but is not a directory — aborting.' % dest)

    subprocess.run('mv out %s' % dest, shell=True, capture_output=True, text=True)
    if plota:
        print('Output saved to', dest)


def namea(N, msa, Ng):
    '''Default output directory name: data/outN-1000*msa-Ng'''
    return 'data/out%d-%d-%d' % (N, 1000 * msa, Ng)


# ---------------------------------------------------------------------------
# jaxions command-line helpers  (MPI / OMP / GPU-aware)
# ---------------------------------------------------------------------------

def generic_jax(msa, N, R=None, Ng=2, Np=1, dump=100, gpu=True, verb=0, options=''):
    '''Build jaxions command strings.

    Returns (JAXI, GRID, N) where:
        JAXI : simulation + physics + IC + output flags
        GRID : grid/decomposition flags
        N    : grid size (passed through for convenience)
    R    : loop radius in code units; sets zf=1.7*R to stop shortly after collapse.
           If None, falls back to zf=N (run to end of box).
    '''
    zf = int(1.7 * R) if R is not None else N
    GRID = " --nx %d --nz %d --zgrid %d" % (N, N // Np, Np)
    if gpu:
        SIMU = " --device gpu --measCPU  --steps 20000000 --wDz 1.0 --lap %d" % Ng
    else:
        SIMU = " --steps 20000000 --wDz 1.0 --lap %d" % Ng
    PHYS = " --vqcd0 --mink --notheta --msa %f --lsize %d  --zf %d " % (msa, N, zf)
    INCO = " --ctype smooth --zi 0.1 --sIter 0 --nncore "
    OUTP = " --p2DmapXZ --dump %d --meas 0 --nologmpi --verbose %d %s" % (dump, verb, options)
    return SIMU + PHYS + INCO + OUTP, GRID, N


def run_jax(JAXI, Np=1, omp=1, r_file='run.sh', o_file='log-con.txt'):
    '''Write and execute a run script with MPI + OMP settings.'''
    subprocess.run('rm out/m/axion.m.*', shell=True, capture_output=True, text=True)
    with open(r_file, 'w') as rsh:
        rsh.write('''\
        #! /bin/bash
        export OMP_NUM_THREADS=%d
        mpirun -np %d caxion3d %s > %s 2>&1
        ''' % (omp, Np, JAXI, o_file))
    subprocess.run('chmod u+x %s' % r_file, shell=True, capture_output=True, text=True)
    subprocess.run('./%s'           % r_file, shell=True, capture_output=True, text=True)


def create_jax(JAXI, Np=1, omp=1):
    '''Run jaxions with --steps 0 to create the HDF5 file skeleton.'''
    os.makedirs("out/m", exist_ok=True)
    run_jax(JAXI + ' --steps 0 --p3D 1 ', Np=Np, omp=omp,
            r_file='create.sh', o_file='log-create.txt')


# ---------------------------------------------------------------------------
# HDF5 IC writer
# ---------------------------------------------------------------------------

def copyics(phi, filename='out/m/axion.00000', n=0):
    Nz, Nx, c = phi.shape
    print('saving jaxions file', phi.shape)
    f1   = h5py.File(filename, 'r+')
    data = f1['/m']
    data[...] = np.reshape(phi, Nz * Nx * 2)
    vata = f1['/v']
    vata[...] = np.reshape(phi * n, Nz * Nx * 2)
    f1.close()


# ---------------------------------------------------------------------------
# Radius / velocity measurement
# ---------------------------------------------------------------------------

def buildr(mf, sigma=50):
    '''Extract loop radius, velocity, and Lorentz gamma vs time.

    Uses Gaussian-weighted centroid (findmer2) for all frames.
    Returns (t, r, v, gamma).
    '''
    N   = pa.gm(mf[0], 'N')
    msa = pa.gm(mf[0], 'msa')
    t     = pa.gml(mf, 'ct')
    r     = np.zeros_like(t)
    v     = np.zeros_like(t)
    gamma = np.zeros_like(t)
    for it in range(len(mf)):
        li   = np.reshape(pa.gm(mf[it], 'da/chunk/m/'), (N, 2))[:, 0]
        vel2 = np.reshape(pa.gm(mf[it], 'da/chunk/v/'), (N, 2))[:, 0] ** 2
        r[it], v[it], gamma[it] = findmer2(li, vel2, msa, sigma, 're_complex')
    return t, r, v, gamma


def getR_2D(mf, sigma=30):
    '''Extract R, v, gamma from 2D measurement files using a Gaussian weight.

    Returns (ct, R, V, G) with ct as absolute simulation time.
    '''
    N   = pa.gm(mf[0], 'N')
    msa = pa.gm(mf[0], 'msa')
    ct  = pa.gml(mf, 'ct')
    rr  = np.arange(N)
    R, V, G = [], [], []
    for m in mf:
        li   = np.reshape(pa.gm(m, 'da/chunk/m/'), (N, 2))[:, 0]
        vel2 = np.reshape(pa.gm(m, 'da/chunk/v/'), (N, 2))[:, 0] ** 2
        w  = np.exp(-li ** 2 * sigma)
        W  = np.sum(w)
        r  = np.sum(rr * w) / W
        vgamma2 = np.sum((vel2 / (.43 ** 2 * msa ** 2)) * w) / W
        v  = np.sqrt(vgamma2 / (1 + vgamma2))
        g  = np.sqrt(1 + vgamma2)
        R.append(r);  V.append(v);  G.append(g)
    return ct, np.array(R), np.array(V), np.array(G)


def getR_2D_interp(mf):
    '''Extract R from 2D measurement files by interpolating the zero crossing.

    Returns (ct, R) with ct as absolute simulation time.
    '''
    N  = pa.gm(mf[0], 'N')
    ct = pa.gml(mf, 'ct')
    R  = []
    for m in mf:
        li = np.reshape(pa.gm(m, 'da/chunk/m/'), (N, 2))[:, 0]
        r0 = np.nan
        for i in range(N - 1):
            f1, f2 = li[i], li[i + 1]
            if f1 == 0:
                r0 = float(i);  break
            elif f1 < 0 and f2 > 0:
                r0 = i - f1 / (f2 - f1);  break
        if np.isnan(r0) and li[-1] == 0:
            r0 = float(N - 1)
        R.append(r0)
    return ct, np.array(R)


# ---------------------------------------------------------------------------
# Zero-crossing / slope finder (sign-change method)
# ---------------------------------------------------------------------------

def findmer(li, ftype='re_complex'):
    '''Find coordinate where a 1D field changes sign (complex) or jumps by pi (theta).

    Returns (R, slope).
    '''
    R, m = -1, -1
    if ftype == 're_complex':
        for i in range(len(li) - 1):
            if (li[i + 1] > 0) and (li[i] < 0):
                m = i
                R = m - li[m] / (li[m + 1] - li[m])
                break
    if ftype == 'theta':
        for i in range(len(li) - 1):
            if np.abs(li[i + 1] - li[i]) > 3.1415:
                m = i
                R = m + 0.5
                break
    if m > 0:
        fr   = 3
        nmin = max(0, m - fr)
        b, c = np.polyfit(np.arange(nmin, m + fr) - R, li[nmin:m + fr], 1)
        return R, b
    return 0, 1


def findmer2(li, vel2, msa, sigma=50, ftype='re_complex'):
    '''Gaussian-weighted centroid finder; also computes v and Lorentz gamma.

    Returns (R, v, gamma).
    '''
    if ftype == 're_complex':
        tmp  = np.exp(-li ** 2 * sigma)
        tmp /= np.sum(tmp)
        rrange  = np.arange(len(tmp))
        R       = np.sum(rrange * tmp)
        vgamma2 = np.sum((vel2 / .43 ** 2 / msa ** 2) * tmp)
        v       = np.sqrt(vgamma2 / (1 + vgamma2))
        gamma   = np.sqrt(1 + vgamma2)
        return R, v, gamma

    # theta branch (fallback)
    R, m = -1, -1
    if ftype == 'theta':
        for i in range(len(li) - 1):
            if np.abs(li[i + 1] - li[i]) > 3.1415:
                m = i;  R = m + 0.5;  break
    if m > 0:
        fr   = 3
        nmin = max(0, m - fr)
        b, c = np.polyfit(np.arange(nmin, m + fr) - R, li[nmin:m + fr], 1)
        return R, b, 1.0
    return 0, 1, 1.0


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def rhof(r):
    '''Radial profile rho(r) for a straight string (r in units of 1/ms).'''
    r2 = r * r;  r4 = r2 * r2
    return (0.43 * r + 0.164 * r2 + 0.036 * r4) / (1 + 0.039 * r + 0.2 * r2 + 0.036 * r4)


def phiics(theta, msa):
    '''Build phi = rho * exp(i*theta) with rho minimising the EOM.'''
    R, s    = findmer(theta[0, :], ftype='theta')
    nz, nrh = theta.shape
    rh      = np.arange(nrh)
    z       = np.arange(nz)
    RH, Z   = np.meshgrid(rh, z)
    phi     = np.zeros((nz, nrh, 2))
    rho     = rhof(msa * np.sqrt((RH - R) ** 2 + Z ** 2))
    phi[:, :, 0] = rho * np.cos(theta)
    phi[:, :, 1] = rho * np.sin(theta)
    return phi


def thetaics(nrh, nz, R, plota=False, readIC=True):
    '''Create a nrh x nz theta field for a loop of radius R.

    Integrates the static B-field along z.
    readIC=True  : load cached ICs from aux/ if available.
    readIC=False : always recompute.
    '''
    name     = 'theta_%dx%dR%d.pkl' % (nrh, nz, R)
    b_create = True

    if readIC:
        try:
            with open('aux/' + name, 'rb') as f:
                print('read from file')
                theta    = pickle.load(f)
                b_create = False
        except IOError:
            print('ICs not found, will create them')

    if b_create:
        rh    = np.arange(nrh)
        z     = np.arange(nz)
        theta = np.zeros((nz, nrh))

        theta[0, (rh <= R)] = np.pi

        RH, Z = np.meshgrid(rh, z)

        with open('aux/tableszetat1t2_4.pkl', 'rb') as f:
            dica = pickle.load(f)
        zeta, t1, t2 = dica['zeta'], dica['t1'], dica['t2']
        f1 = CubicSpline(zeta, t1)
        f2 = CubicSpline(zeta, t2)

        def I1f(z):
            return (3 * np.pi / 4 * z + 2 ** 1.5 * z ** 3 / (1 - z ** 2)) * f1(z)

        def I2f(z):
            return (np.pi + 2 ** 1.5 * z ** 2 / (1 - z ** 2)) * f2(z)

        def Bz(RH, Z, R):
            ZETA = 2 * R * RH / (R ** 2 + RH ** 2 + Z ** 2)
            return (R * RH * I1f(ZETA) - R ** 2 * I2f(ZETA)) / (R ** 2 + RH ** 2 + Z ** 2) ** (3 / 2)

        Bzfie = Bz(RH, Z, R)

        def filltheta(theta, Bzfie, z, rh, R, Z, threshold=10, calculate=True):
            check = np.zeros(len(rh))
            for i in range(nrh):
                for j in range(1, nz):
                    if (rh[i] - R) ** 2 + z[j] ** 2 > threshold ** 2:
                        theta[j, i] = theta[j - 1, i] + (Bzfie[j - 1, i] + Bzfie[j, i]) * (Z[j, i] - Z[j - 1, i]) / 2
                    else:
                        if calculate:
                            def Bzr(zi, Ri):
                                return Bz(rh[i], zi, Ri)
                            res, err = quad(Bzr, z[j - 1], z[j], args=(R,))
                            theta[j, i] = theta[j - 1, i] + res
                            check[i]   += 1
                        else:
                            theta[j, i] = np.arctan2(z[j], (rh[i] - R))
            return check

        check = filltheta(theta, Bzfie, z=z, rh=rh, R=R, Z=Z, threshold=10, calculate=True)

        with open('aux/' + name, 'wb') as f:
            pickle.dump(theta, f)

    if plota:
        fig, ax = plt.subplots(1, 2, figsize=(20, 20))
        i = ax[0].imshow(theta, cmap=pa.thetacmap, origin='lower', vmax=np.pi, vmin=-np.pi)
        pa.colorbar(i)
        i = ax[1].imshow(theta, origin='lower')
        pa.colorbar(i)
        ax[1].set_xlim(R - 2 * R / 10, R + 2 * R / 10)
        ax[1].set_ylim(0, 2 * R / 10)

    return theta


# ---------------------------------------------------------------------------
# Interpolation table builder
# ---------------------------------------------------------------------------

def buildf1f2(ninterp=10000):
    '''Build and cache the zeta/t1/t2 interpolation tables used by thetaics.'''
    name = 'tableszetat1t2_%d.pkl' % np.log10(ninterp)

    def I1i(x, zet):
        return np.cos(x) / (1 - zet * np.cos(x)) ** (3 / 2)

    def I2i(x, zet):
        return 1 / (1 - zet * np.cos(x)) ** (3 / 2)

    zeta = np.linspace(0, 1, ninterp)[:-1]
    I1t  = zeta * 0
    I2t  = zeta * 0
    for i in range(len(zeta)):
        res1, _ = quad(I1i, 0, np.pi, args=(zeta[i],))
        I1t[i]  = res1
        res2, _ = quad(I2i, 0, np.pi, args=(zeta[i],))
        I2t[i]  = res2

    t1, t2 = np.ones(len(zeta) + 1), np.ones(len(zeta) + 1)
    zc = zeta[1:]
    t1[1:-1] = I1t[1:] / (3 * np.pi / 4 * zc + 2 ** 1.5 * zc ** 3 / (1 - zc ** 2))
    t2[1:-1] = I2t[1:] / (np.pi * zc / zc    + 2 ** 1.5 * zc ** 2 / (1 - zc ** 2))
    zeta = np.linspace(0, 1, ninterp)

    dica = {'zeta': zeta, 't1': t1, 't2': t2}
    with open('aux/' + name, 'wb') as f:
        pickle.dump(dica, f)
    print(name, 'saved')
    return zeta, t1, t2
