#JAXIONS MODULE TO RUN SIMULATIONS FROM PYTHON
import os
import time
import re

#auxiliary function to find the last config file
def last_mfile():
    cwd = os.getcwd()
    out_m_dir = cwd + '/out/m'
    files = os.listdir(out_m_dir)

    indices = [int(filename.split('.')[-1]) for filename in files if filename.startswith('axion.') and not filename.startswith('axion.m.')]

    if indices:
        return max(indices)
    else:
        return None

def runsim(JAX, MODE='run', RANK=1, THR=1, USA=' --bind-to socket --mca btl_base_warn_component_unused  0', IDX=False, OUT_CON='out1', CON_OPTIONS='', PAX_OPTIONS='', VERB=False, BONDEN=False):
    """
    runsim(JAX, MODE='run', RANK=1, THR=1, USA=' --bind-to socket --mca btl_base_warn_component_unused  0', IDX=False, OUT_CON='out1', CON_OPTIONS='', PAX_OPTIONS='')

    1 - cleans the out/m directory of axion.m.files
    2 - creates, runs or continues jaxions simulations as (equivalent to the definitons in the "standard" 'vax-ex.sh' scripts):

            - MODE = "create":  mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} --steps 0 --p3D 1 2>&1 | tee log-create.txt
            - MODE = "run":     mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} 2>&1 | tee log.txt (DEFAULT)
            - MODE = "con":     mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} --index {IDX or last_mfile()} {CON_OPTIONS} 2>&1 | tee log-con.txt
            - MODE = "paxion":  mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} paxion3d --index {IDX or last_mfile()} {GRID+SIMU flags from JAX} {PAX_OPTIONS} 2>&1 | tee log-paxion.txt

    *Note that for "con" and "paxion", the user can either specify a specific index or the last config file will be used.
    *For "paxion", the flags --size, --depth, --zgrid, --prec, --fftplan, --lap are extracted from JAX automatically.
     PAX_OPTIONS should contain paxion-specific flags only (e.g. --Rc, --zf, --ftype, --steps, --wDz, --meas, --dump, --p3D, ...).

    JAX         string  vaxion3d flags generated with simgen
    MODE        str     'run' (default), 'create', 'con', or 'paxion'
    RANK        int     number of MPI processes
    THR         int     number of OMP threads
    USA         str     mpirun options
    IDX         int     config file index for 'con' and 'paxion' modes (uses last_mfile() if False)
    OUT_CON     str     output folder for 'con' mode (default: 'out1')
    CON_OPTIONS str     additional flags for 'con' mode
    PAX_OPTIONS str     paxion-specific flags, e.g. '--Rc 16.0 --zf 150.0 --ftype axion ...'
    VERB        bool    print full mpirun command if True
    BONDEN      bool    use mpiexec instead of mpirun
    """
    if VERB:
        print('')
        print('')
        print('--------------------------------------------------------------------------------------------')
        print(f'Mode: {MODE} ')
        print('')

    # Get important parameters from JAX input string
    read_params = JAX
    cwd = os.getcwd()

    # Read specific values from the input JAX string (for printout and reuse)
    N0_match      = re.search(r'--size (\d+)',       read_params)
    depth_match   = re.search(r'--depth (\d+)',      read_params)
    zgrid_match   = re.search(r'--zgrid (\d+)',      read_params)
    L0_match      = re.search(r'--lsize (\d+\.\d+)', read_params)
    msa0_match    = re.search(r'--msa (\d+\.\d+)',   read_params)
    prec_match    = re.search(r'--prec (\S+)',        read_params)
    fftplan_match = re.search(r'--fftplan (\d+)',    read_params)
    lap_match     = re.search(r'--lap (\d+)',         read_params)

    N0    = int(N0_match.group(1))
    depth = int(depth_match.group(1))
    L0    = float(L0_match.group(1))
    if msa0_match:
        msa0 = float(msa0_match.group(1))

    # for mpiexec usage on bonden
    os.environ['OMP_NUM_THREADS'] = str(THR)

    if MODE == 'create':
        # Clear "out" folder, in case it exists
        output = os.popen(f'rm out/m/axion.m.*')
        output.read()

        if VERB:
            print('Overview: N=%d, MPI_RANKS=%d, L=%f, msa=%f' % (N0, RANK, L0, msa0))
            print(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} --steps 0 --p3D 1 2>&1 | tee log-create.txt')

        if BONDEN:
            output = os.popen(f'mpiexec {USA} -n {RANK} vaxion3d {JAX} --steps 0 --p3D 1 2>&1 | tee log-create.txt')
        else:
            output = os.popen(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} --steps 0 --p3D 1 2>&1 | tee log-create.txt')
        output.read()
        if VERB:
            print('')
            print('Done!')

    elif MODE == 'run':
        # Clear "out" folder, in case it exists
        output = os.popen(f'rm out/m/axion.m.*')
        output.read()

        if VERB:
            print('Overview: N=%d, MPI_RANKS=%d, L=%f, msa=%f' % (N0, RANK, L0, msa0))
            print(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} 2>&1 | tee log-run.txt')

        if BONDEN:
            output = os.popen(f'mpiexec {USA} -n {RANK} vaxion3d {JAX} 2>&1 | tee log-run.txt')
        else:
            output = os.popen(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} 2>&1 | tee log-run.txt')
        output.read()
        if VERB:
            print('')
            print('Done!')

    elif MODE == 'con':
        # Define con options as the minimally needed ones + whatever is specified by the user in CON_OPTIONS
        extra_con_options = ''
        if '--size' not in CON_OPTIONS:
            extra_con_options += '--size %d ' % N0
        if '--depth' not in CON_OPTIONS:
            extra_con_options += '--depth %d ' % depth
        if '--msa' not in CON_OPTIONS:
            extra_con_options += '--msa %f ' % msa0

        # add additional params from CON_OPTIONS to the minimally required ones
        extra_con_options += CON_OPTIONS

        # Create new dir for mfiles
        os.makedirs(OUT_CON, exist_ok=True)
        os.makedirs(f'{OUT_CON}/m', exist_ok=True)

        # Set the AXIONS_OUTPUT environment variable for the Python process and its children
        os.environ['AXIONS_OUTPUT'] = f"{cwd}/{OUT_CON}/m"

        # Either use the user-specified index or use the last config file
        index = IDX if IDX else last_mfile()

        # properly link config files
        find = f'{index:05d}'
        symlink_src = f'{cwd}/out/m/axion.{find}'
        symlink_dst = f'{cwd}/{OUT_CON}/m/axion.{find}'
        if not os.path.exists(symlink_src):
            raise FileNotFoundError(
                f"Source config file not found: {symlink_src}\n"
                f"Make sure 'out/m/axion.{find}' exists before running con mode."
            )
        if not os.path.exists(symlink_dst):
            os.symlink(symlink_src, symlink_dst)

        if VERB:
            print(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} --index {index} {extra_con_options} 2>&1 | tee log-con.txt')
        else:
            print('Overview: N=%d, MPI_RANKS=%d, L=%f, msa=%f (data in %s)' % (N0, RANK, L0, msa0, OUT_CON))

        if BONDEN:
            output = os.popen(f'mpiexec {USA} -n {RANK} vaxion3d {JAX} --index {index} {extra_con_options} 2>&1 | tee log-con.txt')
        else:
            output = os.popen(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} --index {index} {extra_con_options} 2>&1 | tee log-con.txt')
        output.read()
        if VERB:
            print('')
            print('Done!')

    elif MODE == 'paxion':
        # Extract compatible grid/simu flags from JAX to pass to paxion3d
        pax_base = ''
        if N0_match:
            pax_base += f' --size {N0}'
        if depth_match:
            pax_base += f' --depth {depth}'
        if zgrid_match:
            pax_base += f' --zgrid {zgrid_match.group(1)}'
        if prec_match:
            pax_base += f' --prec {prec_match.group(1)}'
        if fftplan_match:
            pax_base += f' --fftplan {fftplan_match.group(1)}'
        if lap_match:
            pax_base += f' --lap {lap_match.group(1)}'

        # Create output dir for paxion mfiles (default: pout/m)
        pax_out = OUT_CON if OUT_CON != 'out1' else 'pout'
        os.makedirs(pax_out, exist_ok=True)
        os.makedirs(f'{pax_out}/m', exist_ok=True)

        # Set AXIONS_OUTPUT for paxion3d
        os.environ['AXIONS_OUTPUT'] = f"{cwd}/{pax_out}/m"

        # Either use the user-specified index or use the last config file
        index = IDX if IDX else last_mfile()

        # Link the config file from out/m/
        find = f'{index:05d}'
        symlink_src = f'{cwd}/out/m/axion.{find}'
        symlink_dst = f'{cwd}/{pax_out}/m/axion.{find}'
        if not os.path.exists(symlink_src):
            raise FileNotFoundError(
                f"Source config file not found: {symlink_src}\n"
                f"Make sure 'out/m/axion.{find}' exists before running paxion mode."
            )
        if not os.path.exists(symlink_dst):
            os.symlink(symlink_src, symlink_dst)

        if VERB:
            print(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} paxion3d --index {index}{pax_base} {PAX_OPTIONS} 2>&1 | tee log-paxion.txt')
        else:
            print('Overview: N=%d, MPI_RANKS=%d, L=%f (paxion, data in %s)' % (N0, RANK, L0, pax_out))

        if BONDEN:
            output = os.popen(f'mpiexec {USA} -n {RANK} paxion3d --index {index}{pax_base} {PAX_OPTIONS} 2>&1 | tee log-paxion.txt')
        else:
            output = os.popen(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} paxion3d --index {index}{pax_base} {PAX_OPTIONS} 2>&1 | tee log-paxion.txt')
        output.read()
        if VERB:
            print('')
            print('Done!')

    if VERB:
        print('--------------------------------------------------------------------------------------------')

def simgen (N=256,zRANKS=1,prec='single',dev='cpu', fftplan = 64, lowmem=False,prop='rkn4', spec=False, fspec=False, steps=1000000,wDz=1.0,sst0=10,lap=1,
            nqcd=7.0, fA = -1, msa=1.0,lamb=-1.0,ctf=128.,L=256.0, ind3=1.0,notheta=False,wkb=-1.,gam=0.0,dwgam=1.0,
            vqcd='vqcdC',vpq=0, mink = False, xtr='',prep=False,ic='lola',logi=0.0,cti=-1.11,
            index=-100,ict='lola',dump=10,meas=0,p3D=0,spmask=1,rmask=1.5,redmp=-1.0,wTime=-1.0,
            spKGV=15,printmask=False,ng0calib=1.25,cummask=0, normcore=False,
            p2Dmap=False,p2DmapE=False,p2DmapPE=False,p2DmapPE2=False, p2DmapYZ=False, slc=-1, strmeas=-1, measloop=-1,
            nologmpi=True,verbose=1, ftype='saxion', verb=False, **kwargs):
    """
    simgen creates a string of command line flags to select options for vaxion3d

    returns int (number of ranks), string (flags)

    options:

    N         int    256      number of grid points in x,y directions
    zRANKS    int    1        number of MPI processes along z direction
    prec      str    single   single or double
    dev       str    cpu      cpu or gpu (use --measCPU)
    lowmem    bool   False
    prop      str    rkn4     propagator/time integrator
    spec      str    False    spectral propagator 1 (only CPU atm)
    fspec     str    False    spectral propagator 2 (only CPU atm)
    steps     int    10000    number of time steps (max if --wDz is used)
    wDz       float  1.0      time interval set to dt = wDz/w_max
    sst0      int    10       time steps without strings before switch to theta
    lap       int    1        number of neighbours in Laplacian
    fftplan   int    64       specify FFT plan to speed up initialisation
    nqcd      float  7.0      index of ct-dependence of Topological Susceptibility
    fA        int    -1       if fA > 0 is given, the code uses the --qcd qcd option with fA as specified
    msa       float  1.0      use PRS strings with ms = msa/(dx R); overrides lambda!
    lamb      float  -1.0     use Physical strings with SI lambda; make >0; is overridden by --msa
    ctf       float  128.     final conformal time of simulations
    L         float  256.0    Physical length of x,y directions in ADM units
    ind3      float  1.0      coefficient multiplying axion mass^2
    notheta   bool   False    False/True will switch/not switch to theta-only simulations after strings
    wkb       float  -1.0     the final fields are wkb'ed to this c-time
    gam       float  0.0      damping
    dwgam     float  1.0      rho damping activated after artificial DW destruction
    vqcd      str             vqcd type: 'vqcdC','vqcdV','vqcd0','vqcdL','N2'
    vpq       str    0        2 for vPQ2
    prep      bool   False
    ic        str    lola     type of initial conditions
    logi      float  0.0      use this log ms/H to set the initial c-time
    cti       float  -1.11    initial c-time
    index     int    -100     will read axion.int initial conditions to continue the simulation
    ict
    dump
    meas
    p3D
    spmask
    rmask
    redmp
    wTime
    spKGV
    printmask
    ng0calib
    cummask
    p2Dmap
    p2DmapE
    p2DmapPE
    p2DmapPE2
    p2DmapYZ
    slc
    strmeas
    measloop
    nologmpi
    verbose
    verb
    kwargs
    """
    ####################################################
    GRID=" --size %d --depth %d --zgrid %d"%(N,N//zRANKS,zRANKS)
    ####################################################
    SIMU=" --prec %s --device %s --prop %s --steps %d --wDz %f --sst0 %d --fftplan %d --ftype %s"%(prec,dev,prop,steps,wDz,sst0,fftplan, ftype)
    if verb:
        print('Using --ftype %s'%ftype)

    if not (spec or fspec):
        SIMU += ' --lap %d'%lap

    if lowmem:
        SIMU += ' --lowmem'
    if dev == 'gpu':
        SIMU += ' --measCPU'

    if spec:
        SIMU += ' --spec'
        if verb:
            print('Using spec propagator.')
    if fspec:
        SIMU += ' --fspec'
        if verb:
            print('Using fspec propagator.')
    ####################################################
    VQCP=''
    if vqcd in ['vqcdC','vqcdV','vqcd0','vqcdL','N2']:
        VQCP += ' --%s'%vqcd
    else :
        if verb:
            print('Warning: VQCD not recognised!')
    if vpq == 2:
        VQCP += ' --vPQ2'
    if vqcd == 'vqcd0' and ind3 > 0:
        ind3 = 0.0
        if verb:
            print('Warning: vqcd0 -> ind3 reset to %f'%ind3)
    ####################################################
    if lamb>0:
        tension = ' --llcf %f'%lamb
    else :
        tension = ' --msa %f'%msa
    PHYS="%s --lsize %f --zf %f --ind3 %f"%(tension,L,ctf,ind3)
    noth='';wkbs='';gams='';dwgams='';mnk=''
    if fA > 0:
        qcd = ' --qcd qcd --fA %d'%fA
    else: qcd = ' --qcd %f'%nqcd
    if notheta:
        noth = ' --notheta '
    if wkb > 0:
        wkbs = ' --wkb %f'%wkb
    if gam > 0:
        gams = ' --gam %f'%gam
    if dwgam > 0:
        dwgams = ' --dwgam %f'%dwgam
    if mink:
        mnk = ' --mink'
        if verb:
            print('Minkowski!')
    PHYS += qcd+noth+wkbs+gams+dwgams+mnk+xtr

    if normcore:
        kwargs['normcore'] = True
    #################################################### IC condition 1 by 1
    if index >= 0:
        # READ CONF
        INCO = ' --index %d'%index
    else:
        if cti == -1.11:
            it = ' --logi %f'%logi
        else :
            it = ' --zi %f'%cti

        INCO= it + INCOgen(ict,verb,**kwargs)
    ####################################################
    OUT0=''
    if p2Dmap:
        OUT0+=' --p2Dmap'
    if p2DmapE:
        OUT0+=' --p2DmapE'
    if p2DmapPE and not p2DmapPE2:
        OUT0+=' --p2DmapPE'
    if p2DmapPE2 and not p2DmapPE:
        OUT0+=' --p2DmapPE2'
    if p2DmapYZ:
        OUT0+=' --p2DmapYZ'
    if slc >= 0:
        OUT0+=' --sliceprint %d'%slc
    if strmeas >= 0:
        OUT0+=' --strmeas %d'%strmeas
    if measloop >= 0:
        OUT0+=' --measloop %d'%measloop

    OUT1=" --dump %d --meas %d --p3D %d "%(dump,meas,p3D)
    if redmp > 0:
        OUT1 += ' --redmp %d'%redmp

    OUTM=" --spmask %d --rmask %s --spKGV %d"%(spmask,str(rmask),spKGV)
    if printmask:
        OUTM += ' --printmask'
    if ng0calib != 1.25:
        OUTM += ' --ng0calib %f'%ng0calib
    if cummask != 0:
        OUTM += ' --cummask %d'%cummask

    OUT2=" --verbose %d"%(verbose)
    if wTime > 0:
        OUT2 += '  --wTime %d'%wTime

    if nologmpi:
        OUT2 +=' --nologmpi'
    OUT=OUT0+OUT1+OUTM+OUT2
    if verb==True:
        print('PHYS =',PHYS)
        print('GRID =', GRID)
        print('SIMU = ',SIMU)
        print('VQCD =',VQCP)
        print('INCO =',INCO)
        print('OUT =',OUT)
    return zRANKS, PHYS+GRID+SIMU+VQCP+INCO+OUT

def INCOgen(ict,verb=False,**kwargs):
    def fif(ka,ja,xic):
        if ka in kwargs:
            xic += ' --%s '%ja + str(kwargs[ka])
            if verb:
                print(xic)
        else:
            if verb:
                print('%s missing in kwargs: jaxion defaults will be used'%ka)
        return xic

    INCO = ''
    if ict == 'lola':
        INCO = ' --ctype %s'%ict
        if 'lola_string_multiplier' in kwargs:
            lr = 1
            if 'lolarandom' in kwargs:
                if kwargs['lolarandom']:
                    lr=2
            INCO += ' --sIter %d --kcr %f'%(lr,kwargs['lola_string_multiplier'])
    if ict == 'spax':
        INCO = ' --ctype %s'%ict
    if ict == 'km':
        INCO = ' --ctype %s --mode0 1'%ict
    if ict == 'smooth':
        INCO = ' --ctype %s'%ict
        if 'smvar' in kwargs:
            INCO += ' --smvar %s'%(kwargs['smvar'])
        INCO = fif('mode0','mode0',INCO)
        INCO = fif('kMax','kMax',INCO)
        INCO = fif('kcr','kcr',INCO)
    if ict == 'cole':
        INCO = ' --ctype %s'%ict
        INCO = fif('kMax','kMax',INCO)
    if ict == 'tkachev':
        INCO = ' --ctype %s'%ict
        INCO = fif('kMax','kMax',INCO)
        INCO = fif('kcr','kcr',INCO)

    if ict == 'string':
        INCO = ' --ctype %s'%ict
        INCO = fif('sIter','sIter',INCO)
        if 'kmax' in kwargs:
            INCO = fif('kmax','kmax',INCO)

    if ict == 'thermal':
        INCO = ' --ctype %s'%ict
        INCO = fif('RPQ','RPQ',INCO)
        INCO = fif('kcr','kcr',INCO)

    if 'kickalpha' in kwargs:
        INCO += ' --kickalpha '+str(kwargs['kickalpha'])
    if 'extrav' in kwargs:
        INCO += ' --extrav '+str(kwargs['extrav'])
    if kwargs.get('normcore', False):
        INCO += ' --nncore'
    PREP = ''
    if 'preprop' in kwargs:
        PREP += ' --preprop'
        # preprequires damping?
        PREP = fif('prepcoe','prepcoe',PREP)
        PREP = fif('lz2e','lz2e',PREP)
        PREP = fif('prevqcdtype','prevqcdtype',PREP)
        PREP = fif('pregam','pregam',PREP)
        if 'icstudy' in kwargs:
            if kwargs['icstudy']:
                PREP += ' --icstudy'
    return INCO+PREP
