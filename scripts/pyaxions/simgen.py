# JAXIONS MODULE TO RUN SIMULATIONS FROM PYTHON
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

def runsim(JAX, MODE='run', RANK=1, THR=1, USA=' --bind-to socket --mca btl_base_warn_component_unused  0', IDX = False, OUT_CON='out1', CON_OPTIONS='', VERB = False, BONDEN = False):
    """
    runsim(JAX, MODE='run', RANK=1, THR=1, USA=' --bind-to socket --mca btl_base_warn_component_unused  0', IDX = False, OUT_CON='out1', CON_OPTIONS='')

    1 - cleans the out/m directory of axion.m.files
    2 - creates, runs or continues jaxions simulations as (equivalent to the definitons in the "standard" 'vax-ex.sh' scripts):

            - MODE = "create": mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} --steps 0 --p3D 1 2>&1 | tee log-create.txt
            - MODE = "run":    mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} 2>&1 | tee log.txt (DEFAULT)
            - MODE = "con":    mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} --index {IDX or last_mfile()} {CON_OPTIONS} 2>&1 | tee log-con.txt

    *Note that for "con", the user can either specify a specific index when calling the function or if not, the last config file will be used

    JAX  is a string of vaxion3d flags generated with the simgen program, see help(simgen)
    MODE specifies if the user want to 'run' (default), 'create' or 'con' a simulation
    RANK is the number of MPI processes
    THR  is the number of OMP processes
    USA  are mpirun options
    IDX is either 'False' or an int. It is used only for MODE='con' and specifies which config file should be continued
    OUT_CON is the name of the out folder where the simulation is continued
    CON_OPTIONS are additional settings/flags such as --size N etc. that are needed to appropriately continue/rescale the simulations
    VERB allows for more verbosity in the output by printing the full mpirun command with all the flags. If FALSE, only the used size, depth, lsize and msa will be given
    BONDEN is used to slightly modify the mpi commands for the use of our new computer
    """

    #Clear "out" folder
    output = os.popen(f'rm out/m/axion.m.*')
    output.read()
    print('')
    print('')
    print('--------------------------------------------------------------------------------------------')
    print(f'Mode: {MODE} ')
    print('')

    #Get important parameters from JAX input string
    read_params = JAX
    cwd = os.getcwd()

    #Read specific values from the input JAX string (for printout and rescaling)
    N0_match = re.search(r'--size (\d+)', read_params)
    depth_match = re.search(r'--depth (\d+)', read_params)
    L0_match = re.search(r'--lsize (\d+\.\d+)', read_params)
    msa0_match = re.search(r'--msa (\d+\.\d+)', read_params)

    N0 = int(N0_match.group(1))
    depth = int(depth_match.group(1))
    L0 = float(L0_match.group(1))
    msa0 = float(msa0_match.group(1))

    #for mpiexec usage on bonden
    os.system(f'export OMP_NUM_THREADS={THR}')

    if MODE == 'create':
        if VERB:
            print(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} --steps 0 --p3D 1 2>&1 | tee log-create.txt')
        else:
            print('Overview: N=%d, MPI_RANKS=%d, L=%f, msa=%f'%(N0, RANK, L0, msa0))
        if BONDEN:
            output = os.popen(f'mpirun {USA} -n {RANK} vaxion3d {JAX} --steps 0 --p3D 1 2>&1 | tee log-create.txt')
        else:
            output = os.popen(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} --steps 0 --p3D 1 2>&1 | tee log-create.txt')
        output.read()
        print('')
        print('Done!')

    elif MODE == 'run':
        if VERB:
            print(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} 2>&1 | tee log.txt')
        else:
            print('Overview: N=%d, MPI_RANKS=%d, L=%f, msa=%f'%(N0, RANK, L0, msa0))

        if BONDEN:
            output = os.popen(f'mpirun {USA} -n {RANK} vaxion3d {JAX} --p3D 1 2>&1 | tee log-run.txt')
        else:
            output = os.popen(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX}--p3D 1 2>&1 | tee log-run.txt')
        output.read()
        print('')
        print('Done!')

    elif MODE == 'con':

        #Define con options as the minimally needed ones + whatever is specified by the user in CON_OPTIONS
        extra_con_options = ''
        if '--size' not in CON_OPTIONS:
            extra_con_options += '--size %d '%N0

        if '--depth' not in CON_OPTIONS:
            extra_con_options += '--depth %d '%depth

        if '--msa' not in CON_OPTIONS:
            extra_con_options += '--msa %f '%msa0

        #add additional params from CON_OPTIONS to the minimally required ones
        extra_con_options += CON_OPTIONS

        #Create new dir for mfiles
        os.makedirs(OUT_CON, exist_ok=True)
        os.makedirs(f'{OUT_CON}/m', exist_ok=True)

        #Continue simulation in OUT_CON
        os.system(f'export AXIONS_OUTPUT="{cwd}/{OUT_CON}/m"')

        #Either use the user-specified index or use the last config file
        if IDX:
            index = IDX
        else:
            index = last_mfile()

        #create symbolic link between the config file in out and the new folder in OUT_CON
        find = f'{index:05d}'
        os.symlink(f'{cwd}/out/m/axion.{find}', f'{cwd}/{OUT_CON}/m/axion.{find}')

        if VERB:
            print(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} --index {index} {extra_con_options} 2>&1 | tee log-con.txt')
        else:
            print('Overview: N=%d, MPI_RANKS=%d, L=%f, msa=%f (data in %s)'%(N0, RANK, L0, msa0, OUT_CON))

        if BONDEN:
            output = os.popen(f'mpirun {USA} -n {RANK} vaxion3d {JAX} --index {index} {extra_con_options} 2>&1 | tee log-con.txt')
        else:
            output = os.popen(f'mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} --index {index} {extra_con_options} 2>&1 | tee log-con.txt')

        output.read()
        print('')
        print('Done!')
    print('--------------------------------------------------------------------------------------------')

def runstring(JAX, RANK=1, THR=1, USA=' --bind-to socket --mca btl_base_warn_component_unused  0', OUT_CON='out1', CON_OPTIONS ='', VERB=False, BONDEN = False):
    """
    runstring(JAX, RANK=1, THR=1, USA=' --bind-to socket --mca btl_base_warn_component_unused  0', OUT_CON='out1', CON_OPTIONS='', VERB=False)

    1 - Creates "string" initial conditions in N=256 specified by "string.dat" file with appropriately rescaled variables (JAX_INIT)
    2 - Continues and initial setup with the "correct" parameters JAX

    JAX  is a string of vaxion3d flags generated with the simgen program, see help(simgen)
    MODE specifies if the user want to 'run' (default), 'create' or 'con' a simulation
    RANK is the number of MPI processes
    THR  is the number of OMP processes
    USA  are mpirun options
    OUT_CON is the name of the out folder where the simulation is continued
    CON_OPTIONS are additional settings/flags such as --size N etc. that are needed to appropriately continue/rescale the simulations
    VERB allows for more verbosity in the output by printing the full mpirun command with all the flags. If FALSE, only the used size, depth, lsize and msa will be given
    BONDEN is used to slightly modify the mpi commands for the use of our new computer
    """
    cwd = os.getcwd()

    if 'string.dat' not in next(os.walk(cwd))[2]:
        print('No string.dat file found. Use randomstrings tools to create ICs first (N=256)!')

    else:
        #Copy initial command line flags (generated with simgen)
        JAX_INIT = JAX

        #Find value of N in JAX and use it to rescale the initial configuration
        N0_match = re.search(r'--size (\d+)', JAX_INIT)
        N0 = int(N0_match.group(1))

        #size
        JAX_INIT = JAX_INIT.replace('--size %d'%N0, '--size 256')

        #depth
        JAX_INIT = JAX_INIT.replace('--depth %d'%(N0//RANK), '--depth %d'%(256//RANK))

        #msa
        msa0_match = re.search(r'--msa (\d+\.\d+)', JAX_INIT)
        msa0 = float(msa0_match.group(1))

        JAX_INIT = JAX_INIT.replace('--msa %f'%msa0, '--msa %f'%(msa0*(N0/256)))

        #create string IC using JAX_INIT (N=256)
        runsim(JAX_INIT, MODE='create', RANK=RANK, THR=THR, USA=USA, IDX = 0, OUT_CON=OUT_CON, CON_OPTIONS='', VERB=VERB, BONDEN=BONDEN)

        print('')
        print('Succesfully created string configuration in N=256! (out)')
        #continue simulation with original parameters
        runsim(JAX, MODE='con', RANK=RANK, THR=THR, USA=USA, IDX = 0, OUT_CON=OUT_CON, CON_OPTIONS='--msa %f'%msa0, VERB=VERB, BONDEN=BONDEN)
        print('')
        print('Running ...')
        print('Finished simulation with N=%d! (%s)'%(N0, OUT_CON))
        print('--------------------------------------------------------------------------------------------')

def simgen (N=256,zRANKS=1,prec='single',dev='cpu', fftplan = 64, lowmem=False,prop='rkn4', spec=False, fspec=False, steps=1000000,wDz=1.0,sst0=10,lap=1,
            nqcd=7.0,msa=1.0,lamb=-1.0,ctf=128.,L=256.0, ind3=1.0,notheta=False,wkb=-1.,gam=0.0,dwgam=1.0,
            vqcd='vqcdC',vpq=0, mink = False, xtr='',prep=False,ic='lola',logi=0.0,cti=-1.11,
            index=-100,ict='lola',dump=10,meas=0,p3D=0,spmask=1,rmask=1.5,redmp=-1.0,wTime=-1.0,
            spKGV=15,printmask=False,ng0calib=1.25,cummask=0,
            p2Dmap=False,p2DmapE=False,p2DmapPE=False,p2DmapPE2=False, p2DmapYZ=False, slc=-1, strmeas =-1,
            nologmpi=True,verbose=1,
            verb=False,**kwargs):
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
    nologmpi
    verbose
    verb
    kwargs
    """
    ####################################################
    GRID=" --size %d --depth %d --zgrid %d"%(N,N//zRANKS,zRANKS)
    ####################################################
    SIMU=" --prec %s --device %s --prop %s --steps %d --wDz %f --sst0 %d --lap %d --fftplan %d"%(prec,dev,prop,steps,wDz,sst0,lap, fftplan)
    if lowmem:
        SIMU += ' --lowmem'
    if dev == 'gpu':
        SIMU += ' --measCPU'

    if spec:
        SIMU += ' --spec'
        if verb:
            print('Be careful: --spec overwrites --lap now!')
    if fspec:
        SIMU += ' --fspec'
        if verb:
            print('Be careful: --fspec overwrites --spec and --lap now!')
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
    PHYS=" --qcd %f %s --lsize %f --zf %f --ind3 %f"%(nqcd,tension,L,ctf,ind3)
    noth='';wkbs='';gams='';dwgams='';mnk=''
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
    PHYS += noth+wkbs+gams+dwgams+mnk+xtr
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

    if 'kickalpha' in kwargs:
        INCO += ' --kickalpha '+str(kwargs['kickalpha'])
    if 'extrav' in kwargs:
        INCO += ' --extrav '+str(kwargs['extrav'])
    if 'nncore' in kwargs:
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

#Need to add all the missing (and newly implemeted) features ..
def multisimgen (N=256,zRANKS=1,prec='single',dev='cpu', fftplan = 64, lowmem=False,prop='rkn4', spec=False, fspec=False, steps=1000000,wDz=1.0,sst0=10,lap=1,
            nqcd=7.0,msa=1.0,lamb=-1.0,ctf=128.,L=256.0, ind3=1.0,notheta=False,wkb=-1.,gam=0.0,dwgam=1.0,
            vqcd='vqcdC',vpq=0, mink = False, xtr='',prep=False,ic='lola',logi=0.0,cti=-1.11,
            index=-100,ict='lola',dump=10,meas=0,p3D=0,spmask=1,rmask=1.5,redmp=-1.0,wTime=-1.0,
            spKGV=15,printmask=False,ng0calib=1.25,cummask=0,
            p2Dmap=False,p2DmapE=False,p2DmapPE=False,p2DmapPE2=False, p2DmapYZ=False, slc=-1, strmeas=-1,
            nologmpi=True,verbose=1,
            verb=False,**kwargs):
    """
    multisimgen creates a list of strings of command line flags to select options for vaxion3d. The length of the returned list depends on the number of different configurations that the user decides to provide.

    returns list of ints (number of ranks), list of strings (flags)

    IMPORTANT: All input can be lists of the respective datatype too, e.g: N = [128, 256, 512] with all other parameters fixed to a certain value/option, results in a list of three different strings  with the respective value of N.

    options:

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
    nologmpi
    verbose
    verb
    kwargs
    """

    input = locals() #MUST be defined before creating the empty lists as they would conlict with my tests ..

    #To store the results
    ranks = []
    jaxs = []

    #Check if params with list as entry have same length + create lists of this length for single value params
    lens = []

    configs = max([len(input[x]) if isinstance(input[x], list) else 1 for x in input.keys()])

    for param in input.keys():
        if isinstance(input[param], list):
            lens.append(len(input[param]))
        else:
            input[param] = [input[param]]*configs

    #Crosscheck if all input lists have the same length
    for length in lens:
        if length != configs:
            raise ValueError("Error: All lists that are used as input for variables must be of the same length.")


    #Go through all the configs and generate the corresponding string of flags
    for config in range(configs):
        rank, jax = simgen(N=input["N"][config],zRANKS=input["zRANKS"][config],prec=input["prec"][config],dev=input["dev"][config], fftplan = input["fftplan"][config],lowmem=input["lowmem"][config],prop=input["prop"][config],spec=input["spec"][config],fspec=input["fspec"][config],
                    steps=input["steps"][config],wDz=input["wDz"][config],sst0=input["sst0"][config],lap=input["lap"][config],nqcd=input["nqcd"][config],msa=input["msa"][config],lamb=input["lamb"][config],
                    ctf=input["ctf"][config],L=input["L"][config],ind3=input["ind3"][config],notheta=input["notheta"][config],wkb=input["wkb"][config],gam=input["gam"][config],dwgam=input["dwgam"][config],
                    vqcd=input["vqcd"][config],vpq=input["vpq"][config],mink=input["mink"][config],xtr=input["xtr"][config],prep=input["prep"][config],ic=input["ic"][config],logi=input["logi"][config],cti=input["cti"][config],index=input["index"][config],
                    ict=input["ict"][config],dump=input["dump"][config],meas=input["meas"][config],p3D=input["p3D"][config],spmask=input["spmask"][config],rmask=input["rmask"][config],redmp=input["redmp"][config],
                    wTime=input["wTime"][config],spKGV=input["spKGV"][config],printmask=input["printmask"][config],ng0calib=input["ng0calib"][config],cummask=input["cummask"][config],p2Dmap=input["p2Dmap"][config],
                    p2DmapE=input["p2DmapE"][config],p2DmapPE=input["p2DmapPE"][config],p2DmapPE2=input["p2DmapPE2"][config],p2DmapYZ=input["p2DmapYZ"][config],slc=input["slc"][config],strmeas=input["strmeas"][config],nologmpi=input["nologmpi"][config],verbose=input["verbose"][config],verb=input["verb"][config],**kwargs)
        ranks.append(rank)
        jaxs.append(jax)

    return ranks, jaxs

#Only works for MODE='run', easy to use combnations of runsim in a loop!
def multirun(JAX:list, RANK:list = 1,THR:int=1,USA:str=' --bind-to socket --mca btl_base_warn_component_unused  0', STAT:int=1, NAME:str='new', VERB = False, BONDEN = False):
     """
     multirun(JAX,RANK=1,THR=1,USA=' --bind-to socket --mca btl_base_warn_component_unused  0', STAT=1,  NAME:str='new', VERB = False)
     1 - runs jaxions as
     mpirun {USA} -np {RANK} -x OMP_NUM_THREADS={THR} vaxion3d {JAX} 2>&1 | tee log.txt
     2 - repeats the simulation with the same configuration STAT times
     3 - repeats steps 1 and 2, if JAX is a list of configurations instead of a single configuration

     To avoid overwriting, the respective "out" folders are renamed dynamically after every simulation.

     JAX  is a list of strings of vaxion3d flags generated with the multisimgen program, see help(multisimgen), can be a list
     RANK is the number of MPI processes
     THR  is the number of OMP processes
     USA  are mpirun options
     STAT is the number of repitions per configuration (used to collect statistics)
     NAME is a string that is used to store the data of the simulations in a structured manner
     VERB allows for more verbosity in the output by printing the full mpirun command with all the flags. If FALSE, only the used size, depth, lsize and msa will be given
     BONDEN is used to slightly modify the mpi commands for the use of our new computer
     """

     #Four different cases need to be considered
     #single configuration (in principle the same as the "old" runsim)
     if not len(JAX) > 1 and not STAT > 1:
         print('')
         print('Simulating single configuration.')
         start = time.time()
         runsim(JAX=JAX[0], MODE='run',RANK=RANK[0],THR=THR,USA=USA,IDX = False,OUT_CON='tmp',CON_OPTIONS='',VERB=VERB, BONDEN=BONDEN)
         end = time.time()

         #Better ideas for unique renaming to avoid overwriting?
         os.system("mv out out_%s"%NAME)
         os.system("mv axion.log.0 out_%s"%NAME)
         os.system("mv log.txt out_%s"%NAME)
         print('Simulation done. Data stored in out_%s. Runtime:%s seconds'%(NAME, round(end-start,1)))

     #single configuration with STAT repetitions
     if not len(JAX) > 1 and STAT > 1:
         print('')
         print('Simulating single configuration %s times.'%STAT)
         for rep in range(STAT):
             start = time.time()
             runsim(JAX=JAX[0], MODE='run',RANK=RANK[0],THR=THR,USA=USA,IDX = False,OUT_CON='tmp',CON_OPTIONS='',VERB=VERB, BONDEN=BONDEN)
             end = time.time()

             os.system("mv out out_%s_%s"%(NAME,rep+1))
             os.system("mv axion.log.0 out_%s_%s"%(NAME,rep+1))
             os.system("mv log.txt out_%s_%s"%(NAME,rep+1))
             print('Simulation %s/%s done. Data stored in out_%s_%s. Runtime:%s seconds'%(rep+1, STAT, NAME,rep+1, round(end-start,1)))

     #multiple configurations
     if len(JAX) > 1 and not STAT > 1:
         print('')
         print('Simulating %s configurations.'%len(JAX))
         for config in range(len(JAX)):
             start = time.time()
             runsim(JAX=JAX[config], MODE='run',RANK=RANK[config],THR=THR,USA=USA,IDX = False,OUT_CON='tmp',CON_OPTIONS='',VERB=VERB, BONDEN=BONDEN)
             end = time.time()

             os.system("mv out out_%s_config%s"%(NAME,config+1))
             os.system("mv axion.log.0 out_%s_config%s"%(NAME,config+1))
             os.system("mv log.txt out_%s_config%s"%(NAME,config+1))
             print('Configuration %s/%s done. Data stored in out_%s_config%s. Runtime:%s seconds'%(config+1, len(JAX), NAME,config+1, round(end-start,1)))

     #multiple configurations with STAT repetitions each#
    if len(JAX) > 1 and STAT > 1:
         print('')
         print('Simulating %s configurations %s times each.'%(len(JAX),STAT))
         for config in range(len(JAX)):
             for rep in range(STAT):
                 start = time.time()
                 runsim(JAX=JAX[config], MODE='run',RANK=RANK[config],THR=THR,USA=USA,IDX = False,OUT_CON='tmp',CON_OPTIONS='',VERB=VERB, BONDEN=BONDEN)
                 end = time.time()

                 os.system("mv out out_%s_config%s_%s"%(NAME,config+1, rep+1))
                 os.system("mv axion.log.0 out_%s_config%s_%s"%(NAME,config+1,rep+1))
                 os.system("mv log.txt out_%s_config%s_%s"%(NAME,config+1,rep+1))

                 print('Configuration %s/%s: Simulation %s/%s done. Data stored in out_%s_config%s_%s. Runtime:%s seconds'%(config+1,len(JAX), rep+1, STAT, NAME,config+1,rep+1, round(end-start,1)))
