# JAXIONS MODULE TO RUN SIMULATIONS FROM PYTHON
import os
import time

def runsim(JAX,RANK=1,THR=1,USA=' --bind-to socket --mca btl_base_warn_component_unused  0'):
    """
    runsim(JAX,RANK=1,THR=1,USA=' --bind-to socket --mca btl_base_warn_component_unused  0')
    1 - cleans the out/m directory of axion.m.files
    2 - runs jaxions as
    !mpirun $USA -np $RANK -x OMP_NUM_THREADS=$THR vaxion3d $JAX > log.txt

    USA  are mpirun options
    RANK is the number of MPI processes
    THR  is the number of OMP processes
    JAX  is a string of vaxion3d flags generated with the simgen program, see help(simgen)

    """
#     !rm out/m/axion.m.*
#     !mpirun $USA -np $RANK -x OMP_NUM_THREADS=$THR vaxion3d $JAX > log.txt
    output = os.popen('rm out/m/axion.m.*')
    output.read()
    print('')
    print('')
    print('--------------------------------------------------------------------------------------------')
    print('running: ')
    print('')
    print('mpirun %s -np %s -x OMP_NUM_THREADS=%s vaxion3d %s > log.txt'%(USA,RANK,THR,JAX))
    print('')
    print('--------------------------------------------------------------------------------------------')

    output = os.popen('mpirun %s -np %s -x OMP_NUM_THREADS=%s vaxion3d %s > log.txt'%(USA,RANK,THR,JAX))
    output.read()

def simgen (N=256,zRANKS=1,prec='single',dev='cpu',lowmem=False,prop='rkn4', spec=False, fspec=False, steps=1000000,wDz=1.0,sst0=10,lap=1,
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
    SIMU=" --prec %s --device %s --prop %s --steps %d --wDz %f --sst0 %d --lap %d"%(prec,dev,prop,steps,wDz,sst0,lap)
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
    noth='';wkbs='';gams=''
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
def multisimgen (N=256,zRANKS=1,prec='single',dev='cpu',lowmem=False,prop='rkn4', spec=False, fspec=False, steps=1000000,wDz=1.0,sst0=10,lap=1,
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
        rank, jax = simgen(N=input["N"][config],zRANKS=input["zRANKS"][config],prec=input["prec"][config],dev=input["dev"][config],lowmem=input["lowmem"][config],prop=input["prop"][config],spec=input["spec"][config],fspec=input["fspec"][config],
                    steps=input["steps"][config],wDz=input["wDz"][config],sst0=input["sst0"][config],lap=input["lap"][config],nqcd=input["nqcd"][config],msa=input["msa"][config],lamb=input["lamb"][config],
                    ctf=input["ctf"][config],L=input["L"][config],ind3=input["ind3"][config],notheta=input["notheta"][config],wkb=input["wkb"][config],gam=input["gam"][config],dwgam=input["dwgam"][config],
                    vqcd=input["vqcd"][config],vpq=input["vpq"][config],mink=input["mink"][config],xtr=input["xtr"][config],prep=input["prep"][config],ic=input["ic"][config],logi=input["logi"][config],cti=input["cti"][config],index=input["index"][config],
                    ict=input["ict"][config],dump=input["dump"][config],meas=input["meas"][config],p3D=input["p3D"][config],spmask=input["spmask"][config],rmask=input["rmask"][config],redmp=input["redmp"][config],
                    wTime=input["wTime"][config],spKGV=input["spKGV"][config],printmask=input["printmask"][config],ng0calib=input["ng0calib"][config],cummask=input["cummask"][config],p2Dmap=input["p2Dmap"][config],
                    p2DmapE=input["p2DmapE"][config],p2DmapPE=input["p2DmapPE"][config],p2DmapPE2=input["p2DmapPE2"][config],p2DmapYZ=input["p2DmapYZ"][config],slc=input["slc"][config],strmeas=input["strmeas"][config],nologmpi=input["nologmpi"][config],verbose=input["verbose"][config],verb=input["verb"][config],**kwargs)
        ranks.append(rank)
        jaxs.append(jax)

    return ranks, jaxs

def multirun(JAX:list,RANK:list = 1,THR:int=1,USA:str=' --bind-to socket --mca btl_base_warn_component_unused  0', STAT:int=1, NAME:str='new'):
    """
    multirun(JAX,RANK=1,THR=1,USA=' --bind-to socket --mca btl_base_warn_component_unused  0', STAT=1,  NAME:str='new')
    1 - runs jaxions as
    !mpirun $USA -np $RANK -x OMP_NUM_THREADS=$THR vaxion3d $JAX > log.txt
    2 - repeats the simulation with the same configuration STAT times
    3 - repeats steps 1 and 2, if JAX is a list of configurations instead of a single configuration

    To avoid overwriting, the respective "out" folders are renamed dynamically after every simulation.

    USA  are mpirun options
    RANK is the number of MPI processes
    THR  is the number of OMP processes
    JAX  is a list of strings of vaxion3d flags generated with the multisimgen program, see help(multisimgen), can be a list
    STAT is the number of repitions per configuration (used to collect statistics)
    NAME is a string that is used to store the data of the simulations in a structured manner
    """
    #Four different cases need to be considered

    #single configuration (in principle the same as the "old" runsim)
    if not len(JAX) > 1 and not STAT > 1:
        print('')
        print('Simulating single configuration.')
        start = time.time()
        runsim(JAX[0],RANK[0],THR=THR,USA=USA)
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
            runsim(JAX[0],RANK[0],THR=THR,USA=USA)
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
            runsim(JAX[config],RANK[config],THR=THR,USA=USA)
            end = time.time()

            os.system("mv out out_%s_config%s"%(NAME,config+1))
            os.system("mv axion.log.0 out_%s_config%s"%(NAME,config+1))
            os.system("mv log.txt out_%s_config%s"%(NAME,config+1))
            print('Configuration %s/%s done. Data stored in out_%s_%s. Runtime:%s seconds'%(config+1, len(JAX), NAME,config+1, round(end-start,1)))

    #multiple configurations with STAT repetitions each
    if len(JAX) > 1 and STAT > 1:
        print('')
        print('Simulating %s configurations %s times each.'%(len(JAX),STAT))
        for config in range(len(JAX)):
            for rep in range(STAT):
                start = time.time()
                runsim(JAX[config],RANK[config],THR=THR,USA=USA)
                end = time.time()

                os.system("mv out out_%s_config%s_%s"%(NAME,config+1, rep+1))
                os.system("mv axion.log.0 out_%s_config%s_%s"%(NAME,config+1,rep+1))
                os.system("mv log.txt out_%s_config%s_%s"%(NAME,config+1,rep+1))
                print('Configuration %s/%s: Simulation %s/%s done. Data stored in out_%s_config%s_%s. Runtime:%s seconds'%(config+1,len(JAX), rep+1, STAT, NAME,config+1,rep+1, round(end-start,1)))
