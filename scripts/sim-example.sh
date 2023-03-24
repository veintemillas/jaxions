N=2048                         # Resolution
RANKS=24                       # Number of MPI ranks. Total cpu cores/ openMP threads
OMP=12                         # Number of openMP threads

PREC=single                    # Field precisions: single/double 
DEVI=cpu                       # Device for propagation: cpu/gpu (gpu support only for NVIDIA or CUDA enabled chips)
PROP=rkn4                      # Propagator: rkn4,leap,mleap,...
LAP=1                          # Number of neighboirs for laplacian finite difference calculations
STEP=20000                     # Number of propagator steps
WDZ=1.0                        # Ratio between time-step and maximum energy to resolve  
SST0=0                         # Number of time steps before switching ot axion-only simulations
#NOTH='--notheta'              # Do not switch to axion-only

QCD=6.68                       # Topological susceptiblity power-law coefficient 
MSA=1.25                       # Inverse string-core width in grid units
L=8.0                          # Physical length of the box in ADM units
ZEN=4.5                        # Final simulation time
WKB=20.0                       # Time to apply WKB approximation
RC=16.0                        # Time when axion mass reaches zero-T value

IC='lola'                      # Initial conditions type: lola,smooth,spax,tkachev,kmax. Use --icinfo for more details    
LOGI=0.0                       # Initial value of the log for lola IC type
ZI=0.1                         # Initial simulation time 
KCR=1.0                        # Critical momentum for k-space IC types 
KMAX=0.0                       # Maximum momentum for k-space IC types
SMV=axnoise                    # Smoothing variation, if IC=smooth is selected. axnoise,parres,mc,stringwave,...
ITER=0                         # Iteration steps if smoothing is selected 

GRA=5e-10                      # Gravity coefficient in SP system (~R1/Req)
BETA=0.0                       # Self interaction coefficient in SP system (GPP)
#SAT='--sat_gravity'           # Saturate gravitational evolution when phase field is unresolved 
#HYB='--hybrid_gravity'        # Use the hybrid tuning for gravity propagation

L1=0.025                       # ADM length in parsec, related to axion mass
GTY=void                       # Gadget configuration particle mapping, options halo/void
SIG=0.1                        # For the halo mapping, variance of random displacement 
PVEL='--part_vel'              # For the void mapping add velocities
#SVEL='--sm_vel'               # For the void mapping smooth velocities, requires twice the memory 
#PDI='--part_disp'             # For the void mapping give a random diplacement

AXNUM=1000                     # Maximum number of axitons/halos to track 
CONTHR=200                     # Density contrast threshold for halo finding
CTTHR=10000                    # Time to start seaching for halos 

STR=0                          # Save string+wall data          0:NO, 1:YES
PSP=1                          # Save axion power spectrum      0:NO, 1:YES
NSP=0                          # Save axion number spectrum     0:NO, 1:YES
ENE=0                          # Save axion+saxion energies     0:NO, 1:YES
DUMP=100                       # Dumps measurement file every $DUMP steps
KGV=7                          # Number spectrum type 1:KIN, 2:GRAD, 4:POT, 7:K+G+V
SAVE=32                        # Dump configuration 0:Initial, ... , 32:Saturationyy

WTIM=10.0                      # Set wall time (hours) to save the configuration 
#MAP='--p2Dmap'                # Save 2D slice plot of the field 
MAPP='--p2DmapPE'              # Save 3D->2D projection plot of the energy field 
LO='--nologmpi'                # Only comms from main MPI rank
FFT=64                         # Plan for FFTW 
VERB=1                         # Verbosity level in the logger 

#############################################################################################################################
export OMP_NUM_THREADS=$OMP

DEPTH=$(echo $N/$RANKS | bc)
GRID=" --size $N --depth $DEPTH --zgrid $RANKS"
SIMU=" --prec $PREC --device $DEVI --prop $PROP --steps $STEP --wDz $WDZ --sst0 $SST0 --lap $LAP"
PHYS="--qcd $QCD --msa $MSA --lsize $L  --zf $ZEN $XTR --Rc $RC"
if [[ $LOGI ]]; then INCO=" --ctype $IC --logi $LOGI --sIter $ITER --kcr $KCR"
else                 INCO=" --ctype $IC --zi $ZI --sIter $ITER --kcr $KCR --smvar $SMV --mode0 $MOD --kmax $KMAX $NOTH";fi
PAX="--gravity $GRA --beta $BETA $SAT $HYBR" 
GADG="--gadtype $GTY --L1_pc $L1 $PVEL $SVEL $PDI --kcr $SIG"
TRACK="--axitontracker $AXNUM --axitontracker.con_threshold $CONTHR --axitontracker.ct_threshold $CTTHR "
# Add here the rest 
if [[ $STR == 1 ]] ; then STR=32; fi; if [[ $ENE == 1 ]] ; then ENE=256;   fi
if [[ $PSP == 1 ]] ; then PSP=16384; fi; if [[ $NSP == 1 ]] ; then STR=65536; NSPT="--spKGV $KGV"; fi
MEAS=$(echo $STR+$ENE+$PSP+$NSP | bc )
OUTP="--dump $DUMP --meas $MEAS $NSPT $MAP $MAPP $LO --p3D $SAVE --fftplan $FFT --verbose $VERB --wTime"

case "$1" in
  create)
    echo "Create run in out/m (or default) and save IC"
    rm out/m/axion.*
    rm axion.log.*
    export AXIONS_OUTPUT="out/m"
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP --steps 0 --p3D 1 2>&1 | tee log-create.txt
    ;;

  run)
    echo "Run"
    rm out/m/axion.*; rm axion.log.*
    export AXIONS_OUTPUT="out/m"
    echo Options: $GRID $SIMU $PHYS $INCO $PREP $OUTP 
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP  2>&1 | tee log-run.txt
    ;;

  continue)
    echo "continue Run with index $2 in out/m"
    echo mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $OUTP --index $2 $3
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $OUTP --index $2 $3 2>&1 | tee log-continue.txt
    ;;
  
  con)
    echo "continue Run with index $2 in out/m until time $3 in directory $4 with extra options $5!"
    mkdir $4 ; mkdir $4/m
    rm $4/m/axion.m.*
    export AXIONS_OUTPUT="$4/m"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    cdir=$(pwd)
    find=$(printf "%05d" $2)
    ln -s $cdir/out/m/axion.$find $cdir/$4/m/axion.$find
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $PREP $OUTP --dump 200 --beta 0.0 --index $2  --zf $3 $5   2>&1 | tee log-con.txt
    ;;

  restart)
    echo "restart Run $AXIONS_OUTPUT/axion.restart"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    WTIM=12
    echo mpirun -np $RANKS vaxion3d --restart $GRID $SIMU $PHYS $OUTP --wTime $WTIM
    mpirun $USA -np $RANKS vaxion3d --restart $GRID $SIMU $PHYS $OUTP --wTime $WTIM 2>&1 | tee log-restart.txt
    ;;

  redu)
    echo "redo file with index $2 to n = $3"
    echo mpirun -np $RANKS redu $GRID $SIMU $PHYS $OUTP --index $2 --redmp $3
    mpirun $USA -np $RANKS redu $GRID $SIMU $PHYS $OUTP --index $2 --redmp $3 2>&1 | tee log-redu.txt
    ;;

  wkb)
    echo "WKB the configuration $AXIONS_OUTPUT/axion.$2 until time --zf $3 in logarithmic --steps $4 "
    mkdir wout ; mkdir wout/m
    rm wout/m/axion.m.*
    export AXIONS_OUTPUT="wout/m"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    cdir=$(pwd)
    find=$(printf "%05d" $2)
    ln -s $cdir/out/m/axion.$find $cdir/wout/m/axion.$find
    mpirun $USA -np $RANKS WKVaxion $GRID $SIMU $PHYS $PREP $OUTP --ftype axion --zf $3 --steps $4 --index $2 2>&1 | tee log-wkb.txt
    ;;

  pax)
    echo "Paxion evolution with settings $PAX"
    mkdir pout ; mkdir pout/m
    rm pout/m/axion.*
    export AXIONS_OUTPUT="pout/m"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    cdir=$(pwd)
    find=$(printf "%05d" $2)
    #ln -s $cdir/out/m/axion.$find $cdir/pout/m/axion.$find
    ln -s $cdir/wout/m/axion.$find $cdir/pout/m/axion.$find #for WKB 
    mpirun $USA -np $RANKS paxion3d $GRID $SIMU $PHYS $PREP $OUTP $PAX $TRACK --ftype axion--index $2 --zf $3 $4 2>&1 | tee log-pax.txt
    ;;

  gad)
    echo "gadget file $2 with $3^3 particles with settings $GADG"
    mkdir gadout ;
    rm gadout/axion.*
    export AXIONS_OUTPUT="gadout"
    echo "AXIONS_OUTPUT"=$AXIONS_OUTPUT
    cdir=$(pwd)
    find=$(printf "%05d" $2)
    ln -s $cdir/pout/m/axion.$find $cdir/gadout/axion.$find
    mpirun $USA -np $RANKS gagdetme $GADG --index $2 --size $3 --zgrid $RANKS | tee log-gad.txt 
    ;;

  meas)
    vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP --dump 8 --measlistlog 2>&1 | tee log-meas.txt
    ;;

esac
