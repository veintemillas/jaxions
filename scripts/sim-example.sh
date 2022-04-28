N=3072 ; RANKS=24 ; DEPTH=$(echo $N/$RANKS | bc)
GRID=" --size $N --depth $DEPTH --zgrid $RANKS"

LOW=" --lowmem"  ; PREC=" --prec single" ; DEVI=" --device cpu"
PROP=" --prop  rkn4"   ;
STEP=20000   ;   WDZ=1.0   ;   SST0=0  ; LAP=1
SIMU=" $PREC $DEVI $PROP --steps $STEP --wDz $WDZ --sst0 $SST0 --lap $LAP"

QCD=4.0   ;   MSA=1.50   ;   L=18.0    ;   ZEN=5.5   ;   WKB=20.0
PHYS="--qcd $QCD --msa $MSA --lsize $L  --zf $ZEN $XTR"

INCO=" --ctype lola --logi 0.0 --sIter 0 --kcr 1.0"
#INCO=" --ctype spax --zi 4.0 --sIter 0 "

GRA=5e-10;   BETA= 0.0;   GTY=gadgrid;   L1=0.0362;   SAVE="--p3D 32"
PAX="--gravity $GRA --beta $BETA --sat_gravity $SAVE" #--hybrid_gravity  
GADG="--gadtype $GTY --L1_pc $L1 --part_vel" #--smooth_vel --kcr 0.25

DUMP=100
WTIM=1.0
MEAS=$(echo 8+16384 | bc )
SPMA=$(echo 1 | bc )
SKGV=$(echo 1 | bc )

OUTP="--dump $DUMP --meas $MEAS --p2Dmap --p2DmapPE --p3D 2 --nologmpi --verbose 1 "

echo "vaxion3d   $PHYS"
echo "         " $GRID
echo "         " $SIMU
echo "         " $INCO
echo "         " $PREP
echo "         " $OUTP

export OMP_NUM_THREADS=10

case "$1" in
  create)
    echo "Create run in out/m (or default) and save IC"
    rm out/m/axion.*
    rm axion.log.*
    export AXIONS_OUTPUT="out/m"
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP --steps 0 --p3D 1 2>&1 | tee out/log-create.txt
    ;;
  run)
    echo "Run"
    rm out/m/axion.*
    rm axion.log.*
    export AXIONS_OUTPUT="out/m"
    echo mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP $2
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP $2 2>&1 | tee log-run.txt
    ;;
  continue)
    echo "continue Run with index $2 in out/m"
    echo mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $OUTP --index $2 $3
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $OUTP --index $2 $3 2>&1 | tee out/log-continue.txt
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
    mpirun $USA -np $RANKS vaxion3d --restart $GRID $SIMU $PHYS $OUTP --wTime $WTIM 2>&1 | tee out/log-restart.txt
    ;;
  redu)
    echo "redo file with index $2 to n = $3"
    echo mpirun -np $RANKS redu $GRID $SIMU $PHYS $OUTP --index $2 --redmp $3
    mpirun $USA -np $RANKS redu $GRID $SIMU $PHYS $OUTP --index $2 --redmp $3
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
    #echo " ln -s $cdir/out/m/axion.$find $cdir/wout/m/axion.$find"
    mpirun $USA -np $RANKS WKVaxion $GRID $SIMU $PHYS $PREP $OUTP --zf $3 --steps $4 --index $2 2>&1 | tee log-wkb.txt
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
    mpirun $USA -np $RANKS paxion3d $GRID $SIMU $PHYS $PREP $OUTP $PAX --index $2 --zf $3 $4 2>&1 | tee log-pax.txt
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
    mpirun $USA -np $RANKS gagdetme $GADG --index $2 --size $3 --zgrid $RANKS --nologmpi | tee log-gad.txt 
    ;;
  measfile)
    vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP --dump 8 --measlistlog 2>&1 | tee log-meas.txt
    ;;

esac