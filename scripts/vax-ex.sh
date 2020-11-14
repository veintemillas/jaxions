#%%%%%%%%%%%%%%%%%%%%%%%%# GRID %
N=256 ; RANKS=4 ; DEPTH=$(echo $N/$RANKS | bc)
GRID=" --size $N --depth $DEPTH --zgrid $RANKS"
#%%%%%%%%%%%%%%%%%%%%%%%%# simulation parameters %
LOW=" --lowmem"  ; PREC=" --prec single" ; DEVI=" --device cpu"
PROP=" --prop  rkn4"   ;   SPEC=" --spec"
STEP=20000   ;   WDZ=1.0   ;   SST0=10  ; LAP=1
SIMU=" $PREC $DEVI $PROP --steps $STEP --wDz $WDZ --sst0 $SST0 --lap 1"
#%%%%%%%%%%%%%%%%%%%%%%%%# physical parameters %
QCD=4.0   ;   MSA=1.00   ;   L=6.0    ;   ZEN=4.0   ;   WKB=20.0
#XTR=" --llcf 20000 --ind3 0.0 --notheta  --gam .1 --dwgam 0.1  --ind3 0.0 --notheta --wkb $WKB"
#XTR=" --vPQ2 --vqcdL --vqcd0 --vqcd2"
XTR="  "
PHYS="--qcd $QCD --msa $MSA --lsize $L  --zf $ZEN $XTR"
#%%%%%%%%%%%%%%%%%%%%%%%%# initial conditions %
#PCO=2.0  ;
#PREP=" --preprop --prepcoe 4.0 --icstudy --lz2e 8.0 --prevqcdtype 17409 --pregam 0.2 "
#KCR=$(echo "$L * 1.0 / $ZIN  " | bc -l)
#INCO=" --ctype kmax --zi 0.1 --kmax $N --kcr $KCR"
INCO=" --ctype smooth --kcr 1.1 --sIter 5 "
#INCO=" --ctype smooth --smvar axnoise  --mode0 2 --kcr 1.1 --sIter 5 --notheta "
#INCO=" --ctype smooth --smvar parres  --mode0 0.0 --kmax 0 --kcr 1.1 --sIter 0 --notheta --nncore "
#INCO=" --ctype smooth --zi 0.1 --sIter 5"
#INCO=" --ctype lola --logi 0.0 --sIter 0 --kcr 1.0"
#INCO=" --ctype spax --zi 4.0 --sIter 0 "
#%%%%%%%%%%%%%%%%%%%%%%%%# output and extra %
DUMP=10
WTIM=1.0
MEAS=$(echo 1+2+4+8+32+128+65536+16384 | bc )
SPMA=$(echo 1+64 | bc )
#OUTP="--dump $DUMP --meas $MEAS --p3D 2 --p2DmapE --p2DmapPE --p2DmapPE2 --spmask 2 --rmask 4.0/file --redmp 256 --p2Dmap --nologmpi --wTime $WTIM  "
OUTP="--dump $DUMP --meas $MEAS --p2DmapE --p2DmapP --spmask $SPMA --rmask 4.0 --p2Dmap --nologmpi --wTime $WTIM --verbose 1 "
echo "vaxion3d   $PHYS"
echo "         " $GRID
echo "         " $SIMU
echo "         " $INCO
echo "         " $PREP
echo "         " $OUTP

#export OMP_NUM_THREADS=24
export OMP_NUM_THREADS=$(echo 8/$RANKS | bc)
#USA=" --bind-to socket"

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
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP $2 2>&1 | tee out/logrun.txt
    ;;
  continue)
    echo "continue Run with index $2 in out/m"
    echo mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $OUTP --index $2 $3
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $OUTP --index $2 $3 2>&1 | tee out/log-continue.txt
    ;;
  con)
    echo "continue Run with index $2 in out/m with extra options $3 in directory $4!"
    mkdir $4 ; mkdir $4/m
    rm $4/m/axion.m.*
    export AXIONS_OUTPUT="$4/m"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    cdir=$(pwd)
    find=$(printf "%05d" $2)
    ln -s $cdir/out/m/axion.$find $cdir/$4/m/axion.$find
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $PREP $OUTP --index $2  $3    2>&1 | tee log-con.txt
    ;;
  restart)
    echo "restart Run $AXIONS_OUTPUT/axion.restart"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    WTIM=12
    echo mpirun -np $RANKS vaxion3d --restart $GRID $SIMU $PHYS $OUTP --wTime $WTIM
    mpirun $USA -np $RANKS vaxion3d --restart $GRID $SIMU $PHYS $OUTP --wTime $WTIM 2>&1 | tee out/log-restart.txt
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
    echo pax with extra options $4 !
    mkdir pout ; mkdir pout/m
    rm pout/m/axion.*
    export AXIONS_OUTPUT="pout/m"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    cdir=$(pwd)
    find=$(printf "%05d" $2)
    ln -s $cdir/wout/m/axion.$find $cdir/pout/m/axion.$find
    mpirun $USA -np $RANKS paxion3d $GRID $SIMU $PHYS $PREP $OUTP --zf $3 --index $2   $4    2>&1 | tee log-pax.txt
    ;;
  measfile)
    vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP --dump 8 --measlistlog 2>&1 | tee log-meas.txt
    ;;

esac
