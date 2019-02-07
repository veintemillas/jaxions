#%%%%%%%%%%%%%%%%%%%%%%%%# GRID %
N=256 ; RANKS=4 ; DEPTH=$(echo $N/$RANKS | bc)
GRID=" --size $N --depth $DEPTH --zgrid $RANKS"
#%%%%%%%%%%%%%%%%%%%%%%%%# simulation parameters %
LOW=" --lowmem"  ; PREC=" --prec single" ; DEVI=" --device cpu"
PROP=" --prop  rkn4"   ;   SPEC=" --spec"
STEP=20000   ;   WDZ=1.0   ;   SST0=10
SIMU=" $PREC $DEVI $PROP --steps $STEP --wDz $WDZ --sst0 $SST0"
#%%%%%%%%%%%%%%%%%%%%%%%%# physical parameters %
QCD=4.0   ;   MSA=0.75   ;   L=6.0    ;   ZEN=4.0   ;   WKB=20.0
#XTR=" --gam .1 --wkb $WKB --notheta --wkb $WKB "
XTR=" --dwgam 0.1 --zswitchOn 16.0 "
PHYS="--qcd $QCD --msa $MSA --lsize $L  --zf $ZEN $XTR"
#%%%%%%%%%%%%%%%%%%%%%%%%# initial conditions %
#PCO=2.0  ;
#PREP=" --preprop --prepcoe $PCO --pregam 0.2 "
#KCR=$(echo "$L * 1.0 / $ZIN  " | bc -l)
#INCO=" --ctype kmax --kmax $N --kcr $KCR"
#INCO=" --ctype smooth --sIter 5"
INCO=" --ctype vilgor --zi 0.0 "
#%%%%%%%%%%%%%%%%%%%%%%%%# output and extra %
DUMP=10
WTIM=0.01
MEAS=$(echo 1+2+4+8+32+128+65536+16384 | bc )
#OUTP="--dump $DUMP --meas $MEAS --p2DmapE --p2DmapP --spmask 2 --rmask 4.0 --redmp 256 --p2Dmap --nologmpi --wTime $WTIM  "
OUTP="--dump $DUMP --meas $MEAS --p2DmapE --p2DmapP --spmask 2 --rmask 4.0 --p2Dmap --nologmpi --wTime $WTIM  "
echo "vaxion3d   $PHYS"
echo "         " $GRID
echo "         " $SIMU
echo "         " $INCO
echo "         " $OUTP

#export OMP_NUM_THREADS=24
export OMP_NUM_THREADS=$(echo 8/$RANKS | bc)

case "$1" in
  run)
    rm out/m/axion.*
    rm axion.log.*
    export AXIONS_OUTPUT="out/m"
    mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP 2>&1 | tee logrun.txt
    ;;
  continue)
    mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $PREP $OUTP --zf $3 --index $2 2>&1 | tee log-continue.txt
    ;;
  restart)
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    WTIM=12
    mpirun -np $RANKS vaxion3d --restart $GRID $SIMU $PHYS $PREP $OUTP --wTime $WTIM 2>&1 | tee log-continue.txt
    ;;
  wkb)
    echo wkb!
    mkdir wout ; mkdir wout/m
    rm wout/m/axion.m.*
    export AXIONS_OUTPUT="wout/m"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    cdir=$(pwd)
    find=$(printf "%05d" $2)
    ln -s $cdir/out/m/axion.$find $cdir/wout/m/axion.$find
    #echo " ln -s $cdir/out/m/axion.$find $cdir/wout/m/axion.$find"
    mpirun -np $RANKS WKVaxion $GRID $SIMU $PHYS $PREP $OUTP --zf $3 --steps $4 --index $2 2>&1 | tee logwkb.txt
    ;;
  measfile)
    vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP --dump 8 --measlistlog 2>&1 | tee logmeas.txt
    ;;
esac
