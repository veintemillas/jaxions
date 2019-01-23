#%%%%%%%%%%%%%%%%%%%%%%%%# GRID %
N=256 ; RANKS=4 ; DEPTH=$(echo $N/$RANKS | bc)
GRID=" --size $N --depth $DEPTH --zgrid $RANKS"
#%%%%%%%%%%%%%%%%%%%%%%%%# simulation parameters %
LOW=" --lowmem"  ; PREC=" --prec single" ; DEVI=" --device cpu"
PROP=" --prop  rkn4"   ;   SPEC=" --spec"
STEP=20000   ;   WDZ=1.0   ;   SST0=10
SIMU=" $PREC $DEVI $PROP --steps $STEP --wDz $WDZ --sst0 $SST0"
#%%%%%%%%%%%%%%%%%%%%%%%%# physical parameters %
QCD=4.0   ;   MSA=1.0   ;   L=6.0    ;   ZEN=4.0   ;   WKB=20.0
#XTR=" --gam .1 --wkb $WKB --notheta "
XTR=" --dwgam 0.1 --zswitchOn 16.0 --wkb $WKB "
PHYS="--qcd $QCD --msa $MSA --lsize $L  --zf $ZEN $XTR"
#%%%%%%%%%%%%%%%%%%%%%%%%# initial conditions %
#PCO=2.0  ;
#PREP=" --preprop --prepcoe $PCO --pregam 0.2 "
#KCR=$(echo "$L * 1.0 / $ZIN  " | bc -l)
#INCO=" --ctype kmax --kmax $N --kcr $KCR"
#INCO=" --ctype smooth --sIter 5"
INCO=" --ctype vilgor --zi 0.0 "
#%%%%%%%%%%%%%%%%%%%%%%%%# output and extra %
# for all strings to be printed --p3Dstr 1.e+30 [default is 100000]
DUMP=10
WTIM=12
#MEAS=$(echo 1+2+3+32+64+256+2048+16384+65536)
MEAS=$(echo 1+2+4+8+32+128+65536+16384 | bc )
OUTP="--dump $DUMP --meas $MEAS --p2DmapE --p2DmapP --spmask 2 --rmask 4.0 --redmp 256 --p2Dmap --nologmpi --wTime $WTIM  "

echo "vaxion3d   $PHYS"
echo "         " $GRID
echo "         " $SIMU
echo "         " $INCO
echo "         " $OUTP

#export OMP_NUM_THREADS=24
export OMP_NUM_THREADS=$(echo 8/$RANKS | bc)

case "$1" in
  measfile)
    vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP --dump 8 --measlistlog 2>&1 | tee logmeas.txt
    ;;

  wkb)
    echo wkb!
    rm wout/m/axion.m.*
    export AXIONS_OUTPUT="wout/m"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    cdir=$(pwd)
    find=$(printf "%05d" $2)
    ln -s $cdir/out/m/axion.$find $cdir/wout/m/axion.$find
    #echo " ln -s $cdir/out/m/axion.$find $cdir/wout/m/axion.$find"
    mpirun -np $RANKS WKVaxion $GRID $SIMU $PHYS $PREP $OUTP --zf $3 --steps $4 --index $2 2>&1 | tee logwkb.txt
    ;;
  run)
    rm out/m/axion.*
    rm axion.log.*
    export AXIONS_OUTPUT="out/m"
    mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP 2>&1 | tee logrun.txt
    ;;
  continue)
    echo continue!
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $PREP $OUTP --zf $3 --index $2 2>&1 | tee log-continue.txt
    ;;
esac
