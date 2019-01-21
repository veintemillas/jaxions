#%%%%%%%%%%%%%%%%%%%%%%%%# GRID %
N=256 ; RANKS=1 ; DEPTH=$(echo $N/$RANKS | bc)
GRID=" --size $N --depth $DEPTH --zgrid $RANKS"
#%%%%%%%%%%%%%%%%%%%%%%%%# simulation parameters %
LOW=" --lowmem"  ; PREC=" --prec single" ; DEVI=" --device cpu"
PROP=" --prop  rkn4"   ;   SPEC=" --spec"
STEP=20000   ;   WDZ=1.0   ;   SST0=10
SIMU=" $PREC $DEVI $PROP --steps $STEP --wDz $WDZ --sst0 $SST0"
#%%%%%%%%%%%%%%%%%%%%%%%%# physical parameters %
QCD=6.0   ;   MSA=1.0   ;   L=6.0    ;   ZEN=2.5   ;   WKB=6.0
#XTR=" --gam .1 --wkb $WKB"
XTR=" --dwgam 0.5 "
#XTR=" --wkb $WKB --notheta "

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
OUTP="--dump $DUMP --meas $MEAS --p2DmapE --spmask 10 --rmask 2.0 --redmp 128 --p2Dmap --nologmpi --wTime $WTIM  "
echo
echo "vaxion3d   $PHYS"
echo "         " $GRID
echo "         " $SIMU
echo "         " $INCO
echo "         " $OUTP

export OMP_NUM_THREADS=$(echo 8/$RANKS | bc)


if [ $1 == "measfile" ]; then
        vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP --dump 8 --measlistlog 2>&1 | tee logmeas.txt
        exit 1
fi
if [ "$1" == "wkb" ]; then
        echo wkb!
        rm wout/m/axion.m.*
        export AXIONS_OUTPUT="wout/m"
        echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
        mpirun -np $RANKS WKVaxion $GRID $SIMU $PHYS $PREP $OUTP --zf $3 --steps 10 --index $2 2>&1 | tee logwkb.txt
fi
if [ "$1" == "run" ]; then
        rm out/m/axion.*
        rm axion.log.*
        export AXIONS_OUTPUT="out/m"
        mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP 2>&1 | tee logrun.txt
fi
if [ "$1" == "continue" ]; then
        echo continue!
        echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
        mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $PREP $OUTP --zf $3 --index $2 2>&1 | tee log-continue.txt
fi
