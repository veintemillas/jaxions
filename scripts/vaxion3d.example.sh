#%%%%%%%%%%%%%%%%%%%%%%%%# grid %
N=256   ;   RANKS=1   ;   DEPTH=$(echo $N/$RANKS | bc)
GRID=" --size $N --depth $DEPTH --zgrid $RANKS"
#%%%%%%%%%%%%%%%%%%%%%%%%# simulation parameters %
LOW=" --lowmem"  ; PREC=" --prec single" ; DEVI=" --device cpu"
PROP=" --prop  rkn4"   ;   SPEC=" --spec"
STEP=20000   ;   WDZ=1.5   ;   SST0=10
SIMU=" $PREC $DEVI $PROP --steps $STEP --wDz $WDZ --sst0 $SST0"
#%%%%%%%%%%%%%%%%%%%%%%%%# physical parameters %
QCD=3.0   ;   MSA=1.0   ;   L=2.00   ;   ZIN=0.2  ;   ZEN=3.00   ;   WKB=8.0
#XTR=" --dwgam 0.05 --gam .1 --wkb $WKB --notheta --vPQ2 --vQCD2"
XTR=" --dwgam 0.05"
XTR=" "
PHYS=" --qcd $QCD --msa $MSA --lsize $L --zi $ZIN --zf $ZEN $XTR"
#%%%%%%%%%%%%%%%%%%%%%%%%# initial conditions %
PCO=1.5  ;
PREP=" --preprop --prepcoe $PCO --pregam 0.02 "
KCR=$(echo "$L * 0.9 / $ZIN  " | bc -l)
#KCR=$(echo "$L * $PCO / $ZIN  " | bc -l)
INCO=" --ctype kmax --kmax $N --kcr $KCR"
#Dependence on msa .
#INCO=" --ctype smooth --sIter 300"
#%%%%%%%%%%%%%%%%%%%%%%%%# output and extra %
# for all strings to be printed --p3Dstr 1.e+30 [default is 100000]
DUMP=10
WTIM=12
# OUTP=" --icstudy --dump $DUMP --p2Dmap --p2DmapP  --p3D 0 --redmp 128 --nologmpi --wTime $WTIM"
OUTP=" --dump $DUMP --p3D 2 --nologmpi --wTime $WTIM"
echo
echo "vaxion3d $PHYS"

echo "         " $GRID
echo "         " $SIMU
echo "         " $INCO
echo "         " $OUTP

export OMP_NUM_THREADS=12
mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP 2>&1 | tee luera.txt
