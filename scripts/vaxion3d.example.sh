mkdir out/ >/dev/null
mkdir out/m >/dev/null

#%%%%%%%
# grid %
#%%%%%%%
N=512
RANKS=1
DEPTH=$(echo $N/$RANKS | bc)

GRID=" --size $N --depth $N --zgrid 1"
#echo $GRID

#%%%%%%%%%%%%%%%%%%%%%%%%
# simulation parameters %
#%%%%%%%%%%%%%%%%%%%%%%%%
LOW=" --lowmem"
PREC=" --prec single"
DEVI=" --device cpu"
PROP=" --prop  rkn4"       # leap/rkn4/om2/om4
SPEC=" --spec"
STEP=10000                 # maximum number
 WDZ=1.0
SST0=10

SIMU="$LOW $PREC $EVI $PROP --steps $STEP --wDz $WDZ --sst0 $SST0"
#echo $SIMU

#%%%%%%%%%%%%%%%%%%%%%%
# physical parameters %
#%%%%%%%%%%%%%%%%%%%%%%
QCD=4.0
MSA=1.5
  L=6.0
ZIN=0.1
ZEN=5.0
WKB=6.0
XTR=" --vPQ2 "
                 # [--vqcd2   ]
                 # [-—gam 2.0 ]
PHYS=" --qcd $QCD --msa $MSA --lsize $L --zi $ZIN --zf $ZEN --wkb $WKB $XTR"
#echo $PHYS

#%%%%%%%%%%%%%%%%%%%%%
# initial conditions %
#%%%%%%%%%%%%%%%%%%%%%

KCR=$(echo "$L * 9.0" | bc -l)

#INCO=" --ctype smooth --sIter 10 "
INCO=" --ctype kmax --kmax $N --kcr $KCR"

#%%%%%%%%%%%%%%%%%%%
# output and extra %
#%%%%%%%%%%%%%%%%%%%
# for all strings to be printed --p3Dstr 1.e+30 [default is 100000]

DUMP=10

OUTP="--dump $DUMP --p2Dmap --p3D 0 --p3Dstr 0 --verbose 2"
#echo $OUTP

# XTR="—gam 0.0"”


#echo N=$N, Ranks=$RANKS, D=$DEPTH, Cores/TASK=$OMP_NUM_THREADS
#echo L=$L    , msa  =$MSA
#echo z="$ZIN-$ZEN($WKB)"
echo
echo
echo "raxion3d $PHYS"
echo "         " $GRID
echo "         " $SIMU
echo "         " $INCO
echo "         " $OUTP

#echo "raxion3d $GRID $SIMU $PHYS $INCO $OUTP"

raxion3d $GRID $SIMU $PHYS $INCO $OUTP 2>&1 | tee luera.txt
