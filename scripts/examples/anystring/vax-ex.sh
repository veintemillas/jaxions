#%%%%%%%%%%%%%%%%%%%%%%%%# GRID %
N=128 ; RANKS=4 ; DEPTH=$(echo $N/$RANKS | bc)
GRID=" --size $N --depth $DEPTH --zgrid $RANKS"
#%%%%%%%%%%%%%%%%%%%%%%%%# simulation parameters %
LOW=" --lowmem"  ; PREC=" --prec single" ; DEVI=" --device cpu"
PROP=" --prop  mleap"   ;   SPEC=" --spec"
STEP=1000   ;   WDZ=1.0   ;   SST0=10  ; LAP=4
SIMU=" $PREC $DEVI $PROP --steps $STEP --wDz $WDZ --sst0 $SST0 --lap $LAP"
#%%%%%%%%%%%%%%%%%%%%%%%%# physical parameters %
QCD=4.0   ;   MSA=2.0   ;   L=6.0    ;   ZEN=4.0   ;   WKB=20.0
#XTR=" --llcf 20000 --ind3 0.0 --notheta  --gam .1 --dwgam 0.1  --ind3 0.0 --notheta --wkb $WKB"
#XTR=" --vPQ2 --vqcdL --vqcd0 --vqcd2"
XTR=" --vqcd0 --mink "
PHYS="--qcd $QCD --msa $MSA --lsize $L  --zf $ZEN $XTR"
#%%%%%%%%%%%%%%%%%%%%%%%%# initial conditions %
#PCO=2.0  ;
#PREP=" --preprop --prepcoe 4.0 --icstudy --lz2e 8.0 --prevqcdtype 17409 --pregam 0.2 "
#KCR=$(echo "$L * 1.0 / $ZIN  " | bc -l)
#INCO=" --ctype kmax --zi 0.1 --kmax $N --kcr $KCR"
#INCO=" --ctype smooth --kcr 1.1 --sIter 5 "
#INCO=" --ctype smooth --smvar stXY --mode0 0.0 --kcr 0.5 --sIter 0 --notheta "
INCO=" --ctype string --sIter 10 --notheta --zi 0.001 "
#INCO=" --ctype smooth --zi 0.1 --sIter 5"
#INCO=" --ctype lola --logi 0.0 --sIter 0 --kcr 1.0"
#INCO=" --ctype spax --zi 4.0 --sIter 0 "
#%%%%%%%%%%%%%%%%%%%%%%%%# output and extra %
DUMP=5
WTIM=1.0
MEAS=$(echo 32+64 | bc )
SPMA=$(echo 1 | bc )
SKGV=$(echo 1 | bc )
#OUTP="--dump $DUMP --meas $MEAS --p3D 2 --p2DmapE --p2DmapPE --p2DmapPE2 --spmask 2 --rmask 4.0/file --redmp 256 --p2Dmap --nologmpi --wTime $WTIM  "
OUTP="--dump $DUMP --p3D 1 --meas $MEAS --strmeas 3 --p2DmapP --spmask $SPMA --spKGV $SKGV --rmask 4.0 --p2Dmap --sliceprint $(echo $N/2 | bc) --p2DmapYZ --nologmpi --wTime $WTIM --verbose 3 "
echo "vaxion3d   $PHYS"
echo "         " $GRID
echo "         " $SIMU
echo "         " $INCO
echo "         " $PREP
echo "         " $OUTP

export OMP_DYNAMIC=FALSE
export OMP_NUM_THREADS=1
#export OMP_NUM_THREADS=$(echo 8/$RANKS | bc)
#USA=" --bind-to socket --mca btl_base_warn_component_unused  0 "

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
    OMP_NUM_THREADS=1 mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $INCO $PREP $OUTP $2 2>&1 | tee out/logrun.txt
    ;;
  continue)
    echo "continue Run with index $2 in out/m"
    echo mpirun -np $RANKS vaxion3d $GRID $SIMU $PHYS $OUTP --index $2 $3
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $OUTP --index $2 $3 2>&1 | tee out/log-continue.txt
    ;;
  con)
    echo "continue Run with index $2 of folder $3  with extra options $4 in directory $5!"
    if [ "$3" == "$5"]; then 
	echo "melodrama $3 $5" 
    else
	echo "creating $5 ... "
	mkdir $5 ; mkdir $5/m
        rm $5/m/axion.m.*
    fi
    export AXIONS_OUTPUT="$5/m"
    echo "AXIONS_OUTPUT=$AXIONS_OUTPUT"
    cdir=$(pwd)
    find=$(printf "%05d" $2)
    ln -s $cdir/$3/m/axion.$find $cdir/$5/m/axion.$find
    mpirun $USA -np $RANKS vaxion3d $GRID $SIMU $PHYS $PREP $OUTP --index $2  $4    2>&1 | tee log-con.txt
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
