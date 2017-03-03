extern uint sizeN;
extern uint sizeZ;
extern uint zGrid;
extern int  nSteps;
extern int  dump;
extern int  nQcd;
extern int  fIndex;

extern double sizeL;
extern double zInit;
extern double zFinl;
extern double LL;
extern double alpha;
extern double kCrit;
extern double parm2;

extern int kMax;
extern int iter;
extern int parm1;
extern int Ng;
extern double indi3;
extern double zthres;
extern double zrestore;


extern char *initFile;
extern char outName[128];
extern bool lowmem;
extern bool uPrec;

extern FieldPrecision sPrec;
extern DeviceType     cDev;
extern ConfType	      cType;
extern FieldType      fType;

int	parseArgs (int argc, char *argv[]);
