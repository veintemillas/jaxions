extern int sizeN;
extern int sizeZ;
extern int zGrid;
extern int nSteps;
extern int dump;
extern int nQcd;
extern int fIndex;

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


extern char *initFile;
extern bool lowmem;

extern FieldPrecision sPrec;
extern DeviceType     cDev;
extern ConfType	      cType;

int	parseArgs (int argc, char *argv[]);
