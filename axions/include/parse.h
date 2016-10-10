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

extern char *initFile;
extern bool lowmem;

extern FieldPrecision sPrec;
extern DeviceType     cDev;

int	parseArgs (int argc, char *argv[]);
