#include "enum-field.h"

extern uint sizeN;
extern uint sizeZ;
extern uint zGrid;
extern int  nSteps;
extern int  dump;
extern int  fIndex;

extern double nQcd;
extern double sizeL;
extern double zInit;
extern double zFinl;
extern double LL;
extern double mode0;
extern double alpha;
extern double kCrit;
extern double parm2;
extern double gammo;

extern int kMax;
extern int iter;
extern int parm1;
extern int Ng;
extern double indi3;
extern double msa;
extern double wDz;
extern double zthres;
extern double zrestore;

extern double wkb2z ;
extern int endredmap ;
extern int safest0 ;

extern char *initFile;
extern char outName[128];
extern bool lowmem;
extern bool uPrec;
extern bool uQcd;
extern bool uLambda;
extern bool spectral;

extern FieldPrecision sPrec;
extern DeviceType     cDev;
extern ConfType	      cType;
extern ConfsubType    smvarType;
extern FieldType      fTypeP;
extern LambdaType     lType;
extern VqcdType		    vqcdType;
extern PropType       pType;

extern LogMpi	      logMpi;
extern VerbosityLevel verb;

//for output
extern PrintConf      prinoconfo;
extern bool           p2dmapo;
extern bool           pconfinal;
extern bool           pconfinalwkb ;

int	parseArgs (int argc, char *argv[]);
