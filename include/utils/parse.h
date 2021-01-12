#include "enum-field.h"
#include "cosmos/cosmos.h"
#include<vector>

//using namespace std;

extern size_t sizeN;
extern size_t sizeZ;
extern int    zGrid;
extern int    nSteps;
extern int    dump;
extern int    fIndex;
extern int    fIndex2;

//extern double nQcd;
//extern double sizeL;
extern double zInit;
extern double zFinl;
//extern double LL;
extern double frw;
extern double mode0;
extern double alpha;
extern double kCrit;
extern double parm2;
extern double pregammo;

extern double ng0calib;
extern double dwgammo;
extern double p3DthresholdMB;
extern size_t kMax;
extern size_t iter;
extern size_t parm1;
extern size_t wTime;
extern int    Nng;
//extern double indi3;
//extern double msa;
extern double wDz;
//extern double zthres;
//extern double zrestore;
extern size_t nstrings_globale;

extern std::vector<double> rmask_tab;
extern int i_rmask;

extern int    slicepp;

extern double wkb2z ;
extern double prepstL;
extern double prepcoe;
extern int endredmap ;
extern int endredmapwkb ;
extern int safest0 ;

extern char *initFile;
extern char outName[128];
extern char outDir[1024];
extern char wisDir[1024];
extern bool uwDz;
extern bool lowmem;
extern bool uPrec;
//extern bool uQcd;
//extern bool uMsa;
//extern bool uLambda;
//extern bool uGamma;
//extern bool uPot;
extern bool uZin;
extern bool uZfn;
extern bool aMod;
extern bool spectral;
extern bool fpectral;
extern bool mink;
extern bool icstudy;
extern bool preprop;
extern bool coSwitch2theta;
extern bool WKBtotheend;
extern FieldPrecision sPrec;
extern DeviceType     cDev;
extern ConfType	      cType;
extern ConfsubType    smvarType;
extern FieldType      fTypeP;
extern LambdaType     lType;
//extern VqcdType       vqcdType;
//extern VqcdType       vqcdTypeDamp;
//extern VqcdType       vqcdTypeRhoevol;

extern size_t         fftplanType;
extern PropType         pType;
extern SpectrumMaskType spmask;
extern double           rmask;
extern MeasureType      defaultmeasType;
extern MeasureType      rho2thetameasType;
extern SliceType        maty;
extern nRunType         nrt;
extern MeasInfo         deninfa;
extern StringMeasureType strmeas;

extern LogMpi       logMpi;
extern bool         debug ;
extern VerbosityLevel verb;

//for output
extern PrintConf      prinoconfo;
extern bool           p2dmapo;
extern bool           p2dEmapo;
extern bool           p2dPmapo;
extern bool           p3dstrings;
extern bool           p3dwalls;
extern bool           pconfinal;
extern bool           pconfinalwkb ;
extern bool           restart_flag ;
extern bool           cummask ;
int	parseDims (int argc, char *argv[]);
int	parseArgs (int argc, char *argv[]);
Cosmos	createCosmos();
void	createOutput();
