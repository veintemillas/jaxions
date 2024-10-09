#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <sys/stat.h>
#include <vector>

#include "enum-field.h"
#include "utils/logger.h"
#include "cosmos/cosmos.h"

# define PARSE1 { LogMsg(VERB_NORMAL,"%s   ",argv[i]); passed = true; procArgs++; goto endFor; }
# define PARSE2 { LogMsg(VERB_NORMAL,"%s %s",argv[i],argv[i+1]); i++; passed = true; procArgs++; goto endFor; }

size_t sizeN  = 128;
size_t sizeZ  = 128;
int    zGrid  = 1;
int    nSteps = 5;
int    dump   = 100;
double nQcd   = 7.0;
//JAVIER
int    Nng    = -1 ;
double indi3  = 1.0;
double msa    = 1.5;
double lz2e   = 0.0;
double wDz    = 0.8;
int    fIndex = -1;
int    fIndex2 = 0;
int    slicepp = 0;
int    cumumas = -1;

double sizeL = 4.;
double zInit = 0.5;
double zFinl = 1.0;
double kCrit = 1.0;
//JAVIER
double fA  = 1.0e10;
double frw = 1.0;
double mode0 = 10.0;
double alpha = 0.143;
double zthres   = 1000.0;
double zrestore = 1.0e+30;
double LL = 25000.;
double parm2 = 0.;
double pregammo = 0.0;
double dwgammo  = -1.0;
double gammo    = 0.0;
double dectime  = -1.0;
double p3DthresholdMB = 1.e+6;
double wkb2z  = -1.0;
double prepstL = 5.0 ;
double prepcoe = 3.0 ;
double ng0calib = 1.25 ;

int endredmap = -1;
int endredmapwkb = -1;
int safest0   = 20;
size_t nstrings_globale ;

std::vector<double> rmask_tab;
int i_rmask = 0;

bool uwDz     = false;
bool lowmem   = false;
bool lowmemGPU= false;
bool uPrec    = false;
bool uSize    = false;
bool uQcd     = false;
bool uLambda  = false;
bool uMsa     = false;
bool uI3      = false;
bool uPot     = false;
bool uGamma   = false;
bool uGamExp  = false;
bool uZth     = false;
bool uZrs     = false;
bool uZin     = false;
bool uZfn     = false;
bool uLogi    = false;
bool uMI      = false;
bool uFR      = false;
bool ufA      = false;
bool uexCosm  = false;
bool spectral = false;
bool fpectral = false;
bool mink			= false;
bool aMod     = false;
bool icstudy  = false ;
bool preprop  = false ;
bool coSwitch2theta  = true ;
bool WKBtotheend = false;
bool measCPU  = false;
bool maskenergyonly = false;

size_t kMax  = 2;
size_t iter  = 0;
size_t parm1 = 0;
size_t wTime = std::numeric_limits<std::size_t>::max();

size_t       fftplanType     = 0; //FFTW_MEASURE = 0
PropType     pType           = PROP_NONE;
SpectrumMaskType spmask      = SPMASK_NONE;
StringMeasureType strmeas    = STRMEAS_STRING;
double       rmask           = 2.0 ;
ConfType     cType           = CONF_NONE;
ConfsubType  smvarType       = CONF_RAND;
FieldType    fTypeP          = FIELD_SAXION;
LambdaType   lType           = LAMBDA_FIXED;
VqcdType     vqcdType        = V_QCD1;
VqcdType     vpqType         = V_PQ1;
VqcdType     vqcdTypeDamp    = V_NONE;
VqcdType     vqcdTypeEvol    = V_NONE;
GadType      gadType         = VOID;

// Default IC type
IcData icdatst;

// Default measurement type, some options can be chosen with special flags | all with --meas
MeasureType  defaultmeasType   = MEAS_NOTHING  ;
// Default measurement type for the transition to theta
MeasureType  rho2thetameasType = MEAS_NOTHING ; //MEAS_ALLBIN | MEAS_STRING | MEAS_ENERGY | MEAS_2DMAP | MEAS_NSP_A | MEAS_PSP_A;

// map measurement types (applies to all measurements that get PLOT_2D)
SliceType maty;

// measure K, G, V ? the default is given below
nRunType nrt                 = NRUN_NONE;

// Measurement type (defaults)
MeasInfo deninfa;



char outName[128] = "axion\0";
char gadName[128] = "ics\0";
char outDir[1024] = "out/m\0";
char wisDir[1024] = "./\0";

FieldPrecision	sPrec  = FIELD_SINGLE;
DeviceType	cDev   = DEV_CPU;

VerbosityLevel	verb   = VERB_NORMAL;
LogMpi		logMpi = ALL_RANKS;
bool debug        = false;

// PrintConf prinoconfo  = PRINTCONF_NONE;
bool p2dmapo  	  = false;
bool p2dEmapo  	  = false;
bool p2dPmapo  	  = false;
bool p3dstrings	  = false;
bool p3dwalls	  = false;
bool pconfinal 	  = false;
bool pconfinalwkb = false;
bool restart_flag = false;
bool cummask      = false;

bool mCreateOut = false;
bool bopt = true;

bool CreateLogMeas = false;

bool legal_int(char *str) {
    while (*str)
        if (!isdigit(*str++))
            return false;
    return true;
}

void	createOutput() {
	struct stat tStat;

	if (mCreateOut == false)
		return;

	if (stat("out", &tStat) != 0) {
		auto  dErr = mkdir("out", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (dErr == -1) {
			printf("Error: can't write on filesystem\n");
			exit(1);
		}
	}

	if (stat("out/m", &tStat) != 0) {
		auto  dErr = mkdir("out/m", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (dErr == -1) {
			printf("Error: can't write on filesystem\n");
			exit(1);
		}
	}
}

void	createMeasLogList() {

	FILE *cacheFile = nullptr;
	if ( !((cacheFile  = fopen("./measfile.dat", "r")) == nullptr) ){
		printf("[cmll] Error: measfile.dat found. Rename, delete and rerun this generator.\n");
		exit(0);
	}

		/* calculates logif */
		double logf  ;
		double z0 = 0.0;
		double fr = 1/((double) dump);

		switch(lType)
		{
			case LAMBDA_FIXED:
				logf = log(msa*zFinl*zFinl*sizeN/sizeL);
			break;
			default:
			case LAMBDA_Z2:
				logf = log(msa*zFinl*sizeN/sizeL);
			break;
		}

		if (uZin)
			z0 = zInit;
		printf("[cmll] Generates a measfile.dat file with log-spaced measurements and the default measure.\n");
		printf("[cmll] Starts at logi = zi = %f (uses zi as logi=kappa as in vilgor ICs)  \n",zInit);
		printf("[cmll] Ends at logf = log(msa*zi**/delta) = %f \n",logf);
		printf("[cmll] Warning: Uses --dump %d as number of measurements per log10 interval \n",dump);

		//-output txt file

		FILE *file_te ;
		file_te = NULL;
		file_te = fopen("./measfile.dat","w+");

		double zas;
		printf("logi | z | meastype \n");
		for (double lo = z0; lo<logf; lo += fr){
			zas = exp(lo)*sizeL/sizeN/msa;
			if (lType == LAMBDA_FIXED)
				zas = sqrt(zas);

			printf("%f %f %d\n",lo, zas,static_cast<int>(defaultmeasType));
			fprintf(file_te,"%f %d\n",zas, static_cast<int>(defaultmeasType));
		}
		/* last measurement */
		if (zas != zFinl){
			printf("%f %f %d\n",logf, zFinl,static_cast<int>(defaultmeasType));
			fprintf(file_te,"%f %d\n",zFinl, static_cast<int>(defaultmeasType));
		}


		fclose(file_te);
		printf("[cmll] measfile.dat generated. Run vaxion3d at your leisure.");
		exit(0);

}

void	PrintUsage(char *name)
{
	printf("\nUsage: %s [Options]\n\n", name);

	printf("\nOptions:\n");

	printf("\nSize of the grid:\n");
	printf("  --size  [int]                 Number of lattice points along x and y (Lx). Local size is Lx^2 x Lz (default 128).\n");
	printf("  --depth [int]                 Number of lattice points of depth (Lz) (default 128).\n");
	printf("  --zgrid [int]                 Number of mpi processed involved in the computation (default 1).\n");
	printf("                                Splitting occurs in the z-dimension, so the total lattice is Lx^2 x (zgrid * Lz).\n");

	printf("\nSimulation parameters:\n");
	printf("  --lowmem                      Reduces memory usage by 33%%, but decreases performance as well (default false).\n");
	printf("  --prec  double/single         Precision of the axion field simulation (default single)\n");
	printf("  --device cpu/gpu              Uses nVidia Gpus to accelerate the computations (default, use cpu).\n");
	printf("  --prop  leap/rkn4/om2/om4     Numerical propagator to be used for molecular dynamics (default, use rkn4).\n");
	printf("  --steps [int]                 Number of steps of the simulation (default 500).\n");
	printf("  --spec                        Enables the spectral propagator for the laplacian (default, disabled).\n");
 	printf("  --lap   1/2/3/4             	Number of Neighbours of the laplacian [default --lap 1 flag]\n");
	printf("  --wDz   [float]               Adaptive time step dz = wDz/frequency [l/raxion3D].\n");
	printf("  --restart                     searches for out/m/axion.restart and continues a simulation... needs same input parameters!.\n");
	printf("  --fftplan [64/0/32/8]         FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE (default MEASURE) \n\n");

	printf("  --sst0  [int]                 Number of steps between end of strings and switch to theta-only propagation\n");
	printf("  --dwgam [float]               Damping factor used for rho between Moore's time and switching to theta-only\n");
	printf("  --notheta                     Do not switch to theta-only mode when strings have decayed\n");

	printf("\nPhysical parameters:\n");
	printf("  --ftype saxion/axion          Type of field to be simulated, either saxion + axion or lone axion (default saxion, not parsed yet).\n");
	printf("  --mink                        Minkowski (No expansion of the Universe; experimental)\n");
	printf("  --frw   [float]               Expansion of the Universe [R~eta^frw] (default frw = 1.0)\n");
	printf("  --cax                         Uses a compact axion ranging from -pi to pi (default, the axion is non-compact).\n");
	printf("  --zi    [float]               Initial value of the conformal time z (default 0.5).\n");
	printf("  --zf    [float]               Final value of the conformal time z (default 1.0).\n");
	printf("  --Rc    [float]               Critical value of scale factor R, (mass_A^2 = constant for R>Rc) (default 1.e5).\n");
	printf("  --lsize [float]               Physical size of the system (default 4.0).\n");
	printf("  --qcd   [float]               Exponent of topological susceptibility (default 7).\n");
	printf("  --llcf  [float]               Lagrangian coefficient (default 15000).\n");
	printf("  --msa   [float]               [Sets PRS string simulation] msa is the Spacing to core ratio.\n");
	printf("  --ind3  [float]               Factor multiplying axion mass^2 (default, 1).\n");
	printf("                                Setting 0.0 turns on massless Axion mode.\n");
	printf("  --vqcdC                       Cosine QCD potential (default, disabled).\n");
	printf("  --vqcdV                       Variant of QCD potential (default, disabled).\n");
	printf("  --vqcd0                       No QCD potential (default, disabled).\n");
	printf("  --N2                          PQ potential with NDW=2 (default, disabled, experimental).\n");
	printf("  --vPQ2                        Variant of PQ potential (default, disabled).\n");
	printf("  --onlyrho                    	Only rho-evolution, theta frozen (default, disabled)\n");
	printf("  --onlytheta                   Only theta-evolution, rho frozen (default, disabled)\n");
	printf("  --gam   [float]               Saxion damping rate (default 0.0)\n");


	printf("\nInitial conditions:\n");
	printf("  --icinfo                      Prints more info about initial conditions.\n");
	printf("  --ctype smooth/kmax/vilgor    Initial configuration, either with smoothing or with FFT and a maximum momentum\n");
	printf("  --smvar stXY/stYZ/mc0/mc/...  [smooth variants] string, mc's, pure mode, noise... initial conditions.\n");
	printf("\n");
	printf("  --kmax  [int]                 Maximum momentum squared for the generation of the configuration with --ctype kmax/tkachev (default 2)\n");
	printf("  --kcr   [float]               kritical kappa (default 1.0).\n");
	printf("  --mode0 [float]               Value of axion zero mode [rad] (default random).\n");
	printf("\n");
	printf("  --sIter [int]                 Number of smoothing steps for the generation of the configuration with --ctype smooth (default 40)\n");
	printf("  --alpha [float]               alpha parameter for the smoothing (default 0.143).\n");
	printf("  --wkb   [float]               WKB's the final AXION configuration until specified time [l/raxion3D] (default no).\n");
 	printf("  --nncore  										Do not apply rho(x) core normalisation.\n");
	printf("\n");
	printf("  --index [idx]                 Loads HDF5 file at out/dump as initial conditions (default, don't load).\n");

	// printf("\nPrepropagator:\n");
	// printf("  --preprop                     turns on prepropagator -propagates IC-RHO-damping-fixed zi- for zi or given scaling stL (default yes).\n");
	// printf("  --prepcoe [float]             prepropagator starts at zi/prepcoe (default 3.0).\n");
	// printf("  --prepstL [float]             string length/volume at which prepropagator stops [raxion3d?] (default 5).\n");
	// printf("  --icstudy                     Allows printing maps/energy during preprop (default no).\n");

	printf("\nOutput:\n");
	printf("--name  [filename]              Uses filename to name the output files in out/dump, instead of the default \"axion\"\n");
	printf("--dump  [int]                   frequency of the output (default 100).\n");
	printf("--meas  [int]                   MeasuremeType [default ALLBIN|STRING|STRINGMAP|ENERGY|2DMAP|SPECTRUM].\n");
	printf("--measinfo                      Prints more info about measurement options.\n");
	printf("--p3D 0/1/2/3/4/32              Print initial/final configurations (default 0 = no) 1=initial 2=final 3=both 4=wkb 32=paxion at saturation \n");
	printf("--wTime [float]                 Simulates during approx. [float] hours and then writes the configuration to disk.\n");
	printf("--p2Dmap                        Include 2D XY maps in axion.m.files (default no)\n");
	printf("--p2Dslice [int]                Include 2D XY maps of the desired slice in axion.m.files (default no)\n");
	printf("--p2DmapYZ                      Include 2D YZ maps in axion.m.files (default no)\n");
	printf("--p2DmapE                       2D Energy XY map in axion.m.files (default no) \n");
	printf("--p2DmapPE                      2D Projection map of Energy along z direction in axion.m.files (default no) \n");
	printf("--p2DmapPE2                     2D Projection map of Energy^2 along z direction in axion.m.files (default no) \n");
	printf("--p3Dstr                        Include 3D string/Wall maps axion.m.files (default no)\n");
	printf("--p3Def   [int]                 Include 3D axion energy map of size [specified int < N]**3 in final measurement file (default no) (default int = N)\n");
	printf("--p3Dewkb [int]                 Include 3D axion energy map of size [specified int < N]**3 in measurement file after wkb (default no) (default int = N)\n");

	printf("\nLogging:\n");
	printf("--verbose 0/1/2/3/4             Choose verbosity level 0 = silent, 1 = normal (default), 2 = high, ...\n\n");
	printf("--nologmpi                      Disable logging over MPI so only rank 0 logs (default, all ranks log)\n\n");
	printf("--icinfo                        Info about initial conditions.\n");
	printf("--measinfo                      Info about measurement types.\n");
	printf("--gravinfo                      Info about SP module and gadget output.\n");
	printf("--help                          Prints this message.\n");

	// printf("--debug                         Prints some messages\n");
	//printf("--lapla 0/1/2/3/4               Number of Neighbours in the laplacian [only for simple3D] \n");

	return;
}

void	PrintICoptions()
{
	printf("\nOptions for Initial conditions\n\n");

	printf("--ctype smooth/kmax/tkachev/thermal        			   Main IC selector.\n\n");

	printf(" [smooth]                                          Point based.\n");
	printf("-----------------------------------------------------------------------------------------------\n");
	printf("  Random phase at each point (default).\n");
	printf("  --smvar [string]                                 Spectial initial distributions.\n");
	printf("  --smvar axnoise  --mode0 [float] --kcr [float]   theta = mode0+random{-1,1}*kcr.\n");
	printf("  --smvar saxnoise --mode0 [float] --kcr [float]   theta = mode0, rho = 1 + random{-1,1}*kcr.\n");
	printf("  --smvar mc   --mode0 [float] --kcr [float]       theta = mode0 Exp(-kcr*(x-N/2)^2).\n");
	printf("  --smvar mc0  --mode0 [float] --kcr [float]       theta = mode0 Exp(-kcr*(x)^2).\n");
	printf("  --smvar ax1mode  --mode0 [float] --kmax[int]     theta = mode0 cos(2Pi kMax*x/N).\n");
	printf("  --smvar parres   --mode0 [float] --kmax[int]     theta = mode0 cos(kx*x + ky*y + kz*z) k's specified in kkk.dat, \n");
	printf("                   --kcr [float]                   rho = kcr. Alternatively k = (kMax,0,0) if not kkk.dat file. \n\n");
	printf("  --smvar stXY --mode0 [float] --kcr [float]       Circular loop in the XY plane at z=kcr, radius N/4.\n");
	printf("  --smvar stYZ --mode0 [float] --kcr [float]       Circular loop in the YZ plane, radius N/4.\n");
	printf("  --smvar stringwave --mode0 [float]               Straight strings along Z + axion wave \n");
	printf("                                                   of momentum kx,ky,kz specified in kkk.dat and amplitude mode0. \n\n");

	printf(" [kmax]                                            Saxion momentum based.\n");
	printf("-----------------------------------------------------------------------------------------------\n");
	printf("  --kmax [int] --kcr [float]                       Random modes and inverse FFT.\n");
	printf("                                                   for 3D k < kmax = min(kmax, N/2-1).\n");
	printf("                                                   mode ~ exp(I*random) * exp( -(kcr* k x)^2).\n");
	printf("  --mode0 [float]                                  mode[000] = exp(I*mode0).\n\n");

	printf(" [lola]                                            Start in the VGH attractor solution (or close)\n");
	printf("-----------------------------------------------------------------------------------------------\n");
	printf("  --logi [float]                                   Initial log(ms/H) (no time!!!) [default 0.0].\n");
	printf("  --zi                                             Initial time (do not use with logi)\n");
	printf("  --sIter 1 --kcr [float]                          Network overdense by exact factor kcr. \n");
	printf("  --kickalpha [float]                              Initial V velocity; V initalised as Phi(1+kick) (default 0.0)\n");
	printf("  --extrav    [float]                              Extra noise in V (default 0.0)\n\n");

	printf(" [vilgor,vilgork,vilgors]                          Old versions of lola (legacy, to be discontinued)\n");
	printf("-----------------------------------------------------------------------------------------------\n");
	printf("  --zi [float]                                     Initial log(ms/H) (no time!!!) [default 0.0].\n");
	printf("  --sIter 1 --kcr [float]                          Network overdense by exact factor kcr. \n");
	printf("  --sIter 2 --kcr [float]                          Network over/underdense by random factor [*kcr,/kcr]. \n");

	printf(" [cole]                                            Use correlation length to set field\n");
	printf("-----------------------------------------------------------------------------------------------\n");
	printf("  --logi [float]                                   Initial log(ms/H) (no time!!!) [default 0.0].\n");
	printf("  --zi                                             Initial time (do not use with logi)\n");
	printf("  --kcr [float]                                    Correlation length. \n");

	printf(" [tkachev]                                         Axion momentum based.\n");
	printf("-----------------------------------------------------------------------------------------------\n");
	printf("  --kmax [int] --kcr [float]                       Axion modes as in Kolb&Tkachev 92 .\n");
	printf("                                                   <theta^2>=kcr*pi^2/3 \n");

  printf(" [thermal]                                         Temperature based.\n");
	printf("-----------------------------------------------------------------------------------------------\n");
	printf("  --kcr [float]                                    c. Temperature in ADM units sqrt[H1 fA].\n");

	printf(" --preprop                                         prepropagator; currently only works with lola\n");
	printf("-----------------------------------------------------------------------------------------------\n");
	printf("  --preprop                                                                                    \n");
	printf("  --prepropcoe  [float]                            Preevolution starts at zi/prepropcoe        \n");
	printf("  --prevqcdtype [int]                              VQCD type during prepropagation (default V_QCD0_PQ1_DRHO).\n");
	printf("  --pregam      [float]                            Damping factor during prepropagation (default 0.0) .\n");
	printf("                                                   Requires prevqcdtype to include damping, +16384 or +32768.\n");
	printf("  --prelz2e     [float]               	           Makes lambda = lambda/R^lz2e (Default 2.0 in PRS mode).\n");
	printf("  --icstudy                           	           Prints axion.m.xxxxx files during prepropagation (Default no).\n");
	printf("\n-----------------------------------------------------------------------------------------------\n");
	printf("  --nncore                                         Do not normalise rho according to grad but rho=1.\n\n");

	printf("  Test examples:                                                                             .\n\n");
	printf("  --ctype lola --logi 4.0 --sIter 1 --kcr 2.0                                                .\n\n");
	printf("  --ctype lola --logi 4.0 --sIter 1 --kcr 2.0                                                .\n\n");
	return;
}

void    PrintGravoptions()
{
	printf("\nOptions for Gravity evolution\n");
	printf("-------------------------------------------------------------------------------------------------------\n\n");
	printf("--beta [float]                     Coefficient for self-interactions, GPP system [default 1.0].\n");
	printf("--gravity [float]                  Ratio between R_1/R_eq [default 0.0].\n");
	printf("--sat_gravity                      After saturation time evolve with constant conformal axion mass [default: false].\n");
	printf("--hybrid_gravity                   Compute gravitational potential with hybrid method [default: false].\n");
	printf("--L1_pc [float]                    Sets the value of L1 in parsec units [default: 0.036]");
	printf("--gad_type [gad/gadmass/gadgrid]   Gadget mapping type [default: gadmass].\n");
	printf("--part_vel                         Add particle velocities [default: false].\n");
	printf("--smooth_vel (to be implemented)   Smooth velocity field [default: false].\n");
	printf("--kcr [float]                      Displacement for [gad] mapping [default 1.0, recommended 0.25].\n");
	printf("-------------------------------------------------------------------------------------------------------\n\n");
}

void	PrintMEoptions()
{
	printf("\nOptions for Measurement\n\n");

	printf("--meas [int]                Sum of integers.\n\n");

	printf("--------------------------------------------------\n");
	printf("  BINs\n");
	printf("  Nothing                                  0 \n");
	printf("  Theta angle binned                       1 \n");
	printf("  Saxion field/v                           2 \n");
	printf("  Log theta^2                              4 \n");
	printf("  Axion density contrast                   8 \n\n");

	printf("  MAPS\n");
	printf("  String + Wall                           32 \n");
	printf("  String + Wall 3D map                    64 \n\n");
	// printf("  String + Wall coord.     160 \n");
	printf("  Ax. + Sax. Energy                      256 \n");
	printf("  Ax. Energy 3D map                      512 \n");
	printf("  Ax. Energy reduced 3D map             1024 \n");

	printf("  2D slice map (Field+Velocity)         2048 \n");
	printf("  3D configuration                      4096 \n\n");

	printf("  POWER SPECTRA (binned) \n");
	printf("  Number of modes in each mom. bin    262144 \n");
	printf("  Axion Energy spectrum                16384 \n");
	printf("  Sxion Energy spectrum                32768 \n\n");

	printf("  NUMBER SPECTRA (binned) \n");
	printf("  Saxion Number spectrum (K+G+V)      131072 \n\n");

	printf("  Axion Number spectrum (K+G+V)        65536 \n");
	printf("  --spKGV [int]             Sum of integers.\n");
	printf("    Kinetic                                1 \n");
	printf("    Gradient                               2 \n");
	printf("    Potential                              4 \n");
	printf("    K+G+V                                  7 (default)\n\n");
    printf("  --spmask [int]            Sum of integers.\n");
	printf("    Fields unmasked                        1 (default	)\n");
	printf("    Masked with  rho/v                     2 \n");
	printf("    Masked with (rho/v)^2                  4 \n");
	printf("    Red (Top-hat from Gaussian cut)        8 \n");
	printf("    Gaussian                               16 \n");
	printf("    Diffusion from core top hat            32 \n");
	printf("    Ball                                   64 \n");
	printf("    Axiton Ball                            512 \n");
	printf("    Axiton FT                              1024 \n");
	printf("     --rmask [float]                       Mask radius (for strings in 1/m_s units) [default = 2]\n");
	printf("     --rmask file                          Prints different spectra, each masked \n");
	printf("                                           with the values read from rows of a rmasktable.dat file.\n");
	printf("                                           (Red, Gaus, Axit12 modes) \n\n");
	printf("  --printmask                              Prints the mask\n\n");
  printf("  --maskenergyonly                         Skips masked spectra in Red mode and prints masked energies only (default no)\n\n");
	printf("  --ng0calib                               Parameter tunning the exponential masking (default 1.25)\n");
	printf("                                           (Any negative value gives the old calibration)\n\n");
	printf("  --cummask [int > 0]                      Mask region is not reset at each meas. Can only increase. (default no)\n\n");

	printf("  Options for String Measurement \n");
	printf("  --strmeas [int]            Sum of integers.\n");
	printf("    Statistical measurment only            0 (default	)\n");
	printf("    String length                          1 \n");
	printf("    String velocity and gamma              2 \n");
	printf("    String energy (needs Red-Gauss mask)   4 \n");

	printf("--------------------------------------------------\n");
	printf("--measlistlog              Generate measfile log. with defaults.\n\n");
	return;
}

int	parseDims (int argc, char *argv[])
{
	bool	brank, bcdev, blog, bverb = false;
	int	procArgs = 0;

	/* We collect the arguments for initComms:
		1 - the number of MPI processes
		2 - the device
		3 - type of logMpi
		4 - verbosity level */
	for (int i=1; i<argc; i++)
	{
		/* Base arguments to start MPI communications, logger, etc. */

		if (!strcmp(argv[i], "--zgrid"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a number of mpi ranks.\n");
				exit(1);
			}

			zGrid = atoi(argv[i+1]);

			if (zGrid < 1)
			{
				printf("Error: The number of mpi ranks must be larger than 0.\n");
				exit(1);
			}

			i++;
			procArgs++;
			brank = true;
			continue;
		}

		if (!strcmp(argv[i], "--device"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a device name (cpu/gpu).\n");
				exit(1);
			}

			if (!strcmp(argv[i+1], "cpu"))
			{
				cDev = DEV_CPU;
			}
			else if (!strcmp(argv[i+1], "gpu"))
			{
				cDev = DEV_GPU;
			}
			else if (!strcmp(argv[i+1], "xeon"))
			{
				printf("Error: Knights Corner support has been removed\n");
				exit(1);
			}
			else
			{
				printf("Error: unrecognized device %s\n", argv[i+1]);
				exit(1);
			}

			i++;
			procArgs++;
			bcdev = true;
			continue;
		}

		if (!strcmp(argv[i], "--nologmpi"))
		{
			logMpi = ZERO_RANK;

			procArgs++;
			blog = true;
			continue;
		}

		if (!strcmp(argv[i], "--verbose"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a verbosity level.\n");
				exit(1);
			}
			int mocho;
			sscanf(argv[i+1], "%d", reinterpret_cast<int*>(&mocho));

			if (mocho > VERB_HIGH)      verb = VERB_HIGH;      //2
			if (mocho > VERB_PARANOID)  verb = VERB_PARANOID;  //3
			if (mocho < VERB_SILENT)    verb = VERB_SILENT;    //0


			i++;
			procArgs++;
			bverb = true;

			continue;
		}

		/* Arguments to display info, they interrupt program */

		if (!strcmp(argv[i], "--help"))
		{
			PrintUsage(argv[0]);
			exit(0);
		}

		if (!strcmp(argv[i], "--icinfo"))
		{
			PrintICoptions();
			exit(0);
		}

		if (!strcmp(argv[i], "--gravinfo"))
		{
			PrintGravoptions();
			exit(0);
		}

		if (!strcmp(argv[i], "--measinfo"))
		{
			PrintMEoptions();
			exit(0);
		}

	}

	if (!brank || !bcdev || !blog || !bverb)
	{
		printf("Jaxions will use defauls for %s %s %s %s", !brank? "rank":"", !bcdev? "device":"", !blog? "logmpi":"", !bverb? "verbosity":"");
	}

	return procArgs;
}

int	parseArgs (int argc, char *argv[])
{
	LogMsg(VERB_HIGH,"Parsing ...");
	bool	passed;
	int	procArgs = 0;

	// defaults
	icdatst.Nghost    = 1;
	icdatst.icdrule   = false;
	icdatst.preprop   = false;
	icdatst.icstudy   = false;
	icdatst.prepstL   = 5.0 ;
	icdatst.prepcoe   = 3.0 ;
	icdatst.pregammo  = 0.0;
	icdatst.prelZ2e   = 0.0;
	icdatst.prevtype  = V_QCD1_PQ1_RHO;
	icdatst.normcore  = true;
	icdatst.alpha     = 0.143;
	icdatst.siter     = 40;
	icdatst.kcr       = 1.0;
	icdatst.kMax      = 2;
	icdatst.mode0     = 0.0;
	icdatst.beta      = 1.0;
	icdatst.grav      = -1.0;
	icdatst.L1_pc     = 0.036;
  	icdatst.grav_hyb  = false;
  	icdatst.grav_sat  = false;
	icdatst.part_vel  = false;
	icdatst.sm_vel    = false;
	icdatst.part_disp = false;
	icdatst.zi        = 0.5;
	icdatst.logi      = 0.0;
	icdatst.kickalpha = 0.0;
	icdatst.extrav    = 0.0;
	icdatst.cType     = CONF_KMAX;
	icdatst.smvarType = CONF_RAND;
	icdatst.mocoty    = MOM_MEXP2;
	icdatst.randommom = true;
	icdatst.fieldindex=FIELD_NO;
	// Axiton tracker info. default: disabled
	icdatst.axtinfo.nMax = -1;

	/* Default measurements */
	deninfa.idxprint  = 0;
  deninfa.nbinsspec = -1;              // (natural width bin width = 2pi/L0)
	deninfa.printconf = PRINTCONF_NONE;  // no configuration

	for (int i=1; i<argc; i++)
	{
		passed = false;

		/* These are parsed before */

		if (!strcmp(argv[i], "--zgrid"))
			PARSE2;
		if (!strcmp(argv[i], "--device"))
			PARSE2;
		if (!strcmp(argv[i], "--nologmpi"))
			PARSE1;
		if (!strcmp(argv[i], "--verbose"))
			PARSE2;
		if (!strcmp(argv[i], "--help"))
			PARSE1;
		if (!strcmp(argv[i], "--icinfo"))
			PARSE1;
		if (!strcmp(argv[i], "--measinfo"))
			PARSE1;
		/* Parse jaxion arguments */

    if (!strcmp(argv[i], "--ftype"))
		{
      LogMsg(VERB_NORMAL,"WARNING: Currently setting field type is experimental! Mostly for reading files into other formats.");
			if (i+1 == argc)
			{
				printf("I need a field type!.\n");
				exit(1);
			}
			if (!strcmp(argv[i+1], "saxion"))
				fTypeP = FIELD_SAXION;
			else if (!strcmp(argv[i+1], "axion"))
        fTypeP = FIELD_AXION;
        else if (!strcmp(argv[i+1], "axionmod"))
        fTypeP = FIELD_AXION_MOD;
      else if (!strcmp(argv[i+1], "Paxion"))
        fTypeP = FIELD_PAXION;
      else {
        printf("field type not recognised!.\n");
				exit(1);
      }

			PARSE2;
		}

		if (!strcmp(argv[i], "--mink"))
		{
			mink = true;
			uMI  = true;
			frw  = 0.0;
			PARSE1;
		}

		if (!strcmp(argv[i], "--debug"))
		{
			debug = true;
			PARSE1;
		}

		if (!strcmp(argv[i], "--frw"))
		{

			if (i+1 == argc)
			{
				printf("Error: I need a value for the walltime.\n");
				exit(1);
			}

			frw = atof(argv[i+1]);

			if (frw < 0.)
			{
				printf("Warning: Contracting Universe?\n");
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--p3D"))
		{
			if (i+1 == argc)
			{
				printf("Warning: p3D set by default to 0 [no 00000 and final configuration files].\n");
				deninfa.printconf = PRINTCONF_NONE ;
				PARSE1
			}

			sscanf(argv[i+1], "%d", reinterpret_cast<int*>(&(deninfa.printconf)));

			PARSE2;
		}

		if (!strcmp(argv[i], "--p2Dmap"))
		{
			p2dmapo = true ;
			defaultmeasType |= MEAS_2DMAP;
			maty |= MAPT_XYMV;
			PARSE1;
		}

		if (!strcmp(argv[i], "--p2DmapYZ"))
		{
			p2dmapo = true ;
			defaultmeasType |= MEAS_2DMAP;
			maty |= MAPT_YZMV;
			PARSE1;
		}

		if (!strcmp(argv[i], "--sliceprint"))
		{
			sscanf(argv[i+1], "%d", reinterpret_cast<int*>(&slicepp));
			PARSE2;
		}

		if (!strcmp(argv[i], "--p2Dslice"))
		{
			p2dmapo = true ;
			defaultmeasType |= MEAS_2DMAP;
			maty |= MAPT_XYMV;

			sscanf(argv[i+1], "%d", reinterpret_cast<int*>(&slicepp));
			PARSE2;
		}

		if (!strcmp(argv[i], "--p2DsliceYZ"))
		{
			p2dmapo = true ;
			defaultmeasType |= MEAS_2DMAP;
			maty |= MAPT_YZMV;

			sscanf(argv[i+1], "%d", reinterpret_cast<int*>(&slicepp));
			PARSE2;
		}

		if (!strcmp(argv[i], "--measaux"))
		{
			defaultmeasType |= MEAS_AUX;
			PARSE1;
		}


		if (!strcmp(argv[i], "--p2DmapE"))
		{
			p2dEmapo = true ;
			defaultmeasType |= MEAS_2DMAP;
			maty |= MAPT_XYE;
			PARSE1;
		}

		if (!strcmp(argv[i], "--p2DmapPE2") || !strcmp(argv[i], "--p2DmapP"))
		{
			p2dPmapo = true ;
			defaultmeasType |= MEAS_2DMAP;
			maty |= MAPT_XYPE2;
			PARSE1;
		}

		if (!strcmp(argv[i], "--p2DmapPE"))
		{
			p2dPmapo = true ;
			defaultmeasType |= MEAS_2DMAP;
			maty |= MAPT_XYPE;
			PARSE1;
		}

		if (!strcmp(argv[i], "--p3Dstr"))
		{
			p3dstrings = true ;
			PARSE1;
		}

		if (!strcmp(argv[i], "--p3Dwal"))
		{
			p3dwalls = true ;
			PARSE1;
		}

		if (!strcmp(argv[i], "--pcon") || !strcmp(argv[i], "--p3Def") )
		{
			if (i+1 == argc)
			{
				pconfinal = true ;
				PARSE1;
			} else
					if (legal_int(argv[i+1])) {
						endredmap = atof(argv[i+1]);
						PARSE2;
					} else {
						pconfinal = true ;
						PARSE1;
					}
		}

		if (!strcmp(argv[i], "--pconwkb")  || !strcmp(argv[i], "--p3Dewkb") )
		{
			if (i+1 == argc)
			{
				pconfinalwkb = true ;
				PARSE1;
			} else
					if (legal_int(argv[i+1])) {
						endredmapwkb = atof(argv[i+1]);
						PARSE2;
					} else {
						pconfinalwkb = true ;
						PARSE1;
					}
		}

    if (!strcmp(argv[i], "--idxprint"))
		{
			sscanf(argv[i+1], "%lu", reinterpret_cast<size_t*>(&deninfa.idxprint));
			PARSE2;
		}

		if (!strcmp(argv[i], "--wTime"))
		{
			double	tTime = 0.;

			if (i+1 == argc)
			{
				printf("Error: I need a value for the walltime.\n");
				exit(1);
			}

			tTime = atof(argv[i+1]);

			if (tTime < 0.)
			{
				printf("Error: Walltime must be larger than or equal to 0.\n");
				exit(1);
			}

			wTime = tTime*3600000000;	// Walltime is processed in microseconds, but the expected precision is much worse
			PARSE2;
		}

		if (!strcmp(argv[i], "--vqcdC"))
		{
			uPot = true;
			vqcdType = V_QCDC ;
			PARSE1;
		}

		if (!strcmp(argv[i], "--vqcdV"))
		{
			uPot = true;
			vqcdType = V_QCDV ;
			PARSE1;
		}

		if (!strcmp(argv[i], "--vqcdL"))
		{
			uPot = true;
			vqcdType = V_QCDL ;
			PARSE1;
		}

		if (!strcmp(argv[i], "--vqcd0"))
		{
			uPot = true;
			vqcdType = V_QCD0 ;
			indi3  = 0.0;
			uI3    = true;
			PARSE1;
		}

		if (!strcmp(argv[i], "--N2"))
		{
			uPot = true;
			vqcdType = V_QCD2 ;
			PARSE1;
		}

    if (!strcmp(argv[i], "--vPQ1"))
		{
			uPot = true;
			vpqType = V_PQ1 ;
			PARSE1;
		}

		if (!strcmp(argv[i], "--vPQ2"))
		{
			uPot = true;
			vpqType = V_PQ2 ;
			PARSE1;
		}

    if (!strcmp(argv[i], "--vPQ3"))
		{
			uPot = true;
			vpqType = V_PQ3 ;
			PARSE1;
		}

		if (!strcmp(argv[i], "--onlyrho"))
		{
			uPot = true;
			vqcdTypeEvol = V_EVOL_RHO;
			PARSE1;
		}

		if (!strcmp(argv[i], "--onlytheta"))
		{
			uPot = true;
			vqcdTypeEvol = V_EVOL_THETA;
			PARSE1;
		}

		if (!strcmp(argv[i], "--lowmem"))
		{
			lowmem = true;
			PARSE1;
		}

		if (!strcmp(argv[i], "--lowmemGPU"))
	        {
	                lowmemGPU = true;
	                PARSE1;
	        }

		if (!strcmp(argv[i], "--notheta"))
		{
			coSwitch2theta = false;
			PARSE1;
		}

		if (!strcmp(argv[i], "--size"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a size.\n");
				exit(1);
			}

			sscanf(argv[i+1], "%zu", &sizeN);

			if (sizeN < 2)
			{
				printf("Error: Size must be larger than 2.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--depth"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a size.\n");
				exit(1);
			}

			sscanf(argv[i+1], "%zu", &sizeZ);
			PARSE2;
		}


		if (!strcmp(argv[i], "--kcr"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the critical kappa.\n");
				exit(1);
			}

			kCrit = atof(argv[i+1]); //legacy
			icdatst.kcr = atof(argv[i+1]);

			if (icdatst.kcr < 0.)
			{
				printf("Error: Critical kappa must be larger than or equal to 0.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--norandommom"))
		{
			icdatst.randommom = false;
		       	PARSE1;
		}

		if (!strcmp(argv[i], "--alpha"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for alpha.\n");
				exit(1);
			}

			alpha = atof(argv[i+1]); //legacy
			icdatst.alpha = atof(argv[i+1]);

			if ((icdatst.alpha < 0.) || (icdatst.alpha > 1.))
			{
				printf("Error: Alpha parameter must belong to the [0,1] interval.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--nncore"))
		{
			icdatst.normcore = false;
			PARSE1;
		}

		if (!strcmp(argv[i], "--wkb"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for z_WKB!.\n");
				exit(1);
			}

			wkb2z = atof(argv[i+1]);

			if (wkb2z < zFinl)
			{
				printf("Error: z_wkb must be > zf. No WKB will be done!\n");
				wkb2z = -1.0	;
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--redmp"))
		{
			if (i+1 == argc)
			{
				endredmap = 256 ;
				printf("No new sizeN input for final reducemap. Set to default = 256\n");
			}
			else{
				endredmap = atof(argv[i+1]);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--redmpwkb"))
		{
			if (i+1 == argc)
			{
				endredmapwkb = 256 ;
				printf("No new sizeN input for final reducemap (wkb). Set to default = 256\n");
			}
			else{
				endredmapwkb = atof(argv[i+1]);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--pregam"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the prepropagator damping factor.\n");
				exit(1);
			}

			pregammo = atof(argv[i+1]); //legacy
			icdatst.pregammo = atof(argv[i+1]);

			if (pregammo < 0.)
			{
				printf("Error: pre-Damping factor should be larger than 0.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--prelz2e"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the lambda PRS exponent.\n");
				exit(1);
			}

			icdatst.prelZ2e = atof(argv[i+1]);

			PARSE2;
		}

		if (!strcmp(argv[i], "--prevqcdtype"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the prevqcd type.\n");
				exit(1);
			}

			sscanf(argv[i+1], "%d", reinterpret_cast<int*>(&icdatst.prevtype));

			PARSE2;
		}

		if (!strcmp(argv[i], "--dwgam"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the DW?string destructor damping factor.\n");
				exit(1);
			}

			dwgammo = atof(argv[i+1]);

			if (dwgammo < 0.)
			{
				printf("Error: pre-Damping factor should be larger than 0.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--gam"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the damping factor.\n");
				exit(1);
			}

			gammo = atof(argv[i+1]);
			vqcdTypeDamp = V_DAMP_RHO ;

			uPot  = true;
			uGamma = true;

			if (gammo < 0.)
			{
				printf("Error: Damping factor should be larger than 0.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--dectime"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the damping factor.\n");
				exit(1);
			}

			dectime = atof(argv[i+1]);
			vqcdTypeDamp = V_DAMP_RHO ;

			uPot  = true;
			uGamExp = true;

			if (dectime < 0.)
			{
				printf("Error: decay time should be larger than 0.\n");
				exit(1);
			}

			PARSE2;
		}

		/* TODO If decay time is to be used, give it before --gamall? */
		if (!strcmp(argv[i], "--gamall"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the theta-rho damping factor.\n");
				exit(1);
			}

			gammo = atof(argv[i+1]);
			vqcdTypeDamp = V_DAMP_ALL ;

			uPot  = true;
			uGamma = true;

			if (gammo < 0.)
			{
				printf("Error: Damping (theta-rho) factor should be larger than 0.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--beta"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the Naxion/Paxion selfinteraction coefficient.\n");
				exit(1);
			}

			icdatst.beta = atof(argv[i+1]);

			PARSE2;
		}

		if (!strcmp(argv[i], "--gravity"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the Naxion/Paxion selfinteraction coefficient.\n");
				exit(1);
			}

			icdatst.grav = atof(argv[i+1]);

			PARSE2;
		}

		if (!strcmp(argv[i], "--L1_pc"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the Naxion/Paxion selfinteraction coefficient.\n");
				exit(1);
			}

			icdatst.L1_pc = atof(argv[i+1]);

			PARSE2;
		}

                if (!strcmp(argv[i], "--hybrid_gravity"))
		{

			icdatst.grav_hyb = true;

			PARSE1;
		}

                if (!strcmp(argv[i], "--sat_gravity"))
		{

			icdatst.grav_sat = true;

			PARSE1;
		}

		if (!strcmp(argv[i], "--part_vel"))
		{

			icdatst.part_vel = true;

			PARSE1;
		}

		if (!strcmp(argv[i], "--sm_vel"))
		{

			icdatst.sm_vel = true;

			PARSE1;
		}

		if (!strcmp(argv[i], "--part_disp"))
		{

			icdatst.part_disp = true;

			PARSE1;
		}

		if (!strcmp(argv[i], "--zi"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the initial conformal time.\n");
				exit(1);
			}

			zInit = atof(argv[i+1]); //legacy
			icdatst.zi = atof(argv[i+1]);

		if (icdatst.zi < 0.)
			{
				printf("Error: Initial conformal time must be larger than 0.\n");
				exit(1);
			}

			uZin = true;

			PARSE2;
		}

		if (!strcmp(argv[i], "--logi"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the initial conformal time.\n");
				exit(1);
			}

			icdatst.logi = atof(argv[i+1]);

			uLogi = true;

			PARSE2;
		}


		if (!strcmp(argv[i], "--kickalpha"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for kickalpha.\n");
				exit(1);
			}

			icdatst.kickalpha = atof(argv[i+1]);

			PARSE2;
		}

		if (!strcmp(argv[i], "--extrav"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for extrav.\n");
				exit(1);
			}

			icdatst.extrav = atof(argv[i+1]);

			PARSE2;
		}

		if (!strcmp(argv[i], "--zf"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the Final conformal time.\n");
				exit(1);
			}

			zFinl = atof(argv[i+1]);

			if (zFinl < 0.)
			{
				printf("Error: Final conformal time must be larger than 0.\n");
				exit(1);
			}

			uZfn = true;

			PARSE2;
		}

		if (!strcmp(argv[i], "--lsize"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the physical size of the universe.\n");
				exit(1);
			}

			sizeL = atof(argv[i+1]);

			if (sizeL <= 0.)
			{
				printf("Error: Physical size must be greater than zero.\n");
				exit(1);
			}

			uSize = true;

			PARSE2;
		}

		if (!strcmp(argv[i], "--llcf"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the lagrangian coefficient.\n");
				exit(1);
			}

			LL = atof(argv[i+1]);

			if (LL <= 0.)
			{
				printf("Error: The lagrangian coefficient must be greater than zero.\n");
				exit(1);
			}

			uLambda = true;
			lType   = LAMBDA_FIXED;

			PARSE2;
		}

		if (!strcmp(argv[i], "--Rc"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the Rcritical.\n");
				exit(1);
			}

			zthres = atof(argv[i+1]);

			if (zthres <= 0.)
			{
				printf("Error: The value of Rc should be larger than zero.\n");
				exit(1);
			}

			uZth   = true;

			PARSE2;
		}

		if (!strcmp(argv[i], "--RcOff"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the re-switch-on RcOff of the axion mass growth.\n");
				exit(1);
			}

			zrestore = atof(argv[i+1]);

			if (zrestore <= 0.)
			{
				printf("Error: The mA2 ~ R^n re-switch must be > 0 .\n");
				exit(1);
			}

			uZrs   = true;

			PARSE2;
		}

		//NEW
		if (!strcmp(argv[i], "--msa"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for Spacing-to-core ratio msa.\n");
				exit(1);
			}

			msa = atof(argv[i+1]);

			if (msa <= 0.)
			{
				printf("Error: The Spacing-to-core must be greater than zero.\n");
				exit(1);
			}

			uMsa  = true;
			lType = LAMBDA_Z2; //obsolete?
      lz2e  = 2.0;
			PARSE2;
		}

    if (!strcmp(argv[i], "--lz2e"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the lambda time-dependence exponent.\n");
				exit(1);
			}

			lz2e = atof(argv[i+1]);

			PARSE2;
		}

		//NEW
		if (!strcmp(argv[i], "--ind3"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for axion mass ind3.\n");
				exit(1);
			}

			indi3 = atof(argv[i+1]);
			uI3   = true;

			if (indi3 == 0.0)
				vqcdType = V_QCD0;

			if (indi3 < 0.)
			{
				printf("Error: Indi3 must be greater than or equal to zero.\n");
				exit(1);
			}

			PARSE2;
		}


		if (!strcmp(argv[i], "--wDz"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the adaptive time step.\n");
				exit(1);
			}

			wDz = atof(argv[i+1]);
			uwDz = true;

			if (wDz <= 0.0)
			{
				printf("Error: backwards propagation?\n");
				exit(1);
			}

			PARSE2;
		}

		// NEW
		if (!strcmp(argv[i], "--mode0"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the axion zero mode.\n");
				exit(1);
			}

			mode0 = atof(argv[i+1]); //obsolete
			icdatst.mode0 =  atof(argv[i+1]);

			PARSE2;
		}



		if (!strcmp(argv[i], "--qcd"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need an exponent for the susceptibility nQcd!.\n");
				exit(1);
			}
			if (!strcmp(argv[i+1], "qcd")){
				uexCosm = true;
				// printf("Detailed Cosmology!")
			}
			else
				nQcd = atof(argv[i+1]);

			if (nQcd < 0)
			{
				printf("Error: The exponent of the top. susceptibility nQcd must be equal or greater than 0.\n");
				exit(1);
			}

			uQcd = true;

			PARSE2;
		}

		if (!strcmp(argv[i], "--fA"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value of the axion decay constant fA [GeV]!.\n");
				exit(1);
			}
			fA = atof(argv[i+1]);
			ufA = true;

			if (fA < 0)
			{
				printf("Error: The axion decay constant is negative!.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--steps"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a number of steps.\n");
				exit(1);
			}
			nSteps = atoi(argv[i+1]);

			if (nSteps < 0)
			{
				printf("Error: Number of steps must be > 0.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--sst0"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a number of steps of st=0 before switching to theta-mode.\n");
				exit(1);
			}

			safest0 = atoi(argv[i+1]);

			if (safest0 < 0)
			{
				printf("WARNING: sst0 < 0 won't switch to theta-mode.\n");
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--dump"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a print rate.\n");
				exit(1);
			}

			dump = atoi(argv[i+1]);

			if (dump < 0)
			{
				printf("Error: Print rate must be equal or greater than zero.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--meas"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a number of steps.\n");
				exit(1);
			}

			MeasureType loco = (MeasureType) atoi(argv[i+1]);

			if (loco < 0)
			{
				printf("Error: Measurement type must be >= 0.\n");
				exit(1);
			}

			defaultmeasType |= loco;

			PARSE2;
		}

		if (!strcmp(argv[i], "--meas2theta"))
		{

			MeasureType loco = (MeasureType) atoi(argv[i+1]);

			if (loco < 0)
			{
				printf("Error: Measurement 2theta type must be >= 0.\n");
				exit(1);
			}

			rho2thetameasType |= loco;

			PARSE2;
		}

		if (!strcmp(argv[i], "--measCPU"))
		{
			measCPU = true;

			PARSE1;
		}

    if (!strcmp(argv[i], "--maskenergyonly"))
		{
			maskenergyonly = true;

			PARSE1;
		}

		if (!strcmp(argv[i], "--spmask"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a spectrum masking type: 1 (flat),2 (Villadoro),3 (Flat+Villadoro), ... .\n");
				exit(1);
			}

			sscanf(argv[i+1], "%d", reinterpret_cast<int*>(&spmask));

			if (spmask < 0)
			{
				printf("Error: Spectrum masking type is a positive integer.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--spKGV"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a nRun output type: 1 (K), 2 (G), 4 (V) or sums (default 7) .\n");
				exit(1);
			}

			sscanf(argv[i+1], "%d", reinterpret_cast<int*>(&nrt));

			if (nrt < 0)
			{
				printf("Error: Spectrum masking type is a positive integer.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--printmask"))
		{
			defaultmeasType |= MEAS_MASK ;

			PARSE1;
		}

		if (!strcmp(argv[i], "--fftplan"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a fftw  plan speed: 0 (MEASURE),64 (ESTIMATE),32 (PATIENT), 8 (EXHAUSTIVE)... .\n");
				exit(1);
			}

			sscanf(argv[i+1], "%lu", reinterpret_cast<size_t*>(&fftplanType));

			PARSE2;
		}

		if (!strcmp(argv[i], "--rmask"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a radius for string masking!\n");
				exit(1);
			}

			if (!strcmp(argv[i+1], "file"))
			{
				// "Read the file amd load it in rmask_tab
				FILE *cacheFile = nullptr;
				if (((cacheFile  = fopen("./rmasktable.dat", "r")) == nullptr)){
					printf("No rmasktable.dat ! Exit!");
					exit(1);
				}
				else
				{					double rmaskaux ;
									while(!feof(cacheFile)){
											fscanf (cacheFile ,"%lf ", &rmaskaux);
											rmask_tab.push_back(rmaskaux);
											// LogMsg(VERB_NORMAL,"[VAX] i_meas=%d read z=%f meas=%d", i_meas, meas_zlist[i_meas], meas_typelist[i_meas]);
											i_rmask++ ;
									}
				}
			}
			else
			{
			rmask = atof(argv[i+1]);
			rmask_tab.push_back(rmask);
			i_rmask++;
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--nbinsspec"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a number of bins \n");
				exit(1);
			}

			sscanf(argv[i+1], "%d", reinterpret_cast<int*>(&deninfa.nbinsspec));

			PARSE2;
		}

		if (!strcmp(argv[i], "--strmeas"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a string measurement options: 0 (nothing), 1 (length), 2 (velocity), 4 (energy).\n");
				exit(1);
			}

			sscanf(argv[i+1], "%d", reinterpret_cast<int*>(&strmeas));

			if (strmeas < 0)
			{
				printf("Error: String measurement type is a positive integer.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--name"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a name for the files.\n");
				exit(1);
			}

			if (strlen(argv[i+1]) > 96)
			{
				printf("Error: name too long, keep it under 96 characters\n");
				exit(1);
			}

			strcpy (outName, argv[i+1]);

			PARSE2;
		}

		if (!strcmp(argv[i], "--gname"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a name for the gadget output file.\n");
				exit(1);
			}

			if (strlen(argv[i+1]) > 96)
			{
				printf("Error: name too long, keep it under 96 characters\n");
				exit(1);
			}

			strcpy (gadName, argv[i+1]);

			PARSE2;
		}

		/* Axiton tracker */

		if (!strcmp(argv[i], "--axitontracker"))
		{
			icdatst.axtinfo.th_threshold = -1.;   //default
			icdatst.axtinfo.ve_threshold = -1.;   //default
			icdatst.axtinfo.ct_threshold = -1.;   //default
			icdatst.axtinfo.printradius  = 1;     //default
			icdatst.axtinfo.gradients    = false; //default

			char ff[] = "-";
			if (argv[i+1][0] == ff[0]){
				icdatst.axtinfo.nMax = -2; //default
				PARSE1;
			}
			else {
				sscanf(argv[i+1], "%zu", &icdatst.axtinfo.nMax);
				PARSE2;
			}

		}

		if (!strcmp(argv[i], "--axitontracker.th_threshold"))
		{
			// sscanf(argv[i+1], "%zu", &icdatst.axtinfo.th_threshold);
      icdatst.axtinfo.th_threshold = atof(argv[i+1]);
			PARSE2;
		}

		if (!strcmp(argv[i], "--axitontracker.vh_threshold"))
		{
			// sscanf(argv[i+1], "%zu", &icdatst.axtinfo.ve_threshold);
      icdatst.axtinfo.ve_threshold = atof(argv[i+1]);
			PARSE2;
		}

    if (!strcmp(argv[i], "--axitontracker.con_threshold"))
		{
			// sscanf(argv[i+1], "%zu", &icdatst.axtinfo.con_threshold);
      icdatst.axtinfo.con_threshold = atof(argv[i+1]);
			PARSE2;
		}

		if (!strcmp(argv[i], "--axitontracker.ct_threshold"))
		{
			sscanf(argv[i+1], "%zu", &icdatst.axtinfo.ct_threshold);
      icdatst.axtinfo.ct_threshold = atof(argv[i+1]);
			PARSE2;
		}

		if (!strcmp(argv[i], "--axitontracker.printradius"))
		{
			sscanf(argv[i+1], "%zu", &icdatst.axtinfo.printradius);
			PARSE2;
		}

		if (!strcmp(argv[i], "--axitontracker.gradients"))
		{
			icdatst.axtinfo.gradients = true;
			PARSE2;
		}

		/* IC's*/

		if (!strcmp(argv[i], "--kmax"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need an integer value for the maximum momentum.\n");
				exit(1);
			}

			sscanf(argv[i+1], "%zu", &kMax); //legacy
			sscanf(argv[i+1], "%zu", &icdatst.kMax);

			if (kMax < 0)
			{
				printf("Error: the maximum momentum must be equal or greater than zero.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--sIter"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a number of iterations for the smoothing.\n");
				exit(1);
			}

			sscanf(argv[i+1], "%zu", &iter);
			sscanf(argv[i+1], "%zu", &icdatst.siter);

			if (iter < 0)
			{
				printf("Error: Number of iterations must be equal or greater than zero.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--index"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need an index for the file.\n");
				exit(1);
			}

			fIndex = atoi(argv[i+1]);

			if (fIndex < 0)
			{
				printf("Error: Filename index must be equal or greater than zero.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--ctype"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the configuration type (smooth/kmax/tkachev/thermal/lola/cole...).\n");
				exit(1);
			}

			if (!strcmp(argv[i+1], "smooth"))
			{
				cType = CONF_SMOOTH; // legacy
				icdatst.cType = CONF_SMOOTH;
			}
			else if (!strcmp(argv[i+1], "kmax"))
			{
				cType = CONF_KMAX; // legacy
				icdatst.cType = CONF_KMAX;
			}
			else if (!strcmp(argv[i+1], "vilgor"))
			{
				cType = CONF_VILGOR; // legacy
				icdatst.cType =  CONF_VILGOR;
			}
			else if (!strcmp(argv[i+1], "vilgork"))
			{
				cType = CONF_VILGORK; // legacy
				icdatst.cType =  CONF_VILGORK;
			}
			else if (!strcmp(argv[i+1], "vilgors"))
			{
				cType = CONF_VILGORS; // legacy
				icdatst.cType =  CONF_VILGORS;
			}
			else if (!strcmp(argv[i+1], "tkachev"))
			{
				cType = CONF_TKACHEV; // legacy
				icdatst.cType = CONF_TKACHEV;
			}
      else if (!strcmp(argv[i+1], "thermal"))
			{
				icdatst.cType = CONF_THERMAL;
			}
			else if (!strcmp(argv[i+1], "lola"))
			{
				cType = CONF_LOLA; // legacy
				icdatst.cType =  CONF_LOLA;
			}
			else if (!strcmp(argv[i+1], "cole"))
			{
				cType = CONF_COLE; // legacy
				icdatst.cType =  CONF_COLE;
			}
			else if (!strcmp(argv[i+1], "spax"))
			{
				cType = CONF_SPAX; // legacy
				icdatst.cType =  CONF_SPAX;
			}
			else if (!strcmp(argv[i+1], "spsax"))
			{
				cType = CONF_SPAX; // legacy
				icdatst.cType =  CONF_SPAX;
			}
      else if (!strcmp(argv[i+1], "string"))
			{
				icdatst.cType = CONF_STRING;
			}
			else
			{
				printf("Error: Unrecognized configuration type %s\n", argv[i+1]);
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--smvar"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the configuration type (stXY/stYZ/mc0/mc).\n");
				exit(1);
			}

			smvarType = CONF_RAND ;//legacy
			icdatst.smvarType = CONF_RAND ;

			if (!strcmp(argv[i+1], "stXY"))
			{
				smvarType = CONF_STRINGXY; //legacy
				icdatst.smvarType = CONF_STRINGXY;
			}
			else if (!strcmp(argv[i+1], "stYZ"))
			{
				smvarType = CONF_STRINGYZ; //legacy
				icdatst.smvarType = CONF_STRINGYZ;
			}
			else if (!strcmp(argv[i+1], "mc0"))
			{
				smvarType = CONF_MINICLUSTER0; //legacy
				icdatst.smvarType = CONF_MINICLUSTER0;;
			}
			else if (!strcmp(argv[i+1], "mc"))
			{
				smvarType = CONF_MINICLUSTER; //legacy
				icdatst.smvarType = CONF_MINICLUSTER;
			}
			else if (!strcmp(argv[i+1], "axnoise"))
			{
				smvarType = CONF_AXNOISE; //legacy
				icdatst.smvarType = CONF_AXNOISE;
			}
			else if (!strcmp(argv[i+1], "saxnoise"))
			{
				smvarType = CONF_SAXNOISE; //legacy
				icdatst.smvarType = CONF_SAXNOISE;
			}
			else if (!strcmp(argv[i+1], "ax1mode"))
			{
				smvarType = CONF_AX1MODE; //legacy
				icdatst.smvarType = CONF_AX1MODE;
			}
			else if (!strcmp(argv[i+1], "parres"))
			{
				icdatst.smvarType = CONF_PARRES;
			}
			else if (!strcmp(argv[i+1], "stringwave"))
			{
				icdatst.smvarType = CONF_STRWAVE;
			}
			else if (!strcmp(argv[i+1], "axiton"))
			{
				icdatst.smvarType = CONF_AXITON;
			}
			else
			{
				printf("Error: Unrecognized configuration type %s, using [random]\n", argv[i+1]);
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--gadtype"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the gadget configuration type (gad/gadvel/gadmass/gadmassvel/gadgrid).\n");
				exit(1);
			}

			if (!strcmp(argv[i+1], "halo"))
			{
				gadType = HALO;
			}
			else if (!strcmp(argv[i+1], "void"))
			{
				gadType = VOID;
			}

			PARSE2;
		}


		if (!strcmp(argv[i], "--prec"))
		{
			uPrec = true;

			if (i+1 == argc)
			{
				printf("Error: I need a value for the precision (double/single/mixed).\n");
				exit(1);
			}

			if (!strcmp(argv[i+1], "double"))
			{
				sPrec = FIELD_DOUBLE;
			}
			else if (!strcmp(argv[i+1], "single"))
			{
				sPrec = FIELD_SINGLE;
			}
			else
			{
				printf("Error: Unrecognized precision %s\n", argv[i+1]);
				exit(1);
			}

			PARSE2;
		}


		if (!strcmp(argv[i], "--prop"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a propagator class (leap/rkn4/om2/om4).\n");
				exit(1);
			}

			if (!strcmp(argv[i+1], "leap"))
			{
				pType |= PROP_LEAP;
			}
			else if (!strcmp(argv[i+1], "mleap"))
			{
				pType |= PROP_MLEAP;
			}
			else if (!strcmp(argv[i+1], "rkn4"))
			{
				pType |= PROP_RKN4;
			}
			else if (!strcmp(argv[i+1], "om2"))
			{
				pType |= PROP_OMELYAN2;
			}
			else if (!strcmp(argv[i+1], "om4"))
			{
				pType |= PROP_OMELYAN4;
			}
			else
			{
				printf("Error: unrecognized propagator %s\n", argv[i+1]);
				exit(1);
			}

			if (spectral)
				pType |= PROP_SPEC;

			if (fpectral)
				pType |= PROP_FSPEC;

			PARSE2;
		}

		if (!strcmp(argv[i], "--cax"))
		{
			aMod = true;

			PARSE1;
		}

		if (!strcmp(argv[i], "--spec"))
		{
			spectral = true;

			pType |= PROP_SPEC;

			PARSE1;
		}

		if (!strcmp(argv[i], "--fspec"))
		{
			fpectral = true;

			pType |= PROP_FSPEC;

			PARSE1;
		}

		if (!strcmp(argv[i], "--restart"))
		{
			restart_flag = true;

			PARSE1;
		}


		if (!strcmp(argv[i], "--preprop"))
		{
			preprop = true ; //legacy
			icdatst.preprop = true ;
			PARSE1;
		}

		if (!strcmp(argv[i], "--icstudy"))
		{
			icstudy = true ; //legacy
			icdatst.icstudy = true ;
			PARSE1;
		}

		if (!strcmp(argv[i], "--prepstL"))
		{
			printf("Warning: th prepstL parameter does nothing at the moment.\n");
			if (i+1 == argc)
			{
				printf("Error: I need a value for the Scaling limit.\n");
				exit(1);
			}

			prepstL = atof(argv[i+1]); //legacy
			icdatst.prepstL = atof(argv[i+1]);

			if (icdatst.prepstL <= 0.)
			{
				printf("Error: Scaling limit must be a positive number.\n");
				exit(1);
			}

			PARSE2;
		}

		if (!strcmp(argv[i], "--prepcoe"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the prepropagator time coefficient.\n");
				exit(1);
			}

			prepcoe = atof(argv[i+1]); //legacy
      icdatst.prepcoe = atof(argv[i+1]);

			if (icdatst.prepcoe <= 1.)
			{
				printf("Error: The prepropagator time coefficient must be larger than 1.\n");
				exit(1);
			}

			PARSE2;
		}

    if (!strcmp(argv[i], "--ng0calib"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for ng0calib.\n");
				exit(1);
			}
			ng0calib = atof(argv[i+1]);

			PARSE2;
		}

		if (!strcmp(argv[i], "--cummask"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for cummask.\n");
				exit(1);
			}
			cumumas = atoi(argv[i+1]);

			PARSE2;
		}


		if (!strcmp(argv[i], "--measlistlog"))
		{
			CreateLogMeas = true;

			PARSE1;
		}

		if (!strcmp(argv[i], "--lap"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a number of neighbours.\n");
				exit(1);
			}

			icdatst.Nghost = atoi(argv[i+1]);
			Nng = icdatst.Nghost;
			bopt = true;
			pType |= PROP_BASE;

 			if (icdatst.Nghost < 0 || icdatst.Nghost > 6)
			{
				printf("Error: The number of laplacian neighbours must be 0,1,2,3,4 or 5\n");
				exit(1);
			}

			PARSE2;
		}

		endFor:

		if (!passed)
		{
			PrintUsage(argv[0]);
			printf("\n\nUnrecognized option %s\n", argv[i]);
			exit(1);
		}

	}

// 	if (Nng*2 > (int) sizeZ) {
// 		printf("Error: current limitation for number of neighbours for the laplacian is Depth/2 (Nng%d,sizeZ%d,%d,%d)\n",Nng,sizeZ,Nng*2, Nng*2> sizeZ);
// 		printf("Error: If you are reading from a file, this exit might not be correct. Check it!\n");
// 		exit(1);
// }

//obsolete!
if (icdatst.cType == CONF_SMOOTH )
	{
		parm1 = iter;
		parm2 = alpha;
	} else if ((icdatst.cType == CONF_KMAX) || (icdatst.cType == CONF_TKACHEV)) {
		parm1 = kMax;
		parm2 = kCrit;
	}
	else if ((icdatst.cType == CONF_VILGOR) || (icdatst.cType == CONF_VILGORK) || (icdatst.cType == CONF_VILGORS)) {
		parm1 = iter;	 // here taken as a flag to randomised the
		parm2 = kCrit; // here taken as multiplicative factor for nN3
	}

	if ((pType & PROP_MASK) == PROP_NONE)
		pType |= PROP_RKN4;

// if lapla is chosen
	if (Nng > 0)
	{
		if ( ((pType & PROP_LAPMASK) & PROP_FSPEC) || ((pType & PROP_LAPMASK) & PROP_SPEC))
		{
			printf("Error: Selected spectral propagator and Nng=%d\n",Nng);
			exit(1);
		}
	}

	// if no laplacian type is been chosen chose optimised BASE
	if ((pType & PROP_LAPMASK) == PROP_NONE)
		pType |= PROP_BASE;



// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// % if msa is used FIX lambda
// % if logi is given FIX zi
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/* Adjust LL depending on (lz2e, frw) such that msa = given value at:
    - end of the simulation if lz2 < 2 (because msa decreases)
    - beggining of the simulation if lz2 <= 2 (because msa increases)

        physical mass^2 =  0.5 L1/R^lz2e
        msa^2 = 2 L1/R^lz2e * (L R/N)^2

   therefore (recall that R = eta^frw )

        LL = 0.5 * msa^2 R^(frw(lz2e-2)) (N/L)^2

   Adjust zi of initial conditions if logi is given
      log is usually log ms/H, but it is not well defined for frw = 0
      replace the definition with log (ms d_H) where
        dH = R * eta = eta^(frw+1) where dt = R deta
        log = log (ms/H) = 0.5 log (ms^2/H^2) = 0.5 log (2 lamda/R^lz2e/H^2)
          H = R'/R^2 = frw*R/eta/R^2 = frw/eta R = frw/eta^(1+frw)
          logi = 0.5 log (2 lambda1/eta^(lz2e*frw) eta^(2+2frw)/frw^2) = 0.5 log (2 lambda1/frw^2 * eta^(2+2frw - frw*lz2e))

   therefore

        zi = (exp(2 logi) (frw^2 /2 LL))^(1/(2+frw(2-lz2e)))

   **WARNING**
      logi is not a proxy of time if ms dH = constant, which happens for (2L/R^lze) (R eta)^2 = eta^(2 + frw(2-lz2e)) = constant
      i.e. when 2 -frw(2-lz2e) = 0 -> frw = 2/(lz2e-2)
        (for instance for lze2 = 4 if frw=1)
      in that case, we cannot adjust eta1 to satisfy a certain logi,
      logi can only be adjusted by LL, but we can use msa to determine zi

      in this special case ...

      Conclusions (on paper)
      if using logi
        if lze2<=2
          LL = (msa)^2 / 2 delta^2 1/eta_f^(frw(2-lz2e))
          etai   = [exp(2 logi)/2LL]^(1/(2+frw(2-lze2)))
        else
        if lze2<=2+2/frw
          etai   = exp(logi) delta /msa
          LL = (msa)^2 / 2 delta^2 1/eta_i^(frw(2-lz2e))
      else if using zi
      if lze2<=2
        LL = (msa)^2 / 2 delta^2 1/eta_f^(frw(2-lz2e))
        logi   = 0.5 log (2 LL/frw^2 * etai^(2+2frw - frw*lz2e))
      else
      if lze2<=2+2/frw
        LL = (msa)^2 / 2 delta^2 1/eta_i^(frw(2-lz2e))
        logi   = 0.5 log (2 LL/frw^2 * etai^(2+2frw - frw*lz2e))
   */

	if (uMsa) {
    lType = LAMBDA_Z2;

    if (uLambda)
			printf("Error: Conflicting options --llcf and --msa. Using msa\n");

    if (uLogi){
      if (lz2e <= 2.0){
        LL         = msa*msa / (2.0*sizeL*sizeL/(sizeN*sizeN) * pow(zFinl,frw*(2.0-lz2e)) );
        icdatst.zi = pow(exp(2*icdatst.logi)/(2*LL),1/(2.0+frw*(2-lz2e)));
        LogMsg(VERB_NORMAL,"[parse] Adjusting LL = %f, zi = %f from zf = %f logi = %f msa = %.f",LL,icdatst.zi,zFinl,icdatst.logi,msa);
      }
      else if (lz2e*frw <= 2*(1+frw)){
        icdatst.zi = exp(icdatst.logi) * sizeL/sizeN / msa;
        LL         = msa*msa / (2.0*sizeL*sizeL/(sizeN*sizeN) * pow(icdatst.zi,frw*(2.0-lz2e)) );
        LogMsg(VERB_NORMAL,"[parse] Adjusting LL = %f, zi = %f from logi = %f msa = %.f",LL,icdatst.zi,icdatst.logi,msa);
      }
      else {
        LogMsg(VERB_NORMAL,"[parse] Core increasing so fast that log decreases with time !");
        icdatst.zi = exp(icdatst.logi) * sizeL/sizeN / msa;
        LL         = msa*msa / (2.0*sizeL*sizeL/(sizeN*sizeN) * pow(icdatst.zi,frw*(2.0-lz2e)) );
        LogMsg(VERB_NORMAL,"[parse] Adjusting LL = %f, zi = %f from logi = %f msa = %.f",LL,icdatst.zi,icdatst.logi,msa);
      }
    } // end use logi
    else // using zi
    {
      if (lz2e <= 2.0){
        LL           = msa*msa / (2.0*sizeL*sizeL/(sizeN*sizeN) * pow(zFinl,frw*(2.0-lz2e)) );
        icdatst.logi = 0.5 * log(2*LL * pow(icdatst.zi,(2+2*frw - frw*lz2e)));
        LogMsg(VERB_NORMAL,"[parse] Adjusting LL = %f, logi = %f from zf = %f zi = %f msa = %.f",LL,icdatst.logi,zFinl,icdatst.zi,msa);
      }
      else if (lz2e*frw <= 2*(1+frw)){
        LL         = msa*msa / (2.0*sizeL*sizeL/(sizeN*sizeN) * pow(icdatst.zi,frw*(2.0-lz2e)) );
        icdatst.logi = 0.5 * log(2*LL * pow(icdatst.zi,(2+2*frw - frw*lz2e)));
        LogMsg(VERB_NORMAL,"[parse] Adjusting LL = %f, logi = %f from zi = %f msa = %.f",LL,icdatst.logi,icdatst.zi,msa);
      }
      else {
        LogMsg(VERB_NORMAL,"[parse] Core increasing so fast that log decreases with time !");
        LL         = msa*msa / (2.0*sizeL*sizeL/(sizeN*sizeN) * pow(icdatst.zi,frw*(2.0-lz2e)) );
        icdatst.logi = 0.5 * log(2*LL * pow(icdatst.zi,(2+2*frw - frw*lz2e)));
        LogMsg(VERB_NORMAL,"[parse] Adjusting LL = %f, logi = %f from zi = %f msa = %.f",LL,icdatst.logi,icdatst.zi,msa);
      }
    } // end zi case
  }
  else // using lambda from command line, interpreted as L1 // user is responsible for getting the correct zi, logi...
  {
    /* In principle we could adjust zi, from logi... */
    if (uLogi){
      icdatst.zi = pow(exp(2*icdatst.logi)/(2*LL),1/(2.0+frw*(2-lz2e)));
      LogMsg(VERB_NORMAL,"[parse] Adjusting zi = %f from logi = %f",icdatst.zi,icdatst.logi);
    }
    msa = sqrt(2*LL)*sizeL/sizeN * pow(icdatst.zi,frw*(1.0-lz2e/2.0));
	}
  zInit = icdatst.zi; //legacy


	vqcdType |= vpqType;
 	vqcdType |= (vqcdTypeDamp | vqcdTypeEvol);


	if (zrestore < zthres) {
		printf("Warning: zrestore = %f < zthres %f. Switch-off disabled.\n", zrestore, zthres);
		zthres = 1.e+20;
		zrestore = 1.e+20;
	}

	/* make sure that endredmap endredmapwkb makes sense */
	{
		int siN = (int) sizeN;
		if (endredmap > siN){
			printf("[Error:1] Reduced map dimensions (%d) set to %d\n ", endredmap,siN);
			endredmap = siN;
		}

		if (endredmapwkb > siN){
			printf("[Error:1] Reduced wkb map dimensions (%d) set to %d\n ", endredmap,siN);
			endredmapwkb = siN;
		}
	}
	/*	Set the output directory, according to an environmental variable	*/

	if (const char *outPath = std::getenv("AXIONS_OUTPUT")) {
		if (strlen(outPath) < 1022) {
			struct stat tStat;
			if (stat(outPath, &tStat) == 0 && S_ISDIR(tStat.st_mode)) {
				strcpy(outDir, outPath);
			} else {
				printf("Path %s doesn't exist, using default\n", outPath);
				mCreateOut = true;
			}
		}
	} else {
		mCreateOut = true;
	}

	/*	Set the directory where the FFTW wisdom is/will be stored		*/

	if (const char *wisPath = std::getenv("AXIONS_WISDOM")) {
		if (strlen(wisPath) < 1022) {
			struct stat tStat;
			if (stat(wisPath, &tStat) == 0 && S_ISDIR(tStat.st_mode)) {
				strcpy(wisDir, wisPath);
			} else {
				printf("Path %s doesn't exist, using default\n", wisPath);
			}
		}
	}

	/*	Remove stop files if present	*/

	FILE *capa = nullptr;
	if (!((capa  = fopen("./stop", "r")) == nullptr)) {
		fclose (capa);
		printf("Stop file detected! ... ");
		if( remove( "./stop" ) != 0 ){
			printf("and cannot be deleted. Exit!\n");
			exit(1);
		} else
			printf("and deleted!\n ");
	}
	if (!((capa  = fopen("./abort", "r")) == nullptr)) {
		fclose (capa);
		printf("Abort file detected! ... ");
		if( remove( "./abort" ) != 0 ){
			printf("and cannot be deleted. Exit!\n");
			exit(1);
		} else
			printf("and deleted!\n ");
	}

	/*	Create measfile.dat if required	*/
	if (CreateLogMeas)
		createMeasLogList();

	if (zGrid == 1)
		logMpi = ZERO_RANK;



  icdatst.fType = fTypeP;

	//- information needs to be passed onto measurement files
 	deninfa.sliceprint = slicepp;
	deninfa.index = 0;
	deninfa.redmap = endredmap;
	deninfa.measCPU  = measCPU;
	deninfa.cTimesec = 0.0;
	deninfa.propstep = 0;
	deninfa.cummask = cumumas;

	// default measurement type is parsed
	deninfa.measdata = defaultmeasType;
	deninfa.strmeas = strmeas;
	/* if measdata included spectra but no mask was speficied, set FLAT*/
	// if ( (deninfa.measdata & ( MEAS_SPECTRUM )) && (spmask == SPMASK_NONE) )
	// 	spmask = SPMASK_FLAT;
	deninfa.mask = spmask;
	deninfa.rmask = rmask;


	std::sort(rmask_tab.begin(), rmask_tab.end());
 	auto last = std::unique(rmask_tab.begin(), rmask_tab.end());
 	rmask_tab.erase(last, rmask_tab.end());
	deninfa.i_rmask = rmask_tab.size();
	deninfa.rmask_tab = rmask_tab;
	deninfa.maty = maty;
	/* if measdata included spectra but no KGV specified, set FLAT*/
	// if ( (deninfa.measdata & ( MEAS_SPECTRUM )) && (nrt == NRUN_NONE) )
	// 	nrt = NRUN_KGV;
	deninfa.nrt = nrt;
  deninfa.maskenergyonly = maskenergyonly;

	LogMsg(VERB_HIGH,"Parse Jaxions completed! %d parsed args",procArgs);
	return	procArgs;
}


Cosmos	createCosmos()
{
	Cosmos myCosmos;

	/* Initial condition data is saved in cosmos; potential problems when reading confs?*/
	myCosmos.SetICData(icdatst);

	/*	I'm reading from disk	*/
	if (fIndex >= 0) {

		if (uMsa || uLambda)
			myCosmos.SetLambda(LL);

		if (uMsa || uLambda)
			myCosmos.SetLamZ2Exp( (lType == LAMBDA_Z2) ? lz2e : 0.0);

		if (uQcd)
			myCosmos.SetQcdExp(nQcd);

		if (uGamma)
			myCosmos.SetGamma(gammo);

		if (uGamExp)
			myCosmos.SetDecTime(dectime);

		if (uPot)
			myCosmos.SetQcdPot(vqcdType);

		if (uSize)
			myCosmos.SetPhysSize(sizeL);

		if (uZth)
			myCosmos.SetZThRes  (zthres);

		if (uZrs)
			myCosmos.SetZRestore(zrestore);

		if (uI3)
			myCosmos.SetIndi3(indi3);

		if (uFR)
			myCosmos.SetFrw(frw);

		if (uMI)
			myCosmos.SetMink(mink);

		if (uexCosm)
			myCosmos.SetUeC(uexCosm);

		if (ufA)
			myCosmos.SetFA(fA);

	} else {
		myCosmos.SetLambda  (LL);
		myCosmos.SetLamZ2Exp( (lType == LAMBDA_Z2) ? lz2e : 0.0);
		myCosmos.SetQcdExp  (nQcd);
		myCosmos.SetGamma   (gammo);
		myCosmos.SetDecTime   (dectime);
		myCosmos.SetQcdPot  (vqcdType);
		myCosmos.SetPhysSize(sizeL);
		myCosmos.SetZThRes  (zthres);
		myCosmos.SetZRestore(zrestore);
		myCosmos.SetIndi3   (indi3);
		myCosmos.SetFrw     (frw);
		myCosmos.SetMink    (mink);
		myCosmos.SetUeC     (uexCosm);
		myCosmos.SetFA      (fA);
	}

	return	myCosmos;
}
