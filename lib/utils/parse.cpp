#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <sys/stat.h>

#include "enum-field.h"
#include "cosmos/cosmos.h"

size_t sizeN  = 128;
size_t sizeZ  = 128;
int    zGrid  = 1;
int    nSteps = 5;
int    dump   = 100;
double nQcd   = 7.0;
//JAVIER
int    Ng     = 1 ;
double indi3  = 1.0;
double msa    = 1.5;
double wDz    = 0.8;
int    fIndex = -1;

double sizeL = 4.;
double zInit = 0.5;
double zFinl = 1.0;
double kCrit = 1.0;
//JAVIER
double mode0 = 10.0;
double alpha = 0.143;
double zthres   = 1000.0;
double zrestore = 1000.0;
double LL = 25000.;
double parm2 = 0.;
double pregammo = 0.;
double gammo = 0.;
double p3DthresholdMB = 1.e+6;
double wkb2z  = -1.0;
double prepstL = 5.0 ;
double prepcoe = 3.0 ;
int endredmap = -1;
int safest0   = 20;
size_t nstrings_globale ;

bool lowmem   = false;
bool uPrec    = false;
bool uSize    = false;
bool uQcd     = false;
bool uLambda  = false;
bool uMsa     = false;
bool uI3      = false;
bool uPot     = false;
bool uGamma   = false;
bool uZth     = false;
bool uZrs     = false;
bool uZin     = false;
bool uZfn     = false;
bool spectral = false;
bool aMod     = false;
bool icstudy  = false ;
bool preprop  = false ;

size_t kMax  = 2;
size_t iter  = 0;
size_t parm1 = 0;
size_t wTime = std::numeric_limits<std::size_t>::max();

PropType     pType     = PROP_NONE;
ConfType     cType     = CONF_NONE;
ConfsubType  smvarType = CONF_RAND;
FieldType    fTypeP    = FIELD_SAXION;
LambdaType   lType     = LAMBDA_FIXED;
VqcdType     vqcdType  = VQCD_1;
VqcdType     vqcdTypeDamp    = VQCD_NONE;
VqcdType     vqcdTypeRhoevol = VQCD_NONE;

char outName[128] = "axion\0";
char outDir[1024] = "out/m\0";
char wisDir[1024] = "./\0";

FieldPrecision	sPrec  = FIELD_SINGLE;
DeviceType	cDev   = DEV_CPU;

VerbosityLevel	verb   = VERB_NORMAL;
LogMpi		logMpi = ALL_RANKS;

PrintConf prinoconfo  = PRINTCONF_NONE;
bool p2dmapo  	  = false;
bool p3dstrings	  = false;
bool p3dwalls	  = false;
bool pconfinal 	  = false;
bool pconfinalwkb = false;
bool restart_flag = false;

bool mCreateOut = false;

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
	printf("  --lowmem                      Reduces memory usage by 33\%, but decreases performance as well (default false).\n");
	printf("  --prec  double/single         Precision of the axion field simulation (default single)\n");
	printf("  --device cpu/gpu              Uses nVidia Gpus to accelerate the computations (default, use cpu).\n");
	printf("  --prop  leap/rkn4/om2/om4     Numerical propagator to be used for molecular dynamics (default, use rkn4).\n");
	printf("  --steps [int]                 Number of steps of the simulation (default 500).\n");
	printf("  --spec                        Enables the spectral propagator for the laplacian (default, disabled).\n");
	printf("  --wDz   [float]               Adaptive time step dz = wDz/frequency [l/raxion3D].\n");
	printf("  --sst0  [int]                 # steps (Saxion mode) after str=0 before switching to theta [l/raxion3D].\n");
	printf("  --restart                     searches for out/m/axion.restart and continues a simulation... needs same input parameters!.\n");

	printf("\nPhysical parameters:\n");
	printf("  --ftype saxion/axion          Type of field to be simulated, either saxion + axion or lone axion (default saxion, not parsed yet).\n");
	printf("  --cax                         Uses a compact axion ranging from -pi to pi (default, the axion is non-compact).\n");
	printf("  --zi    [float]               Initial value of the redshift (default 0.5).\n");
	printf("  --zf    [float]               Final value of the redshift (default 1.0).\n");
	printf("  --lsize [float]               Physical size of the system (default 4.0).\n");
	printf("  --qcd   [float]               Exponent of topological susceptibility (default 7).\n");
	printf("  --llcf  [float]               Lagrangian coefficient (default 15000).\n");
	printf("  --msa   [float]               Spacing to core ratio (Moore parameter) [l/raxion3D].\n");
	printf("  --ind3  [float]               Factor multiplying axion mass^2 (default, 1).\n");
	printf("  --vqcd2                       Variant of QCD potential (default, disabled).\n");
	printf("  --vPQ2                        Variant of PQ potential (default, disabled).\n");


	printf("\nInitial conditions:\n");
	printf("  --icinfo                      Prints more info about initial conditions.\n");
	printf("  --ctype smooth/kmax/tkachev   Initial configuration, either with smoothing or with FFT and a maximum momentum\n");
	printf("  --smvar stXY/stYZ/mc0/mc/...  [smooth variants] string, mc's, pure mode, noise... initial conditions.\n");
	printf("\n");
	printf("  --kmax  [int]                 Maximum momentum squared for the generation of the configuration with --ctype kmax/tkachev (default 2)\n");
	printf("  --kcr   [float]               kritical kappa (default 1.0).\n");
	printf("  --mode0 [float]               Value of axion zero mode [rad] (default random).\n");
	printf("\n");
	printf("  --sIter [int]                 Number of smoothing steps for the generation of the configuration with --ctype smooth (default 40)\n");
	printf("  --alpha [float]               alpha parameter for the smoothing (default 0.143).\n");
	printf("  --wkb   [float]               WKB's the final AXION configuration until specified time [l/raxion3D] (default no).\n");
	printf("\n");
	printf("  --index [idx]                 Loads HDF5 file at out/dump as initial conditions (default, don't load).\n");

	printf("\nPrepropagator:\n");
	printf("  --preprop                     turns on prepropagator -propagates IC-RHO-damping-fixed zi- for zi or given scaling stL (default yes).\n");
	printf("  --prepcoe [float]             prepropagator starts at zi/prepcoe (default 3.0).\n");
	printf("  --prepstL [float]             string length/volume at which prepropagator stops [raxion3d?] (default 5).\n");
	printf("  --icstudy                     Allows printing maps/energy during preprop (default no).\n");

	printf("\nOutput:\n");
	printf("--name  [filename]              Uses filename to name the output files in out/dump, instead of the default \"axion\"\n");
	printf("--dump  [int]                   frequency of the output (default 100).\n");
	printf("--p3D 0/1/2/3                   Print initial/final configurations (default 0 = no) 1=initial 2=final 3=both \n");
	printf("--wTime [float]                 Simulates during approx. [float] hours and then writes the configuration to disk.\n");
	printf("--p2Dmap                        Include 2D maps in axion.m.files (default no)\n");
	printf("--p3Dstr  [Mb]                  Include 3D string/Wall maps axion.m.files always or if expected size below [Mbs] (default no)\n");
	printf("--pcon                          Include 3D contrastmap in final axion.m.  (default no)\n");
	printf("--pconwkb                       Include 3D contrastmap in final wkb axion.m. (default yes)\n");
	printf("--redmp [fint]                  Reduces final density map to [specified n]**3 [l/raxion3D] (default NO or 256 if int not specified).\n");
	printf("                                Includes reduced 3D contrast maps if possible and in final axion.m.file\n");

	printf("\nLogging:\n");
	printf("--verbose 0/1/2                 Choose verbosity level 0 = silent, 1 = normal (default), 2 = high.\n\n");
	printf("--nologmpi                      Disable logging over MPI so only rank 0 logs (default, all ranks log)\n\n");
	printf("--help                          Prints this message.\n");

	//printf("--lapla 0/1/2/3/4               Number of Neighbours in the laplacian [only for simple3D] \n");

	return;
}

void	PrintICoptions()
{
	printf("\Options for Initial conditions\n\n");

	printf("--ctype smooth/kmax/tkachev               			   Main IC selector.\n\n");

	printf(" [smooth]                                          Point based.\n");
	printf("-----------------------------------------------------------------------------------------------\n");
	printf("  Random phase at each point (default).\n");
	printf("  --smvar [string]                                 Spectial initial distributions.\n");
	printf("  --smvar axnoise  --mode0 [float] --kcr [float]   theta = mode0+random{-1,1}*kcr.\n");
	printf("  --smvar saxnoise --mode0 [float] --kcr [float]   theta = mode0, rho = 1 + random{-1,1}*kcr.\n");
	printf("  --smvar ax1mode  --mode0 [float] --kMax[int]     theta = mode0 cos(2Pi kMax*x/N).\n\n");

	printf("  --smvar mc   --mode0 [float] --kcr [float]       theta = mode0 Exp(-kcr*(x-N/2)^2).\n");
	printf("  --smvar mc0  --mode0 [float] --kcr [float]       theta = mode0 Exp(-kcr*(x)^2).\n\n");

	printf("  --smvar stXY --mode0 [float] --kcr [float]       Circular loop in the XY plane, radius N/4.\n");
	printf("  --smvar stYZ --mode0 [float] --kcr [float]       Circular loop in the XY plane, radius N/4.\n\n");

	printf(" [kmax]                                            Saxion momentum based.\n");
	printf("-----------------------------------------------------------------------------------------------\n");
	printf("  --kmax [float] --kcr [float]                     Random modes and inverse FFT.\n");
	printf("                                                   for 3D k < kmax = min(kmax, N/2-1).\n");
	printf("                                                   mode ~ exp(I*random) * exp( -(kcr* k x)^2).\n");
	printf("  --mode0 [float]                                  mode[000] = exp(I*mode0).\n");

	printf(" [tkachev]                                         Axion momentum based.\n");
	printf("-----------------------------------------------------------------------------------------------\n");
	return;
}

int	parseArgs (int argc, char *argv[])
{
	bool	passed;
	int	procArgs = 0;

	for (int i=1; i<argc; i++)
	{
		passed = false;

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

		if (!strcmp(argv[i], "--verbose"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a verbosity level.\n");
				exit(1);
			}

			sscanf(argv[i+1], "%d", &verb);

			if (verb > VERB_HIGH)   verb = VERB_HIGH;
			if (verb < VERB_SILENT) verb = VERB_SILENT;

			i++;
			procArgs++;
			passed = true;

			goto endFor;
		}

		if (!strcmp(argv[i], "--p3D"))
		{
			if (i+1 == argc)
			{
				printf("Warning: p3D set by default to 0 [no 00000 and final configuration files].\n");
				prinoconfo = PRINTCONF_NONE ;
				procArgs++;
				passed = true;
				goto endFor;
			}

			sscanf(argv[i+1], "%d", &prinoconfo);
			//printf("p3D set to %d \n", prinoconfo);

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--p2Dmap"))
		{
			p2dmapo = true ;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--p3Dstr"))
		{
			p3dstrings = true ;

			// p3DthresholdMB=1.8e+21;
			p3DthresholdMB = atof(argv[i+1]);
			i++;

			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--p3Dwal"))
		{
			p3dwalls = true ;

			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--pcon"))
		{
			pconfinal = true ;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--pconwkb"))
		{
			pconfinalwkb = true ;
			procArgs++;
			passed = true;
			goto endFor;
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

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--vqcd2"))
		{
			uPot = true;
			vqcdType = VQCD_2 ;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--vPQ2"))
		{
			uPot = true;
			vqcdType = VQCD_1_PQ_2 ;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--onlyrho"))
		{
			uPot = true;
			vqcdTypeRhoevol = VQCD_EVOL_RHO;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--lowmem"))
		{
			lowmem = true;
			procArgs++;
			passed = true;
			goto endFor;
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

			if (endredmap == -1)
				endredmap = sizeN;

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--depth"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a size.\n");
				exit(1);
			}

			sscanf(argv[i+1], "%zu", &sizeZ);

			if (sizeZ < 2)
			{
				printf("Error: Size must be larger than 2.\n");
				exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

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
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--kcr"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the critical kappa.\n");
				exit(1);
			}

			kCrit = atof(argv[i+1]);

			if (kCrit < 0.)
			{
				printf("Error: Critical kappa must be larger than or equal to 0.\n");
				exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--alpha"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for alpha.\n");
				exit(1);
			}

			alpha = atof(argv[i+1]);

			if ((alpha < 0.) || (alpha > 1.))
			{
				printf("Error: Alpha parameter must belong to the [0,1] interval.\n");
				exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
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

			i++;
			procArgs++;
			passed = true;
			goto endFor;
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


			// if ((endredmap == sizeN))
			// {
			// 	printf("Warning: reducedN == sizeN, Gaussian filtering at most\n");
			// }
			//
			// if (endredmap < 0)
			// {
			// 	printf("Error: reducedN should be in the interval [0 < size]. Set to 256\n");
			// 	endredmap = 256	;
			// }
			//
			// if ((endredmap > sizeN))
			// {
			// 	printf("Error: reducedN should be in the interval [0 < size]. Set to sizeN\n");
			// 	endredmap = sizeN	;
			// }


			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--pregam"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the prepropagator/string destructor damping factor.\n");
				exit(1);
			}

			pregammo = atof(argv[i+1]);
			//vqcdTypeDamp = VQCD_DAMP_RHO ;

			//uPot  = true;
			//uGamma = true;

			if (pregammo < 0.)
			{
				printf("Error: pre-Damping factor should be larger than 0.\n");
				exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--gam"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the damping factor.\n");
				exit(1);
			}

			gammo = atof(argv[i+1]);
			vqcdTypeDamp = VQCD_DAMP_RHO ;

			uPot  = true;
			uGamma = true;

			if (gammo < 0.)
			{
				printf("Error: Damping factor should be larger than 0.\n");
				exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--zi"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the initial redshift.\n");
				exit(1);
			}

			zInit = atof(argv[i+1]);

			if (zInit < 0.)
			{
				printf("Error: Initial redshift must be larger than 0.\n");
				exit(1);
			}

			uZin = true;

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--zf"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the Final redshift.\n");
				exit(1);
			}

			zFinl = atof(argv[i+1]);

			if (zFinl < 0.)
			{
				printf("Error: Final redshift must be larger than 0.\n");
				exit(1);
			}

			uZfn = true;

			i++;
			procArgs++;
			passed = true;
			goto endFor;
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

			i++;
			procArgs++;
			passed = true;
			goto endFor;
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

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--zswitchOn"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the switch-off z.\n");
				exit(1);
			}

			zthres = atof(argv[i+1]);

			if (zthres <= 0.)
			{
				printf("Error: The z switch-off must happen at nonzero redshift.\n");
				exit(1);
			}

			uZth   = true;

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--zswitchOff"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the re-switch-on z.\n");
				exit(1);
			}

			zrestore = atof(argv[i+1]);

			if (zrestore <= 0.)
			{
				printf("Error: The z re-switch must happen at nonzero redshift.\n");
				exit(1);
			}

			uZrs   = true;

			i++;
			procArgs++;
			passed = true;
			goto endFor;
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
			lType = LAMBDA_Z2;

			i++;
			procArgs++;
			passed = true;
			goto endFor;
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

			if (indi3 < 0.)
			{
				printf("Error: Indi3 must be greater than or equal to zero.\n");
				exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}


		if (!strcmp(argv[i], "--wDz"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the adaptive time step.\n");
				exit(1);
			}

			wDz = atof(argv[i+1]);

			if (wDz <= 0.)
			{
				printf("Error: backwards propagation?\n");
				exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		// NEW
		if (!strcmp(argv[i], "--mode0"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the axion zero mode.\n");
				exit(1);
			}

			mode0 = atof(argv[i+1]);

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}



		if (!strcmp(argv[i], "--qcd"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need an exponent for the susceptibility nQcd!.\n");
				exit(1);
			}

			nQcd = atof(argv[i+1]);

			if (nQcd < 0)
			{
				printf("Error: The exponent of the top. susceptibility nQcd must be equal or greater than 0.\n");
				exit(1);
			}

			uQcd = true;

			i++;
			procArgs++;
			passed = true;
			goto endFor;
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

			i++;
			procArgs++;
			passed = true;
			goto endFor;
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

			i++;
			procArgs++;
			passed = true;
			goto endFor;
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

			i++;
			procArgs++;
			passed = true;
			goto endFor;
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

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--kmax"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need an integer value for the maximum momentum.\n");
				exit(1);
			}

			sscanf(argv[i+1], "%zu", &kMax);

			if (kMax < 0)
			{
				printf("Error: the maximum momentum must be equal or greater than zero.\n");
				exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--sIter"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a number of iterations for the smoothing.\n");
				exit(1);
			}

			sscanf(argv[i+1], "%zu", &iter);

			if (iter < 0)
			{
				printf("Error: Number of iterations must be equal or greater than zero.\n");
				exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
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

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--ctype"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the configuration type (smooth/kmax/tkachev).\n");
				exit(1);
			}

			if (!strcmp(argv[i+1], "smooth"))
			{
				cType = CONF_SMOOTH;
			}
			else if (!strcmp(argv[i+1], "kmax"))
			{
				cType = CONF_KMAX;
			}
			else if (!strcmp(argv[i+1], "tkachev"))
			{
				cType = CONF_TKACHEV;
			}
			else
			{
				printf("Error: Unrecognized configuration type %s\n", argv[i+1]);
				exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--smvar"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the configuration type (stXY/stYZ/mc0/mc).\n");
				exit(1);
			}

			smvarType = CONF_RAND ;

			if (!strcmp(argv[i+1], "stXY"))
			{
				smvarType = CONF_STRINGXY;
			}
			else if (!strcmp(argv[i+1], "stYZ"))
			{
				smvarType = CONF_STRINGYZ;
			}
			else if (!strcmp(argv[i+1], "mc0"))
			{
				smvarType = CONF_MINICLUSTER0;
			}
			else if (!strcmp(argv[i+1], "mc"))
			{
				smvarType = CONF_MINICLUSTER;
			}
			else if (!strcmp(argv[i+1], "axnoise"))
			{
				smvarType = CONF_AXNOISE;
			}
			else if (!strcmp(argv[i+1], "saxnoise"))
			{
				smvarType = CONF_SAXNOISE;
			}
			else if (!strcmp(argv[i+1], "ax1mode"))
			{
				smvarType = CONF_AX1MODE;
			}
			else
			{
				printf("Error: Unrecognized configuration type %s, using [random]\n", argv[i+1]);
				exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
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

			i++;
			procArgs++;
			passed = true;
			goto endFor;
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
			passed = true;
			goto endFor;
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

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--cax"))
		{
			aMod = true;

			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--spec"))
		{
			spectral = true;

			pType |= PROP_SPEC;

			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--restart"))
		{
			restart_flag = true;

			procArgs++;
			passed = true;
			goto endFor;
		}


		if (!strcmp(argv[i], "--preprop"))
		{
			preprop = true ;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--icstudy"))
		{
			icstudy = true ;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--prepstL"))
		{
			printf("Warning: th prepstL parameter does nothing at the moment.\n");
			if (i+1 == argc)
			{
				printf("Error: I need a value for the Scaling limit.\n");
				exit(1);
			}

			prepstL = atof(argv[i+1]);

			if (prepstL <= 0.)
			{
				printf("Error: Scaling limit must be a positive number.\n");
				exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--prepcoe"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a value for the prepropagator time coefficient.\n");
				exit(1);
			}

			prepcoe = atof(argv[i+1]);

			if (prepcoe <= 1.)
			{
				printf("Error: The prepropagator time coefficient must be larger than 1.\n");
				exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		if (!strcmp(argv[i], "--nologmpi"))
		{
			logMpi = ZERO_RANK;

			procArgs++;
			passed = true;
			goto endFor;
		}
		//JAVIER added gradient
		if (!strcmp(argv[i], "--lapla"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a number of neighbours.\n");
				exit(1);
			}

			Ng = atoi(argv[i+1]);

			if (Ng < 0 || Ng > 4 )
			{
				printf("Error: The number of neighbours must be 0,1,2,3. Set to 1.\n");
				//exit(1);
			}

			i++;
			procArgs++;
			passed = true;
			goto endFor;
		}

		endFor:

		if (!passed)
		{
			PrintUsage(argv[0]);
			printf("\n\nUnrecognized option %s\n", argv[i]);
			exit(1);
		}

	}

	if (cType == CONF_SMOOTH)
	{
		parm1 = iter;
		parm2 = alpha;
	} else if (cType == CONF_KMAX) {
		parm1 = kMax;
		parm2 = kCrit;
	}

	if ((pType & PROP_MASK) == PROP_NONE)
		pType |= PROP_RKN4;

	if (uMsa) {
		if (uLambda)
			printf("Error: Conflicting options --llcf and --msa. Using msa\n");

		double tmp = (msa*sizeN)/sizeL;

		LL    = 0.5*tmp*tmp;
		lType = LAMBDA_Z2;
	} else {
		double tmp = sizeL/sizeN;

		msa = sqrt(2*LL)*tmp;
	}

 	vqcdType |= (vqcdTypeDamp | vqcdTypeRhoevol);

	if (zrestore < zthres) {
		printf("Warning: zrestore = %f < zthres %f. Switch-off disabled.\n", zrestore, zthres);
		zthres = 100.;
		zrestore = 100.;
	}

	/*	Set the output directory, according to an environmental variable	*/

	if (const char *outPath = std::getenv("AXIONS_OUTPUT")) {
		if (strlen(outPath) < 1022) {
			struct stat tStat;
			if (stat(outPath, &tStat) == 0 && S_ISDIR(tStat.st_mode)) {
				strcpy(outDir, outPath);
			} else {
				printf("Path %s doesn't exists, using default\n", outPath);
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
				printf("Path %s doesn't exists, using default\n", wisPath);
			}
		}
	}

	if (zGrid == 1)
		logMpi = ZERO_RANK;

	return	procArgs;
}

Cosmos	createCosmos()
{
	Cosmos myCosmos;

	/*	I'm reading from disk	*/
	if (fIndex >= 0.) {
		if (uMsa || uLambda)
			myCosmos.SetLambda(LL);

		if (uQcd)
			myCosmos.SetQcdExp(nQcd);

		if (uGamma)
			myCosmos.SetGamma(gammo);

		if (uPot)
			myCosmos.SetQcdPot(vqcdType);

		if (uSize)
			myCosmos.SetPhysSize(sizeL);

		if (uZth)
			myCosmos.SetZThRes  (zthres);

		if (uZrs)
			myCosmos.SetZRestore(zrestore);

		if (uI3)
			myCosmos.SetZRestore(zrestore);
	} else {
		myCosmos.SetLambda  (LL);
		myCosmos.SetQcdExp  (nQcd);
		myCosmos.SetGamma   (gammo);
		myCosmos.SetQcdPot  (vqcdType);
		myCosmos.SetPhysSize(sizeL);
		myCosmos.SetZThRes  (zthres);
		myCosmos.SetZRestore(zrestore);
		myCosmos.SetIndi3   (indi3);
	}

	return	myCosmos;
}
