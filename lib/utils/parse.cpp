#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "enum-field.h"

size_t sizeN = 128;
size_t sizeZ = 128;
int  zGrid = 1;
int  nSteps = 500;
int  dump = 100;
int  nQcd = 3;
//JAVIER
int  Ng = 1 ;
double indi3 = 1.0;
double msa  = 1.5;
double wDz  = 0.8;
int  fIndex = -1;

double sizeL = 4.;
double zInit = 0.5;
double zFinl = 1.0;
double kCrit = 1.0;
//JAVIER
double mode0 = 10.0;
double alpha = 0.143;
double zthres   = 1.0;
double zrestore = 1.0;
double LL = 15000.;
double parm2 = 0.;


bool lowmem = false;
bool uPrec  = false;

size_t kMax  = 2;
//JAVIER played with the following number
size_t iter  = 40;
size_t parm1 = 0;

ConfType  cType = CONF_NONE;
FieldType fType = FIELD_SAXION;

char *initFile = NULL;
char outName[128] = "axion\0";

FieldPrecision	sPrec = FIELD_DOUBLE;
DeviceType	cDev  = DEV_CPU;

void	printUsage(char *name)
{
	printf("\nUsage: %s [Options]\n\n", name);

	printf("\nOptions:\n\n");

	printf("--size  [int]                   Number of lattice points along x and y (Lx). Local size is Lx^2 x Lz (default 128).\n");
	printf("--depth [int]                   Number of lattice points of depth (Lz) (default 128).\n");
	printf("--zgrid [int]                   Number of gpus involved in the computation (default 1).\n");
	printf("                                Splitting occurs in the z-dimension, so the total lattice is Lx^2 x (zgrid * Lz).\n");
	printf("--prec  double/single           Precision of the axion field simulation (default double)\n");
	printf("--ftype saxion/axion            Type of field to be simulated, either saxion + axion or lone axion (default saxion)\n");

	printf("--qcd   [int]                   Exponent of topological susceptibility (default 3).\n");
	printf("--lsize [float]                 Physical size of the system (default 4.0).\n");
	printf("--zi    [float]                 Initial value of the redshift (default 0.5).\n");
	printf("--zf    [float]                 Final value of the redshift (default 1.0).\n");
	printf("--llcf  [float]                 Lagrangian coefficient (default 15000).\n");
	printf("--msa   [float]                 Spacing to core ratio (Moore parameter) [laxion3D].\n");
	printf("--wDz   [float]                 Adaptive time step dz = wDz/frequency [laxion3D].\n");
	printf("--steps [int]                   Number of steps of the simulation (default 500).\n");
	printf("--ctype smooth/kmax/tkachev     Initial configuration, either with smoothing or with FFT and a maximum momentum\n");
	printf("--kmax  [int]                   Maximum momentum squared for the generation of the configuration with --ctype kmax/tkachev (default 2)\n");
	printf("--kcr   [float]                 kritical kappa (default 1.0).\n");
	printf("--mode0 [float]               	Value of axion zero mode [rad] (default random).\n");
	printf("--sIter [int]                   Number of smoothing steps for the generation of the configuration with --ctype smooth (default 40)\n");
	printf("--alpha [float]                 alpha parameter for the smoothing (default 0.143).\n");
	printf("--dump  [int]                   frequency of the output (default 100).\n");
	printf("--name  [filename]              Uses filename to name the output files in out/dump, instead of the default \"axion\"\n");
	printf("--index [idx]                   Loads HDF5 file at out/dump as initial conditions (default, don't load).\n");
	printf("--lowmem                        Reduces memory usage by 33\%, but decreases performance as well (default false).\n");
	printf("--device cpu/gpu/xeon           Uses nVidia Gpus or Intel Xeon Phi to accelerate the computations (default, use cpu).\n");
	printf("--lapla 0/1/2/3/4               Number of Neighbours in the laplacian [only for simple3D] \n");
	printf("--help                          Prints this message.\n");

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
			printUsage(argv[0]);
			exit(0);
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
				printf("Error: I need a number of gpus.\n");
				exit(1);
			}

			zGrid = atoi(argv[i+1]);

			if (zGrid < 1)
			{
				printf("Error: The number of gpus must be larger than 0.\n");
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

			nQcd = atoi(argv[i+1]);

			if (nQcd < 0)
			{
				printf("Error: The exponent of the top. susceptibility nQcd must be equal or greater than 0.\n");
				exit(1);
			}

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
				printf("Error: Number of steps must be greater than or equal to zero.\n");
				exit(1);
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

		if (!strcmp(argv[i], "--load"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a file to load.\n");
				exit(1);
			}

			initFile = argv[i+1];

			if (fIndex != -1)
			{
				printf("Error: You must use either --load or --index, they are mutually exclusive.\n");
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
				printf("Error: Name too long, keep it under 96 characters\n");
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
				printf("Error: The maximum momentum must be equal or greater than zero.\n");
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

			if (initFile != NULL)
			{
				printf("Error: You must use either --load or --index, they are mutually exclusive.\n");
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
				printf("Error: I need a device name (cpu/gpu/xeon).\n");
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
				cDev = DEV_XEON;
			}
			else
			{
				printf("Error: Unrecognized device %s\n", argv[i+1]);
				exit(1);
			}

			i++;
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
			printUsage(argv[0]);
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

	return	procArgs;
}
