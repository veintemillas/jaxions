#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "enum-field.h"

int sizeN = 128;
int sizeZ = 128;
int zGrid = 1;
int nSteps = 500;
int dump = 100;
int nQcd = 3;
int fIndex = -1;
//int kMax = 8;

double sizeL = 4.;
double zInit = 0.5;
double zFinl = 1.0;
double kCrit = 1.0;
//JAVIER
double alpha = 0.857;
double LL = 15000.;
double parm2 = 0.;

bool lowmem = false;

int kMax  = 0;
//JAVIER played with the following number
int iter  = 40;
int parm1 = 0;

ConfType cType = CONF_NONE;

char *initFile = NULL;

FieldPrecision	sPrec = FIELD_DOUBLE;
DeviceType	cDev  = DEV_CPU;

void	printUsage(char *name)
{
	printf("\nUsage: %s [Options]\n\n", name);

	printf("\nOptions:\n\n");

	printf("--size  [int]                   Defines the size of the lattice Lx. Local size is Lx^2 x Lz (default 128).\n");
	printf("--depth [int]                   Defines the local depth of the lattice Lz (default 128).\n");
	printf("--zgrid [int]                   Defines the number of gpus involved in the computation (default 1).\n");
	printf("                                Splitting occurs in the z-dimension, so the total lattice is Lx^2 x (zgrid * Lz).\n");
	printf("--zi    [float]                 Defines the initial value of the redshift (default 0.5).\n");
	printf("--zf    [float]                 Defines the final value of the redshift (default 1.0).\n");
	printf("--lsize [float]                 Defines the physical size of the system (default 4.0).\n");
	printf("--llcf  [float]                 Defines the lagrangian coefficient (default 15000).\n");
	printf("--kcr   [float]                 Defines the critical kappa (default 1.0).\n");
	printf("--qcd   [int]                   Defines the number of QCD colors (default 3).\n");
	printf("--prec  double/single           Defines the precision of the axion field simulation (default double)\n");
	printf("--ctype smooth/kmax             Defines now to calculate the initial configuration, either with smoothing or with FFT and a maximum momentum\n");
	printf("--kMax  [int]                   Defines the maximum momentum squared for the generation of the configuration with --ctype kmax (default 8)\n");
	printf("--sIter [int]                   Defines the number of smoothing steps for the generation of the configuration with --ctype smooth (default 10)\n");
	printf("--steps [int]                   Defines the number of steps of the simulation (default 500).\n");
	printf("--dump  [int]                   Defines the frequency of the output (default 100).\n");
	printf("--load  [filename]              Loads filename as initial conditions (default out/initial_conditions_m(_single).txt).\n");
	printf("--index [idx]                   Loads HDF5 file at out/dump as initial conditions (default, don't load).\n");
	printf("--lowmem                        Reduces memory usage by 33\%, but decreases performance as well (default false).\n");
	printf("--device cpu/gpu/xeon           Uses nVidia Gpus or Intel Xeon Phi to accelerate the computations (default, use cpu).\n");
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

			sizeN = atoi(argv[i+1]);

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

			sizeZ = atoi(argv[i+1]);

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
				printf("Error: Critical kappa must be larger than 0.\n");
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

		if (!strcmp(argv[i], "--qcd"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need a number of colors.\n");
				exit(1);
			}

			nQcd = atoi(argv[i+1]);

			if (nQcd < 0)
			{
				printf("Error: The number of colors must be equal or greater than 0.\n");
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

			if (nSteps <= 0)
			{
				printf("Error: Number of steps must be greater than zero.\n");
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

		if (!strcmp(argv[i], "--kMax"))
		{
			if (i+1 == argc)
			{
				printf("Error: I need an integer value for the maximum momentum.\n");
				exit(1);
			}

			kMax = atoi(argv[i+1]);

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

			iter = atoi(argv[i+1]);

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
				printf("Error: I need a value for the configuration type (smooth/kMax).\n");
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
