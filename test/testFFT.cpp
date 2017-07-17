#include<cmath>
#include<chrono>

#include<complex>
#include<vector>

#include"scalar/scalar.h"
#include"propagator/allProp.h"
#include"utils/utils.h"
#include"io/readWrite.h"
#include"comms/comms.h"
#include"fft/fftCode.h"

using namespace std;

#ifdef	USE_XEON
	__declspec(target(mic)) char *mX, *vX, *m2X;
#endif

int	main (int argc, char *argv[])
{
	initAxions(argc, argv);

	LogOut("\n-------------------------------------------------\n");
	LogOut("\n          CREATING MINICLUSTERS!                \n\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS       
	//--------------------------------------------------

	FlopCounter *fCount = new FlopCounter;

	Scalar *axion;
	char fileName[256];

	axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, FIELD_SAXION, CONF_NONE, 0, 0, NULL);
	readConf(&axion, 0);

	//--------------------------------------------------
	//          SETTING BASE PARAMETERS
	//-------------------------------------------------- 

	double dz = (zFinl - zInit)/((double) nSteps);
	double delta = sizeL/sizeN;

	LogOut("--------------------------------------------------\n");
	LogOut("           INITIAL CONDITIONS                     \n\n");

	LogOut("Length =  %2.5f\n", sizeL);
	LogOut("N      =  %d\n",    sizeN);
	LogOut("Nz     =  %d\n",    sizeZ);
	LogOut("zGrid  =  %d\n",    zGrid);
	LogOut("dx     =  %2.5f\n", delta);  
	LogOut("dz     =  %2.5f\n", dz);
	LogOut("LL     =  %2.5f\n", LL);
	LogOut("--------------------------------------------------\n");

	const int S0 = sizeN*sizeN;
	const int SF = sizeN*sizeN*(sizeZ+1)-1;
	const int V0 = 0;
	const int VF = axion->Size()-1;

	LogOut("INITIAL CONDITIONS LOADED\n");
	if (sPrec != FIELD_DOUBLE)
	{
		LogOut("Example mu: m[0] = %f + %f*I, m[N3-1] = %f + %f*I\n", ((complex<float> *) axion->mCpu())[S0].real()/zInit, ((complex<float> *) axion->mCpu())[S0].imag()/zInit,
									        ((complex<float> *) axion->mCpu())[SF].real()/zInit, ((complex<float> *) axion->mCpu())[SF].imag()/zInit);
		LogOut("Example  v: v[0] = %f + %f*I, v[N3-1] = %f + %f*I\n", ((complex<float> *) axion->vCpu())[V0].real(), ((complex<float> *) axion->vCpu())[V0].imag(),
									        ((complex<float> *) axion->vCpu())[VF].real(), ((complex<float> *) axion->vCpu())[VF].imag());
	}
	else
	{
		LogOut("Example mu: m[0] = %lf + %lf*I, m[N3-1] = %lf + %lf*I\n", ((complex<double> *) axion->mCpu())[S0].real()/zInit, ((complex<double> *) axion->mCpu())[S0].imag()/zInit,
										    ((complex<double> *) axion->mCpu())[SF].real()/zInit, ((complex<double> *) axion->mCpu())[SF].imag()/zInit);
		LogOut("Example  v: v[0] = %lf + %lf*I, v[N3-1] = %lf + %lf*I\n", ((complex<double> *) axion->vCpu())[V0].real(), ((complex<double> *) axion->vCpu())[V0].imag(),
										    ((complex<double> *) axion->vCpu())[VF].real(), ((complex<double> *) axion->vCpu())[VF].imag());
	}

	LogOut("Ez     =  %d\n",    axion->eDepth());

	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------  

	LogOut("--------------------------------------------------\n");
	LogOut("           STARTING COMPUTATION                   \n");
	LogOut("--------------------------------------------------\n");

	std::chrono::high_resolution_clock::time_point start, current, old;

	int counter = 0;
	int index = 0;

	//LogOut ("Dumping configuration %05d...\n", index);
	//fflush (stdout);
	//writeConf(axion, index);


	LogOut ("Start FFT\n");
	fflush (stdout);

	commSync();

	start = std::chrono::high_resolution_clock::now();
	old = start;
	std::chrono::milliseconds elapsed;

	auto &myPlan = AxionFFT::fetchPlan("Init");
	myPlan.run(FFT_FWD);

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	index++;
	writeConf(axion, index);

	LogOut("\n PROGRAMM FINISHED\n");

	if (sPrec == FIELD_DOUBLE)
	{
		LogOut("\n Examples m: m[0]= %f + %f*I, m[N3-1]= %f + %f*I\n",  ((complex<double> *) axion->mCpu())[S0].real(), ((complex<double> *) axion->mCpu())[S0].imag(),
		 								  ((complex<double> *) axion->mCpu())[SF].real(), ((complex<double> *) axion->mCpu())[SF].imag());
		LogOut("\n Examples v: v[0]= %f + %f*I, v[N3-1]= %f + %f*I\n\n",((complex<double> *) axion->vCpu())[V0].real(), ((complex<double> *) axion->vCpu())[V0].imag(),
									 	  ((complex<double> *) axion->vCpu())[VF].real(), ((complex<double> *) axion->vCpu())[VF].imag());
	}
	else
	{
		LogOut("\n Examples m: m[0]= %f + %f*I, m[N3-1]= %f + %f*I\n",  ((complex<float> *) axion->mCpu())[S0].real(), ((complex<float> *) axion->mCpu())[S0].imag(),
										  ((complex<float> *) axion->mCpu())[SF].real(), ((complex<float> *) axion->mCpu())[SF].imag());
		LogOut("\n Examples v: v[0]= %f + %f*I, v[N3-1]= %f + %f*I\n\n",((complex<float> *) axion->vCpu())[V0].real(), ((complex<float> *) axion->vCpu())[V0].imag(),
										  ((complex<float> *) axion->vCpu())[VF].real(), ((complex<float> *) axion->vCpu())[VF].imag());
	}

	LogOut("z_final = %f\n", *axion->zV());
	LogOut("Total time: %2.3f s\n", elapsed.count()*1.e-3);
	LogOut("--------------------------------------------------\n");

	delete fCount;

	endAxions();
    
	return 0;
}
