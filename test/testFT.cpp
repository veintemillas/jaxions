#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "scalar/scaleField.h"

#include "fft/fftCode.h"

using namespace std;


/* Program */

int	main (int argc, char *argv[])
{
	Cosmos myCosmos = initAxions(argc, argv);

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	commSync();

	Scalar *axion;

	double t_0 = (double) Timer()*1.0e-6;
	LogMsg(VERB_NORMAL,"Generating Complex Scalar ... t=%f \n",t_0);
	axion = new Scalar (&myCosmos, sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, FIELD_SAXION, lType, myCosmos.ICData().Nghost);

	char oName[2048];
	sprintf (oName, "cardio%d.txt", sizeN);
	FILE *file ;
	file = NULL;
	file = fopen(oName,"a+");

	/* Plan r2c */

	double t_P = (double) Timer()*1.0e-6;
	LogMsg(VERB_NORMAL,"Fetch plan pSpecAx (r2c inplace)... t=%f \n",t_P);
	auto &myPlan = AxionFFT::fetchPlan("pSpecAx");

	std::vector<double>	store;
	store.resize(8);

	std::vector<double>	data;
	data.resize(nSteps);
	double t_li,t_lf,mean,std;

	double t_F = (double) Timer()*1.0e-6;
	LogMsg(VERB_NORMAL,"FFTs ... t=%f\n",t_F);
	for (int i=0; i< nSteps; i++) {
		t_li = (double) Timer()*1.0e-6;
		myPlan.run(FFT_FWD);
		t_lf = (double) Timer()*1.0e-6;
		data.at(i) = t_lf-t_li;
		LogMsg(VERB_NORMAL,"%f",data.at(i));
	}

	mean =0;
	std =0;
	for (int i=0; i< nSteps; i++) {
	mean += data.at(i);
	std += data.at(i)*data.at(i);
	}
	mean /= nSteps;
	store.at(0) = mean;
	store.at(1) = sqrt(std/nSteps-mean*mean);

	double t_iF = (double) Timer()*1.0e-6;
	LogMsg(VERB_NORMAL,"FFTs ... t=%f\n",t_iF);
	for (int i=0; i< nSteps; i++){
		t_li = (double) Timer()*1.0e-6;
		myPlan.run(FFT_BCK);
		t_lf = (double) Timer()*1.0e-6;
		data.at(i) = t_lf-t_li;
		LogMsg(VERB_NORMAL,"%f",data.at(i));
	}

	mean =0;
	std =0;
	for (int i=0; i< nSteps; i++) {
	mean += data.at(i);
	std += data.at(i)*data.at(i);
	}
	mean /= nSteps;
	store.at(2) = mean;
	store.at(3) = sqrt(std/nSteps-mean*mean);

	double t_FE = (double) Timer()*1.0e-6;
	LogMsg(VERB_NORMAL,"pSpecAx FWD/BCK took %f/%f s\n",(t_iF-t_F)/nSteps, (t_FE-t_iF)/nSteps);
	double a1 = (t_iF-t_F)/nSteps;
	double b1 = (t_FE-t_iF)/nSteps;

	/* Plan c2c */

	double t_P2 = (double) Timer()*1.0e-6;
	LogMsg(VERB_NORMAL,"Fetch plan SpSx (c2c outofplace)... t=%f\n",t_P2);
	auto &myflan = AxionFFT::fetchPlan("SpSx");

	double t_F2 = (double) Timer()*1.0e-6;
	LogMsg(VERB_NORMAL,"FFTs ... t=%f\n",t_F2);
	for (int i=0; i< nSteps; i++){
		t_li = (double) Timer()*1.0e-6;
		myflan.run(FFT_FWD);
		t_lf = (double) Timer()*1.0e-6;
		data.at(i) = t_lf-t_li;
		LogMsg(VERB_NORMAL,"%f",data.at(i));
	}
	mean =0;
	std =0;
	for (int i=0; i< nSteps; i++) {
	mean += data.at(i);
	std += data.at(i)*data.at(i);
	}
	mean /= nSteps;
	store.at(4) = mean;
	store.at(5) = sqrt(std/nSteps-mean*mean);

	double t_iF2 = (double) Timer()*1.0e-6;
	LogMsg(VERB_NORMAL,"FFTs ... t=%f\n",t_iF2);
	for (int i=0; i< nSteps; i++){
		t_li = (double) Timer()*1.0e-6;
		myflan.run(FFT_BCK);
		t_lf = (double) Timer()*1.0e-6;
		data.at(i) = t_lf-t_li;
		LogMsg(VERB_NORMAL,"%f",data.at(i));
	}

	mean =0;
	std =0;
	for (int i=0; i< nSteps; i++) {
	mean += data.at(i);
	std += data.at(i)*data.at(i);
	}
	mean /= nSteps;
	store.at(6) = mean;
	store.at(7) = sqrt(std/nSteps-mean*mean);

	double t_FE2 = (double) Timer()*1.0e-6;
	LogMsg(VERB_NORMAL,"SpSx FWD/BCK took %f/%f s\n",(t_iF2-t_F2)/nSteps, (t_FE2-t_iF2)/nSteps);

	if (commRank() == 0) {
		LogOut("Nx %d Nx %d FFTp %d MPI %d OMP %d %f %f %f %f ", axion->Length(), axion->Depth(), fftplanType, commSize(), commThreads(), a1, b1, (t_iF2-t_F2)/nSteps, (t_FE2-t_iF2)/nSteps);
		fprintf(file, "%d %d %d %d %d %f %f %f %f ", axion->Length(), axion->Depth(), fftplanType, commSize(), commThreads(), a1, b1, (t_iF2-t_F2)/nSteps, (t_FE2-t_iF2)/nSteps);
		for (int i=0; i<8; i++){
			LogOut("%f ", store.at(i));
			fprintf(file,"%f ", store.at(i));
		}
		fprintf(file," %f %f \n ",t_P-t_0, t_FE2-t_P);
		LogOut(" took %f (scalar) %f (FFts) \n", t_P-t_0, t_FE2-t_P);
	}

	delete axion;

	endAxions();

	return 0;
}
