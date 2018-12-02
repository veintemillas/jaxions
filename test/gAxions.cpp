#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "propagator/allProp.h"
#include "energy/energy.h"
#include "utils/utils.h"
#include "io/readWrite.h"
#include "comms/comms.h"
#include "map/map.h"
#include "strings/strings.h"
#include "scalar/scalar.h"
#include "scalar/scaleField.h"
#include "spectrum/spectrum.h"

#include "meas/meas.h"
#include "WKB/WKB.h"

using namespace std;
using namespace AxionWKB;

double	findZDoom   (Scalar *axion);
void	checkTime   (Scalar *axion, int index);
//void	printSample (FILE *aFile, Scalar *axion, double LLL, size_t idxprint, MeasData &meas);

int	main (int argc, char *argv[])
{
	Cosmos myCosmos = initAxions(argc, argv);

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	commSync();

	LogOut("\n-------------------------------------------------\n");
	LogOut("\n          JgAxion 3D\n\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	start = std::chrono::high_resolution_clock::now();

	Scalar *axion;

	double zInitSave = zInit;
	double zPreInit = zInit;

	if(preprop)
		zPreInit = zInit/prepcoe;

	if ((fIndex == -1) && (cType == CONF_NONE) && (!restart_flag)) {
		LogOut("Error: Neither initial conditions nor configuration to be loaded selected. Empty field.\n");
		endAxions();
	} else {
		if ( (fIndex == -1) && !restart_flag)
		{
			//This generates initial conditions
			LogOut("Generating scalar ... ");
			axion = new Scalar (&myCosmos, sizeN, sizeZ, sPrec, cDev, zPreInit, lowmem, zGrid, fTypeP, lType, cType, parm1, parm2);
			LogOut("Done! \n");
		}
		else
		{
			//This reads from an axion.00000 file
			readConf(&myCosmos, &axion, fIndex, restart_flag);

			if (axion == nullptr)
			{
				LogOut ("Error reading HDF5 file\n");
				endAxions();
				exit (0);
			}
			// prepropagation tends to mess up reading initial conditions
			// configurations are saved before prepropagation and have z<zInit, which readConf reverses
			// the following line fixes the issue, but a more elegant solution could be devised
			if( (preprop) && !restart_flag) {
				zInit = zInitSave;
				*axion->zV() = zPreInit;
			}
		}
	}

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	LogOut("ICtime %f min\n",elapsed.count()*1.e-3/60.);
/*
	FILE *myFile = nullptr;

	if (!restart_flag)
		myFile = fopen("out/sample.txt","w+");
	else
		myFile = fopen("out/sample.txt","a+");

	if (myFile == nullptr) {
		LogError ("Couldn't open sample file\n");
		delete axion;
		endAxions();
	}
*/
	MeasData mData;

	double zNow;
	double axMassNow;
	double delta = axion->Delta();
	double dz;
	double dzAux;
	double llPhys = myCosmos.Lambda();

	if (nSteps == 0)
		dz = 0.;
	else
		dz = (zFinl - zInit)/((double) nSteps);

	if (endredmap > sizeN)
	{
		endredmap = sizeN;
		LogOut("[Error:1] Reduced map dimensions set to %d\n ", endredmap);
	}

	if (sizeN%endredmap != 0 )
	{
		int schei =  sizeN/endredmap;
		endredmap = sizeN/schei;
		LogOut("[Error:2] Reduced map dimensions set to %d\n ", endredmap);
	}

	bool coD = true;

	int strCount = 0;
	double LL1 = myCosmos.Lambda();

	LogOut("--------------------------------------------------\n");
	if (!restart_flag)
		LogOut("           Starting computation\n");
	else
		LogOut("          Continuing computation\n");
	LogOut("--------------------------------------------------\n");

	int counter = 0;
	int index = 0;

	void *eRes;
	trackAlloc(&eRes, 128);
	memset(eRes, 0, 128);
	double *eR = static_cast<double *> (eRes);

	commSync();

	if (icstudy)
	{
		if (fIndex == -1) {
			if (prinoconfo%2 == 1 ) {
				LogOut ("Dumping configuration %05d ...", index);
				writeConf(axion, index);
				LogOut ("Done!\n");
			}
		} else
			index = fIndex;
	}


	double saskia = 0.0;
	double zShift = 0.0;

	Folder munge(axion);

	if (cDev != DEV_GPU)
		munge(FOLD_ALL);

	if (cDev != DEV_CPU)
		axion->transferDev(FIELD_MV);

	if (dump > nSteps)
		dump = nSteps;

	int nLoops;

	if (dump == 0)
		nLoops = 0;
	else
		nLoops = (int)(nSteps/dump);


	LogOut("--------------------------------------------------\n");
	LogOut("           Parameters\n\n");
	LogOut("  Length =  %2.2f\n", myCosmos.PhysSize());
	LogOut("  nQCD   =  %2.2f\n", myCosmos.QcdExp());

	if (myCosmos.ZRestore() > myCosmos.ZThRes())
		LogOut("       =  0 in (%3.3f, %3.3f)   \n", myCosmos.ZThRes(), myCosmos.ZRestore());

	LogOut("  N      =  %ld\n",   axion->Length());
	LogOut("  Nz     =  %ld\n",   axion->Depth());
	LogOut("  zGrid  =  %ld\n",   zGrid);
	LogOut("  dx     =  %2.5f\n", axion->Delta());
	LogOut("  dz     =  %2.2f/FREQ\n", wDz);

	if (LAMBDA_FIXED == axion->Lambda()){
		LogOut("  LL     =  %.0f (msa=%1.2f-%1.2f in zInit,3)\n\n", myCosmos.Lambda(),
		sqrt(2.*myCosmos.Lambda())*zInit*axion->Delta(),sqrt(2.*myCosmos.Lambda())*3*axion->Delta());
	}
	else
		LogOut("  LL     =  %1.3e/z^2 Set to make ms*delta =%.2f \n\n", myCosmos.Lambda(), axion->Msa());

	if	((myCosmos.QcdPot() & VQCD_TYPE) == VQCD_1)
		LogOut("  VQcd I, shift, continuous theta  \n\n");
	else if	((myCosmos.QcdPot() & VQCD_TYPE) == VQCD_2)
		LogOut("  VQcd II, no shift, continuous theta  \n\n");
	else if	((myCosmos.QcdPot() & VQCD_TYPE) == VQCD_1_PQ_2)
		LogOut("  VQcd PQ2, shift, continuous theta  \n\n");

	LogOut("  VQcd flag %d\n", myCosmos.QcdPot());
	LogOut("  Damping %d gam = %f\n", myCosmos.QcdPot() & VQCD_DAMP, myCosmos.Gamma());
	LogOut("--------------------------------------------------\n\n");
	LogOut("           Estimates\n\n");

	double zDoom;

	if ((myCosmos.QcdPot() & VQCD_TYPE) == VQCD_1_PQ_2)
		zDoom = pow(2.0*0.1588*axion->Msa()/axion->Delta(), 2./(myCosmos.QcdExp()+2.));
	else
		zDoom = pow(    0.1588*axion->Msa()/axion->Delta(), 2./(myCosmos.QcdExp()+2.));

	double zDoom2 = findZDoom(axion);
	double zAxiq  = pow(1.00/axion->Delta(), 2./(myCosmos.QcdExp()+2.));
	double zNR    = pow(3.46/axion->Delta(), 2./(myCosmos.QcdExp()+2.));
	LogOut("zDoomsDay %f(%f) \n", zDoom, zDoom2);
	LogOut("zAxiquenc %f \n", zAxiq);
	LogOut("z_NR       %f \n", zNR);
	LogOut("--------------------------------------------------\n\n");

	commSync();

	//--------------------------------------------------
	// prepropagator with relaxing strong damping [RELAXATION of GAMMA DOES NOT WORK]
	//--------------------------------------------------
	// only if preprop and if z smaller or equal than zInit
	// When z>zInit, it is understood that prepropagation was done
	// NEW it takes the pregam value (if is > 0, otherwise gam )
	if (preprop && ((*axion->zV()) < zInit)) {
		LogOut("Preprocessing...\n\n",
			(*axion->zV()), zInit, (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO, myCosmos.Gamma(),pregammo);
		// gammo is reserved for long-time damping
		// use pregammo for prepropagation damping
		double gSave = myCosmos.Gamma();
		double zCur  = *(axion->zV());

		if (pregammo > 0)
			myCosmos.SetGamma(pregammo);

		// prepropagation is always with rho-damping
		LogOut("Prepropagator always with damping VQcd flag %d\n", (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO);
		initPropagator (pType, axion, (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO);
		tunePropagator (axion);

		while (zCur < zInit){
			dzAux = axion->dzSize(zInit)/2.;

//			printSample(myFile, axion, myCosmos.Lambda(), 0, mData);

			if (icstudy){
				// string control
				mData = Measure (axion, index, MEAS_STRING | MEAS_ENERGY | MEAS_2DMAP);

				if (axion->Lambda() == LAMBDA_Z2)
					llPhys = LL1/((zCur)*(zCur));

				axMassNow = axion->AxionMass();
				saskia    = axion->Saskia();
				zShift    = zCur * saskia;

				index++;
			} else
				LogOut("z %f (gamma %f)\n", zCur, myCosmos.Gamma());

			propagate (axion, dzAux);
		}

		myCosmos.SetGamma(gSave);
	}

	if (!icstudy)
	{
		if (fIndex == -1){
			if (prinoconfo%2 == 1 ){
				LogOut ("Dumping configuration (after prep) %05d ...", index);
				writeConf(axion, index);
				LogOut ("Done!\n");
			}
		} else
			index = fIndex;
	}

	if (!restart_flag && (fIndex == -1)) {
		LogOut("First measurement file %d \n",index);
		mData = Measure (axion, index, MEAS_STRING | MEAS_ENERGY | MEAS_2DMAP | MEAS_ALLBIN) ;
	} else
		LogOut("last measurement file was %d \n",index);

	LogOut("Running ...\n\n");
	LogOut("Init propagator VQcd flag %d\n", myCosmos.QcdPot());
	initPropagator (pType, axion, myCosmos.QcdPot());
	tunePropagator (axion);

	LogOut ("Start redshift loop\n\n");
	fflush (stdout);

	start = std::chrono::high_resolution_clock::now();
	old = start;

	for (int zloop = 0; zloop < nLoops; zloop++)
	{
		index++;

		for (int zsubloop = 0; zsubloop < dump; zsubloop++)
		{
			old = std::chrono::high_resolution_clock::now();

			dzAux = axion->dzSize();
			propagate (axion, dzAux);

			zNow = (*axion->zV());

//			printSample(myFile, axion, myCosmos.Lambda(), 0, mData);

			if (axion->Field() == FIELD_SAXION)
			{
				if (mData.str.strDen < 1000)
					mData.str = strings(axion);

				if ((zNow > zDoom2*0.95) && (coD) && pregammo > 0.)
				{
					myCosmos.SetGamma(pregammo);
					LogOut("-----------------------------------------\n");
					LogOut("Damping (gam = %f, z ~ 0.95*zDoom %f)\n", myCosmos.Gamma(), 0.95*zDoom2);
					LogOut("-----------------------------------------\n");

					initPropagator (pType, axion, (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO);
					coD = false ;
					// possible problem!! if gamma is needed later, as it is written pregammo will stay
				}

				if (mData.str.strDen == 0)
					strCount++;

				if (smvarType != CONF_SAXNOISE)
					if (mData.str.strDen == 0 && strCount > safest0)
					{
						if (axion->Lambda() == LAMBDA_Z2)
							llPhys = myCosmos.Lambda()/(zNow*zNow);

						axMassNow = axion->AxionMass();
						saskia    = axion->Saskia();
						zShift    = zNow * saskia;

						mData = Measure (axion, 10000, MEAS_2DMAP | MEAS_ENERGY | MEAS_ALLBIN);

						LogOut("--------------------------------------------------\n");
						LogOut(" Theta transition @ z %.4f\n",zNow);

						cmplxToTheta (axion, zShift);

						mData = Measure (axion, 10001, MEAS_2DMAP | MEAS_ENERGY | MEAS_ALLBIN ) ;

						LogOut("--------------------------------------------------\n");

						tunePropagator (axion);
					}
			}

			current = std::chrono::high_resolution_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);

			counter++;

			if (zNow > zFinl)
				break;

			checkTime(axion, index);
		}

		if (axion->Lambda() == LAMBDA_Z2)
			llPhys = myCosmos.Lambda()/(zNow*zNow);

		axMassNow = axion->AxionMass();
		saskia    = axion->Saskia();
		zShift    = zNow * saskia;

		auto	cTime = Timer();

		if (zNow > zFinl)
		{
			LogOut("Final z reached\n");
			break;
		}

		mData = Measure (axion, index, MEAS_ALLBIN | MEAS_STRING | MEAS_STRINGMAP | MEAS_ENERGY | MEAS_2DMAP | MEAS_SPECTRUM);

		profiler::Profiler &prof = profiler::getProfiler(PROF_PROP);

		auto pFler = prof.Prof().cbegin();
		auto pName = pFler->first;
		profiler::printMiniStats(zNow, mData.str, PROF_PROP, pName);

		checkTime(axion, index);
	}

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	LogOut("\n");
	LogOut("--------------------------------------------------\n");
	LogOut("              Evolution finished\n");
	LogOut("--------------------------------------------------\n");
	fflush(stdout);

	LogOut ("Final measurement file is: %05d \n", index);
	munge(UNFOLD_ALL);

	if (axion->Field() == FIELD_AXION)
	{
		MeasureType mesa = MEAS_2DMAP | MEAS_ALLBIN | MEAS_SPECTRUM | MEAS_ENERGY;

		if ((prinoconfo >= 2) && (wkb2z < 0)) {
			LogOut ("Dumping final configuration %05d ...", index);
			mesa = mesa | MEAS_3DMAP  ;
		}
		if (pconfinal)
			mesa = mesa | MEAS_ENERGY3DMAP ;
		if (endredmap > 0)
			mesa = mesa | MEAS_REDENE3DMAP ;

		mData = Measure (axion, index, mesa);

		if (wkb2z >= zFinl) {
			WKB wonka(axion, axion);

			LogOut ("WKBing @ %.4f to %.4f ... ", index, zNow, index+1, wkb2z);

			wonka(wkb2z);
			zNow = (*axion->zV());

			index++;

			MeasureType mesa = MEAS_2DMAP | MEAS_ALLBIN | MEAS_SPECTRUM | MEAS_ENERGY;

			if (prinoconfo >= 2) {
				LogOut ("Dumping final WKBed configuration %05d ...", index);
				mesa = mesa | MEAS_3DMAP;
			}

			if (pconfinalwkb)
				mesa = mesa | MEAS_ENERGY3DMAP;

			if (pconfinal)
				mesa = mesa | MEAS_ENERGY3DMAP;

			if (endredmap > 0)
				mesa = mesa | MEAS_REDENE3DMAP;

			mData = Measure (axion, index, mesa);
		}
	} else {
		if ((prinoconfo >= 2)) {
			LogOut ("Dumping final Saxion onfiguration %05d ...", index);
			writeConf(axion, index);
			LogOut ("Done!\n");
		}
	}
	LogOut("z final = %f\n", zNow);
	LogOut("#Steps = %i\n",  counter);
	LogOut("#Prints = %i\n", index);
	LogOut("Total time: %2.3f min\n", elapsed.count()*1.e-3/60.);
	LogOut("Total time: %2.3f h\n", elapsed.count()*1.e-3/3600.);

	trackFree(eRes);
//	fclose(myFile);

	delete axion;

	endAxions();

	return 0;
}
/*
void printSample(FILE *fichero, Scalar *axion, double LLL, size_t idxprint, MeasData &meas)
{
	double zNow = (*axion->zV());
	double llPhys = LLL;
	if (axion->Lambda() == LAMBDA_Z2)
		llPhys = LLL/(zNow*zNow);

	size_t S0 = sizeN*sizeN ;
	if (commRank() == 0 && sPrec == FIELD_SINGLE) {
		if (axion->Field() == FIELD_SAXION) {
			double axMassNow = axion->AxionMass();
			double saskia    = axion->Saskia();

			if (axion->Precision() == FIELD_DOUBLE) {
				fprintf(fichero,"%lf %lf %lf %lf %lf %lf %lf %ld %lf %e\n", zNow, axMassNow, llPhys,
				static_cast<complex<double>*> (axion->mCpu())[idxprint + S0].real(),
				static_cast<complex<double>*> (axion->mCpu())[idxprint + S0].imag(),
				static_cast<complex<double>*> (axion->vCpu())[idxprint].real(),
				static_cast<complex<double>*> (axion->vCpu())[idxprint].imag(),
				meas.str.strDen, meas.maxTheta, saskia);
			} else {
				fprintf(fichero,"%f %f %f %f %f %f %f %ld %f %e\n", zNow, axMassNow, llPhys,
				static_cast<complex<float> *> (axion->mCpu())[idxprint + S0].real(),
				static_cast<complex<float> *> (axion->mCpu())[idxprint + S0].imag(),
				static_cast<complex<float> *> (axion->vCpu())[idxprint].real(),
				static_cast<complex<float> *> (axion->vCpu())[idxprint].imag(),
				meas.str.strDen, meas.maxTheta, saskia);
			}
		} else {
			if (axion->Precision() == FIELD_DOUBLE) {
				fprintf(fichero,"%f %f %f %f %f\n", zNow, axion->AxionMass(),
				static_cast<double*> (axion->mCpu())[idxprint + S0],
				static_cast<double*> (axion->vCpu())[idxprint], meas.maxTheta);
			} else {
				fprintf(fichero,"%f %f %f %f %f\n", zNow, axion->AxionMass(),
				static_cast<float *> (axion->mCpu())[idxprint + S0],
				static_cast<float *> (axion->vCpu())[idxprint], meas.maxTheta);
			}
		}

		fflush(fichero);
	}
}
*/
double findZDoom(Scalar *axion)
{
	double ct = zInit ;
	double DWfun;
	double meas ;
	while (meas < 0.001)
	{
		DWfun = 40*axion->AxionMassSq(ct)/(2.0*axion->BckGnd()->Lambda()) ;
		if (axion->Lambda() == LAMBDA_Z2)
			DWfun *= ct*ct;
		meas = DWfun - 1 ;
		ct += 0.001 ;
	}
	LogOut("Real zDoom %f ", ct );
	return ct ;
}

void	checkTime (Scalar *axion, int index) {
	auto	cTime = Timer();
	int	cSize = commSize();
	int	flag  = 0;
	std::vector<int> allFlags(cSize);

	bool	done  = false;

	if (wTime <= cTime)
		flag = 1;

	MPI_Allgather(&flag, 1, MPI_INT, allFlags.data(), 1, MPI_INT, MPI_COMM_WORLD);

	for (const int &val : allFlags) {
		if (val == 1) {
			done = true;
			break;
		}
	}

	if (done) {
		if (cDev == DEV_GPU)
			axion->transferCpu(FIELD_MV);

		LogOut ("Walltime reached, dumping configuration...");
		writeConf(axion, index, 1);
		LogOut ("Done!\n");

		LogOut("z Final = %f\n", *axion->zV());
		LogOut("nPrints = %i\n", index);

		LogOut("Total time: %2.3f min\n", cTime*1.e-6/60.);
		LogOut("Total time: %2.3f h\n", cTime*1.e-6/3600.);
//		trackFree(eRes);	FIXME!!!!

		delete axion;

		endAxions();

		exit(0);
	}
}
