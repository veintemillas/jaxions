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
#include "powerCpu.h"
#include "scalar/scalar.h"
#include "spectrum/spectrum.h"
#include "scalar/mendTheta.h"

#include "WKB/WKB.h"

//#include<mpi.h>

using namespace std;
using namespace AxionWKB;

int	main (int argc, char *argv[])
{
	Cosmos myCosmos = initAxions(argc, argv);

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	commSync();
	LogOut("\n-------------------------------------------------\n");
	LogOut("\n          CREATING MINICLUSTERS!                \n\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	start = std::chrono::high_resolution_clock::now();

	Scalar *axion;
	char fileName[256];

	if ((fIndex == -1) && (myCosmos.ICData().cType == CONF_NONE))
		LogOut("Error: Neither initial conditions nor configuration to be loaded selected. Empty field.\n");
	else
	{
		if (fIndex == -1)
		{
			//This generates initial conditions
			LogOut("Generating scalar ... ");
			axion = new Scalar (&myCosmos, sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fTypeP, lType);
			LogOut("Done! \n");
		}
		else
		{
			//This reads from an Axion.00000 file
			readConf(&myCosmos, &axion, fIndex);
			if (axion == NULL)
			{
				LogOut ("Error reading HDF5 file\n");
				exit (0);
			}
		}
	}

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	LogOut("ICtime %f min\n",elapsed.count()*1.e-3/60.);

	//--------------------------------------------------
	//          OUTPUTS FOR CHECKING
	//--------------------------------------------------

	double Vr, Vt, Kr, Kt, Grz, Gtz;
	size_t nstrings = 0;
	size_t nstrings_global = 0;

  double nstringsd = 0. ;
	double nstringsd_global = 0. ;
	double maximumtheta = 3.141597;
	size_t sliceprint = 0 ; // sizeN/2;

	// Axion spectrum
	const int kmax = axion->Length()/2 -1;
	int powmax = floor(1.733*kmax)+2 ;

	double  *binarray	 ;
	trackAlloc((void**) (&binarray),  10000*sizeof(size_t));
	double *bA = static_cast<double *> (binarray);

	complex<float> *mC = static_cast<complex<float> *> (axion->mCpu());
	complex<float> *vC = static_cast<complex<float> *> (axion->vCpu());
	//Pointers
	float *m = static_cast<float *> (axion->mCpu());
	float *v = static_cast<float *> (axion->mCpu())+axion->eSize();


	FILE *file_samp ;
	file_samp = NULL;
	file_samp = fopen("out/sample.txt","w+");
	size_t idxprint = 0 ;

	LogOut("Bins allocated! \n");

	double z_now ;
	double axmass_now;

	//--------------------------------------------------
	//          SETTING BASE PARAMETERS
	//--------------------------------------------------

	double delta = axion->Delta();
	double dz;
	double dzaux;
	double llphys = myCosmos.Lambda();

	if (nSteps == 0)
		dz = 0.;
	else
		dz = (zFinl - zInit)/((double) nSteps);

	if (endredmap > sizeN)
	{
		endredmap = sizeN;
		LogOut("[Error:1] Reduced map dimensions set to %d\n ", endredmap);
	}

	LogOut("nena %d moda %d\n", sizeN/endredmap, sizeN%endredmap);
	if (sizeN%endredmap != 0 )
	{
		int schei =  sizeN/endredmap;
		endredmap = sizeN/schei;
		LogOut("[Error:2] Reduced map dimensions set to %d\n ", endredmap);
	}


	const size_t S0 = axion->Surf();
	const size_t SF = axion->Size()-1+S0;
	const size_t V0 = 0;
	const size_t VF = axion->Size()-1;

	//	--------------------------------------------------
	//	TRICK PARAMETERS TO RESPECT STRINGS
	//	--------------------------------------------------

	bool coZ = 1;
	bool coS = 1;
	bool coA = 1;
	bool coD = true;
	int strcount = 0;

	int numaxiprint = 10 ;
	StringData rts ;

	double llconstantZ2 = myCosmos.Lambda();

	if (LAMBDA_FIXED == axion->LambdaT())
		LogOut ("Lambda in FIXED mode\n");
	else {
		LogOut ("Lambda in Z2 mode\n");
		LogOut ("llconstantZ2 = %f - LL = llconstantZ2/Z^2 \n", llconstantZ2);
	}

	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------

	LogOut("--------------------------------------------------\n");
	LogOut("           STARTING REALLIGHT COMPUTATION         \n");
	LogOut("--------------------------------------------------\n");


	int counter = 0;
	int index = 0;

	commSync();

	void *eRes;			// Para guardar la energia y las cuerdas
	trackAlloc(&eRes, 128);
	memset(eRes, 0, 128);
	double *eR = static_cast<double *> (eRes);

	commSync();


	if (fIndex == -1) {
		if (prinoconfo%2 == 1 ) {
			LogOut ("Dumping configuration %05d ...", index);
			writeConf(axion, index);
			LogOut ("Done!\n");
		} else
			LogOut ("Bypass configuration writting!\n");
	} else
		index = fIndex;

	double saskia = 0.0;
	double shiftz = 0.0;

//	--------------------------------------------------
//	--------------------------------------------------

	Folder munge(axion);

	if (cDev != DEV_GPU)
	{
		LogOut ("Folding configuration ... ");
		munge(FOLD_ALL);
	}

	if (cDev != DEV_CPU)
	{
		LogOut ("Transferring configuration to device\n");
		axion->transferDev(FIELD_MV);
	}
	LogOut ("Done! \n");

//	if (cDev != DEV_GPU)
		double	strDen;

	if (dump > nSteps)
		dump = nSteps;

	int nLoops;

	if (dump == 0)
		nLoops = 0;
	else
		nLoops = (int)(nSteps/dump);


	LogOut("--------------------------------------------------\n");
	LogOut("           PARAMETERS  						                \n\n");
	LogOut("Length =  %2.2f\n", myCosmos.PhysSize());
	LogOut("nQCD   =  %2.2f\n", myCosmos.QcdExp());
	LogOut("N      =  %ld\n",   axion->Length());
	LogOut("Nz     =  %ld\n",   axion->Depth());
	LogOut("zGrid  =  %ld\n",   zGrid);
	LogOut("dx     =  %2.5f\n", axion->Delta());
	LogOut("dz     =  %2.2f/FREQ\n", wDz);

	if (LAMBDA_FIXED == axion->LambdaT())
		LogOut("LL     =  %f \n\n", myCosmos.Lambda());
	else
		LogOut("LL     =  %1.3e/z^2 Set to make ms*delta =%f \n\n", myCosmos.Lambda(), axion->Msa());

	if	((myCosmos.QcdPot() & VQCD_TYPE) == VQCD_1)
		LogOut("VQCD1PQ1,shift,continuous theta  \n\n");
	else if	((myCosmos.QcdPot() & VQCD_TYPE) == VQCD_2)
		LogOut("VQCD2PQ1,no shift, continuous theta  \n\n");
	else if	((myCosmos.QcdPot() & VQCD_TYPE) == VQCD_1_PQ_2)
		LogOut("VQCD1PQ2,shift, continuous theta  \n\n");
	LogOut("Vqcd flag %d\n", myCosmos.QcdPot());
	LogOut("Damping %d gam = %f\n", myCosmos.QcdPot() & VQCD_DAMP, myCosmos.Gamma());
	LogOut("--------------------------------------------------\n\n");
	LogOut("           ESTIMATES\n\n");

	double z_doom;

	if ((myCosmos.QcdPot() & VQCD_TYPE) == VQCD_1_PQ_2)
		z_doom = pow(2.0*0.1588*axion->Msa()/axion->Delta(),2./(myCosmos.QcdExp()+2.));
	else
		z_doom = pow(    0.1588*axion->Msa()/axion->Delta(),2./(myCosmos.QcdExp()+2.));

	double z_axiq = pow(1.00/axion->Delta(),2./(myCosmos.QcdExp()+2.));
	double z_NR   = pow(3.46/axion->Delta(),2./(myCosmos.QcdExp()+2.));
	LogOut("z_doomsday %f \n", z_doom);
	LogOut("z_axiquenc %f \n", z_axiq);
	LogOut("z_NR       %f \n", z_NR);
	LogOut("--------------------------------------------------\n\n");


	createMeas(axion, index);

	if(p2dmapo)
		writeMapHdf5s (axion,sliceprint);

	{
		float z_now = *axion->zV();
		Binner<100,complex<float>> rhoBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
						  [z=z_now] (complex<float> x) { return (double) abs(x)/z; } );
		rhoBin.run();
		writeBinner(rhoBin, "/bins", "rhoB");

		Binner<100,complex<float>> thBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
						 [] (complex<float> x) { return (double) arg(x); });
		thBin.run();
		writeBinner(thBin, "/bins", "thetaB");
	}

	destroyMeas();

	LogOut ("Start redshift loop\n\n");
	fflush (stdout);

	commSync();


	//--------------------------------------------------
	// prepropagator
	//--------------------------------------------------

	LogOut("pppp Preprocessing ... %d \n\n", (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO);
	double *zaza = axion->zV();
	initPropagator (pType, axion, (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO);
	double dzcontrol = 0.0;
	double strdensn ;

	LogOut("--------------------------------------------------\n");
	LogOut("            TUNING PROPAGATOR                     \n");
	LogOut("--------------------------------------------------\n");

	tunePropagator (axion);

	for (int zloop = 0; zloop < nLoops; zloop++)
	{
		dzaux = axion->dzSize(zInit);
		propagate (axion, dzaux);
		*zaza = zInit;
		dzcontrol += dzaux;
		rts = strings(axion);
		nstrings_global = rts.strDen;
		strdensn = 0.75*axion->Delta()*nstrings_global*zInit*zInit/(myCosmos.PhysSize()*myCosmos.PhysSize()*myCosmos.PhysSize());
		LogOut("dzcontrol %f strings %ld [Lt^2/V] %f\n", dzcontrol, nstrings_global, strdensn);
		if (strdensn < 5.0)
		{
			LogOut("                   ...  done!\n\n");
			break;
		}

	}

	// LL is LL      in FIXED MODE
	// LL is LL(z=1) in Z2 MODE (computed from msa in parse.cpp)
	// damping only from zst1000
	LogOut("Running ...\n\n");
	initPropagator (pType, axion, myCosmos.QcdPot() & VQCD_TYPE);

	start = std::chrono::high_resolution_clock::now();
	old = start;



    //--------------------------------------------------
		// THE TIME ITERATION LOOP
		//--------------------------------------------------

	for (int zloop = 0; zloop < nLoops; zloop++)
	{
		//--------------------------------------------------
		// THE TIME ITERATION SUB-LOOP
		//--------------------------------------------------

		index++;

		for (int zsubloop = 0; zsubloop < dump; zsubloop++)
		{

			z_now = (*axion->zV());

			old = std::chrono::high_resolution_clock::now();


			//--------------------------------------------------
			// DYAMICAL deltaz
			//--------------------------------------------------

			dzaux = axion->dzSize();

			//--------------------------------------------------
			// PROPAGATOR
			//--------------------------------------------------

			propagate (axion, dzaux);

			if (commRank() == 0 && sPrec == FIELD_SINGLE) {
				z_now = (*axion->zV());
					if (axion->Field() == FIELD_SAXION) {
						// LAMBDA_Z2 MODE assumed!
							if (axion->LambdaT() == LAMBDA_Z2)
								llphys = llconstantZ2/(z_now*z_now);
							//if (vqcdType == VQCD_1 || vqcdType == VQCD_1_PQ_2)
							//	saskia = saxionshift(z_now, nQcd, zthres, zrestore, llphys);
							axmass_now = axion->AxionMass();
							saskia = axion->Saskia();
							fprintf(file_samp,"%f %f %f %f %f %f %f %ld %f %e\n", z_now, axmass_now, llphys,
							mC[idxprint + S0].real(), mC[idxprint + S0].imag(), vC[idxprint].real(), vC[idxprint].imag(), nstrings_global, maximumtheta, saskia);
					} else {
							fprintf(file_samp,"%f %f %f %f %f\n", z_now, axion->AxionMass(),
							m[idxprint + S0], v[idxprint], maximumtheta);
					}
				fflush(file_samp);
			}

			if (axion->Field() == FIELD_SAXION)
			{
				// compute this 500 qith an educated guess

				if (nstrings_global < 1000)
				{
					rts = strings(axion);
					nstrings_global = rts.strDen ;
					LogOut("  str extra check (string = %d, wall = %d)\n",rts.strDen, rts.wallDn);
				}

				//if ( (nstrings_global < 1000) && (coD) && (vqcdType | VQCD_DAMP) )
				if ((z_now > z_doom*0.95) && (coD) && ((myCosmos.QcdPot() & VQCD_DAMP) != VQCD_NONE))
				{
					LogOut("---------------------------------------\n");
					LogOut("  DAMPING! G = %f (0.95*z_doom %f)\n", myCosmos.Gamma(), 0.95*z_doom);
					LogOut("---------------------------------------\n");
					initPropagator (pType, axion, myCosmos.QcdPot());
					coD = false ;
				}

				//--------------------------------------------------
				// TRANSITION TO THETA
				//--------------------------------------------------

				if (nstrings_global == 0) {
					LogOut("  no st counter %d\n", strcount);
					strcount++;
				}

				if (smvarType != CONF_SAXNOISE) // IF CONF_SAXNOISE we do not ever switch to theta to follow the evolution of saxion field
					if (nstrings_global == 0 && strcount > safest0)
					{
						z_now = (*axion->zV());
						if (axion->LambdaT() == LAMBDA_Z2)
							llphys = llconstantZ2/(z_now*z_now);
						axmass_now = axion->AxionMass();
						saskia = axion->Saskia();
						double shiftz = z_now * saskia;

						createMeas(axion, 10000);
						// IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS
						if(p2dmapo)
							writeMapHdf5s (axion,sliceprint);
						//ENERGY
						energy(axion, eRes, EN_ENE, shiftz);
						writeEnergy(axion, eRes);
						// BIN THETA
						{
							float z_now = *axion->zV();
							Binner<100,complex<float>> rhoBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
											  [z=z_now] (complex<float> x) { return (double) abs(x)/z; });
							rhoBin.run();
							writeBinner(rhoBin, "/bins", "rhoB");

							Binner<100,complex<float>> thBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
											 [] (complex<float> x) { return (double) arg(x); });
							thBin.run();
							writeBinner(thBin, "/bins", "thetaB");
						}

						destroyMeas();

						// TRANSITION TO THETA
						LogOut("--------------------------------------------------\n");
						LogOut("              TRANSITION TO THETA (z=%.4f)\n",z_now);
						LogOut("              shift = %f 			\n", saskia);

						cmplxToTheta (axion, shiftz);

						createMeas(axion, 10001);
						// IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS
						if(p2dmapo)
						  	writeMapHdf5s (axion,sliceprint);
						//ENERGY
						energy(axion, eRes, EN_ENE, 0.);
						writeEnergy(axion, eRes);
						// BIN THETA
						Binner<100,float> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
						 [z=z_now] (float x) -> float { return (float) (x/z); });
						thBin2.run();
						writeBinner(thBin2, "/bins", "thetaB");
						destroyMeas();

						LogOut("--------------------------------------------------\n");

						LogOut("--------------------------------------------------\n");
						LogOut("            TUNING PROPAGATOR                     \n");
						LogOut("--------------------------------------------------\n");

						tunePropagator (axion);
					}
			}

			current = std::chrono::high_resolution_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);

			counter++;

			if ((*axion->zV()) > zFinl)
			{
				LogOut("zf reached! ENDING ... \n"); fflush(stdout);
				break;
			}

		} // ZSUBLOOP

		//--------------------------------------------------
		// PARTIAL ANALISIS
		//--------------------------------------------------

		z_now = (*axion->zV());
		if (axion->LambdaT() == LAMBDA_Z2)
			llphys = llconstantZ2/(z_now*z_now);
		axmass_now = axion->AxionMass();
		saskia = axion->Saskia();

		if ((*axion->zV()) > zFinl)
		{
			// THE LAST MEASURE IS DONE AT THE END
			LogOut("zf reached! ENDING FINALLY... \n");
			break;
		}

		createMeas(axion, index);

		if (axion->Field() == FIELD_SAXION)
		{
			//THETA
			{
				float z_now = *axion->zV();
				Binner<100,complex<float>> rhoBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
								  [z=z_now] (complex<float> x) -> float { return (float) abs(x)/z; });
				rhoBin.run();
				writeBinner(rhoBin, "/bins", "rhoB");

				Binner<100,complex<float>> thBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
								 [] (complex<float> x) -> float { return (float) arg(x); });
				thBin.run();
				writeBinner(thBin, "/bins", "thetaB");
			}
			// maximumtheta = axion->thetaDIST(100, binarray); // note that bins rho 100-200
			// writeArray(bA, 100, "/bins", "theta");
			// writeArray(bA+100, 100, "/bins", "rho");
			// writeBinnerMetadata (maximumtheta, 0., 100, "/bins");
											// old shit still works
			//ENERGY
			energy(axion, eRes, EN_ENE, shiftz);
			//DOMAIN WALL KILLER NUMBER
			double maa = 40*axion->AxionMassSq()/(2.*llphys);
			if (axion->LambdaT() == LAMBDA_Z2 )
				maa = maa*z_now*z_now;
		//STRINGS
			rts = strings(axion);
			nstrings_global = rts.strDen;
			if (p3DthresholdMB/((double) nstrings_global) > 1.)
				writeString(axion, rts, true);
			else
				writeString(axion, rts, false);
			LogOut("%d/%d | z=%f | dz=%.3e | LLaux=%.3e | 40ma2/ms2=%.3e ", zloop, nLoops, (*axion->zV()), dzaux, llphys, maa );
			LogOut("strings %ld [Lt^2/V] %f\n", nstrings_global, 0.75*axion->Delta()*nstrings_global*z_now*z_now/(myCosmos.PhysSize()*myCosmos.PhysSize()*myCosmos.PhysSize()));
		} else {
			//BIN THETA
			Binner<100,float> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
						 [z=z_now] (float x) -> float { return (float) (x/z);});
			thBin2.run();
			writeBinner(thBin2, "/bins", "thetaB");
			maximumtheta = max(abs(thBin2.min()),thBin2.max());

			LogOut("%d/%d | z=%f | dz=%.3e | maxtheta=%f | ", zloop, nLoops, (*axion->zV()), dzaux, maximumtheta);
			fflush(stdout);

			LogOut("DensMap");
			SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);

			// computes energy and creates map
			energy(axion, eRes, EN_MAP, 0.);

			{
				double *eR = static_cast<double*>(eRes);
				float eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
				Binner<3000,float> contBin(static_cast<float *>(axion->m2Cpu()), axion->Size(),
							    [eMean = eMean] (float x) -> float { return (double) (log10(x/eMean) );});
				contBin.run();
				writeBinner(contBin, "/bins", "contB");
			}

			specAna.pRun();
			writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");

			specAna.nRun();
			writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
			writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
			writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");


			LogOut("| \n");
			fflush(stdout);

		}

		if(p2dmapo)
			writeMapHdf5s(axion,sliceprint);
		writeEnergy(axion, eRes);
		destroyMeas();

		// if ((*axion->zV()) > zFinl)
		// {
		// 	LogOut("zf reached! ENDING FINALLY... \n");
		// 	break;
		// }
	} // ZLOOP

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	LogOut("\n");
	LogOut("--------------------------------------------------\n");
	LogOut("              EVOLUTION FINISHED \n");
	LogOut("--------------------------------------------------\n");
	fflush(stdout);

	LogOut ("Final measurement file is: %05d \n", index);
	LogOut("Unfold ... ");
	munge(UNFOLD_ALL);
	LogOut("| ");

	//index++	; // LAST MEASUREMENT IS NOT PRINTED INSIDE THE LOOP, IT IS DONE HERE INSTEAD
	z_now = (*axion->zV());

	if (axion->Field() == FIELD_AXION)
	{
		if ( (prinoconfo >= 2) && (wkb2z < 0)  ){
			LogOut ("Dumping final configuration %05d ...", index);
			writeConf(axion, index);
			LogOut ("Done!\n");
		}


		createMeas(axion, index);
		if(p2dmapo)
			writeMapHdf5s(axion, sliceprint);

		LogOut("n Spectrum ... ");
		/*	Test		*/
		SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);
		specAna.nRun();
		writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
		writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
		writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");

		LogOut("DensMap ... ");

		energy(axion, eRes, EN_MAP, 0.);
		{
			float eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
			Binner<3000,float> contBin(static_cast<float *>(axion->m2Cpu()), axion->Size(),
						  [eMean = eMean] (float x) -> float { return (double) (log10(x/eMean) );});
			contBin.run();
			writeBinner(contBin, "/bins", "contB");
		}
		//bins density
		//axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;
		//write binned distribution
		//writeArray(bA, 10000, "/bins", "cont");
		if (pconfinal)
			writeEDens(axion);
		writeEnergy(axion, eRes);

		LogOut("p Spectrum ... ");
		specAna.pRun();
		writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");

		if ( endredmap > 0){
			LogOut("filtering to reduce map with %d neighbours ... ", sizeN/endredmap);
			int nena = sizeN/endredmap ;
			specAna.filter(nena);
			writeEDensReduced(axion, index, endredmap, endredmap/zGrid);
		}

		LogOut("|  ");
		LogOut("Theta bin ... ");

		double z_now = *axion->zV();
		Binner<100,float> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
					 [z=z_now] (float x) -> float { return (float) (x/z); });
		thBin2.run();
		//writeArray(thBin2.data(), 100, "/bins", "testTh");
		writeBinner(thBin2, "/bins", "thetaB");

		/*	Fin test	*/

		destroyMeas();

		//--------------------------------------------------
		// FINAL WKB
		//--------------------------------------------------

		if (wkb2z >= zFinl)
		{
			WKB wonka(axion, axion);

			LogOut ("WKBing %d (z=%.4f) to %d (%.4f) ... ", index, z_now, index+1, wkb2z);

			wonka(wkb2z);
			z_now = (*axion->zV());
			LogOut(" done! (z=%.4f)\n", z_now);

			index++;
			if (prinoconfo >= 2) {
				LogOut ("Dumping final WKBed configuration %05d ...", index);
				writeConf(axion, index);
				LogOut ("Done!\n");
			}

			LogOut ("Printing measurement file %05d ... ", index);
			createMeas(axion, index);
			SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);

			LogOut("theta ");
			Binner<100,float> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
						 [z=z_now] (float x) -> float { return (float) (x/z);});
			thBin2.run();
			//writeArray(thBin2.data(), 100, "/bins", "testTh");
			writeBinner(thBin2, "/bins", "thetaB");

			LogOut ("spec ");
			specAna.nRun();
			writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
			writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
			writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");

			if(p2dmapo) {
				LogOut ("2D ");
				writeMapHdf5s(axion,sliceprint);
				LogOut ("Done!\n");
			}

			// computes energy and creates map
			LogOut ("en ");
			energy(axion, eRes, EN_MAP, 0.);
			{
				float eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
				Binner<3000,float> contBin(static_cast<float *>(axion->m2Cpu()), axion->Size(),
							    [eMean = eMean] (float x) -> float { return (double) (log10(x/eMean) );});
				contBin.run();
				writeBinner(contBin, "/bins", "contB");
			}
			//bins density
			//LogOut ("con ");
			//axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;
			//write binned distribution
			//LogOut ("bin ");
			//writeArray(bA, 3000, "/bins", "cont");
			if (pconfinalwkb) {
				LogOut ("MAP ");
				writeEDens(axion);
			}

			LogOut ("tot ");
			writeEnergy(axion, eRes);
			//computes power spectrum
			LogOut ("pow ");
			specAna.pRun();
			writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");

			if (endredmap > 0) {
				LogOut("redmap ");
				int nena = sizeN/endredmap ;
				specAna.filter(nena);
				writeEDensReduced(axion, index, endredmap, endredmap/zGrid);
			}

			destroyMeas();
		}


		// --------------------------------------------------
		// FINAL REDUCE MAP (note it reads from file)
		// THIS IS REDUNDANT
		// --------------------------------------------------
		//
		// if ( endredmap > 0)
		// {
		// 	// LogOut ("Reducing map %d to %d^3 ... ", index, endredmap);
		// 	// 	char mirraa[128] ;
		// 	// 	strcpy (mirraa, outName);
		// 	// 	strcpy (outName, "./out/m/axion\0");
		// 	// 	reduceEDens(index, endredmap, endredmap) ;
		// 	// 	strcpy (outName, mirraa);
		// 	// LogOut ("Done!\n");
      //
		// 	createMeas(axion, index+1);
		// 	writeEnergy(axion, eRes);
		// 	writeEDensReduced(axion, index+1, endredmap, endredmap/zGrid);
		// 	destroyMeas();
      //
		// }

	}

	LogOut("z_final = %f\n", *axion->zV());
	LogOut("#_steps = %i\n", counter);
	LogOut("#_prints = %i\n", index);
	LogOut("Total time: %2.3f min\n", elapsed.count()*1.e-3/60.);
	LogOut("Total time: %2.3f h\n", elapsed.count()*1.e-3/3600.);

	trackFree(eRes);
	trackFree((void*) binarray);
	fclose(file_samp);


	delete axion;

	endAxions();

	return 0;
}
