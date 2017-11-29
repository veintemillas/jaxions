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
#include "strings/strings.h"
#include "scalar/scalar.h"
#include "reducer/reducer.h"
#include "spectrum/spectrum.h"
#include "WKB/WKB.h"

#include "utils/parse.h"

#define	StrFrac 4e-08

using namespace std;
using namespace AxionWKB;

int	main (int argc, char *argv[])
{
	initAxions(argc, argv);

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	commSync();

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	start = std::chrono::high_resolution_clock::now();

	Scalar *axion;

	LogOut("Axions molecular dynamics code started\n\n");

	if ((fIndex == -1) && (cType == CONF_NONE)) {
		LogError("Error: neither initial conditions nor configuration to be loaded selected\n");
		endAxions();
		return	1;
	} else {
		if (fIndex == -1) {
			LogOut("Generating axion field (this might take a while) ... ");
			axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fTypeP, lType, cType, parm1, parm2);
			if (axion == nullptr) {
				LogError("Error: couldn't generate axion field\n");
				endAxions();
				return	1;
			}
			LogOut("Success!\n\n");
		} else {
			LogOut("Reading file index %d ... ", fIndex);
			readConf(&axion, fIndex);
			if (axion == nullptr) {
				LogError ("Error: can't read HDF5 file\n");
				endAxions();
				return	1;
			}
			LogOut("Success!\n\n");
		}
	}

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	LogFlush();
	LogOut ("Field set up in %lu ms\n", elapsed.count());

	complex<float> *mC = static_cast<complex<float> *> (axion->mCpu());
	complex<float> *vC = static_cast<complex<float> *> (axion->vCpu());
	float *m = static_cast<float *> (axion->mCpu());
	float *v = static_cast<float *> (axion->mCpu())+axion->eSize();

	const size_t S0  = axion->Surf();

	double delta     = sizeL/axion->Length();
	double dz        = 0.;
	double dzAux     = 0.;
	double llPhys    = LL;
	double llConstZ2 = LL;

	double saskia    = 0.;
	double zShift    = 0.;
	double maxTheta  = M_PI;

	bool   dampSet   = false;

	if (nSteps != 0)
		dz = (zFinl - zInit)/((double) nSteps);

	if (endredmap > axion->Length()) {
		endredmap = axion->Length();
		LogError ("Error: can't reduce from %lu to %lu, will reduce to %lu", endredmap, axion->Length(), axion->Length());
	}

	zthres 	 = 100.0 ;
	zrestore = 100.0 ;

	LogOut("Lambda is in %s mode\n", (axion->Lambda() == LAMBDA_FIXED) ? "fixed" : "z2");

	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------

	int counter = 0;
	int index = fIndex+1;

	commSync();

	void *eRes, *str;			// Para guardar la energia y las cuerdas
	trackAlloc(&eRes, 128);
	memset(eRes, 0, 128);
	double *eR = static_cast<double *> (eRes);

	alignAlloc(&str, axion->DataAlign(), (axion->Size()));
	memset(str, 0, axion->Size());

	commSync();

	if (fIndex == -1) {
		if (prinoconfo%2 == 1) {
			LogOut ("Dumping configuration %05d ...", index);
			writeConf(axion, index);
			LogOut ("Done!\n");
		}
	}

	Folder munge(axion);

	LogOut ("Folding configuration ... \n");
	munge(FOLD_ALL);

	if (cDev != DEV_CPU)
	{
		LogOut ("Transferring configuration to device\n");
		axion->transferDev(FIELD_MV);
	}


	if (dump > nSteps)
		dump = nSteps;

	int nLoops = 0;

	if (dump != 0)
		nLoops = (int)(nSteps/dump);

	LogOut("\n");
	LogOut("-------------------------------------------------\n");
	LogOut("             Simulation parameters               \n");
	LogOut("-------------------------------------------------\n");
	LogOut("  Length =  %2.2f\n", sizeL);
	LogOut("  nQCD   =  %2.2f\n", nQcd);
	LogOut("  N      =  %ld\n",   axion->Length());
	LogOut("  Nz     =  %ld\n",   axion->Depth());
	LogOut("  zGrid  =  %ld\n",   zGrid);
	LogOut("  dx     =  %2.5f\n", delta);
	LogOut("  dz     =  %2.2f/FREQ\n", wDz);

	if (axion->Lambda() == LAMBDA_FIXED)
		LogOut("  LL     =  %f \n\n", LL);
	else
		LogOut("  LL     =  %1.3e/z^2 Set to make ms*delta =%f\n\n", llConstZ2, msa);

	switch (vqcdType & VQCD_TYPE) {
		case	VQCD_1:
			LogOut("  VQcd 1 PQ 1, shift, continuous theta, flag %d\n\n", vqcdType);
			break;
		case	VQCD_2:
			LogOut("  VQcd 2 PQ 1, no shift, continuous theta, flag %d\n\n", vqcdType);
			break;
		case	VQCD_1_PQ_2:
			LogOut("  VQcd 1 PQ 2, shift, continuous theta, flag %d\n\n", vqcdType);
			break;
	}

	if ((vqcdType & VQCD_DAMP) != 0)
		LogOut("  Damping enabled with friction constant %e\n\n", gammo);

	LogOut("-------------------------------------------------\n\n\n\n");

	LogOut("-------------------------------------------------\n");
	LogOut("                   Estimates                     \n");
	LogOut("-------------------------------------------------\n");

	double axMassNow = 0.;

	double zNow  = *(axion->zV());
	double zDoom = 0.;

	if ((vqcdType & VQCD_TYPE) == VQCD_1_PQ_2)
		zDoom = pow(0.1588*msa/delta*2., 2./(nQcd+2.));
	else
		zDoom = pow(0.1588*msa/delta,    2./(nQcd+2.));

	double zAxiq  = pow(1.00/delta, 2./(nQcd+2.));
	double zNR    = pow(3.46/delta, 2./(nQcd+2.));

	LogOut("  z Doomsday %f \n", zDoom);
	LogOut("  z Axiquenc %f \n", zAxiq);
	LogOut("  z NR       %f \n", zNR);

	LogOut("-------------------------------------------------\n\n");

	size_t       curStrings  = 0;
	const size_t fineStrings = (size_t) (floor(((double) axion->TotalSize())*StrFrac));

	createMeas(axion, index);

	if(p2dmapo)
		writeMapHdf5 (axion);

	if (axion->Precision() == FIELD_SINGLE) {
		Binner<100, complex<float>> rhoBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
						  [z = zNow] (complex<float> x) { return (double) abs(x)/z; } );
		rhoBin.run();
		writeBinner(rhoBin, "/bins", "rho");

		Binner<100, complex<float>> thBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
						 [] (complex<float> x) { return (double) arg(x); });
		thBin.run();
		writeBinner(thBin, "/bins", "theta");
	} else {
		Binner<100, complex<double>> rhoBin(static_cast<complex<double> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
						    [z = zNow] (complex<double> x) { return (double) abs(x)/z; } );
		rhoBin.run();
		writeBinner(rhoBin, "/bins", "rho");

		Binner<100, complex<double>> thBin(static_cast<complex<double> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
						 [] (complex<double> x) { return (double) arg(x); });
		thBin.run();
		writeBinner(thBin,  "/bins", "theta");
	}

	destroyMeas();

	commSync();
	LogFlush();

	/*	We run a few iterations with damping too smooth the rho field	*/

	initPropagator (pType, axion, nQcd, delta, LL, (vqcdType & VQCD_TYPE) | VQCD_DAMP_RHO);

	LogOut ("Tuning propagator...\n");
	tunePropagator(axion);
	LogOut ("Tuned\n");

	double dzControl = 0.0;

	LogOut ("Damping rho...\n");

	for (int zLoop = 0; zLoop < nLoops; zLoop++)
	{
		dzAux = dzSize(zInit, axion->Field(), axion->Lambda(), vqcdType);

		propagate (axion, dzAux);

		axion->setZ(zInit);
		dzControl += dzAux;

		auto   rts     = strings(axion, str);
		curStrings     = rts.strDen;
		double strDens = 0.75*delta*curStrings*zInit*zInit/(sizeL*sizeL*sizeL);

		LogOut("dzControl %f nStrings %lu [Lt^2/V] %f\n", dzControl, rts.strDen, strDens);

		if (strDens < 5.0)
			break;

	}

	initPropagator (pType, axion, nQcd, delta, LL, vqcdType & VQCD_TYPE);

	start = std::chrono::high_resolution_clock::now();
	old = start;

	FILE *history = nullptr;

	if ((history = fopen("sample.txt", "w+")) == nullptr)
		LogError ("Couldn't open history file");

	LogFlush();
	LogOut ("Start redshift loop\n\n");

	commSync();

	/*	These vectors store data that will be written to a file at the end	*/
	/*	of the simulation. We keep z, axion mass, zRestore, m, v, and the	*/
	/*	maximum value of theta							*/
	std::vector<std::tuple<double, double, double, complex<double>, complex<double>, double>> sxPoints;
	std::vector<std::tuple<double, double, double, double,          double,          double>> axPoints;

	int strCount = 0;
	StringData rts;

	for (int zLoop = 0; zLoop < nLoops; zLoop++) {

		index++;

		for (int zSubloop = 0; zSubloop < dump; zSubloop++) {

			zNow = (*axion->zV());
			old  = std::chrono::high_resolution_clock::now();
			dzAux = dzSize(zNow, axion->Field(), axion->Lambda(), vqcdType);

			propagate (axion, dzAux);

			zNow = (*axion->zV());

			if (axion->Field() == FIELD_SAXION) {
				llPhys = (axion->Lambda() == LAMBDA_Z2) ? llConstZ2/(zNow*zNow) : llConstZ2;

				axMassNow = axionmass(zNow, nQcd, zthres, zrestore);
				saskia    = saxionshift(axMassNow, llPhys, vqcdType);
				zShift    = zNow * saskia;

				/*	If there are a few strings, we compute them every small step		*/
				if (curStrings < 1000) {//fineStrings) {
					rts   = strings(axion, str);
					curStrings = rts.strDen;
				}

				/*	Enable damping given the right conditions. We only do this once,	*/
				/*	and it's controlled by zDomm and the dampSet boolean			*/
				if ((zNow > zDoom*0.95) && !dampSet && ((vqcdType & VQCD_DAMP) != VQCD_NONE)) {
					LogOut("Reaching doomsday (z %.3f, zDoom %.3f)\n", zNow, zDoom);
					LogOut("Enabling damping with gamma %.4f\n\n", gammo);
					initPropagator (pType, axion, nQcd, delta, LL, vqcdType );
					dampSet = true;
				}

				if (commRank() == 0) {
					complex<double> m = complex<double>(0.,0.);
					complex<double> v = complex<double>(0.,0.);
					if (axion->Precision() == FIELD_DOUBLE) {
						m = static_cast<complex<double>*>(axion->mCpu())[axion->Surf()];
						v = static_cast<complex<double>*>(axion->vCpu())[0];
//						maxTheta = find<FIND_MAX,complex<double>>(static_cast<complex<double>*>(axion->mCpu()) + axion->Surf(), axion->Size(),
//											 [] (complex<double> x) { return (double) abs(arg(x)); });
					} else {
						m = static_cast<complex<float>*> (axion->mCpu())[axion->Surf()];
						v = static_cast<complex<float>*> (axion->vCpu())[0];
//						maxTheta = find<FIND_MAX,complex<float>> (static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
//											 [] (complex<float>  x) { return (double) abs(arg(x)); });
					}
					if (history != nullptr) {
						fprintf (history, "%f %f %f %f %f %f %f %lu %f %e\n", zNow, axMassNow, llPhys, m.real(), m.imag(), v.real(), v.imag(),
							 curStrings, maxTheta, saskia);
						fflush  (history);
					}
					sxPoints.emplace_back(make_tuple(zNow, axMassNow, zrestore, m, v, maxTheta));
				}

				/*	If we didn't see strings for a while, go to axion mode			*/
				if (curStrings == 0) {
					strCount++;

					/*	CONF_SAXNOISE	will keep the saxion field forever		*/
					if (strCount > safest0 && smvarType != CONF_SAXNOISE) {

						createMeas(axion, 10000);

						if(p2dmapo)
							writeMapHdf5 (axion);

				  		energy(axion, eRes, false, delta, nQcd, llPhys, vqcdType, zShift);
						writeEnergy(axion, eRes);

						if (axion->Precision() == FIELD_SINGLE) {

							Binner<100,complex<float>> rhoBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
											  [z = zNow] (complex<float> x) { return (double) abs(x)/z; });
							rhoBin.run();
							writeBinner(rhoBin, "/bins", "rho");

							Binner<100,complex<float>> thBin(static_cast<complex<float> *> (axion->mCpu()) + axion->Surf(), axion->Size(),
											 [] (complex<float> x) { return (double) arg(x); });
							thBin.run();
							maxTheta = max(abs(thBin.min()),thBin.max());
							writeBinner(thBin, "/bins", "theta");
						} else {
							Binner<100,complex<double>>rhoBin(static_cast<complex<double>*>(axion->mCpu()) + axion->Surf(), axion->Size(),
											   [z = zNow] (complex<double>x) { return (double) abs(x)/z; });
							rhoBin.run();
							writeBinner(rhoBin, "/bins", "rho");

							Binner<100,complex<double>>thBin(static_cast<complex<double>*> (axion->mCpu()) + axion->Surf(), axion->Size(),
											  [] (complex<double>x) { return (double) arg(x); });
							thBin.run();
							maxTheta = max(abs(thBin.min()),thBin.max());
							writeBinner(thBin, "/bins", "theta");
						}

						destroyMeas();

						LogOut("--------------------------------------------------\n");
						LogOut("           TRANSITION TO THETA (z=%.4f)           \n", zNow);
						LogOut("                  shift = %f                      \n", saskia);
						LogOut("--------------------------------------------------\n");

						cmplxToTheta (axion, zShift);

						createMeas(axion, 10001);

						if(p2dmapo)
						  	writeMapHdf5 (axion);

						energy(axion, eRes, false, delta, nQcd, 0., vqcdType, 0.);
						writeEnergy(axion, eRes);

						if (axion->Precision() == FIELD_SINGLE) {
							Binner<100,float> thBin(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
										 [z = zNow] (float x)  -> double { return (float) (x/z); });
							thBin.run();
							writeBinner(thBin, "/bins", "theta");
						} else {
							Binner<100,double>thBin(static_cast<double*>(axion->mCpu()) + axion->Surf(), axion->Size(),
										 [z = zNow] (double x) -> double { return (double) (x/z); });
							thBin.run();
							writeBinner(thBin, "/bins", "theta");
						}
						destroyMeas();

						LogOut ("Tuning propagator...\n");
						tunePropagator(axion);
						LogOut ("Tuned\n");
					}
				}
			} else {
				if (commRank() == 0) {
					double m = 0., v = 0.;

					if (axion->Precision() == FIELD_DOUBLE) {
						m = static_cast<double*>(axion->mCpu())[axion->Surf()];
						v = static_cast<double*>(axion->vCpu())[0];
//						maxTheta = find<FIND_MAX, double>(static_cast<double*>(axion->mCpu()) + axion->Surf(), axion->Size(),
//										 [] (double x) { return (double) abs(x); });
					} else {
						m = static_cast<float *>(axion->mCpu())[axion->Surf()];
						v = static_cast<float *>(axion->vCpu())[0];
//						maxTheta = find<FIND_MAX, float> (static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
//										 [] (float x) { return (double) abs(x); });
					}

					if (history != nullptr) {
						fprintf (history, "%f %f %f %f %f\n", zNow, axMassNow, m, v, maxTheta);
						fflush  (history);
					}

					axPoints.emplace_back(make_tuple(zNow, axMassNow, zrestore, m, v, maxTheta));
				}
			}

			current = std::chrono::high_resolution_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);

			counter++;

			if (zNow > zFinl)
			{
				LogOut("Redshift z = %.3f reached target value %.3f\n\n", zNow, zFinl);
				break;
			}

		} // zSubloop iteration

		/*	We perform now an online analysis	*/

		createMeas(axion, index);

		if (axion->Field() == FIELD_SAXION) {
			if (axion->Precision() == FIELD_SINGLE) {

				Binner<100,complex<float>> rhoBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
								  [z = zNow] (complex<float> x) { return (float) abs(x)/z; });
				rhoBin.run();
				writeBinner(rhoBin, "/bins", "rho");

				Binner<100,complex<float>> thBin (static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
								  [] (complex<float> x) { return (float) arg(x); });
				thBin.run();
				maxTheta = max(abs(thBin.min()),thBin.max());
				writeBinner(thBin,  "/bins", "theta");
			} else {
				Binner<100,complex<double>>rhoBin(static_cast<complex<double>*>(axion->mCpu()) + axion->Surf(), axion->Size(),
								  [z = zNow] (complex<double> x) { return (double) abs(x)/z; });
				rhoBin.run();
				writeBinner(rhoBin, "/bins", "rho");

				Binner<100,complex<double>>thBin (static_cast<complex<double>*>(axion->mCpu()) + axion->Surf(), axion->Size(),
								  [] (complex<double> x) { return (double) arg(x); });
				thBin.run();
				maxTheta = max(abs(thBin.min()),thBin.max());
				writeBinner(thBin,  "/bins", "theta");
			}

			energy(axion, eRes, false, delta, nQcd, llPhys, vqcdType, zShift);

			double maa = 40.*axionmass2(zNow, nQcd, zthres, zrestore)/(2*llPhys);

			if (axion->Lambda() == LAMBDA_Z2)
				maa = maa*zNow*zNow;

			rts   = strings(axion, str);
			curStrings = rts.strDen;

			if (p3DthresholdMB/((double) curStrings) > 1.)
				writeString(str, rts, true);
			else
				writeString(str, rts, false);

			LogOut("%05d | dz %.3e\tLambda %.3e\t40ma2/ms2 %.3e\t[Lt^2/V] %.3f\t\t", zLoop, dzAux, llPhys, maa, 0.75*delta*curStrings*zNow*zNow/(sizeL*sizeL*sizeL));
			profiler::Profiler &prof = profiler::getProfiler(PROF_PROP);

			auto pFler = prof.Prof().cbegin();
			auto pName = pFler->first;
			profiler::printMiniStats(zNow, rts, PROF_PROP, pName);
		} else {
			energy(axion, eRes, true, delta, nQcd, 0., vqcdType, 0.);

			if (axion->Precision() == FIELD_SINGLE) {
				Binner<100, float> thBin(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
							 [z = zNow] (float x)  -> double { return (double) (x/z);});
				thBin.run();
				maxTheta = max(abs(thBin.min()),thBin.max());

				float eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
				Binner<3000, float> contBin(static_cast<float *>(axion->m2Cpu()), axion->Size(),
							    [eMean = eMean] (float x) -> double { return (double) (log10(x/eMean) );});
				contBin.run();

				writeBinner(contBin, "/bins", "cont");
				writeBinner(thBin,   "/bins", "theta");
			} else {
				Binner<100, double>thBin(static_cast<double*>(axion->mCpu()) + axion->Surf(), axion->Size(),
							 [z = zNow] (double x) -> double { return (double) (x/z);});
				thBin.run();
				maxTheta = max(abs(thBin.min()),thBin.max());

				double eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
				Binner<3000, double>contBin(static_cast<double*>(axion->m2Cpu()), axion->Size(),
							    [eMean = eMean] (double x) -> double { return (double) (log10(x/eMean) );});
				contBin.run();

				writeBinner(contBin, "/bins", "cont");
				writeBinner(thBin,   "/bins", "theta");
			}

			LogOut("%05d | dz %.3e\tMaxTheta %f\t\t", zLoop, dzAux, maxTheta);
			profiler::Profiler &prof = profiler::getProfiler(PROF_PROP);
			auto pFler = prof.Prof().cbegin();
			auto pName = pFler->first;
			profiler::printMiniStats(zNow, rts, PROF_PROP, pName);

			SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);
			specAna.pRun();
			writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");

			specAna.nRun();
			writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
			writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
			writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");
		}
	
		if(p2dmapo)
			writeMapHdf5(axion);

		writeEnergy(axion, eRes);

		if (zNow >= zFinl)
		{
			LogOut("Redshift z = %.3f reached target value %.3f\n\n", zNow, zFinl);
			break;
		} else {
			destroyMeas();
		}
		LogFlush();
	} // zLoop

	fclose (history);

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	LogOut("Propagation finished\n");

	munge(UNFOLD_ALL);

//	index++	;
	if (axion->Field() == FIELD_AXION) {
		if (pconfinal)
			energy(axion, eRes, true, delta, nQcd, 0., vqcdType, 0.);

		if (pconfinal) {
			if (endredmap > 0) {
				double ScaleSize = ((double) axion->Length())/((double) endredmap);
				double eFc  = 0.5*M_PI*M_PI*(ScaleSize*ScaleSize)/((double) axion->Surf());
				size_t nLz = endredmap / commSize();

				if (axion->Precision() == FIELD_DOUBLE) {
					reduceField(axion, endredmap, nLz, FIELD_M2, [eFc = eFc] (int px, int py, int pz, complex<double> x) -> complex<double>
						   { return x*exp(-eFc*(px*px + py*py + pz*pz)); });
				} else {
					reduceField(axion, endredmap, nLz, FIELD_M2, [eFc = eFc] (int px, int py, int pz, complex<float>  x) -> complex<float>
						   { return x*((float) exp(-eFc*(px*px + py*py + pz*pz))); });
				}
			}

			writeEDens (axion);
		}

		destroyMeas();

		if ((prinoconfo >= 2) && (wkb2z < 0)) {
			LogOut ("Dumping final configuration %05d ...", index);

			if (cDev == DEV_GPU)
				axion->transferCpu(FIELD_MV);

			writeConf(axion, index);
			LogOut ("Done!\n");
		}

		LogFlush();

		/*	If needed, go on with the WKB approximation	*/
		if (wkb2z >= zFinl) {
			LogOut ("\n\nWKB approximation propagating from %.4f to %.4f ... ", zFinl, wkb2z);

			WKB wonka(axion, axion);
			wonka(wkb2z);

			zNow = (*axion->zV());

			index++;

			if (prinoconfo >= 2) {
				LogOut ("Dumping final WKBed configuration %05d ...", index);
				writeConf(axion, index);
				LogOut ("Done!\n");
			}

			LogOut ("Printing last measurement file %05d ... ", index);
			createMeas(axion, index);

			if (p2dmapo)
				writeMapHdf5(axion);

			SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);
			specAna.nRun();
			writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
			writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
			writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");

			energy(axion, eRes, true, delta, nQcd, 0., vqcdType, 0.);

			if (axion->Precision() == FIELD_SINGLE) {
				float eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
				Binner<3000,float> contBin(static_cast<float *>(axion->m2Cpu()), axion->Size(),
							   [eMean = eMean] (float x) -> double { return (double) (log10(x/eMean) );});
				contBin.run();

				Binner<100, float> thBin(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
							 [z=zNow] (float x) -> double { return (double) (x/z);});
				thBin.run();

				writeBinner(contBin, "/bins", "cont");
				writeBinner(thBin,   "/bins", "theta");
			} else {
				double eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
				Binner<3000,double>contBin(static_cast<double *>(axion->m2Cpu()), axion->Size(),
							   [eMean = eMean] (double x) -> double { return (double) (log10(x/eMean) );});
				contBin.run();

				Binner<100, double>thBin(static_cast<double*>(axion->mCpu()) + axion->Surf(), axion->Size(),
							[z = zNow] (double x) -> double { return (double) (x/z); });
				thBin.run();

				writeBinner(contBin, "/bins", "cont");
				writeBinner(thBin,   "/bins", "theta");
			}

			writeEnergy(axion, eRes);

			if (pconfinalwkb) {
				if (endredmap > 0) {
					double ScaleSize = ((double) axion->Length())/((double) endredmap);
					double eFc  = 0.5*M_PI*M_PI*(ScaleSize*ScaleSize)/((double) axion->Surf());
					size_t nLz = endredmap / commSize();

					if (axion->Precision() == FIELD_DOUBLE) {
						reduceField(axion, endredmap, nLz, FIELD_M2, [eFc = eFc] (int px, int py, int pz, complex<double> x) -> complex<double>
							   { return x*exp(-eFc*(px*px + py*py + pz*pz)); });
					} else {
						reduceField(axion, endredmap, nLz, FIELD_M2, [eFc = eFc] (int px, int py, int pz, complex<float>  x) -> complex<float>
							   { return x*((float) exp(-eFc*(px*px + py*py + pz*pz))); });
					}
				}

				writeEDens (axion);
			}

			specAna.pRun();
			writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");

			destroyMeas();
		}  // End WKB stuff


		/*	For Jens	*/
		if (endredmap > 0) {
			energy(axion, eRes, true, delta, nQcd, 0., vqcdType, 0.);
			double ScaleSize = ((double) axion->Length())/((double) endredmap);
			double eFc  = 0.5*M_PI*M_PI*(ScaleSize*ScaleSize)/((double) axion->Surf());
			size_t nLz = endredmap / commSize();

			if (axion->Precision() == FIELD_DOUBLE) {
				reduceField(axion, endredmap, nLz, FIELD_M2, [eFc = eFc] (int px, int py, int pz, complex<double> x) -> complex<double>
					   { return x*exp(-eFc*(px*px + py*py + pz*pz)); });
			} else {
				reduceField(axion, endredmap, nLz, FIELD_M2, [eFc = eFc] (int px, int py, int pz, complex<float>  x) -> complex<float>
					   { return x*((float) exp(-eFc*(px*px + py*py + pz*pz))); });
			}

			createMeas(axion, index+1);
			writeEnergy(axion, eRes);
			writeEDens (axion);
			destroyMeas();
		}
		LogFlush();
	}  // End axion stuff, for the saxion it seems we don't care

	LogOut("z Final = %f\n", *axion->zV());
	LogOut("nSteps  = %i\n", counter);
	LogOut("nPrints = %i\n", index);

	LogOut("Total time: %2.3f min\n", elapsed.count()*1.e-3/60.);
	LogOut("Total time: %2.3f h\n", elapsed.count()*1.e-3/3600.);

	trackFree(&eRes, ALLOC_TRACK);
	trackFree(&str,  ALLOC_ALIGN);

	delete axion;

	endAxions();

	return 0;
}
