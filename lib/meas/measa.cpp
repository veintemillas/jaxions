#include <cmath>
#include <chrono>
#include <complex>

#include "energy/energy.h"
#include "utils/utils.h"
#include "io/readWrite.h"
#include "comms/comms.h"
#include "map/map.h"
#include "strings/strings.h"
#include "scalar/scalar.h"
#include "reducer/reducer.h"
#include "spectrum/spectrum.h"
#include "projector/projector.h"
#include "scalar/fourier.h"

using namespace std;
using namespace profiler;

double xivilgor(double logi);
double xivilgorphys(double logi);

template<typename Float>
MeasData	Measureme  (Scalar *axiona, MeasInfo info)
{
	MeasureType measa = info.measdata ;

	MeasData MeasDataOut;

	MeasDataOut.maxTheta    = -1 ;
	MeasDataOut.str.strDen  = 0 ;
	MeasDataOut.str.wallDn  = 0 ;
	MeasDataOut.eA = 0 ;
	MeasDataOut.eS = 0 ;


	size_t sliceprint = info.sliceprint;
	int indexa = info.index;
	SpectrumMaskType mask = info.mask ;
	int redmap = info.redmap;
	StringMeasureType strmeas = info.strmeas;
	double radius_mask = info.rmask ;
	/* This is a change with respect to previous behaviour
	   Changes the definition of mask to be in units of ms^-1 */
		if (axiona->Lambda() == LAMBDA_Z2)
		 		radius_mask /= axiona->Msa(); // radius is a/msa in a units
		if (axiona->Lambda() == LAMBDA_FIXED){
			double si = sqrt(2.0*axiona->BckGnd()->Lambda())* (*axiona->RV())*axiona->BckGnd()->PhysSize()/axiona->Length() ;
				LogMsg(VERB_HIGH,"msa calaculated %f mask-parameter resized with 1/msa",si);
				radius_mask /= si;
		}

	auto	cTime = Timer();

	Profiler &prof = getProfiler(PROF_MEAS);
	/* marks the begginin*/
	LogOut("~");
	LogMsg(VERB_NORMAL, "[Meas %d] Measurement %d", indexa, measa);

	/* Save configuration, placed here to avoid running any test if MEAS_NOTHING
	but to save the configuration */
	if (measa & MEAS_3DMAP){
		LogMsg(VERB_NORMAL, "[Meas %d] Sav3 3D configuration", indexa);
		writeConf(axiona, indexa);
		MeasureType mesa = measa;
		measa = measa ^ MEAS_3DMAP ; // removes the map
		// LogOut("mesa %d measa %d MEAS3DMAP %d\n", mesa, measa, MEAS_3DMAP);
	}

	double z_now     = *axiona->zV();
	double R_now     = *axiona->RV();
	double saskia	   = axiona->Saskia();
	// fix?
	double shiftz	   = R_now * saskia;

	if (axiona->Field() == FIELD_AXION)
	 	shiftz = 0.0;

	if (measa != MEAS_NOTHING)
	{

	createMeas(axiona, indexa);

	if	( axiona->MMomSpace() || axiona->VMomSpace() )
	{
		{
			LogMsg(VERB_NORMAL, "[Meas %d] bin FS acceleration",indexa);
			float *ms = static_cast<float *>(axiona->m2Cpu()) ;
			//  LogOut("acceleration %f %f %f %f \n",ms[0],ms[1],ms[2],ms[3]);
			// JARE possible problem m2 saved as double in _DOUBLE?
			Binner<3000,Float> contBin(static_cast<Float *>(axiona->m2Cpu()), 2*axiona->Size(),
							[] (Float x) -> float { return (double) ( x ) ;});
			contBin.run();
			writeBinner(contBin, "/bins", "fsacceleration");
		}
		FTfield pelota(axiona);
		pelota(FIELD_MV, FFT_BCK); // FWD is to send to POSITION space
	}

	if (measa & MEAS_2DMAP){
			if(p2dmapo)
				writeMapHdf5s (axiona,sliceprint);
	}

	if (measa & MEAS_NEEDENERGY)
	{
		void *eRes;
		trackAlloc(&eRes, 128);
		memset(eRes, 0, 128);
		double *eR = static_cast<double *> (eRes);

		if (measa & MEAS_NEEDENERGYM2)
		{
			// LogOut("energy (map->m2) ");
			LogMsg(VERB_NORMAL, "[Meas %d] called energy + map->m2", indexa);
			energy(axiona, eRes, true, shiftz);

			MeasDataOut.eA = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]) ;
			MeasDataOut.eS = (eR[5] + eR[6] + eR[7] + eR[8] + eR[9]) ;

			if (measa & MEAS_ENERGY3DMAP){
				// LogOut("write eMap ");
				LogMsg(VERB_NORMAL, "[Meas %d] called writeEDens",indexa);
				writeEDens(axiona);
			}

			if (measa & MEAS_BINDELTA)
			{
				// LogOut("bindelta ");
				LogMsg(VERB_NORMAL, "[Meas %d] bin energy axion (delta)",indexa);
				// JARE possible problem m2 saved as double in _DOUBLE?
				float eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
				Binner<3000,Float> contBin(static_cast<Float *>(axiona->m2Cpu()), axiona->Size(),
								[eMean = eMean] (Float x) -> float { return (double) (log10(x/eMean)) ;});
				contBin.run();
				writeBinner(contBin, "/bins", "contB");
			}

			if (measa & MEAS_2DMAP)
			{
				if(p2dEmapo){
					LogMsg(VERB_NORMAL, "[Meas %d] 2D energy map",indexa);
					writeEMapHdf5s (axiona,sliceprint);
				}

				if(p2dPmapo){
					LogMsg(VERB_NORMAL, "[Meas %d] Proyection",indexa);
					if (axiona->Precision() == FIELD_DOUBLE){
						projectField	(axiona, [] (double x) -> double { return x*x ; } );
					}
					else{
						projectField	(axiona, [] (float x) -> float { return x*x ; } );
					}
					writePMapHdf5 (axiona);
				}

			}

			LogMsg(VERB_NORMAL, "[meas] M2 status %d", axiona->m2Status());
			if (measa & (MEAS_PSP_A | MEAS_REDENE3DMAP | MEAS_PSP_A))
			{

				SpecBin specAna(axiona, (pType & PROP_SPEC) ? true : false);

				if (measa & (MEAS_PSP_A | MEAS_REDENE3DMAP))
				{


					if( (axiona->Field() == FIELD_AXION) && (mask & SPMASK_AXIT)){
						prof.start();
						LogMsg(VERB_NORMAL, "[Meas %d] PSPA (masked axitons)",indexa);

						specAna.masker(radius_mask, SPMASK_AXIT);
						writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sPmasked");

						prof.stop();
						prof.add(std::string("PSPA_mask"), 0.0, 0.0);
					}

					if( (axiona->Field() == FIELD_AXION) && (mask & SPMASK_AXIT2)){
						prof.start();
						LogMsg(VERB_NORMAL, "[Meas %d] PSPA (masked axitons 2 radius_mask = %f)",indexa,radius_mask);

						specAna.masker(radius_mask, SPMASK_AXIT2);

						writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sPmasked2");

						prof.stop();
						prof.add(std::string("PSPA_mask"), 0.0, 0.0);
					}

						prof.start();
						LogMsg(VERB_NORMAL, "[Meas %d] PSPA",indexa);
						// at the moment runs PA and PS if in saxion mode
						// perhaps we should create another psRun() YYYEEEESSSSS
						specAna.pRun();
						writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");

						prof.stop();
						prof.add(std::string("PSPA"), 0.0, 0.0);

						//FIXME issue with endredmap?
						if (measa & MEAS_REDENE3DMAP){
								if ( redmap > 0 ){
									// prof.start();
									// size_t nena = sizeN/ ((size_t) redmap) ;
									// LogMsg(VERB_NORMAL, "[Meas %d] reduced energy map to N=%d by smoothing %d neig",indexa, redmap, nena);
									// specAna.filter(nena);
									// writeEDensReduced(axiona, indexa, redmap, redmap/((int) zGrid));
									// prof.stop();
									// prof.add(std::string("Reduced PSPA"), 0.0, 0.0);
										LogMsg(VERB_NORMAL, "[Meas %d] reduced energy map to N=%d by smoothing",sizeN, redmap);
										double ScaleSize = ((double) axiona->Length())/((double) redmap);
										double eFc  = 0.5*M_PI*M_PI*(ScaleSize*ScaleSize)/((double) axiona->Surf());
										size_t nLz = redmap / commSize();

										if (axiona->Precision() == FIELD_DOUBLE) {
											reduceField(axiona, redmap, nLz, FIELD_M2, [eFc = eFc] (int px, int py, int pz, complex<double> x) -> complex<double>
													 { return x*exp(-eFc*(px*px + py*py + pz*pz)); });
										} else {
											reduceField(axiona, redmap, nLz, FIELD_M2, [eFc = eFc] (int px, int py, int pz, complex<float>  x) -> complex<float>
													 { return x*((float) exp(-eFc*(px*px + py*py + pz*pz))); });
										}
									writeEDens (axiona);
								}
						}
				}
				if ( (measa & MEAS_PSP_S) && (axiona->Field() == FIELD_SAXION))
				{
						// LogOut("PSPS ");
						// LogMsg(VERB_NORMAL, "[Meas %d] PSPS",index);
						// has been computed before
						// JAVI : SURE PROBLEM OF PSA PSS FILTER
						// specAna.pSRun();
						// writeArray(specSAna.data(SPECTRUM_PS), specSAna.PowMax(), "/pSpectrum", "sPS");
				}
			}
		} // no m2 map
		else{
			// LogOut("energy (sum)");
			LogMsg(VERB_NORMAL, "[Meas %d] called energy (no map)",indexa);
			energy(axiona, eRes, false, shiftz);
		}

		LogMsg(VERB_NORMAL, "[Meas %d] write energy",indexa);
		writeEnergy(axiona, eRes);

		trackFree(eRes);
	}

	if(axiona->Field() == FIELD_SAXION){
		if ( (measa & (MEAS_STRING | MEAS_STRINGMAP | MEAS_STRINGCOO | MEAS_MASK)) || (mask & SPMASK_REDO))
		{

			if ( !(measa & MEAS_STRINGCOO)){
					LogMsg(VERB_NORMAL, "[Meas %d] string",indexa);
					MeasDataOut.str = strings(axiona);
					MeasDataOut.str = stringlength(axiona,MeasDataOut.str,strmeas);

					if ( measa & MEAS_STRINGMAP )
					{
						// LogOut("+map ");
						if (p3DthresholdMB/((double) MeasDataOut.str.strDen) > 1.)
						{
							LogMsg(VERB_NORMAL, "[Meas %d] string map",indexa);
							writeString(axiona, MeasDataOut.str, true);
						}
					}
					// else {
					else if ( !(measa & MEAS_MASK)) {
						writeString(axiona, MeasDataOut.str, false);
					}
			}
			else if (measa & MEAS_STRINGCOO){
				LogMsg(VERB_NORMAL, "[Meas %d] string2",indexa);
				MeasDataOut.str = strings2(axiona);
				MeasDataOut.str = stringlength(axiona,MeasDataOut.str,strmeas);
				if ( measa & MEAS_STRINGMAP ){
					LogMsg(VERB_NORMAL, "[Meas %d] string map'",indexa);
					writeString(axiona, MeasDataOut.str, true);
				}
				LogMsg(VERB_NORMAL, "[Meas %d] string coordinates",indexa);
				writeStringCo(axiona, MeasDataOut.str, true);
				//saves strings in m2//problem with energy
			}

		}
	}


	// if we are computing any spectrum, prepare the instance
	if (measa & MEAS_SPECTRUM)
	{
		SpecBin specAna(axiona, (pType & PROP_SPEC) ? true : false);

		/* this is an experimental print that uses axion energy plot2D and could use plot3D energy
		   is incompatible with a real output of energy density... */
		if (measa & MEAS_MASK)
		{
			LogMsg(VERB_NORMAL, "[Meas %d] Calculating MASK in m2",indexa);
				prof.start();
				specAna.masker(radius_mask, SPMASK_REDO);
				prof.stop();
				prof.add(std::string("Masker"), 0.0, 0.0);
				/* activate to export premask -- needs changes in spectrum.cpp too*/
					// writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/mSpectrum", "W0");
				/* activate this to print a 2D smooth mask */
				 	// writeEMapHdf5s (axiona,sliceprint);
				/* activate this to see the smooth mask */
					//writeEDens(axiona);
				/* activate this to see the binary mask */
				// writeString(axiona, MeasDataOut.str, true);
				writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/mSpectrum", "W_Red");
				if(strmeas & STRMEAS_ENERGY) {
					// measure the energy density of strings by using masked points
					MeasDataOut.strE = stringenergy(axiona);
					writeStringEnergy(axiona,MeasDataOut.strE);
				}
		}

		/* If Axion mode this is the only spectrum. In saxion an option */
		if ( ((axiona->Field() == FIELD_SAXION) && (measa & MEAS_NSP_A)) ||
	 			 ((axiona->Field() == FIELD_AXION) && (measa & (MEAS_NSP_A | SPMASK_VIL| SPMASK_REDO |SPMASK_VIL2 ))))
		{

			if ( ((axiona->Field() == FIELD_SAXION) && (mask & SPMASK_FLAT)) ||
						(axiona->Field() == FIELD_AXION) )
			{
				// LogOut("NSPA ");
				LogMsg(VERB_NORMAL, "[Meas %d] NSPA UNMASKED",indexa);
				prof.start();
				specAna.nRun(SPMASK_FLAT);
				prof.stop();
				prof.add(std::string("NSPA_FLAT"), 0.0, 0.0);

				writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
				writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
				// if (axiona->Field() == FIELD_AXION)
				// 	writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");
				if (axiona->AxionMassSq() > 0.0)
					writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");
			}

			if ( (axiona->Field() == FIELD_SAXION) && (mask & SPMASK_REDO))
			{
				// LogOut("NSPA ");
				LogMsg(VERB_NORMAL, "[Meas %d] NSPA MASK_RED",indexa);

				if ( !(measa & (MEAS_MASK)) ){
						LogMsg(VERB_NORMAL, "[Meas %d] MASK_TEST inside NSPA",indexa);
						prof.start();
						specAna.masker(radius_mask, SPMASK_REDO);
						prof.stop();
						prof.add(std::string("Masker"), 0.0, 0.0);

					  writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/mSpectrum", "W_Red");

						if(strmeas & STRMEAS_ENERGY) {
							// measure the energy density of strings by using masked points
							MeasDataOut.strE = stringenergy(axiona);
							writeStringEnergy(axiona,MeasDataOut.strE);
						}
				}
				LogMsg(VERB_NORMAL, "[Meas %d] Now the spectrum",indexa);
				prof.start();
				specAna.nRun(SPMASK_REDO);
				prof.stop();
				prof.add(std::string("NSPA_M"), 0.0, 0.0);

				writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK_Red");
				writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG_Red");
				// if (axiona->Field() == FIELD_AXION)
				// 	writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV_Red");
				if (axiona->AxionMassSq() > 0.0)
					writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV_Red");

				prof.start();
				specAna.matrixbuilder();
				prof.stop();
				prof.add(std::string("Matrix Builder"), 0.0, 0.0);

				writeArray(static_cast<double *>(axiona->m2Cpu()), specAna.PowMax()*specAna.PowMax(), "/mSpectrum", "M_Red");
			}

			if ( (axiona->Field() == FIELD_SAXION) && (mask & SPMASK_VIL))
			{
				// if (mask & SPMASK_FLAT)
				// 	specAna.reset0();

				// LogOut("NSPA ");
				LogMsg(VERB_NORMAL, "[Meas %d] NSPA masked-Villadoro",indexa);
					prof.start();
				specAna.nRun(SPMASK_VIL);
					prof.stop();
					prof.add(std::string("NSPA_Vi"), 0.0, 0.0);
				writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK_Vi");
				writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG_Vi");
				if (axiona->AxionMassSq() > 0.0)
					writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV_Vi");

				LogMsg(VERB_NORMAL, "[Meas %d] producing correction matrix",indexa);
					prof.start();
				specAna.wRun(SPMASK_VIL);
					prof.stop();
					prof.add(std::string("w Run"), 0.0, 0.0);
				writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/mSpectrum", "W_Vi");

					prof.start();
				specAna.matrixbuilder();
					prof.stop();
					prof.add(std::string("Matrix Builder"), 0.0, 0.0);
				writeArray(static_cast<double *>(axiona->m2Cpu()), specAna.PowMax()*specAna.PowMax(), "/mSpectrum", "M_Vi");
			}

			if ((axiona->Field() == FIELD_SAXION) && (mask & SPMASK_VIL2) )
			{
				// if ( (mask & SPMASK_FLAT) || (mask & SPMASK_VIL))
				// 	specAna.reset0();

				// LogOut("NSPA ");
				LogMsg(VERB_NORMAL, "[Meas %d] NSPA masked-Villadoro squared",indexa);
					prof.start();
				specAna.nRun(SPMASK_VIL2);
					prof.stop();
					prof.add(std::string("NSPA_Vi2"), 0.0, 0.0);

				writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK_Vi2");
				writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG_Vi2");
				// if (axiona->Field() == FIELD_AXION)
				// 	writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV_Vi2");
				if (axiona->AxionMassSq() > 0.0)
					writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV_Vi2");

				LogMsg(VERB_NORMAL, "[Meas %d] producing correction matrix",indexa);

					prof.start();
				specAna.wRun(SPMASK_VIL2);
					prof.stop();
					prof.add(std::string("w Run"), 0.0, 0.0);

					prof.start();
				specAna.matrixbuilder();
					prof.stop();
					prof.add(std::string("Matrix Builder"), 0.0, 0.0);

				writeArray(static_cast<double *>(axiona->m2Cpu()), specAna.PowMax()*specAna.PowMax(), "/mSpectrum", "M_Vi2");
			}

			if ( (axiona->Field() == FIELD_SAXION) && (mask & SPMASK_SAXI) )
			{
				// if ( (mask & SPMASK_FLAT) || (mask & SPMASK_VIL) || (mask & SPMASK_VIL2))
				// 	specAna.reset0();
				// LogOut("NSPA ");
				LogMsg(VERB_NORMAL, "[Meas %d] NSP real and imaginary",indexa);
				specAna.nRun(SPMASK_SAXI);
				writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK_Im");
				writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sK_Re");
				writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG_Im");
				writeArray(specAna.data(SPECTRUM_PS), specAna.PowMax(), "/nSpectrum", "sG_Re");
			}

		}

		if ( (axiona->Field() == FIELD_SAXION) && (measa & MEAS_NSP_S))
		{
			if (axiona->Field() == FIELD_SAXION){
				// LogOut("NSPS ");
				LogMsg(VERB_NORMAL, "[Meas %d] NSPS ",indexa);
				specAna.nSRun();
				writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sKS");
				writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sGS");
				writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sVS");
			}
		}

		if ( (indexa == 0) || (measa & MEAS_NNSPEC) )
		{
			// LogOut("Nmod ");
			LogMsg(VERB_NORMAL, "[Meas %d] Nmod ", indexa);
				prof.start();
			specAna.nmodRun();
				prof.stop();
				prof.add(std::string("nmod Run"), 0.0, 0.0);
			writeArray(specAna.data(SPECTRUM_NN), specAna.PowMax(), "/nSpectrum", "nmodes");
		}

		if ( (indexa == 0) || (measa & MEAS_NNSPEC) )
		{
			// LogOut("Nmod ");
			LogMsg(VERB_NORMAL, "[Meas %d] average K ", indexa);
				prof.start();
			specAna.avekRun();
				prof.stop();
				prof.add(std::string("avek Run"), 0.0, 0.0);

			writeArray(specAna.data(SPECTRUM_AK), specAna.PowMax(), "/nSpectrum", "averagek");
		}
	}




	if (axiona->Field() == FIELD_SAXION){

			if (measa & MEAS_BINTHETA)
			{
				// LogOut("binT ");
				LogMsg(VERB_NORMAL, "[Meas %d] bin theta",indexa);
					// Float shs = shiftz;
					// complex<Float> shhhs = (shs,0.);
					// Binner<3000,complex<Float>> thBin(static_cast<complex<Float> *>(axiona->mCpu()) + axiona->Surf(), axiona->Size(),
					// 				 [s=shhhs] (complex<Float> x) { return (double) arg(x-s); });
					Binner<3000,complex<Float>> thBin(static_cast<complex<Float> *>(axiona->mCpu()) + axiona->Surf(), axiona->Size(),
									 [] (complex<Float> x) { return (double) arg(x); });
					thBin.run();
					writeBinner(thBin, "/bins", "thetaB");
					MeasDataOut.maxTheta = max(abs(thBin.min()),thBin.max());
			}
				if (measa & MEAS_BINRHO)
				{
					// LogOut("binR ");
					LogMsg(VERB_NORMAL, "[Meas %d] bin rho",indexa);
					// Float z_now = *axiona->zV();
					// Float shs = shiftz;
					// complex<Float> shhhs = (shs,0.);
					// Binner<3000,complex<Float>> rhoBin(static_cast<complex<Float> *>(axiona->mCpu()) + axiona->Surf(), axiona->Size(),
					// 					[z=z_now,s=shhhs] (complex<Float> x) { return (double) abs(x-s)/z; } );
					Binner<3000,complex<Float>> rhoBin(static_cast<complex<Float> *>(axiona->mCpu()) + axiona->Surf(), axiona->Size(),
										[z=R_now] (complex<Float> x) { return (double) abs(x)/z; } );
					rhoBin.run();
					writeBinner(rhoBin, "/bins", "rhoB");
				}
					if (measa& MEAS_BINLOGTHETA2)
					{
						// LogOut("binL ");
						LogMsg(VERB_NORMAL, "[Meas %d] bin log10 theta^2 ",indexa);
						Binner<3000,complex<Float>> logth2Bin(static_cast<complex<Float> *>(axiona->mCpu()) + axiona->Surf(), axiona->Size(),
										 [] (complex<Float> x) { return (double) log10(1.0e-10+pow(arg(x),2)); });
						logth2Bin.run();
						writeBinner(logth2Bin, "/bins", "logtheta2B");
					}
	}
	else{ // FIELD_AXION
		if (measa & MEAS_BINTHETA)
		{
			// LogOut("binthetha ");
			LogMsg(VERB_NORMAL, "[Meas %d] bin theta ",indexa);
				Binner<3000,Float> thBin(static_cast<Float *>(axiona->mCpu()) + axiona->Surf(), axiona->Size(),
								 [z=R_now] (Float x) { return (double) (x/z); });
				thBin.run();
				writeBinner(thBin, "/bins", "thetaB");
				MeasDataOut.maxTheta = max(abs(thBin.min()),thBin.max());
		}
			// if (measa & MEAS_BINRHO)
			// {
			// 	LogMsg(VERB_NORMAL, "[Meas %d] bin rho called in axion mode. Ignored.",indexa);
			// }
				if (measa& MEAS_BINLOGTHETA2)
				{
					// LogOut("bintt2 ");
					LogMsg(VERB_NORMAL, "[Meas %d] bin log10 theta^2 ",indexa);
					Binner<3000,Float> logth2Bin2(static_cast<Float *>(axiona->mCpu()) + axiona->Surf(), axiona->Size(),
									 [z=R_now] (Float x) -> float { return (double) log10(1.0e-10+pow(x/z,2)); });
					logth2Bin2.run();
					writeBinner(logth2Bin2, "/bins", "logtheta2B");
				}
	}

	LogMsg(VERB_HIGH, "destroying meas",indexa);
	LogFlush();
	destroyMeas();
	}

	if ((indexa-1) % 10 == 0){
		LogOut("ctime  |  index |  cmeas |  wtime  | mass \n");
		LogOut(".");
		}

	LogOut("%2.3f  | ",z_now);

	if (cTime*1.e-6/3600. < 1.0 )
		LogOut("  %3d  | %6d | %2.3f m | ", indexa, info.measdata, cTime*1.e-6/60.);
		else
		LogOut("  %3d  | %6d | %2.3f h | ", indexa, info.measdata, cTime*1.e-6/3600.);

	double DWfun = 40*axiona->AxionMassSq()/(2.0*axiona->BckGnd()->Lambda()) ;
	if (axiona->Lambda() == LAMBDA_Z2)
		DWfun *= R_now*R_now;
	LogOut("%.1e %.1e (%.1e) ", axiona->AxionMass(), sqrt(axiona->SaxionMassSq()), DWfun );

	if ( axiona->Field() == FIELD_SAXION)
	{
		if ( measa & (MEAS_STRING | MEAS_STRINGMAP | MEAS_STRINGCOO) ){
			if (axiona->Lambda() == LAMBDA_Z2) {
		double loks = log(axiona->Msa()*z_now/axiona->Delta());
		double Le = axiona->BckGnd()->PhysSize();
			LogOut("log(%.1f) xi_t(%f) xi(%f) #_st %ld ", loks, xivilgor(loks),
				(1/6.)*axiona->Delta()*( (double) MeasDataOut.str.strDen)*z_now*z_now/(Le*Le*Le),
				MeasDataOut.str.strDen );
			} else if (axiona->Lambda() == LAMBDA_FIXED) {
				double loks = log(axiona->Msa()*z_now*z_now/axiona->Delta());
				double Le = axiona->BckGnd()->PhysSize();
					LogOut("log(%.1f) xi_t(%f) xi(%f) #_st %ld ", loks, xivilgorphys(loks),
						(1/6.)*axiona->Delta()*( (double) MeasDataOut.str.strDen)*z_now*z_now/(Le*Le*Le),
						MeasDataOut.str.strDen );
			}
		} else {
			// LogOut("str not measured (%ld, %ld) ",MeasDataOut.str.strDen, -1);
			LogOut("str not measured ");
		}
	} else {
		LogOut("maxth=%f ", MeasDataOut.maxTheta);
		LogOut(" ... ");
	}

	LogOut("\n");


return MeasDataOut;
}

MeasData	Measureme  (Scalar *axiona,  MeasInfo infa)
{
	if (axiona->Precision() == FIELD_SINGLE)
	{
		return Measureme<float> (axiona,  infa);
	}
	else
	{
		return Measureme<double>(axiona,  infa);
	}
}

double xivilgor(double logi){
	return (249.48 + 38.8431*logi + 1086.06* logi*logi)/(21775.3 + 3665.11*logi)  ;
}

double xivilgorphys(double logi){
	return (9.31021 + 1.38292e-6*logi + 0.713821*logi*logi)/(42.8748 + 0.788167*logi)  ;
}
