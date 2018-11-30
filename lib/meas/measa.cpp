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
#include "spectrum/spectrum.h"
#include "projector/projector.h"

using namespace std;

double xivilgor(double logi);

template<typename Float>
MeasData	Measureme  (Scalar *axiona, MeasInfo info, MeasureType measa)
{
	MeasData MeasDataOut;

	MeasDataOut.maxTheta       = -1 ;
	MeasDataOut.str.strDen = -1 ;
	MeasDataOut.str.wallDn = -1 ;
	MeasDataOut.eA = 0 ;
	MeasDataOut.eS = 0 ;

	size_t sliceprint = info.sliceprint;
	int indexa = info.index;
	SpectrumMaskType mask = info.mask ;

	auto	cTime = Timer();

	if (measa & MEAS_3DMAP){
		LogOut("3D conf ");
		writeConf(axiona, indexa);
	}

	double z_now     = *axiona->zV();
	double R_now     = *axiona->RV();
	double saskia	   = axiona->Saskia();
	// fix?
	double shiftz	   = R_now * saskia;

	if (axiona->Field() == FIELD_AXION)
	 	shiftz = 0.0;

	createMeas(axiona, indexa);

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

			if (measa & MEAS_ENERGY3DMAP){
				// LogOut("write eMap ");
				LogMsg(VERB_NORMAL, "[Meas %d] called writeEDens",indexa);
				writeEDens(axiona);
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

		} // no m2 map
		else{
			// LogOut("energy (sum)");
			LogMsg(VERB_NORMAL, "[Meas %d] called energy (no map)",indexa);
			energy(axiona, eRes, false, shiftz);
		}

		LogMsg(VERB_NORMAL, "[Meas %d] write energy",indexa);
		writeEnergy(axiona, eRes);
	}

	// if we are computing any spectrum, prepare the instance
	if (measa & MEAS_SPECTRUM)
	{
		SpecBin specAna(axiona, (pType & PROP_SPEC) ? true : false);

		if (measa & MEAS_PSP_A)
		{
				// LogOut("PSPA ");
				LogMsg(VERB_NORMAL, "[Meas %d] PSPA",indexa);
				// at the moment runs PA and PS if in saxion mode
				// perhaps we should create another psRun() YYYEEEESSSSS
				specAna.pRun();
				writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");
		}
		// move after PSP!
		if (measa & MEAS_REDENE3DMAP){
				if ( endredmap > 0){
					// LogOut("redMap->%d! ",sizeN/endredmap);
					LogMsg(VERB_NORMAL, "[Meas %d] reduced energy map to %d neig",indexa,sizeN/endredmap);
					int nena = sizeN/endredmap ;
					specAna.filter(nena);
					writeEDensReduced(axiona, indexa, endredmap, endredmap/zGrid);
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

		if (measa & MEAS_NSP_A)
		{

			if (mask & SPMASK_FLAT)
			{
				// LogOut("NSPA ");
				LogMsg(VERB_NORMAL, "[Meas %d] NSPA",indexa);
				specAna.nRun(SPMASK_FLAT);
				writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
				writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
				if (axiona->Field() == FIELD_AXION)
					writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");
			}

			if (mask & SPMASK_VIL)
			{
				// if (mask & SPMASK_FLAT)
				// 	specAna.reset0();

				// LogOut("NSPA ");
				LogMsg(VERB_NORMAL, "[Meas %d] NSPA masked-Villadoro",indexa);
				specAna.nRun(SPMASK_VIL);
				writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sKVi");
				writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sGVi");
				if (axiona->Field() == FIELD_AXION)
					writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sVVi");
			}

			if (mask & SPMASK_VIL2)
			{
				// if ( (mask & SPMASK_FLAT) || (mask & SPMASK_VIL))
				// 	specAna.reset0();

				// LogOut("NSPA ");
				LogMsg(VERB_NORMAL, "[Meas %d] NSPA masked-Villadoro squared",indexa);
				specAna.nRun(SPMASK_VIL2);
				writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sKVi2");
				writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sGVi2");
				if (axiona->Field() == FIELD_AXION)
					writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sVVi2");
			}

			if (mask & SPMASK_SAXI)
			{
				// if ( (mask & SPMASK_FLAT) || (mask & SPMASK_VIL) || (mask & SPMASK_VIL2))
				// 	specAna.reset0();

				// LogOut("NSPA ");
				LogMsg(VERB_NORMAL, "[Meas %d] NSP real and imaginary",indexa);
				specAna.nRun(SPMASK_SAXI);
				writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sKIm");
				writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sKRe");
				writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sGIm");
				writeArray(specAna.data(SPECTRUM_PS), specAna.PowMax(), "/nSpectrum", "sGRe");
			}
		}

		if (measa & MEAS_NSP_S)
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

		if (measa & MEAS_NNSPEC)
		{
			// LogOut("Nmod ");
			LogMsg(VERB_NORMAL, "[Meas %d] Nmod ", indexa);
			specAna.nmodRun();
			writeArray(specAna.data(SPECTRUM_PS), specAna.PowMax(), "/nSpectrum", "nmodes");
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

			if (measa & MEAS_STRING)
			{

				// LogOut("string ");
				LogMsg(VERB_NORMAL, "[Meas %d] string",indexa);
				MeasDataOut.str = strings(axiona);

				if (measa & MEAS_STRINGMAP)
				{
					// LogOut("+map ");
					LogMsg(VERB_NORMAL, "[Meas %d] string map",indexa);
					if (p3DthresholdMB/((double) MeasDataOut.str.strDen) > 1.)
						writeString(axiona, MeasDataOut.str, true);
					else
						writeString(axiona, MeasDataOut.str, false);
				}
				else if (measa & MEAS_STRINGCOO){
					LogMsg(VERB_NORMAL, "[Meas %d] string coordinates",indexa);
					writeString2(axiona, MeasDataOut.str, true);
				}
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

	if ((indexa-1) % 10 == 0)
		LogOut("ctime  |  index |  cmeas |  wtime  | mass \n");

	LogOut("%2.3f  | ",z_now);

	if (cTime*1.e-6/3600. < 1.0 )
		LogOut("  %3d  | %d | %2.3f m | ", indexa, measa, cTime*1.e-6/60.);
		else
		LogOut("  %3d  | %d | %2.3f h | ", indexa, measa, cTime*1.e-6/3600.);

	double DWfun = 40*axiona->AxionMassSq()/(2.0*axiona->BckGnd()->Lambda()) ;
	if (axiona->Lambda() == LAMBDA_Z2)
		DWfun *= R_now*R_now;
	LogOut("%.1e %.1e (%.1e) ", axiona->AxionMass(), sqrt(axiona->SaxionMassSq()), DWfun );

	if ( axiona->Field() == FIELD_SAXION)
	{
		double loks = log(axiona->Msa()*z_now/axiona->Delta());
		double Le = axiona->BckGnd()->PhysSize();
			LogOut("log(%.1f) xi_t(%f) xi(%f) #_st %ld ", loks, xivilgor(loks),
				(1/6.)*axiona->Delta()*( (double) MeasDataOut.str.strDen)*z_now*z_now/(Le*Le*Le),
				MeasDataOut.str.strDen );
	} else {
		LogOut("maxth=%f ", MeasDataOut.maxTheta);
		LogOut(" ... ");
	}

	LogOut("\n");
destroyMeas();

return MeasDataOut;
}

MeasData	Measureme  (Scalar *axiona,  MeasInfo infa, MeasureType measa)
{
	if (axiona->Precision() == FIELD_SINGLE)
	{
		return Measureme<float> (axiona,  infa,  measa);
	}
	else
	{
		return Measureme<double>(axiona,  infa,  measa);
	}
}

double xivilgor(double logi){
	return (249.48 + 38.8431*logi + 1086.06* logi*logi)/(21775.3 + 3665.11*logi)  ;
}
