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

using namespace std;

template<typename Float>
MeasData	Measure	(Scalar *axion, int index, MeasureType meas)
{
	MeasData	retMeas;

	retMeas.maxTheta   = M_PI;
	retMeas.str.strDen = -1;
	retMeas.str.wallDn = -1;

//	auto	cTime = Timer();

	if	(meas & MEAS_3DMAP) {
		LogOut("Dumping configuration to disk\n");
		writeConf(axion, index);
	}

	double zNow   = *axion->zV();
	double saskia =  axion->Saskia();
	double zShift = zNow * saskia;

	if (axion->Field() == FIELD_SAXION)
	 	zShift = 0.0;

	createMeas(axion, index);

	if	(meas & MEAS_2DMAP) {
		//LogOut("Dumping 2D slice\n");
		if(p2dmapo)
			writeMapHdf5s (axion, 0);	// I set sliceprint to 0
	}

	if (meas & MEAS_NEEDENERGY) {

		void *eRes;
		trackAlloc(&eRes, 128);
		memset(eRes, 0, 128);
		double *eR = static_cast<double *> (eRes);

		if 	(meas & MEAS_NEEDENERGYM2) {
			// LogOut("energy (map->m2) ");
//			LogMsg(VERB_NORMAL, "[Meas %d] called energy + map->m2",index);
			energy(axion, eRes, true, zShift);

			if	(meas & MEAS_BINDELTA) {
				// LogOut("bindelta ");
//				LogMsg(VERB_NORMAL, "[Meas %d] bin energy axion (delta)",index);
				// JARE possible problem m2 saved as double in _DOUBLE?
				double eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
				Binner<3000,Float> contBin(static_cast<Float *>(axion->m2Cpu()), axion->Size(),
								[eMean = eMean] (Float x) -> double { return (double) (log10(x/eMean) );});
				contBin.run();
				writeBinner(contBin, "/bins", "contB");
			}

			if (meas & MEAS_ENERGY3DMAP) {
				// LogOut("write eMap ");
//				LogMsg(VERB_NORMAL, "[Meas %d] called writeEDens",index);
				writeEDens(axion);
			}
		} else { // no m2 map
			// LogOut("energy (sum)");
//			LogMsg(VERB_NORMAL, "[Meas %d] called energy (no map)", index);
			energy(axion, eRes, false, zShift);
		}

//		LogMsg(VERB_NORMAL, "[Meas %d] write energy", index);
		writeEnergy(axion, eRes);
	}

	// if we are computing any spectrum, prepare the instance
	if	(meas & MEAS_SPECTRUM) {
		SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);

		if	(meas & MEAS_PSP_A) {
			// LogOut("PSPA ");
//			LogMsg(VERB_NORMAL, "[Meas %d] PSPA",index);
			// at the moment runs PA and PS if in saxion mode
			// perhaps we should create another psRun() YYYEEEESSSSS
			specAna.pRun();
			writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");
		}
		// move after PSP!
		if	(meas & MEAS_REDENE3DMAP) {
			if	(endredmap > 0) {
				// LogOut("redMap->%d! ",sizeN/endredmap);
//				LogMsg(VERB_NORMAL, "[Meas %d] reduced energy map to %d neig",index,sizeN/endredmap);
				int nena = sizeN/endredmap ;
				specAna.filter(nena);
				writeEDensReduced(axion, index, endredmap, endredmap/zGrid);
			}
		}

		if	((meas & MEAS_PSP_S) & (axion->Field() == FIELD_SAXION)) {
			// LogOut("PSPS ");
//			LogMsg(VERB_NORMAL, "[Meas %d] PSPS",index);
			// has been computed before
			// JAVI : SURE PROBLEM OF PSA PSS FILTER
			// specAna.pSRun();
			// writeArray(specSAna.data(SPECTRUM_PS), specSAna.PowMax(), "/pSpectrum", "sPS");
		}

		if	(meas & MEAS_NSP_A) {
			// LogOut("NSPA ");
//			LogMsg(VERB_NORMAL, "[Meas %d] NSPA",index);
			specAna.nRun();
			writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
			writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");

			if (axion->Field() == FIELD_AXION)
				writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");
		}

		if	(meas & MEAS_NSP_S) {
			// LogOut("NSPS ");
//			LogMsg(VERB_NORMAL, "[Meas %d] NSPS ",index);
			specAna.nSRun();
			writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sKS");
			writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sGS");
			writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sVS");
		}

		if	(meas & MEAS_NNSPEC) {
			// LogOut("Nmod ");
//			LogMsg(VERB_NORMAL, "[Meas %d] Nmod ",index);
			specAna.nmodRun();
			writeArray(specAna.data(SPECTRUM_PS), specAna.PowMax(), "/nSpectrum", "nmodes");
		}
	}

	if	(axion->Field() == FIELD_SAXION) {

		if	(meas & MEAS_BINTHETA) {
			// LogOut("binT ");
//			LogMsg(VERB_NORMAL, "[Meas %d] bin theta",index);
			Binner<3000,complex<Float>> thBin(static_cast<complex<Float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
							  [] (complex<Float> x) { return (double) arg(x); });
			thBin.run();
			writeBinner(thBin, "/bins", "thetaB");
			retMeas.maxTheta = max(abs(thBin.min()),thBin.max());
		}

		if	(meas & MEAS_BINRHO) {
			// LogOut("binR ");
//			LogMsg(VERB_NORMAL, "[Meas %d] bin rho",index);
			float zNow = *axion->zV();
			Binner<3000,complex<Float>> rhoBin(static_cast<complex<Float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
							   [z=zNow] (complex<Float> x) { return (double) abs(x)/z; } );
			rhoBin.run();
			writeBinner(rhoBin, "/bins", "rhoB");
		}

		if	(meas& MEAS_BINLOGTHETA2) {
			// LogOut("binL ");
//			LogMsg(VERB_NORMAL, "[Meas %d] bin log10 theta^2 ",index);
			Binner<3000,complex<Float>> logth2Bin(static_cast<complex<Float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
							      [] (complex<Float> x) { return (double) log10(1.0e-10+pow(arg(x),2)); });
			logth2Bin.run();
			writeBinner(logth2Bin, "/bins", "logtheta2B");
		}

		if	(meas & MEAS_STRING) {
			// LogOut("string ");
//			LogMsg(VERB_NORMAL, "[Meas %d] string",index);
			retMeas.str = strings(axion);

			if (meas & MEAS_STRINGMAP)
			{
				// LogOut("+map ");
//				LogMsg(VERB_NORMAL, "[Meas %d] string map",index);
				if (p3DthresholdMB/((double) retMeas.str.strDen) > 1.)
					writeString(axion, retMeas.str, true);
				else
					writeString(axion, retMeas.str, false);
			} else {
				// LogOut("string alone ");
			}
		}

	} else { // FIELD_AXION
		retMeas.str.strDen = 0;
		retMeas.str.wallDn = 0;

		if	(meas & MEAS_BINTHETA)
		{
			// LogOut("binthetha ");
//			LogMsg(VERB_NORMAL, "[Meas %d] bin theta ",index);
			Binner<3000,Float> thBin(static_cast<Float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
						 [z=zNow] (Float x) -> double { return (double) arg(x); });
			thBin.run();
			writeBinner(thBin, "/bins", "thetaB");
			retMeas.maxTheta = max(abs(thBin.min()),thBin.max());
		}

//		if	(meas & MEAS_BINRHO)
//			LogMsg(VERB_NORMAL, "[Meas %d] bin rho called in axion mode. Ignored.",index);

		if	(meas & MEAS_BINLOGTHETA2) {
			// LogOut("bintt2 ");
//			LogMsg(VERB_NORMAL, "[Meas %d] bin log10 theta^2 ",index);
			Binner<3000,Float> logth2Bin2(static_cast<Float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
							[z=zNow] (Float x) -> double { return (double) log10(1.0e-10+pow(x/z,2)); });
			logth2Bin2.run();
			writeBinner(logth2Bin2, "/bins", "logtheta2B");
		}
	}

//	LogOut("z=%2.3f | ", zNow);

	// No tío, no hagas esto, usa un profiler, que acumula todas las medidas
/*
	if	(cTime*1.e-6/3600. < 1.0 )
		LogOut("index %d meas %d wtime %2.3f m ", index, meas, cTime*1.e-6/60.);
	else
		LogOut("index %d meas %d wtime %2.3f h ", index, meas, cTime*1.e-6/3600.);
*/

	if	(axion->Field() == FIELD_SAXION) {
		double DWfun = 40*axion->AxionMassSq()/(2.0*axion->BckGnd()->Lambda()) ;

		if (axion->Lambda() == LAMBDA_Z2)
			DWfun *= zNow*zNow;

		// double maa = 40*axion->AxionMassSq()/(2.*axion->myCosmos.Lambda());
		// if (axion->Lambda() == LAMBDA_Z2 )
		// 	maa = maa*zNow*zNow;
//		double Le = axion->BckGnd()->PhysSize();
//		LogOut("alpha=%.3e ", DWfun );
//		LogOut("#_st %ld xi(%f) ", nStrings, (1/6.)*axion->Delta()*( (double) nStrings)*zNow*zNow/(Le*Le*Le));
	} else {
		// LogOut("z=%f | dz=%.3e | maxtheta=%f | ", zloop, nLoops, (*axion->zV()), dzaux, maximumtheta);
//		LogOut(" ... ");
	}

//	LogOut("\n");
	destroyMeas();

	return	retMeas;
}

MeasData	Measure	(Scalar *axion, int index, MeasureType meas)
{
	if	(axion->Precision() == FIELD_SINGLE)
		return	Measure<float> (axion,  index,  meas);
	else
		return	Measure<double>(axion,  index,  meas);
}

