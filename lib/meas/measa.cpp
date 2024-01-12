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

#include "fft/fftCode.h"

#include "utils/kgvops.h"

using namespace std;
using namespace profiler;

double xivilgor(double logi);

template<typename Float>
MeasData	Measureme  (Scalar *axiona, MeasInfo info)
{
	auto	cTime = Timer();

	Profiler &prof = getProfiler(PROF_MEAS);
	/* marks the begginin*/
	LogOut("~");
	LogMsg(VERB_NORMAL, "\n ");
	LogMsg(VERB_NORMAL, "[Meas %d] MEASUREMENT %d, MAP %d, NRT %d, SPMASK %d ctime %2.3f\n", info.index, info.measdata,
	info.maty, info.nrt, info.mask, *axiona->zV());

	bool wasGPU = false;
	if (cDev == DEV_GPU){
		LogMsg (VERB_HIGH, "[Meas ] Transferring configuration to CPU");
		axiona->transferCpu(FIELD_MV);
		axiona->setFolded(false);
		if (info.measCPU){
			axiona->setDev(DEV_CPU);
			wasGPU = true;
		}
	}

	/* Define nicer variables for the rest of the file for readability */

	MeasureType measa = info.measdata ;
	MeasData MeasDataOut;

	MeasDataOut.maxTheta    = -1 ;
	MeasDataOut.str.strDen  = 0 ;
	MeasDataOut.str.wallDn  = 0 ;
	MeasDataOut.eA = 0 ;
	MeasDataOut.eS = 0 ;

	size_t sliceprint = info.sliceprint;
	int indexa = info.index;

	/* Prepare spectrum mask issues */

	SpectrumMaskType mask = info.mask ;
	LogMsg(VERB_HIGH,"[Meas ...] spmtype, mask passed = %d",mask);
	if (axiona->Field() == FIELD_SAXION)
		mask = mask & (SPMASK_FLAT | SPMASK_VIL | SPMASK_VIL2 | SPMASK_REDO | SPMASK_GAUS | SPMASK_DIFF | SPMASK_BALL);
	else if (axiona->Field() == FIELD_AXION){
		// if (mask & (SPMASK_VIL | SPMASK_VIL2 | SPMASK_REDO | SPMASK_GAUS | SPMASK_DIFF))
		// 	mask = mask | SPMASK_FLAT;
		mask = mask & (SPMASK_FLAT | SPMASK_AXIT | SPMASK_AXIT2 | SPMASK_AXITV);
		LogMsg(VERB_HIGH,"[Meas ...] spmtype, mask corrected = %d",mask);
		}

	int redmap = info.redmap;
	StringMeasureType strmeas = info.strmeas;
	double radius_mask = info.rmask ; //obsolete?
	int irmask = info.i_rmask;
	std::vector<double> rmasktab = info.rmask_tab ;
	std::vector<double> rmasklabel = info.rmask_tab ; // use rmasklabels from command line (1/ms units) rather than lattice units
	/* Patch to make the cases where no radius is needed to work */
	if ( irmask == 0 ){
		irmask = 1;
		rmasktab.push_back(0.0);
	}
	LogMsg(VERB_PARANOID,"[Meas ...] rmask %.2f ",radius_mask);
	LogMsg(VERB_HIGH,    "[Meas ...] number of mask radii %d ",irmask);
	for (int ii=0; ii<irmask; ii++)
		LogMsg(VERB_PARANOID,    "[Meas ...] rmask #%d %.2f ",ii,rmasktab[ii]);
	nRunType nruntype = info.nrt;
	
	bool onlymaskenergy = info.maskenergyonly;

	/* This is a change with respect to previous behaviour
	   Changes the definition of mask to be in units of ms^-1 (saxion)
		 or ma^-1 (axion)
		 only if there is something to mask, of course all masks have mask > 1; 1 is FLAT*/
		if (mask > 1){
			double msa_aux ;
			if (axiona->Field() == FIELD_SAXION){
					if (axiona->LambdaT() == LAMBDA_Z2)
							msa_aux = axiona->Msa();
					if ((axiona->LambdaT() == LAMBDA_FIXED) || (axiona->LambdaT() == LAMBDA_CONF))
							msa_aux = sqrt(2.0*axiona->LambdaP())*(*axiona->RV())*axiona->BckGnd()->PhysSize()/axiona->Length() ;

				LogMsg(VERB_HIGH,"[Meas ...] msa = %f rmask-parameter interpreted in 1/ms units. ",msa_aux);
				LogMsg(VERB_HIGH,"           Internally converted to lattice units by multiplying with 1/msa",msa_aux);

				radius_mask /= msa_aux;
				for (size_t i=0;i<irmask;i++){
					LogMsg(VERB_HIGH,"[Meas ...] rmask %f -> %f ",rmasktab[i],rmasktab[i]/msa_aux);
					rmasktab[i] /= msa_aux;
				}
			} else { //AXION< PAXION NAXION
				// msa_aux = axiona->AxionMass()*(*axiona->RV())*axiona->BckGnd()->PhysSize()/axiona->Length();
				// LogMsg(VERB_HIGH,"[Meas ...] msa = %f rmask-parameter interpreted in 1/ma_c units. ",msa_aux);
				// LogMsg(VERB_HIGH,"           Internally converted to lattice units by multiplying with 1/maa",msa_aux);
				// LogMsg(VERB_HIGH,"           We use min r*(2 + 1/ma_c)");
				//
				// radius_mask *= 2. + 1./msa_aux;
				// for (size_t i=0;i<irmask;i++){
				// 	LogMsg(VERB_HIGH,"[Meas ...] rmask %f -> %f ",rmasktab[i],rmasktab[i]*(1.+1./msa_aux));
				// 	rmasktab[i] *= 2. + 1./msa_aux;
				// }
				// msa_aux = axiona->AxionMass()*(*axiona->RV())*axiona->BckGnd()->PhysSize()/axiona->Length();
				LogMsg(VERB_HIGH,"[Meas ...] rmask-parameter interpreted in delta units ");
			}
	LogMsg(VERB_NORMAL,"cummask %d %d %d",info.cummask,deninfa.cummask,cummask);




		}



	/* Save configuration, placed here to avoid running any test if MEAS_NOTHING
	but to save the configuration */
	if (measa & MEAS_3DMAP){
		LogMsg(VERB_NORMAL, "[Meas %d] Sav3 3D configuration", indexa);
		writeConf(axiona, indexa);
#ifdef USE_NYX_OUTPUT
		LogMsg(VERB_NORMAL, "[Meas %d] Save NYX 3D conf", indexa);
		writeConfNyx(axiona,indexa);
#endif
		MeasureType mesa = measa;
		measa = measa ^ MEAS_3DMAP ; // removes the map
		// LogOut("mesa %d measa %d MEAS3DMAP %d\n", mesa, measa, MEAS_3DMAP);
	}

	double z_now     = *axiona->zV();
	double R_now     = *axiona->RV();
	double saskia	   = axiona->Saskia();
	// fix?
	double shiftz	   = R_now * saskia;

	if (axiona->Field() != FIELD_SAXION)
	 	shiftz = 0.0;

	if (measa != MEAS_NOTHING)
	{

	createMeas(axiona, indexa);

	/* Use M2, M2h, SD before they are marked as garbage */

		/* If map of mendtheta is present use it */
		if ( (axiona->Field() == FIELD_AXION) && (axiona->sDStatus() == SD_MENDMAP) && (measa & MEAS_MASK)) {
			writeString(axiona, MeasDataOut.str, true);
			}


		LogMsg(VERB_HIGH, "[Meas %d] set aux fields to dirty",indexa);
		axiona->setM2(M2_DIRTY);
		axiona->setM2h(M2_DIRTY);
		axiona->setSD(SD_DIRTY);

	writeAttribute(&info.cTimesec, "Wall time [s]", H5T_NATIVE_DOUBLE);
	writeAttribute(&info.propstep, "Prop step #", H5T_NATIVE_INT);

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
		pelota(FIELD_MV, FFT_BCK); // FWD is to send to MOMENTUM space
	}

	// LogOut("[MS*] m  values %.2e %.2e %.2e %.2e \n",mmm[0],mmm[1],mmm[2],mmm[3]);
	// LogOut("[MS*] ms values %.2e %.2e %.2e %.2e \n",mss[0],mss[1],mss[2],mss[3]);
	// LogOut("[MS*] v  values %.2e %.2e %.2e %.2e \n",vvv[0],vvv[1],vvv[2],vvv[3]);
	// LogOut("[MS*] m2 values %.2e %.2e %.2e %.2e \n",mm2[0],mm2[1],mm2[2],mm2[3]);


	if( info.maty & MAPT_XYMV)
		writeMapHdf5s  (axiona,sliceprint);
	if( info.maty & MAPT_YZMV)
		writeMapHdf5s2 (axiona,sliceprint);

	//	--------------------------------------------------------------------------
	//
	//	ENERGY BLOCK
	//
	//	--------------------------------------------------------------------------

	bool mapsneedenergy = (info.maty & MAPT_XYPE2) || (info.maty & MAPT_XYPE) || (info.maty & MAPT_XYE);

	if ( (measa & MEAS_NEEDENERGY) || mapsneedenergy)
	{
		void *eRes;
			trackAlloc(&eRes, 512);
				memset(eRes, 0, 512);
					double *eR = static_cast<double *> (eRes);

		if ((measa & MEAS_NEEDENERGYM2) || mapsneedenergy)
		{
			// LogOut("energy (map->m2) ");
			LogMsg(VERB_NORMAL, "[Meas %d] called energy + map->m2", indexa);
				energy(axiona, eRes, EN_MAP, shiftz);

				MeasDataOut.eA = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]) ;
					MeasDataOut.eS = (eR[5] + eR[6] + eR[7] + eR[8] + eR[9]) ;

			if (measa & MEAS_ENERGY3DMAP){
				// LogOut("write eMap ");
				LogMsg(VERB_NORMAL, "[Meas %d] called writeEDens",indexa);
					writeEDens(axiona);
			}

			if (measa & MEAS_BINDELTA){
				// LogOut("bindelta ");
				LogMsg(VERB_NORMAL, "[Meas %d] bin energy axion (delta)",indexa);
					float eMean = MeasDataOut.eA;
						Binner<3000,Float> contBin(static_cast<Float *>(axiona->m2Cpu()), axiona->Size(),
							[eMean = eMean] (Float x) -> float { return (double) (log10(x/eMean)) ;});
								contBin.run();
									writeBinner(contBin, "/bins", "contB");
			}

			if (mapsneedenergy) {

				if(info.maty & MAPT_XYE){
					LogMsg(VERB_NORMAL, "[Meas %d] 2D energy map",indexa);
						writeEMapHdf5s (axiona,sliceprint);
				}

				if(info.maty & MAPT_XYPE){
					LogMsg(VERB_NORMAL, "[Meas %d] Proyection",indexa);
						if (axiona->Precision() == FIELD_DOUBLE){
							projectField	(axiona, [] (double x) -> double { return x ; } );
						}
							else {
								projectField	(axiona, [] (float x) -> float { return x ; } );
							}
								writePMapHdf5 (axiona);
				}

				if(info.maty & MAPT_XYPE2){
					LogMsg(VERB_NORMAL, "[Meas %d] Proyection energy squared",indexa);
						if (axiona->Precision() == FIELD_DOUBLE){
							projectField	(axiona, [] (double x) -> double { return x*x ; } );
						}
							else {
								projectField	(axiona, [] (float x) -> float { return x*x ; } );
							}
								writePMapHdf5 (axiona);
				}
			}

			LogMsg(VERB_NORMAL, "[meas] M2 status %d M2h status %d", axiona->m2Status(), axiona->m2hStatus());

			if (measa & (MEAS_PSP_A | MEAS_REDENE3DMAP | MEAS_PSP_S | MEAS_MULTICON)){

 				SpecBin specAna(axiona, (pType & (PROP_SPEC | PROP_FSPEC)) ? true : false, info);

				if (measa & (MEAS_PSP_A | MEAS_REDENE3DMAP | MEAS_MULTICON))
				{

					if( (axiona->Field() == FIELD_AXION) && (mask & SPMASK_AXIT)){
						LogMsg(VERB_NORMAL, "[Meas %d] PSPA (masked axitons)",indexa);

						for(int ii=0; ii < irmask; ii++){
							LogMsg(VERB_NORMAL, "[Meas %d] PSPA (masked axitons 1 radius_mask = %f)",indexa,rmasktab[ii]);
								char PRELABEL[256];
									sprintf(PRELABEL, "%s_%.2f", "sPmasked",rmasklabel[ii]);
										specAna.masker(rmasktab[ii], SPMASK_AXIT, M2_ENERGY, cummask);
											writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", PRELABEL);
										}
					}

					if( (axiona->Field() == FIELD_AXION) && (mask & SPMASK_AXIT2)){
						for(int ii=0; ii < irmask; ii++){
							LogMsg(VERB_NORMAL, "[Meas %d] PSPA (masked axitons 2 radius_mask = %f)",indexa,rmasktab[ii]);
								char PRELABEL[256];
									sprintf(PRELABEL, "%s_%.2f", "sPmasked2",rmasklabel[ii]);
										specAna.masker(rmasktab[ii], SPMASK_AXIT2, M2_ENERGY, cummask);
											writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", PRELABEL);
										}
					}

					if( (axiona->Field() == FIELD_AXION) && (mask & SPMASK_AXITV)){
						for(int ii=0; ii < irmask; ii++){
							LogMsg(VERB_NORMAL, "[Meas %d] PSPA (masked axitons V radius_mask = %f)",indexa,rmasktab[ii]);
								char PRELABEL[256];
									sprintf(PRELABEL, "%s", "sPmaskedV");
										specAna.masker(rmasktab[ii], SPMASK_AXITV, M2_ENERGY, cummask);
											writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", PRELABEL);
										}
					}

						LogMsg(VERB_NORMAL, "[Meas %d] PSPA",indexa);
							specAna.pRun();
								writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");


					if (measa & MEAS_MULTICON){
						LogMsg(VERB_NORMAL, "[Meas %d] Multi contrast tool",indexa);LogFlush();
						prof.start();
							/* requires M2_ENERGY or M2_ENERGY_FFT in M2 or M2h > here it gets it */
							/* loop over filters? */
							/* loop over interesting distances */
							size_t Nx = axiona->Length();
							double delta = axiona->BckGnd()->PhysSize()/Nx;
							int dsteps = 4*std::log2(Nx);

							{
							for (int id = 0; id < dsteps+1; id++){
								LogMsg(VERB_NORMAL,"Multicon %id\n",id);LogFlush();
								/* this is calibrated to make sense for FILTER_TOPHAT*/
								double smthi = 0.25*std::exp( std::log(4*Nx)*id/dsteps )	;
									double smth = smthi*delta;
										specAna.smoothFourier(smth, FILTER_TOPHAT);
											char FIL[256],LAB[256];
												sprintf(FIL, "Top Hat");
													/* bin delta */
														float eMean = MeasDataOut.eA;
															LogMsg(VERB_NORMAL, "[Meas %d] bin energy axion (delta) eA %.6e",indexa, MeasDataOut.eA);LogFlush();
																Binner<3000,Float> conaBin(static_cast<Float *>(axiona->m2Cpu()), axiona->Size()+2*Nx*axiona->Depth(),
																	[eMean = eMean] (Float x) -> float { return (double) (log10(x/eMean)) ;});
																		conaBin.setpad(Nx+2,Nx);
																			conaBin.run();
																					sprintf(LAB, "cont%03dB", id);
																						writeBinner(conaBin, "/bins", LAB);
																							sprintf(LAB, "/bins/cont%03dB", id);
																								writeAttributeg(&smth,LAB,"Smoothing Length",H5T_NATIVE_DOUBLE);
																									writeAttributeg(FIL,LAB,"Filter Type",H5T_C_S1);
/* print somehting if needed */
specAna.unpad(PFIELD_M2,PFIELD_M2);
sprintf(LAB, "/map/P%03d",id);
axiona->setM2(M2_ENERGY_SMOOTH);
if (axiona->Precision() == FIELD_DOUBLE){
projectField	(axiona, [] (double x) -> double { return x ; } );
} else {
projectField	(axiona, [] (float x) -> float { return x ; } );}
writePMapHdf5s (axiona, LAB);
							}}
							prof.stop();
								prof.add(std::string("Multiple contrast"), 0.0, 0.0);

						}

						if (measa & MEAS_REDENE3DMAP){
								if ( redmap > 0 ){
										size_t nLz = (axiona->Depth()*redmap)/axiona->Length();
										LogMsg(VERB_NORMAL, "[Meas %d] reduced energy map from Nx,Nz=%d,%d to Nx'Nz'=%d,%d by smoothing",indexa, axiona->Depth()*commSize(),
										redmap, nLz*commSize());
										double ScaleSize = ((double) axiona->Length())/((double) redmap);
										double eFc  = 0.5*M_PI*M_PI*(ScaleSize*ScaleSize)/((double) axiona->Surf());
										// size_t nLz = redmap / commSize();


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
			energy(axiona, eRes, EN_ENE, shiftz);
		}

		LogMsg(VERB_NORMAL, "[Meas %d] write energy",indexa);
		writeEnergy(axiona, eRes);

		if (axiona->Field() == FIELD_PAXION)
			MeasDataOut.maxTheta = static_cast<double *>(eRes)[TH_KIN];

		trackFree(eRes);
	}

	//	--------------------------------------------------------------------------
	//
	//	STRING BLOCK
	//
	//	--------------------------------------------------------------------------


	if(axiona->Field() == FIELD_SAXION){
		if ( (measa & (MEAS_STRING | MEAS_STRINGMAP | MEAS_STRINGCOO | MEAS_MASK)) || (mask & SPMASK_REDO | SPMASK_GAUS | SPMASK_DIFF) )
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


	//	--------------------------------------------------------------------------
	//
	//	SPECTRUM BLOCK
	//
	//	--------------------------------------------------------------------------


	/* Spectra, masks, nmodes, averaged k2 ... */
	if (measa & MEAS_SPECTRUM)
	{
 		SpecBin specAna(axiona, (pType & (PROP_SPEC | PROP_FSPEC)) ? true : false, info);

		/* this is an experimental print that uses axion energy plot2D and could use plot3D energy
		   is incompatible with a real output of energy density... */
		bool printedmask[8] = {false,false,false,false,false,false,false,false};

		if (measa & MEAS_MASK)
		{
			LogMsg(VERB_NORMAL, "[Meas %d] MEAS_MASK in m2 ",indexa);LogFlush();

				char PRELABEL[256];
				char LABEL[256];
				string           masklab[8] = {"Vi", "Vi2", "Bal", "Red", "Gau", "Dif", "Axit", "Axit2"};
				SpectrumMaskType maskara[8] = {SPMASK_VIL,SPMASK_VIL2,SPMASK_BALL,SPMASK_REDO,SPMASK_GAUS,SPMASK_DIFF,SPMASK_AXIT,SPMASK_AXIT2};
				bool             mulmask[8] = {false,false,true,true,true,true,true,true};

				LogMsg(VERB_NORMAL, "[Meas %d] masks are %d",indexa,mask);LogFlush();
				for (size_t i=0; i < 8; i++)
				{
					LogMsg(VERB_HIGH, "[Meas %d] maskara[%d]=%d",indexa,i,maskara[i]);LogFlush();
				}
				bool wEm = false;
				for (size_t i=0; i < 8; i++)
				{

					if ( !(mask & maskara[i])){
						LogMsg(VERB_HIGH, "[Meas %d] mask %s (%d) skipped",indexa,masklab[i].c_str(),i);LogFlush();
						continue;
					}
						LogMsg(VERB_HIGH, "[Meas %d] Measuring mask %s (%d) ... ",indexa,masklab[i].c_str(),i);LogFlush();

						for(int ii=0; ii < irmask; ii++)
						{
							if (mulmask[i])
								sprintf(PRELABEL, "%s_%.2f", masklab[i].c_str(),rmasklabel[ii]);
							else
								sprintf(PRELABEL, "%s", masklab[i].c_str());

							LogMsg(VERB_NORMAL, "[Meas %d] mask %s rmask %f [%d/%d]",indexa,masklab[i].c_str(),rmasktab[ii],ii+1,irmask);LogFlush();
							// prof.start();
							specAna.masker(rmasktab[ii], maskara[i], M2_ANTIMASK,cummask); // produces antimask in M2 to export
							// prof.stop();
							sprintf(LABEL, "Masker %s", masklab[i].c_str());
							// prof.add(std::string(LABEL), 0.0, 0.0);

							sprintf(LABEL, "W_%s", PRELABEL);
							writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/mSpectrum", LABEL);
							printedmask[i] = true;
							/* DANGER ZONE !*/
								/* activate to export premask -- needs changes in spectrum.cpp too*/
									// writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/mSpectrum", "W0");
								/* activate this to print a 2D smooth mask */


							if (axiona->Precision() == FIELD_DOUBLE){
								projectField	(axiona, [] (double x) -> double { return x ; } );
							}
							else{
								projectField	(axiona, [] (float x) -> float { return x ; } );
							}
							sprintf(LABEL, "/map/W_%s", PRELABEL);
							writePMapHdf5s (axiona, LABEL);
								/* activate this to see the smooth mask */
								// if ( !(measa & (MEAS_NSP_A | MEAS_NSP_S | MEAS_NNSPEC)) && !(measa & MEAS_ENERGY3DMAP) && !wEm)
								// {
								// 	writeEDens(axiona);
								// 	wEm = true;
								// }
								/* activate this to see the binary mask */
								// writeString(axiona, MeasDataOut.str, true);

								// if(strmeas & STRMEAS_ENERGY) {
								// 	// measure the energy density of strings by using masked points
								// 	MeasDataOut.strE = stringenergy(axiona);
								// 	MeasDataOut.strE.rmask = rmasktab[ii]; // this is not written by stringenergy();
								// 	writeStringEnergy(axiona,MeasDataOut.strE);
								// }
								if ( !mulmask[i])
									break;
							} //end for mask length
						} // end for mask type
				}  // end if mask






		if (measa & (MEAS_NSP_A | MEAS_NSP_S | MEAS_NNSPEC))
		{
			LogMsg(VERB_NORMAL, "\n",indexa);
			LogMsg(VERB_NORMAL, "[Meas %d] N_SPECTRA",indexa);
			LogFlush();

			char PRELABEL[256];
			char LABEL[256];
			string           masklab[9] = {"0", "Vi", "Vi2", "Bal", "Red", "Gau", "Dif", "Axit", "Axit2"};
			SpectrumMaskType maskara[9] = {SPMASK_FLAT,SPMASK_VIL,SPMASK_VIL2,SPMASK_BALL,SPMASK_REDO,SPMASK_GAUS,SPMASK_DIFF,SPMASK_AXIT,SPMASK_AXIT2};
			bool             prntmsk[9] = {false,true,true,true,true,true,true,true};
			bool             mulmask[9] = {false,false,false,true,true,true,true,true,true};

			LogMsg(VERB_NORMAL, "[Meas %d] masks are %d",indexa,mask);LogFlush();
			for (size_t i=0; i < 9 ; i++)
			{
				LogMsg(VERB_HIGH, "[Meas %d] maskara[%d]=%d",indexa,i,maskara[i]);LogFlush();
			}

			for (size_t i=0; i < 9; i++)
			{
				LogMsg(VERB_HIGH,   "[Meas %d] mask %s (%d) irmask %d",indexa,masklab[i].c_str(),i,irmask);LogFlush();
				if ( !(mask & maskara[i])){
					LogMsg(VERB_HIGH, "          ... skipped",indexa,masklab[i].c_str(),i);LogFlush();
					continue;
				}
				/* Place to set limitations and incompatibilities between saxion and axion spectra */

					for(int ii=0; ii < irmask; ii++)
					{
						if (mulmask[i])
							sprintf(PRELABEL, "%s_%.2f", masklab[i].c_str(),rmasklabel[ii]);
						else
							sprintf(PRELABEL, "%s", masklab[i].c_str());


						if (prntmsk[i]){
							LogMsg(VERB_NORMAL, "[Meas %d] mask %s rmask %f [%d/%d]",indexa,masklab[i].c_str(),rmasktab[ii],ii+1,irmask);LogFlush();
								// prof.start();
									specAna.masker(rmasktab[ii], maskara[i], M2_MASK, cummask);
										// prof.stop();
											sprintf(LABEL, "Masker %s", masklab[i].c_str());
												// prof.add(std::string(LABEL), 0.0, 0.0);
											sprintf(LABEL, "W_%s", PRELABEL);
											if (printedmask[i-1])
													LogMsg(VERB_NORMAL,"[meas %d] mask already printed");
												else
													writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/mSpectrum", LABEL);



								/* Kenichi's only active in REDO mode */

								if( (strmeas & STRMEAS_ENERGY) && (maskara[i] & SPMASK_REDO) ) {
									// measure the energy density of strings by using masked points
									MeasDataOut.strE = stringenergy(axiona);
									MeasDataOut.strE.rmask = rmasklabel[ii]; // this is not written by stringenergy();
									writeStringEnergy(axiona,MeasDataOut.strE);
								}

								/* Redondo's could be active in any mode */
								if (maskara[i] == SPMASK_REDO){
										void *eRes;
										trackAlloc(&eRes, 512);
										memset(eRes, 0, 512);
										double *eR = static_cast<double *> (eRes);
									energy(axiona, eRes, EN_MASK, shiftz); // EN_MAPMASK possible
									writeEnergy(axiona, eRes, rmasklabel[ii]);
												// if(p2dEmapo){ writeEMapHdf5s (axiona,sliceprint) }; //Needs EN_MAPMASK
									trackFree(eRes);
								}
						}
						
						if((maskara[i] == SPMASK_REDO) && onlymaskenergy) {
							LogMsg(VERB_NORMAL,"[Meas %d] Spectrum %s rmask %f skipped",indexa,masklab[i].c_str(),rmasktab[ii]);
							continue; // skip spectra in Red mode when onlymaskenergy is true
						}

						/* For spectra we have two options:
						1 - with LUT correction (slow more accurate)
						2 - without LUT correction */
						if (nruntype & (NRUN_K | NRUN_G | NRUN_V | NRUN_S))
						{
							LogMsg(VERB_NORMAL, "[Meas %d] Spectrum %s rmask %f [%d/%d]",indexa,masklab[i].c_str(),rmasktab[ii],ii+1,irmask);LogFlush();
								// prof.start();
									specAna.nRun(maskara[i], nruntype & (NRUN_K | NRUN_G | NRUN_V | NRUN_S));
										// prof.stop();
											sprintf(LABEL, "NSPA_%s", masklab[i].c_str());
												// prof.add(std::string(LABEL), 0.0, 0.0);

							if (nruntype & NRUN_K){
								sprintf(LABEL, "sK_%s",PRELABEL);
									writeArray(specAna.data(SPECTRUM_KK), specAna.PowMax(), "/eSpectrum", LABEL);
#ifdef USE_NN_BINS
									writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", LABEL);
#endif
								}
							if (nruntype & NRUN_G){
							sprintf(LABEL, "sG_%s",PRELABEL);
								writeArray(specAna.data(SPECTRUM_GG), specAna.PowMax(), "/eSpectrum", LABEL);
#ifdef USE_NN_BINS
								writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", LABEL);
#endif
								// // TEMP INDIVIDUAL XYZ
								// sprintf(LABEL, "sGX_%s",PRELABEL);
								// writeArray(specAna.data(SPECTRUM_GGy), specAna.PowMax(), "/eSpectrum", LABEL);
								// sprintf(LABEL, "sGY_%s",PRELABEL);
								// writeArray(specAna.data(SPECTRUM_GGz), specAna.PowMax(), "/eSpectrum", LABEL);
							}
							if ( (nruntype & NRUN_V) && axiona->AxionMassSq() > 0.0 ){
									sprintf(LABEL, "sV_%s",PRELABEL);
									writeArray(specAna.data(SPECTRUM_VV), specAna.PowMax(), "/eSpectrum", LABEL);
#ifdef USE_NN_BINS
									writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", LABEL);
#endif
							}
							if ( (nruntype & NRUN_S) && axiona->AxionMassSq() > 0.0 ){
									sprintf(LABEL, "sS_%s",PRELABEL);
									writeArray(specAna.data(SPECTRUM_VVNL), specAna.PowMax(), "/eSpectrum", LABEL);
#ifdef USE_NN_BINS
									writeArray(specAna.data(SPECTRUM_VNL), specAna.PowMax(), "/nSpectrum", LABEL);
#endif
							}
						} // END IF NSPECTRA WITH LUT CORRECTION

						// NSPECTRA WITHOUT LUT CORRECTION
						if (nruntype & (NRUN_CK | NRUN_CG | NRUN_CV | NRUN_CS))
						{
							LogMsg(VERB_NORMAL, "[Meas %d] Spectrum %s rmask %f [%d/%d] (old version)",indexa,masklab[i].c_str(),rmasktab[ii],ii+1,irmask);LogFlush();
							// prof.start();
							nRunType aux = nruntype & (NRUN_CK | NRUN_CG | NRUN_CV | NRUN_CS) ;
								specAna.nRun(maskara[i], aux);
									// prof.stop();
										sprintf(LABEL, "NSPA_%s (pure)", masklab[i].c_str());
											// prof.add(std::string(LABEL), 0.0, 0.0);
						if (nruntype & NRUN_CK){
							sprintf(LABEL, "sCK_%s",PRELABEL);
								writeArray(specAna.data(SPECTRUM_KK), specAna.PowMax(), "/eSpectrum", LABEL);
#ifdef USE_NN_BINS
								writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", LABEL);
#endif
						}
						if (nruntype & NRUN_CG){
						sprintf(LABEL, "sCG_%s",PRELABEL);
							writeArray(specAna.data(SPECTRUM_GG), specAna.PowMax(), "/eSpectrum", LABEL);
#ifdef USE_NN_BINS
							writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", LABEL);
#endif
						}
							/* Note that there is NO DIFFERENCE between CV CS and V,S */

						if ( (nruntype & NRUN_CV) && axiona->AxionMassSq() > 0.0 ){
							LogMsg(VERB_NORMAL,"[Meas %d] Warning: You asked for nspV without LUT correction (CV) which is the same as V. Waste of resources?",indexa);
							sprintf(LABEL, "sCV_%s",PRELABEL);
								writeArray(specAna.data(SPECTRUM_VV), specAna.PowMax(), "/eSpectrum", LABEL);
#ifdef USE_NN_BINS
								writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", LABEL);
#endif
							}
						if ( (nruntype & NRUN_CS) && axiona->AxionMassSq() > 0.0 ){
							LogMsg(VERB_NORMAL,"[Meas %d] Warning: You asked for nspS without LUT correction (CS) which is the same as S. Waste of resources?",indexa);
							sprintf(LABEL, "sCS_%s",PRELABEL);
								writeArray(specAna.data(SPECTRUM_VVNL), specAna.PowMax(), "/eSpectrum", LABEL);
#ifdef USE_NN_BINS
								writeArray(specAna.data(SPECTRUM_VNL), specAna.PowMax(), "/nSpectrum", LABEL);
#endif
							}

					} // END IF NSPECTRA WITHOUT CORRECTION

					// SAXION SPECTRA
					if ( (axiona->Field() == FIELD_SAXION) && (measa & MEAS_NSP_S))
					{
						LogMsg(VERB_NORMAL, "[Meas %d] Spectrum %s rmask %f [%d/%d] (saxion)",indexa,masklab[i].c_str(),rmasktab[ii],ii+1,irmask);LogFlush();
						nRunType aux = nruntype & (NRUN_K | NRUN_G | NRUN_V);
						specAna.nSRun(maskara[i], aux);
						
						if (nruntype & NRUN_K){
							sprintf(LABEL, "sKS_%s",PRELABEL);
								writeArray(specAna.data(SPECTRUM_KK), specAna.PowMax(), "/eSpectrum", LABEL);
#ifdef USE_NN_BINS
								writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", LABEL);
#endif
						}
						if (nruntype & NRUN_G){
							sprintf(LABEL, "sGS_%s",PRELABEL);
								writeArray(specAna.data(SPECTRUM_GG), specAna.PowMax(), "/eSpectrum", LABEL);
#ifdef USE_NN_BINS
								writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", LABEL);
#endif
						}
						if (nruntype & NRUN_V){
							sprintf(LABEL, "sVS_%s",PRELABEL);
								writeArray(specAna.data(SPECTRUM_VV), specAna.PowMax(), "/eSpectrum", LABEL);
#ifdef USE_NN_BINS
								writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", LABEL);
#endif
						}
					} // END IF SAXION SPECTRA

					if (prntmsk[i]){
						LogMsg(VERB_NORMAL, "[Meas %d] m-matrix build and write skipped; uncomment if you wish otherwise!",indexa);
					// 		prof.start();
					// 			specAna.matrixbuilder();
					// 				prof.stop();
					// 					prof.add(std::string("Matrix Builder"), 0.0, 0.0);
					// 						// sprintf(LABEL, "M_%s_%.2f", masklab[i].c_str(), rmasktab[ii]);
					// 						sprintf(LABEL, "M_%s",PRELABEL);
					// 							writeArray(static_cast<double *>(axiona->m2Cpu()), specAna.PowMax()*specAna.PowMax(), "/mSpectrum", LABEL);
					}
						/* If only one masking radius makes sense (like no masking) */
						if ( !mulmask[i])
							break;
				} //end for mask length
			} // end for mask type

		/* Some special features */

			if ( (axiona->Field() == FIELD_SAXION) && (mask & SPMASK_SAXI) )
			{
				// if ( (mask & SPMASK_FLAT) || (mask & SPMASK_VIL) || (mask & SPMASK_VIL2))
				// 	specAna.reset0();
				// LogOut("NSPA ");
				LogMsg(VERB_NORMAL, "[Meas %d] NSP real and imaginary",indexa);
				specAna.nRun(SPMASK_SAXI, nruntype);
				writeArray(specAna.data(SPECTRUM_KK), specAna.PowMax(), "/nSpectrum", "sK_Im");
				writeArray(specAna.data(SPECTRUM_VV), specAna.PowMax(), "/nSpectrum", "sK_Re");
				writeArray(specAna.data(SPECTRUM_GG), specAna.PowMax(), "/nSpectrum", "sG_Im");
				writeArray(specAna.data(SPECTRUM_PS), specAna.PowMax(), "/nSpectrum", "sG_Re");
			}


		// saxion spectra moved into mask loop
		//if ( (axiona->Field() == FIELD_SAXION) && (measa & MEAS_NSP_S))
		//{
		//	if (axiona->Field() == FIELD_SAXION){
		//		// LogOut("NSPS ");
		//		LogMsg(VERB_NORMAL, "[Meas %d] NSPS ",indexa);
		//		specAna.nSRun();
		//		writeArray(specAna.data(SPECTRUM_KK), specAna.PowMax(), "/eSpectrum", "sKS");
		//		writeArray(specAna.data(SPECTRUM_GG), specAna.PowMax(), "/eSpectrum", "sGS");
		//		writeArray(specAna.data(SPECTRUM_VV), specAna.PowMax(), "/eSpectrum", "sVS");
		//#ifdef USE_NN_BINS
		//		writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sKS");
		//		writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sGS");
		//		writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sVS");
		//#endif
		//	}
		//}

		if ( (indexa == 0) || (measa & MEAS_NNSPEC) )
		{
			// LogOut("Nmod ");
			LogMsg(VERB_NORMAL, "[Meas %d] Nmod ", indexa);
				// prof.start();
			specAna.nmodRun();
				// prof.stop();
				// prof.add(std::string("nmod Run"), 0.0, 0.0);
			writeArray(specAna.data(SPECTRUM_NN), specAna.PowMax(), "/nSpectrum", "nmodes");
		}

		if ( (indexa == 0) || (measa & MEAS_NNSPEC) )
		{
			// LogOut("Nmod ");
			LogMsg(VERB_NORMAL, "[Meas %d] average K ", indexa);
				// prof.start();
			specAna.avekRun();
				// prof.stop();
				// prof.add(std::string("avek Run"), 0.0, 0.0);

			writeArray(specAna.data(SPECTRUM_AK), specAna.PowMax(), "/nSpectrum", "averagek");
		}
	} // end if spectrum
}   // end if spectrum or mask




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
	else if (axiona->Field() == FIELD_AXION) { // FIELD_AXION
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
	if (axiona->Field() == FIELD_NAXION)
	{
		if (measa & MEAS_BINTHETA)
		{
			// LogOut("binT ");
			LogMsg(VERB_NORMAL, "[Meas %d] bin |P|^2",indexa);
				// Float shs = shiftz;
				// complex<Float> shhhs = (shs,0.);
				// Binner<3000,complex<Float>> thBin(static_cast<complex<Float> *>(axiona->mCpu()) + axiona->Surf(), axiona->Size(),
				// 				 [s=shhhs] (complex<Float> x) { return (double) arg(x-s); });
				Binner<3000,complex<Float>> thBin(static_cast<complex<Float> *>(axiona->mStart()), axiona->Size(),
								 [] (complex<Float> x) { return (double) abs(x);});
				thBin.run();
				writeBinner(thBin, "/bins", "thetaP2");
				MeasDataOut.maxTheta = thBin.max();
		}
	}
	if (axiona->Field() == FIELD_PAXION)
	{
		// if (measa & MEAS_BINTHETA)
		// {
		// 	// LogOut("binT ");
		// 	LogMsg(VERB_NORMAL, "[Meas %d] bin |P|^2",indexa);
		// 		// Float shs = shiftz;
		// 		// complex<Float> shhhs = (shs,0.);
		// 		// Binner<3000,complex<Float>> thBin(static_cast<complex<Float> *>(axiona->mCpu()) + axiona->Surf(), axiona->Size(),
		// 		// 				 [s=shhhs] (complex<Float> x) { return (double) arg(x-s); });
		// 		Binner<3000,Float> thBin1(static_cast<Float*>(axiona->mStart()), axiona->Size(),
		// 						 [] (Float x) { return (double) x*x;});
		// 		thBin1.run();
		// 		Binner<3000,Float> thBin2(static_cast<Float*>(axiona->vCpu()), axiona->Size(),
		// 						 [] (Float x) { return (double) x*x;});
		// 		thBin2.run();
		// 		for (int g  =0; g <300; g++)
		// 			thBin2.data()[g] += thBin1.data()[g];
		//
		// 		writeBinner(thBin2, "/bins", "thetaP2");
		// 		MeasDataOut.maxTheta = thBin2.max();
		// }
	}



	/* This is a generic placeholder to test new functions or debugging */
	if (measa & MEAS_AUX)
	{
		// if	(axiona->Folded())
		// {
		// 	Folder	munge(axiona);
		// 	munge(UNFOLD_ALL);
		// }
		// buildc_k_map(axiona, true);
		// writeEDens(axiona);
	}




	LogMsg(VERB_HIGH, "destroying meas",indexa);
	LogFlush();
	destroyMeas();
	}

	if ((indexa-1) % 10 == 0){
		LogOut("ctime  |  index |  cmeas |  wtime  | mass \n");
		LogOut(".");
		}
	if (z_now < 100.)
		LogOut("%2.3f  | ",z_now);
	else
		LogOut("%2.1e  | ",z_now);

	if (cTime*1.e-6/3600. < 1.0 )
		LogOut("  %3d  | %6d | %2.3f m | ", indexa, info.measdata, cTime*1.e-6/60.);
		else
		LogOut("  %3d  | %6d | %2.3f h | ", indexa, info.measdata, cTime*1.e-6/3600.);

	double lola = axiona->LambdaP();

	double DWfun = 40*axiona->AxionMassSq()/(2.0*lola) ;
	LogOut("%.1e %.1e (%.1e) ", axiona->AxionMass(), sqrt(axiona->SaxionMassSq()), DWfun );

	if ( axiona->Field() == FIELD_SAXION)
	{
		if ( measa & (MEAS_STRING | MEAS_STRINGMAP | MEAS_STRINGCOO) ){
		double loks = log(sqrt(2*lola)*R_now*R_now);
		double Le = axiona->BckGnd()->PhysSize();
			LogOut("log(%.1f) xi_t(%.3f) xi(%.3f) msa(%.3f) #_st %ld ", loks, xivilgor(loks),
				(1/6.)*axiona->Delta()*( (double) MeasDataOut.str.strDen)*z_now*z_now/(Le*Le*Le),
				axiona->Msa(),
				MeasDataOut.str.strDen );
		} else {
			// LogOut("str not measured (%ld, %ld) ",MeasDataOut.str.strDen, -1);
			LogOut("str not measured ");
		}
	} else if ( axiona->Field() == FIELD_AXION) {
		LogOut("maxth=%f ", MeasDataOut.maxTheta);
		LogOut(" ... ");
	} else if ( axiona->Field() == FIELD_NAXION || axiona->Field() == FIELD_PAXION) {
		// E/m-1
		double k2m2 = 12*pow(((double) axiona->Length())/axiona->BckGnd()->PhysSize(),2)/(axiona->AxionMassSq()*R_now*R_now);
		LogOut("Emax/m-1=%.2e Max P2 %.2e", k2m2/(sqrt(k2m2 + 1.0)+ 1.0), std::sqrt(MeasDataOut.maxTheta) );
		LogOut(" ... ");
	}

	LogOut("\n");

// if (cDev != DEV_CPU){
// LogMsg (VERB_HIGH,"Transferring configuration to device");
// axiona->transferDev(FIELD_MV);
// }

// After the first measurement we switch cumulative mask buiding
if (deninfa.cummask > 0){
	cummask = true;
	LogMsg(VERB_NORMAL, "[Meas %d] Set cummask to %d (false/true = 1/0)",indexa,cummask);
}


// Return to GPU if needed
if (info.measCPU && wasGPU)
		axiona->setDev(DEV_GPU);

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
