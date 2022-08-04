#ifndef	_CLASS_SPECTRUM_
	#define	_CLASS_SPECTRUM_

	#include <vector>
	#include <complex>
	#include <cmath>


	#include "enum-field.h"
	#include "scalar/scalarField.h"
//	#include "scalar/varNQCD.h"

	class	SpecBin {

		private:

		std::vector<double>	binK;
		std::vector<double>	binG;

		// //TEMP INDIVIDUAL XYZ
		// std::vector<double>	binGy;
		// std::vector<double>	binGz;

		std::vector<double>	binV;
		std::vector<double>	binVnl;
#ifdef USE_NN_BINS
		std::vector<double>	binNK;
		std::vector<double>	binNG;
		std::vector<double>	binNV;
		std::vector<double>	binNVnl;
#endif
		std::vector<double>	binP;
		std::vector<double>	binPS;
		std::vector<double>	binNN;
		std::vector<double>	binAK;

		std::vector<double>	cosTable;
		std::vector<double>	cosTable2;

		Scalar			*field;

		size_t			Lx, Ly, Lz, hLx, hLy, hLz, hTz, Tz, V;
		size_t			nPts, nModeshc, kMax, powMax, nbins, controlxyz;
		size_t			LyLy, Ly2Ly, LyLz, dl, pl, dataTotalSize, dataBareSize;
		size_t			zBase;
		double			mass2, mass2Sax; // squared masses (comoving)
		double 			Rscale, Rpp, depta;
		double 			zaskar;
		float				zaskarf;
		double			nbinmul;
		double 			k0;

		std::complex<double>	zaska ;
		std::complex<float>	zaskaf ;

		const bool		spec;
		const FieldPrecision	fPrec;
		const FieldType		fType;

		void			fillCosTable ();

		template<typename Float, FilterIndex filter>
		void	smoothFourier	(double length);



		public:

				SpecBin (Scalar *field, const bool spectral, MeasInfo measinfo) : field(field), Ly(field->Length()), Lz(field->Depth()), Tz(field->TotalDepth()),
									spec(spectral), fPrec(field->Precision()), fType(field->Field()), zBase(commRank()*Ly/commSize()),
									k0(2.0*M_PI/((double) field->BckGnd()->PhysSize())) {

				/* select nbins from maximum k in 2pi/L0 units
				we want measinfo.nbinsspec bins
				we will compute k/k1 = sqrt(kx^2+ky^2+kz^2) in k1=2pi/L units
				and have a total of NB1 = sqrt(2*(Nx/2)^2+(Nz/2)^2) natural bins
				so we need to multiply k/k1 by (nbinsspec/NB1)
				NB = (k/k1)*(nbinsspec/NB1)
				there are plenty of sqwrt and products so create a LUT table of squares
				IDEA: tabulate floor((k/k1)**2), NB
				PROBLEM: how many integers do we need?
				NICETY: Each MPI rank needs only a few of them
				DISCARDED: too much for too little*/


				/* Number of natural bins */
				powMax = floor(sqrt(2.*(Ly>>1)*(Ly>>1) + (Tz>>1)*(Tz>>1)))+1;
				/* Number of user desired bins */
				nbins = measinfo.nbinsspec < 0 ? powMax : (size_t) measinfo.nbinsspec;
				/* Multiplier */
				nbinmul = measinfo.nbinsspec < 0 ? 1.0 : ((double) measinfo.nbinsspec)/((double) powMax);
				/* Prefil the value of k? */

				LogMsg(VERB_NORMAL,"[spe] SpecBin constructor called with powMax %lu nbins %lu nbinmul %f ",powMax,nbins,nbinmul);

				binK.resize(nbins); binK.assign(nbins, 0.);
				binG.resize(nbins); binG.assign(nbins, 0.);
				binV.resize(nbins); binV.assign(nbins, 0.);
				binVnl.resize(nbins); binVnl.assign(nbins, 0.);
				binP.resize(nbins); binP.assign(nbins, 0.);
				binPS.resize(nbins); binPS.assign(nbins, 0.);

#ifdef USE_NN_BINS
				binNK.resize(nbins); binNK.assign(nbins, 0.);
				binNG.resize(nbins); binNG.assign(nbins, 0.);
				binNV.resize(nbins); binNV.assign(nbins, 0.);
				binNVnl.resize(nbins); binNVnl.assign(nbins, 0.);
#endif

				/* kMax for correction tables */
				kMax   = (Ly >=  Tz) ? (Ly>>1) : (Tz>>1);

				mass2    = field->AxionMassSq()*(*field->RV())*(*field->RV());
				mass2Sax = field->SaxionMassSq()*(*field->RV())*(*field->RV());
				Rscale   = *field->RV();
				Rpp      = field->BckGnd()->Rp(*field->zV())*(Rscale); /* this is R' */
				depta    = field->BckGnd()->PhysSize()/Ly;

				zaskar  = field->Saskia()*Rscale;
				zaskarf = (float) zaskar ;
				zaska   = std::complex<double>(zaskar,0.);
				zaskaf  = std::complex<float>(zaskarf,0.f);

				controlxyz = 0;

				fillCosTable();

				hLy = Ly >> 1;
				hLz = Lz >> 1;
				hTz = Tz >> 1;

				switch (fType) {
					// THIS CASE IS ILL DEFINED, WILL NEVER BE USED
					// well... I am starting to use it!
					// I assume the saxion mode will be analised also in real components
					case	FIELD_SAXION:
						// Lx   = Ly;
						// hLx  = Ly >> 1;
						// break;
					case	FIELD_AXION_MOD:
					case	FIELD_AXION:
					case	FIELD_PAXION:
						Lx   = (Ly >> 1)+1;
						hLx  = Lx;
						break;

					case	FIELD_WKB:
						LogError("Warning: WKB fields not supported for analysis");
						Lx = 0; Ly = 0; hLx = 0; nPts = 0;
						return;

					default:
						LogError("Warning: Field not supported for analysis");
						Lx = 0; Ly = 0; hLx = 0; nPts = 0;
						return;
				}

				/* for half-complex Fourier loops */
				nModeshc = Lx*Ly*Lz;

				/* for pad/unpadding loops */
				V             =  field->Size();
				LyLy          =  Ly*Ly;
				LyLz          =  Ly*Lz;
				Ly2Ly         =  Ly*(Ly+2);
				dl            =  Ly*field->Precision(); 					/* data line length */
				pl            =  (Ly+2)*field->Precision(); 			/* padded data line length */
				dataTotalSize =  (Ly+2)*Ly*Lz*field->Precision(); /* total data volume including padding */
				dataBareSize  =  V*field->Precision();            /* total data volume without padding */

				LogMsg(VERB_HIGH,"[spe] SpecBin constructor ended.");
				LogFlush();
		}


		inline const size_t	PowMax() const { return nbins; }
		inline const size_t	NBins() const { return nbins; }

		inline double		operator()(size_t idx, SpectrumType sType)	const;
		inline double&		operator()(size_t idx, SpectrumType sType);

		inline const double*	data(SpectrumType sType)	const;
		inline	     double*	data(SpectrumType sType);

		template<typename cFloat, const SpectrumType sType, const bool spectral>
		void	fillBins	();

		template<typename cFloat>
		void	filterFFT	(double neigh);

		void	nRun		(SpectrumMaskType mask = SPMASK_FLAT, nRunType nrt = NRUN_KGV);
		void	nSRun		(SpectrumMaskType mask = SPMASK_FLAT, nRunType nrt = NRUN_KGV);
		void	pRun		();
		void	nmodRun		();
		void	avekRun		();
		void  wRun(SpectrumMaskType mask);

		template<typename Float, SpectrumMaskType mask>
		void	nRun		(nRunType nrt);
		
		template<typename Float, SpectrumMaskType mask>
		void	nSRun		(nRunType nrt);

		template<typename Float, SpectrumMaskType mask>
		void	wRun		();

		int	pad	  (PadIndex origin, PadIndex dest) ;
		int unpad	(PadIndex origin, PadIndex dest) ;
		char*	chosechar	(PadIndex start);

		void	filter	(size_t neigh);

		void	reset0(){
				LogMsg(VERB_NORMAL,"Reset SpecAna Bins to zero");
				binK.assign(nbins, 0.);
				binG.assign(nbins, 0.);
				binV.assign(nbins, 0.);
				binP.assign(nbins, 0.);
				binPS.assign(nbins, 0.);
#ifdef USE_NN_BINS
				binNK.assign(nbins, 0.);
				binNG.assign(nbins, 0.);
				binNV.assign(nbins, 0.);
				binNVnl.assign(nbins, 0.);
#endif
		}

		void	masker	(double radius_mask, SpectrumMaskType mask = SPMASK_REDO, StatusM2 out = M2_MASK, bool l_cumsum = false);

		template<typename Float, SpectrumMaskType mask>
		void	masker	(double radius_mask, StatusM2 out, bool l_cumsum);

		void	matrixbuilder	();

		template<typename Float>
		void	matrixbuilder	();

		void	maskball	(double radius_mask, char DEFECT_LABEL, char MASK_LABEL) ;

		void	smoothFourier	(double length, FilterIndex filter);

	};




	inline double	SpecBin::operator()(size_t idx, SpectrumType sType)	const	{

		switch(sType) {
			/* energy */
			case	SPECTRUM_KK:
				return binK[idx];
				break;

			case	SPECTRUM_GG:
				return binG[idx];
				break;

				// // TEMP INDIVIDUAL XYZ
				// case	SPECTRUM_GGy:
				// 	return binGy[idx];
				// 	break;
				//
				// case	SPECTRUM_GGz:
				// 	return binGz[idx];
				// 	break;

			case	SPECTRUM_VV:

				return binV[idx];
				break;

			case	SPECTRUM_VVNL:
				return binVnl[idx];
				break;

			/* number */
#ifdef USE_NN_BINS
			case	SPECTRUM_K:
				return binNK[idx];
				break;

			case	SPECTRUM_G:
				return binNG[idx];
				break;

			case	SPECTRUM_V:
				return binNV[idx];
				break;

			case	SPECTRUM_VNL:
				return binNVnl[idx];
				break;
#endif

			/* power-spectrum-axion */
			case	SPECTRUM_P:
				return binP[idx];
				break;

			/* power-spectrum-saxion */
			case	SPECTRUM_PS:
				return binPS[idx];
				break;

			case	SPECTRUM_NN:
				return binNN[idx];
				break;

			case	SPECTRUM_AK:
				return binAK[idx];
				break;


			default:
				return	0.0;
		}
	}

	inline double&	SpecBin::operator()(size_t idx, SpectrumType rType)		{

		switch(rType) {
			/* energy */
			case	SPECTRUM_KK:
				return binK[idx];
				break;

			case	SPECTRUM_GG:
				return binG[idx];
				break;

				// // TEMP INDIVIDUAL XYZ
				// case	SPECTRUM_GGy:
				// 	return binGy[idx];
				// 	break;
				//
				// case	SPECTRUM_GGz:
				// 	return binGz[idx];
				// 	break;

			case	SPECTRUM_VV:
				return binV[idx];
				break;

			case	SPECTRUM_VVNL:
				return binVnl[idx];
				break;

			/* number */
#ifdef USE_NN_BINS
			case	SPECTRUM_K:
				return binNK[idx];
				break;

			case	SPECTRUM_G:
				return binNG[idx];
				break;

			case	SPECTRUM_V:
				return binNV[idx];
				break;

			case	SPECTRUM_VNL:
				return binNVnl[idx];
				break;
#endif
			/* power-spectrum-axion */
			case	SPECTRUM_P:
				return binP[idx];
				break;

			/* power-spectrum-saxion */
			case	SPECTRUM_PS:
				return binPS[idx];
				break;

			case	SPECTRUM_NN:
				return binNN[idx];
				break;

			case	SPECTRUM_AK:
				return binAK[idx];
				break;

			default:
				LogError ("Undefined spectrum requested.");
				return	binK[0];
		}
	}

	inline double*	SpecBin::data(SpectrumType sType) {

		switch(sType) {
			case	SPECTRUM_KK:
				return binK.data();
				break;

			case	SPECTRUM_GG:
				return binG.data();
				break;

				// // TEMP INDIVIDUAL XYZ
				// case	SPECTRUM_GGy:
				// 	return binGy.data();
				// 	break;
				//
				// case	SPECTRUM_GGz:
				// 	return binGz.data();
				// 	break;

			case	SPECTRUM_VV:
				return binV.data();
				break;

			case	SPECTRUM_VVNL:
				return binVnl.data();
				break;

#ifdef USE_NN_BINS
			case	SPECTRUM_K:
				return binNK.data();
				break;

			case	SPECTRUM_G:
				return binNG.data();
				break;

			case	SPECTRUM_V:
				return binNV.data();
				break;

			case	SPECTRUM_VNL:
				return binNVnl.data();
				break;
#endif

			case	SPECTRUM_P:
				return binP.data();
				break;

			case	SPECTRUM_PS:
				return binPS.data();
				break;

			case	SPECTRUM_NN:
				return binNN.data();
				break;

			case	SPECTRUM_AK:
				return binAK.data();
				break;

			default:
				return	nullptr;
		}
	}

	inline const double*	SpecBin::data(SpectrumType sType)	const	{

		switch(sType) {
			case	SPECTRUM_KK:
				return binK.data();
				break;

			case	SPECTRUM_GG:
				return binG.data();
				break;

				// // TEMP INDIVIDUAL XYZ
				// case	SPECTRUM_GGy:
				// 	return binGy.data();
				// 	break;
				//
				// case	SPECTRUM_GGz:
				// 	return binGz.data();
				// 	break;


			case	SPECTRUM_VV:
				return binV.data();
				break;

			case	SPECTRUM_VVNL:
				return binVnl.data();
				break;

#ifdef USE_NN_BINS
			case	SPECTRUM_K:
				return binNK.data();
				break;

			case	SPECTRUM_G:
				return binNG.data();
				break;

			case	SPECTRUM_V:
				return binNV.data();
				break;

			case	SPECTRUM_VNL:
				return binNVnl.data();
				break;
#endif

			case	SPECTRUM_P:
				return binP.data();
				break;

			case	SPECTRUM_PS:
				return binPS.data();
				break;

			case	SPECTRUM_NN:
				return binNN.data();
				break;

			case	SPECTRUM_AK:
				return binAK.data();
				break;

			default:
				return	nullptr;
		}
	}
#endif
