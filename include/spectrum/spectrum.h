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
		std::vector<double>	binV;
		std::vector<double>	binVnl;
		std::vector<double>	binNK;
		std::vector<double>	binNG;
		std::vector<double>	binNV;
		std::vector<double>	binNVnl;
		std::vector<double>	binP;
		std::vector<double>	binPS;
		std::vector<double>	binNN;
		std::vector<double>	binAK;

		std::vector<double>	cosTable;
		std::vector<double>	cosTable2;

		Scalar			*field;

		size_t			Lx, Ly, Lz, hLx, hLy, hLz, hTz, Tz, V;
		size_t			nPts, nModeshc, kMax, powMax, controlxyz;
		size_t			LyLy, Ly2Ly, LyLz, dl, pl, dataTotalSize, dataBareSize;
		size_t			zBase;
		double			mass2, mass2Sax; // squared masses (comoving)
		double 			Rscale, depta;
		double 			zaskar;
		float				zaskarf;
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

				SpecBin (Scalar *field, const bool spectral) : field(field), Ly(field->Length()), Lz(field->Depth()), Tz(field->TotalDepth()),
									spec(spectral), fPrec(field->Precision()), fType(field->Field()), zBase(commRank()*Ly/commSize()),
									k0(2.0*M_PI/((double) field->BckGnd()->PhysSize())) {
				kMax   = (Ly >=  Tz) ? (Ly>>1) : (Tz>>1);
				powMax = floor(sqrt(2.*(Ly>>1)*(Ly>>1) + (Tz>>1)*(Tz>>1)))+1;

				binK.resize(powMax); binK.assign(powMax, 0.);
				binG.resize(powMax); binG.assign(powMax, 0.);
				binV.resize(powMax); binV.assign(powMax, 0.);
				binVnl.resize(powMax); binVnl.assign(powMax, 0.);
				binP.resize(powMax); binP.assign(powMax, 0.);
				binPS.resize(powMax); binPS.assign(powMax, 0.);

				binNK.resize(powMax); binNK.assign(powMax, 0.);
				binNG.resize(powMax); binNG.assign(powMax, 0.);
				binNV.resize(powMax); binNV.assign(powMax, 0.);
				binNVnl.resize(powMax); binNVnl.assign(powMax, 0.);


				mass2    = field->AxionMassSq()*(*field->RV())*(*field->RV());
				mass2Sax = field->SaxionMassSq()*(*field->RV())*(*field->RV());
				Rscale   = *field->RV();
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
		}


		inline const size_t	PowMax() const { return powMax; }

		inline double		operator()(size_t idx, SpectrumType sType)	const;
		inline double&		operator()(size_t idx, SpectrumType sType);

		inline const double*	data(SpectrumType sType)	const;
		inline	     double*	data(SpectrumType sType);

		template<typename cFloat, const SpectrumType sType, const bool spectral>
		void	fillBins	();

		template<typename cFloat>
		void	filterFFT	(double neigh);

		void	nRun		(SpectrumMaskType mask = SPMASK_FLAT, nRunType nrt = NRUN_KGV);
		void	nSRun		();
		void	pRun		();
		void	nmodRun		();
		void	avekRun		();
		void  wRun(SpectrumMaskType mask);

		template<typename Float, SpectrumMaskType mask>
		void	nRun		(nRunType nrt);

		template<typename Float, SpectrumMaskType mask>
		void	wRun		();

		int	pad	  (PadIndex origin, PadIndex dest) ;
		int unpad	(PadIndex origin, PadIndex dest) ;
		char*	chosechar	(PadIndex start);

		void	filter	(size_t neigh);

		void	reset0(){
				LogMsg(VERB_NORMAL,"Reset SpecAna Bins to zero");
				binK.assign(powMax, 0.);
				binG.assign(powMax, 0.);
				binV.assign(powMax, 0.);
				binP.assign(powMax, 0.);
				binPS.assign(powMax, 0.);
				binNK.assign(powMax, 0.);
				binNG.assign(powMax, 0.);
				binNV.assign(powMax, 0.);

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

			case	SPECTRUM_VV:

				return binV[idx];
				break;

			case	SPECTRUM_VVNL:
				return binVnl[idx];
				break;

			/* number */
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

			case	SPECTRUM_VV:
				return binV[idx];
				break;

			case	SPECTRUM_VVNL:
				return binVnl[idx];
				break;

			/* number */
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

			case	SPECTRUM_VV:
				return binV.data();
				break;

			case	SPECTRUM_VVNL:
				return binVnl.data();
				break;

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

			case	SPECTRUM_VV:
				return binV.data();
				break;

			case	SPECTRUM_VVNL:
				return binVnl.data();
				break;

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
