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
		std::vector<double>	binP;
		std::vector<double>	binPS;
		std::vector<double>	binNN;
		std::vector<double>	binAK;

		std::vector<double>	cosTable;
		std::vector<double>	cosTable2;

		Scalar			*field;

		size_t			Lx, Ly, Lz, hLx, hLy, hLz, hTz, Tz, nPts, nModeshc, kMax, powMax, controlxyz;
		double			mass2, mass2Sax; // squared masses (comoving)
		double 			Rscale, depta;
		double 			zaskar;
		float			zaskarf;
		std::complex<double>	zaska ;
		std::complex<float>	zaskaf ;

		void			fillCosTable ();

		const bool		spec;
		const FieldPrecision	fPrec;
		const FieldType		fType;

		public:

				SpecBin (Scalar *field, const bool spectral) : field(field), Ly(field->Length()), Lz(field->Depth()), Tz(field->TotalDepth()),
									spec(spectral), fPrec(field->Precision()), fType(field->Field()) {
				kMax   = (Ly >=  Tz) ? (Ly>>1) : (Tz>>1);
				powMax = floor(sqrt(2.*(Ly>>1)*(Ly>>1) + (Tz>>1)*(Tz>>1)))+1;

				binK.resize(powMax); binK.assign(powMax, 0.);
				binG.resize(powMax); binG.assign(powMax, 0.);
				binV.resize(powMax); binV.assign(powMax, 0.);
				binP.resize(powMax); binP.assign(powMax, 0.);
				binPS.resize(powMax); binPS.assign(powMax, 0.);

				mass2    = field->AxionMassSq()*(*field->zV())*(*field->zV());
				mass2Sax = field->SaxionMassSq()*(*field->zV())*(*field->zV());
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

				nModeshc = Lx*Ly*Lz;

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

		void	filter	(size_t neigh);

		void	reset0(){
				LogMsg(VERB_NORMAL,"Reset SpecAna Bins to zero");
				binK.assign(powMax, 0.);
				binG.assign(powMax, 0.);
				binV.assign(powMax, 0.);
				binP.assign(powMax, 0.);
				binPS.assign(powMax, 0.);
		}

		void	masker	(double radius_mask, SpectrumMaskType mask = SPMASK_REDO);

		template<typename Float, SpectrumMaskType mask>
		void	masker	(double radius_mask);

		void	matrixbuilder	();

		template<typename Float>
		void	matrixbuilder	();

	};




	inline double	SpecBin::operator()(size_t idx, SpectrumType sType)	const	{

		switch(sType) {
			case	SPECTRUM_K:
			case	SPECTRUM_KS:
				return binK[idx];
				break;

			case	SPECTRUM_G:
			case	SPECTRUM_GS:
				return binG[idx];
				break;

			case	SPECTRUM_V:
			case	SPECTRUM_VS:
				return binV[idx];
				break;

			case	SPECTRUM_P:
				return binP[idx];
				break;

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
			case	SPECTRUM_K:
			case	SPECTRUM_KS:
				return binK[idx];
				break;

			case	SPECTRUM_G:
			case	SPECTRUM_GS:
				return binG[idx];
				break;

			case	SPECTRUM_V:
			case	SPECTRUM_VS:
				return binV[idx];
				break;

			case	SPECTRUM_P:
				return binP[idx];
				break;

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
			case	SPECTRUM_K:
			case	SPECTRUM_KS:
				return binK.data();
				break;

			case	SPECTRUM_G:
			case	SPECTRUM_GS:
				return binG.data();
				break;

			case	SPECTRUM_V:
			case	SPECTRUM_VS:
				return binV.data();
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
			case	SPECTRUM_K:
			case	SPECTRUM_KS:
				return binK.data();
				break;

			case	SPECTRUM_G:
			case	SPECTRUM_GS:
				return binG.data();
				break;

			case	SPECTRUM_V:
			case	SPECTRUM_VS:
				return binV.data();
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
