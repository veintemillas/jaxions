#ifndef	_CLASS_SPECTRUM_
	#define	_CLASS_SPECTRUM_

	#include <vector>
	#include <cmath>

	#include "enum-field.h"
	#include "scalar/scalarField.h"
	#include "scalar/varNQCD.h"

	class	SpecBin {

		private:

		std::vector<double>	binK;
		std::vector<double>	binG;
		std::vector<double>	binV;
		std::vector<double>	binP;

		std::vector<double>	cosTable;

		Scalar			*field;

		size_t			Lx, Ly, Lz, hLx, hLy, hLz, hTz, Tz, nPts, kMax, powMax;
		double			mass;

		void			fillCosTable ();

		const bool		spec;
		FieldPrecision		fPrec;
		FieldType		fType;

		public:

				SpecBin (Scalar *field, const bool spectral) : field(field), Lx(field->Length()), Lz(field->Depth()), Tz(field->TotalDepth()),
									       fPrec(field->Precision()), nPts(field->Size()), fType(field->Field()), spec(spectral) {
				kMax   = (Lx>>1)-1;
				powMax = floor(sqrt(3.)*kMax)+2;

				binK.resize(powMax); binK.assign(powMax, 0.);
				binG.resize(powMax); binG.assign(powMax, 0.);
				binV.resize(powMax); binV.assign(powMax, 0.);
				binP.resize(powMax); binP.assign(powMax, 0.);

				mass   = axionmass2((*field->zV()), nQcd, zthres, zrestore)*(*field->zV())*(*field->zV());

				fillCosTable();

				Ly = Lx;

				hLy = Ly >> 1;
				hLz = Lz >> 1;
				hTz = Tz >> 1;

				switch (fType) {
					case	FIELD_AXION:
						Lx   = (Lx >> 1)+1;
						nPts = Lx*Ly*Lz;
						hLx  = Lx;
						break;

					case	FIELD_SAXION:
						hLx  = Lx >> 1;
						break;
				}
			}



		inline double		operator()(size_t idx, SpectrumType sType)	const;
		inline double&		operator()(size_t idx, SpectrumType sType);

		inline const double*	data(SpectrumType sType)	const;
		inline	     double*	data(SpectrumType sType);

		template<typename cFloat, const SpectrumType sType, const bool spectral>
		void		fillBins	();
		void		nRun		();
		void		pRun		();
	};


	inline double	SpecBin::operator()(size_t idx, SpectrumType sType)	const	{

		switch(sType) {
			case	SPECTRUM_K:
				return binK[idx];
				break;

			case	SPECTRUM_G:
				return binG[idx];
				break;

			case	SPECTRUM_V:
				return binV[idx];
				break;

			case	SPECTRUM_P:
				return binP[idx];
				break;
		}
	}

	inline double&	SpecBin::operator()(size_t idx, SpectrumType rType)		{

		switch(rType) {
			case	SPECTRUM_K:
				return binK[idx];
				break;

			case	SPECTRUM_G:
				return binG[idx];
				break;

			case	SPECTRUM_V:
				return binV[idx];
				break;

			case	SPECTRUM_P:
				return binP[idx];
				break;
		}
	}

	inline double*	SpecBin::data(SpectrumType sType) {

		switch(sType) {
			case	SPECTRUM_K:
				return binK.data();
				break;

			case	SPECTRUM_G:
				return binG.data();
				break;

			case	SPECTRUM_V:
				return binV.data();
				break;

			case	SPECTRUM_P:
				return binP.data();
				break;
		}
	}

	inline const double*	SpecBin::data(SpectrumType sType)	const	{

		switch(sType) {
			case	SPECTRUM_K:
				return binK.data();
				break;

			case	SPECTRUM_G:
				return binG.data();
				break;

			case	SPECTRUM_V:
				return binV.data();
				break;

			case	SPECTRUM_P:
				return binP.data();
				break;
		}
	}
#endif
