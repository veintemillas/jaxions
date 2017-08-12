#ifndef	_CLASS_SPECTRUM_
	#define	_CLASS_SPECTRUM_

	#include "enum-field.h"
	#include "scalar/scalarField.h"
	#include "scalar/varQCD.h"

	class	SpecBin : public Tunable {

		private:

		std::vector<double>	binK;
		std::vector<double>	binG;
		std::vector<double>	binV;
		std::vector<double>	binP;

		std::vector<double>	cosTable;

		Scalar			*field;

		size_t			Lx, Ly, Lz, Tz, nPts, kMax;
		double			powMax, mass;

		void			fillCosTable ();

		const bool		spec;

		public:

				SpecBin (Scalar *field, const bool spectral) : field(field), Lx(field->Length()), Lz(field->Depth()), Tz(field->TotalDepth()), fSize(field->DataSize()),
									       fPrec(field->Precision()), nPts(field->Size()), fType(field->Field()), spec(spectral) {
				kMax   = (Lx>>1)-1;
				powMax = floor(sqrt(3.)*kMax)+2;

				binK.resize(powMax); binK.fill(0.);
				binG.resize(powMax); binG.fill(0.);
				binV.resize(powMax); binV.fill(0.);
				binP.resize(powMax); binP.fill(0.);

				mass   = axionmass2((*field->zV()), nQcd, zthres, zrestore)*(*field->zV())*(*field->zV());

				fillCosTable();

				Ly = Lx;

				hLy = Ly >> 1;
				hLz = Tz >> 1;

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



		inline double	SpecBin::operator()(size_t idx, SpectrumType sType)	const;
		inline double&	SpecBin::operator()(size_t idx, SpectrumType sType);

		void		run	();
	}


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

	inline Float&	SpecBin::operator()(size_t idx, SpectrumType rType)		{

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
#endif
