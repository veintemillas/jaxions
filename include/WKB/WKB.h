#ifndef	_WKB_CLASS_
	#define	_WKB_CLASS_

	#include <cmath>
	#include <complex>
	#include <cstring>

	#include "fft/fftCode.h"
	#include "scalar/scalarField.h"
	#include "scalar/scaleField.h"
	#include "scalar/varNQCD.h"
	#include "utils/index.h"
	#include "utils/utils.h"
	#include "energy/energy.h"
	#include "fft/fftCode.h"

	#include <gsl/gsl_sf_hyperg.h>
	#include <gsl/gsl_sf_gamma.h>

	using namespace std;

	namespace AxionWKB {

		const auto im = complex<double>(1.0i);

		//--------------------------------------------------
		// WKB CLASS
		//--------------------------------------------------
		class WKB
		{
			private:

			Scalar	*field;
			Scalar	*tmp;

			const size_t rLx, Ly, Lz, Sm, hLy, hLz, hTz, Tz, nModes;

			const double zIni;
//			const double amass2zini2 = axionmass2(zini, nQcd, zthres, zrestore)*zini*zini ;

			bool firsttime = true;

			FieldPrecision fPrec ;
//			FieldType fType ;

			// pointers for the axion matrices
			// I need :
			// axion1 m+surf, v, m2
			// axion2 m, v

			template<typename cFloat>
			void	doWKB     (double zEnd);

			template<typename cFloat>
			void	doWKBinplace     (double zEnd);

			public:

				WKB       (Scalar* axion, Scalar* axion2);

			void	operator()(double zEnd);

		};
	}
#endif
