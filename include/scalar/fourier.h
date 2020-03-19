#ifndef	_FFTFIELD_CLASS_
	#define _FFTFIELD_CLASS_

	#include "scalarField.h"
	#include "utils/tunable.h"

	class	FTfield : public Tunable
	{
		private:

		Scalar *field;

		const size_t Lz;
		const size_t n1;
		const size_t n2;
		const size_t n3;
		const size_t N;

		size_t shift;
		size_t fSize;

		void	ftField(FieldIndex mvomv);

		void	iftField(FieldIndex mvomv);

		void	padtom2(void* point);
		void	unpadfromm2(void* point);

		public:

			 FTfield(Scalar *scalar);
			~FTfield() {};

		void	operator()(FieldIndex mvomv, FFTdir dir);
	};
#endif
