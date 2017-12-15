#ifndef	_FFT_CLASS_
	#define	_FFT_CLASS_

	#include <string>

	#include "enum-field.h"
	#include "scalar/scalarField.h"
	#include "utils/utils.h"
	#include "utils/tunable.h"

	namespace AxionFFT {

		class	FFTplan	: public Tunable
		{
			private:

			void *		planForward;
			void *		planBackward;

			FFTtype		type;
			FFTdir		dFft;
			FieldPrecision	prec;

			size_t		Lx, Lz;

			void		importWisdom();
			void		exportWisdom();

			public:

					 FFTplan() : planForward(nullptr), planBackward(nullptr), Lx(0), Lz(0), dFft(FFT_NONE), type(FFT_NOTYPE) {}
					 FFTplan(Scalar * axion, FFTtype type, FFTdir dFft);
//					~FFTplan() {};

			void		run	(FFTdir cDir);
			double		GFlops	(FFTdir cDir);

			inline	void		SetDir (FFTdir  newDFFT) { if (dFft == FFT_NONE)   dFft = newDFFT; }
			inline	void		SetType(FFTtype newType) { if (type == FFT_NOTYPE) type = newType; }

			inline	FieldPrecision	Precision() { return prec; }
			inline	FFTdir		Direction() { return dFft; }
			inline	void *		PlanFwd  () { return planForward; }
			inline	void *		PlanBack () { return planBackward; }
		};

		void		initFFT		(FieldPrecision prec);
		void		initPlan	(Scalar * axion, FFTtype type, FFTdir dFft, std::string name);
		FFTplan&	fetchPlan       (std::string name);
		void		removePlan	(std::string name);
		void		closeFFT	();
	}
#endif
