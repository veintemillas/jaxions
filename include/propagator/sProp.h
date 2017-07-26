#ifndef	_SPROPCLASS_
	#define	_SPROPCLASS_

	#include "propagator/propBase.h"
	#include "scalar/scalarField.h"
	#include "utils/utils.h"

	#include "propagator/sPropXeon.h"
	#include "propagator/sPropThetaXeon.h"


	template<const int nStages, const bool lastStage, VqcdType pot>
	class	PropSpec : public PropBase<nStages, lastStage, pot> {

		private:

		const double dz, nQcd, LL;
		const size_t Lx, Lz, V, S;

		FieldPrecision precision;
		LambdaType lType;

		Scalar	*axionField;

		public:

			 SPropagator(Scalar *field, const double LL, const double nQcd, const double dz, VqcdType pot);
			~SPropagator() {};

		void	sRunGpu	();	// Saxion propagator
		void	sRunCpu	();
		void	sRunXeon();

		void	tRunGpu	();	// Axion propagator
		void	tRunCpu	();
		void	tRunXeon();
	};


#endif
