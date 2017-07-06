#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"
#include "propagator/RKParms.h"

#include "propagator/sPropXeon.h"
//#include "propagator/propThetaXeon.h"
/*
#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "propagator/propGpu.h"
	#include "propagator/propThetaGpu.h"
#endif
*/
#include "utils/logger.h"
#include "utils/profiler.h"

class	SPropagator
{
	private:

	const double c1, c2, c3, c4;	// The parameters of the Runge-Kutta-Nyström
	const double d1, d2, d3, d4;
	const double dz;
	const double nQcd, LL;
	const size_t Lx, Lz, V, S;

	FieldPrecision precision;
	LambdaType lType;
	VqcdType pot;

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

	SPropagator::SPropagator(Scalar *field, const double LL, const double nQcd, const double dz, VqcdType pot) : axionField(field), dz(dz), Lx(field->Length()), Lz(field->eDepth()),
		V(field->Size()), S(field->Surf()), c1(C1), d1(D1), c2(C2), d2(D2), c3(C3), d3(D3), c4(C4), d4(D4), precision(field->Precision()), LL(LL), nQcd(nQcd), pot(pot), lType(field->Lambda())
{
}

void	SPropagator::sRunGpu	()
{
}

void	SPropagator::sRunCpu	()
{
	propSpecCpu	(axionField, dz, LL, nQcd, Lx, V, S, precision, pot);
}

void	SPropagator::sRunXeon	()
{
#ifdef	USE_XEON
	propSpecXeon	(axionField, dz, LL, nQcd, Lx, V, S, precision, pot);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	SPropagator::tRunGpu	()
{
}

void    SPropagator::tRunCpu	()
{
//	propSpecThetaCpu(axionField, dz, nQcd, Lx, V, S, precision);
}

void    SPropagator::tRunXeon	()
{
#ifdef  USE_XEON
//	propSpecThetaXeon(axionField, dz, delta2, nQcd, Lx, V, S, precision);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

using namespace profiler;

void	sPropagate	(Scalar *field, const double dz, const double nQcd, const double LL, VqcdType pot)
{
	LogMsg  (VERB_HIGH, "Called propagator");
	Profiler &prof = getProfiler(PROF_PROP);

	SPropagator *prop = new SPropagator(field, LL, nQcd, dz, pot);

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	std::string	name;

	prof.start();

	switch (field->Device())
	{
		case DEV_CPU:
			if (field->Field() == FIELD_SAXION) {
				name.assign("RKN4 Spectral Saxion");
				prop->sRunCpu ();
			} else {
				name.assign("RKN4 Spectral Axion");
				prop->tRunCpu ();
			}
			break;

		case DEV_GPU:
			if (field->Field() == FIELD_SAXION) {
				name.assign("RKN4 Spectral Saxion");
				prop->sRunGpu ();
			} else {
				name.assign("RKN4 Spectral Axion");
				prop->tRunGpu ();
			}
			break;

		case	DEV_XEON:
			if (field->Field() == FIELD_SAXION) {
				name.assign("RKN4 Spectral Saxion");
				prop->sRunXeon();
			} else {
				name.assign("RKN4 Spectral Axion");
				prop->tRunXeon ();
			}
			break;
		default:
			printf ("Not a valid device\n");
			break;
	}

	delete	prop;

	if (field->Field() == FIELD_SAXION) {
		switch (pot)
		{
			case VQCD_1: {
				double	gFlops = (field->Size()*64 + field->Size()*(20.1977*log(field->Size()) + 2))*1e-9;	// Approx
				double	gBytes = field->Size()*field->DataSize()*22e-9;
				prof.add(name, gFlops, gBytes);
				break;
			}

			case VQCD_2: {
				double	gFlops = (field->Size()*76 + field->Size()*(20.1977*log(field->Size()) + 2))*1e-9;	// Approx
				double	gBytes = field->Size()*field->DataSize()*22e-9;
				prof.add(name, gFlops, gBytes);
				break;
			}
		}
	} else {
		double	gFlops = 0.*1e-9;	// Approx
		double	gBytes = 0.*field->Size()*field->DataSize()*22e-9;
		prof.add(name, gFlops, gBytes);
	}

	LogMsg  (VERB_HIGH, "Propagator %s reporting %lf GFlops %lf GBytes", name.c_str(), prof.Prof()[name].GFlops(), prof.Prof()[name].GBytes());

	return;
}
