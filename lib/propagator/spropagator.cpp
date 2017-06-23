#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"
#include "propagator/RKParms.h"

#include "propagator/propXeon.h"
#include "propagator/propThetaXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "propagator/propGpu.h"
	#include "propagator/propThetaGpu.h"
#endif

#include "utils/flopCounter.h"

class	SPropagator
{
	private:

	const double c1, c2, c3, c4;	// The parameters of the Runge-Kutta-Nyström
	const double d1, d2, d3, d4;
	const double delta2, dz;
	const double nQcd, LL;
	const size_t Lx, Lz, V, S;

	FieldPrecision precision;
	LambdaType lType;
	VqcdType pot;

	Scalar	*axionField;

	void	spropLowGpu	(const double c, const double d);

	public:

		 SPropagator(Scalar *field, const double LL, const double nQcd, const double delta, const double dz, VqcdType pot);
		~SPropagator() {};

	void	sRunCpu	();	// Saxion propagator
	void	sRunXeon();

	void	tRunCpu	();	// Axion propagator
	void	tRunXeon();

	void	lowCpu	();	// Lowmem only available for saxion
	void	lowXeon	();
};

	SPropagator::SPropagator(Scalar *field, const double LL, const double nQcd, const double delta, const double dz, VqcdType pot) : axionField(field), dz(dz), Lx(field->Length()), Lz(field->eDepth()),
		V(field->Size()), S(field->Surf()), c1(C1), d1(D1), c2(C2), d2(D2), c3(C3), d3(D3), c4(C4), d4(D4), delta2(delta*delta), precision(field->Precision()), LL(LL), nQcd(nQcd), pot(pot), lType(field->Lambda())
{
}



void	SPropagator::sRunCpu	()
{
	spropagateCpu	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision, pot);
}

void	SPropagator::lowCpu	()
{
	spropLowMemCpu	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision, pot);
}

void	SPropagator::sRunXeon	()
{
#ifdef	USE_XEON
	spropagateXeon	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision, pot);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	SPropagator::lowXeon	()
{
#ifdef	USE_XEON
	spropLowMemXeon	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision, pot);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void    SPropagator::tRunCpu	()
{
	spropThetaCpu(axionField, dz, delta2, nQcd, Lx, V, S, precision);
}

void    SPropagator::tRunXeon	()
{
#ifdef  USE_XEON
	spropThetaXeon(axionField, dz, delta2, nQcd, Lx, V, S, precision);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}


void	spropagate	(Scalar *field, FlopCounter *fCount, const double dz, const double delta, const double nQcd, const double LL, VqcdType pot)
{
	SPropagator *prop = new SPropagator(field, LL, nQcd, delta, dz, pot);

	if	(!field->Folded())
	{
		Folder	munge(field);
		munge(FOLD_ALL);
	}

	switch (field->Device())
	{
		case DEV_CPU:
			if (field->Field() == FIELD_SAXION) {
				if (field->LowMem())
					prop->lowCpu ();
				else
					prop->sRunCpu ();
			} else {
				prop->tRunCpu ();
			}
			break;

		case DEV_GPU:
			if (field->Field() == FIELD_SAXION) {
				if (field->LowMem())
					prop->lowGpu ();
				else
					prop->sRunGpu ();
			} else {
				prop->tRunGpu ();
			}
			break;

		case	DEV_XEON:
			if (field->Field() == FIELD_SAXION) {
				if (field->LowMem())
					prop->lowXeon();
				else
					prop->sRunXeon();
				break;
			} else {
				prop->tRunXeon ();
			}

		default:
			printf ("Not a valid device\n");
			break;
	}

	delete	prop;

	if (field->Field() == FIELD_SAXION) {
		switch (pot)
		{
			case VQCD_1:
				fCount->addFlops(32.*4.*field->Size()*1.e-9, 10.*4.*field->DataSize()*field->Size()*1.e-9);
				break;

			case VQCD_2:
				fCount->addFlops(35.*4.*field->Size()*1.e-9, 10.*4.*field->DataSize()*field->Size()*1.e-9);
				break;
		}
	} else {
		fCount->addFlops(16.*4.*field->Size()*1.e-9, 10.*4.*field->DataSize()*field->Size()*1.e-9);
	}

	return;
}
