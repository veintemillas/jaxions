#include <cstdio>
#include <cstdlib>
#include "scalarField.h"
#include "enum-field.h"
#include "RKParms.h"

#include "propXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "propGpu.h"
#endif

#include "flopCounter.h"

class	Propagator
{
	private:

	const double c1, c2, c3, c4;	// The parameters of the Runge-Kutta-Nyström
	const double d1, d2, d3, d4;
	const double delta2, dz;
	const double nQcd, LL;
	const int Lx, Lz, V, S;

	FieldPrecision precision;

	Scalar	*axionField;

	void	propLowGpu	(const double c, const double d);

	public:

		 Propagator(Scalar *field, const double LL, const double nQcd, const double delta, const double dz);
		~Propagator() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();

	void	lowCpu	();
	void	lowGpu	();
	void	lowXeon	();
};

	Propagator::Propagator(Scalar *field, const double LL, const double nQcd, const double delta, const double dz) : axionField(field), dz(dz), Lx(field->Length()), Lz(field->eDepth()), V(field->Size()),
				S(field->Surf()), c1(C1), d1(D1), c2(C2), d2(D2), c3(C3), d3(D3), c4(C4), d4(D4), delta2(delta*delta), precision(field->Precision()), LL(LL), nQcd(nQcd)
{
}

void	Propagator::runGpu	()
{
#ifdef	USE_GPU
	const int ext = V + S;
	double *z = axionField->zV();

        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, LL, nQcd, Lx, Lz, 2*S, V, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, LL, nQcd, Lx, Lz, S, 2*S, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, LL, nQcd, Lx, Lz, V, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
	*z += dz*d1;

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot

        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, LL, nQcd, Lx, Lz, 2*S, V, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M2);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, LL, nQcd, Lx, Lz, S, 2*S, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, LL, nQcd, Lx, Lz, V, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
	*z += dz*d2;

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot

        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, LL, nQcd, Lx, Lz, 2*S, V, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, LL, nQcd, Lx, Lz, S, 2*S, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, LL, nQcd, Lx, Lz, V, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
	*z += dz*d3;

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot

        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, LL, nQcd, Lx, Lz, 2*S, V, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M2);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, LL, nQcd, Lx, Lz, S, 2*S, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, LL, nQcd, Lx, Lz, V, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
	*z += dz*d4;

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void	Propagator::propLowGpu	(const double c, const double d)
{
#ifdef	USE_GPU
	const int ext = V + S;
	double *z = axionField->zV();

	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, LL, nQcd, Lx, Lz, 2*S, V, precision, ((cudaStream_t *)axionField->Streams())[2]);
	updateMGpu(axionField->mGpu(), axionField->vGpu(), dz, d, Lx, 3*S, V-S, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, LL, nQcd, Lx, Lz, S, 2*S, precision, ((cudaStream_t *)axionField->Streams())[0]);
	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, LL, nQcd, Lx, Lz, V, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
	cudaStreamSynchronize(((cudaStream_t *)axionField->Streams())[2]);
	updateMGpu(axionField->mGpu(), axionField->vGpu(), dz, d, Lx, S, 3*S, precision, ((cudaStream_t *)axionField->Streams())[0]);
	updateMGpu(axionField->mGpu(), axionField->vGpu(), dz, d, Lx, V-S, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	*z += dz*d;

	cudaDeviceSynchronize();
#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void	Propagator::lowGpu	()
{
	propLowGpu(c1, d1);
	propLowGpu(c2, d2);
	propLowGpu(c3, d3);
	propLowGpu(c4, d4);
}

void	Propagator::runCpu	()
{
	propagateCpu(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
}

void	Propagator::lowCpu	()
{
	propLowMemCpu(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
}

void	Propagator::runXeon	()
{
#ifdef	USE_XEON
	propagateXeon(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	Propagator::lowXeon	()
{
#ifdef	USE_XEON
	propLowMemXeon(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	propagate	(Scalar *field, const double dz, const double LL, const double nQcd, const double delta, DeviceType dev, FlopCounter *fCount)
{
	Propagator *prop = new Propagator(field, LL, nQcd, delta, dz);

	switch (dev)
	{
		case DEV_CPU:
			if (field->LowMem())
				prop->lowCpu ();
			else
				prop->runCpu ();
			break;

		case DEV_GPU:
			if (field->LowMem())
				prop->lowGpu ();
			else
				prop->runGpu ();
			break;

		case	DEV_XEON:
			if (field->LowMem())
				prop->lowXeon();
			else
				prop->runXeon();
			break;

		default:
			printf ("Not a valid device\n");
			break;
	}

	delete	prop;

	fCount->addFlops(32.*4.*field->Size()*1.e-9, 10.*4.*field->dataSize()*field->Size()*1.e-9);

	return;
}
