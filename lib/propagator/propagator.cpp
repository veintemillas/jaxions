#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "enum-field.h"
#include "propagator/RKParms.h"

#include "propagator/propXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "propagator/propGpu.h"
#endif

#include "utils/flopCounter.h"

class	Propagator
{
	private:

	const double c1, c2, c3, c4;	// The parameters of the Runge-Kutta-Nyström
	const double d1, d2, d3, d4;
	const double delta2, dz;
	const double nQcd, LL;
	const size_t Lx, Lz, V, S;

	FieldPrecision precision;
	VqcdType pot;

	Scalar	*axionField;

	void	propLowGpu	(const double c, const double d);

	public:

		 Propagator(Scalar *field, const double LL, const double nQcd, const double delta, const double dz, VqcdType pot);
		~Propagator() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();

	void	lowCpu	();
	void	lowGpu	();
	void	lowXeon	();
};

	Propagator::Propagator(Scalar *field, const double LL, const double nQcd, const double delta, const double dz, VqcdType pot) : axionField(field), dz(dz), Lx(field->Length()), Lz(field->eDepth()),
				V(field->Size()), S(field->Surf()), c1(C1), d1(D1), c2(C2), d2(D2), c3(C3), d3(D3), c4(C4), d4(D4), delta2(delta*delta), precision(field->Precision()), LL(LL), nQcd(nQcd), pot(pot)
{
}

void	Propagator::runGpu	()
{
#ifdef	USE_GPU
	const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
	const uint ext = uV + uS;
	double *z = axionField->zV();

        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, LL, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, LL, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, LL, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
	*z += dz*d1;

        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, LL, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M2);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, LL, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, LL, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
	*z += dz*d2;

        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, LL, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, LL, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, LL, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
	*z += dz*d3;

        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, LL, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M2);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, LL, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, LL, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
	*z += dz*d4;
#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void	Propagator::propLowGpu	(const double c, const double d)
{
#ifdef	USE_GPU
	const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
	const uint ext = V + S;
	double *z = axionField->zV();

	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, LL, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	updateMGpu(axionField->mGpu(), axionField->vGpu(), dz, d, Lx, 3*S, V-S, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, LL, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, LL, nQcd, uLx, uLz, uV,  ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
	cudaStreamSynchronize(((cudaStream_t *)axionField->Streams())[2]);
	updateMGpu(axionField->mGpu(), axionField->vGpu(), dz, d, uLx, uS,   3*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
	updateMGpu(axionField->mGpu(), axionField->vGpu(), dz, d, uLx, uV-uS, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

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
	switch (pot)
	{
		case VQCD_1:
			propagateCpu	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
			break;

		case VQCD_2:
			propagateCpuV2	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
			break;
	}
}

void	Propagator::lowCpu	()
{
	switch (pot)
	{
		case VQCD_1:
			propLowMemCpu	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
			break;

		case VQCD_2:
			propLowMemCpuV2	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
			break;
	}
}

void	Propagator::runXeon	()
{
#ifdef	USE_XEON
	switch (pot)
	{
		case VQCD_1:
			propagateXeon	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
			break;

		case VQCD_2:
			propagateXeonV2	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
			break;
	}
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	Propagator::lowXeon	()
{
#ifdef	USE_XEON
	switch (pot)
	{
		case VQCD_1:
			propLowMemXeon	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
			break;

		case VQCD_2:
			propLowMemXeonV2(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
			break;
	}
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	propagate	(Scalar *field, const double dz, const double LL, const double nQcd, const double delta, DeviceType dev, FlopCounter *fCount, VqcdType pot)
{
	Propagator *prop = new Propagator(field, LL, nQcd, delta, dz, pot);

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

	switch (pot)
	{
		case VQCD_1:
			fCount->addFlops(32.*4.*field->Size()*1.e-9, 10.*4.*field->DataSize()*field->Size()*1.e-9);
			break;

		case VQCD_2:
			fCount->addFlops(35.*4.*field->Size()*1.e-9, 10.*4.*field->DataSize()*field->Size()*1.e-9);
			break;
	}

	return;
}
