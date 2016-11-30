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

class	PropTheta
{
	private:

	const double c1, c2, c3, c4;	// The parameters of the Runge-Kutta-Nyström
	const double d1, d2, d3, d4;
	const double delta2, dz;
	const double nQcd;
	const size_t Lx, Lz, V, S;

	FieldPrecision precision;

	Scalar	*axionField;

	void	propLowGpu	(const double c, const double d);

	public:

		 PropTheta(Scalar *field, const double nQcd, const double delta, const double dz);
		~PropTheta() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();

	void	lowCpu	();
	void	lowGpu	();
	void	lowXeon	();
};

	PropTheta::PropTheta(Scalar *field, const double nQcd, const double delta, const double dz) : axionField(field), dz(dz), Lx(field->Length()), Lz(field->eDepth()), V(field->Size()),
				S(field->Surf()), c1(C1), d1(D1), c2(C2), d2(D2), c3(C3), d3(D3), c4(C4), d4(D4), delta2(delta*delta), precision(field->Precision()), nQcd(nQcd)
{
}

void	PropTheta::runGpu	()
{
#ifdef	USE_GPU
	const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
	const uint ext = uV + uS;
	double *z = axionField->zV();

        propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
        propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
	*z += dz*d1;

        propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M2);
        propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
	*z += dz*d2;

        propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
        propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
	*z += dz*d3;

        propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M2);
        propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
	*z += dz*d4;
#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void	PropTheta::propThetaLowGpu	(const double c, const double d)
{
#ifdef	USE_GPU
	const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
	const uint ext = V + S;
	double *z = axionField->zV();

	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	updateMGpu(axionField->mGpu(), axionField->vGpu(), dz, d, Lx, 3*S, V-S, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, nQcd, uLx, uLz, uV,  ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
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

void	PropTheta::lowGpu	()
{
	propThetaLowGpu(c1, d1);
	propThetaLowGpu(c2, d2);
	propThetaLowGpu(c3, d3);
	propThetaLowGpu(c4, d4);
}

void	PropTheta::runCpu	()
{
	propThetaCpu(axionField, dz, delta2, nQcd, Lx, V, S, precision);
}

void	PropTheta::lowCpu	()
{
	propThetaLowMemCpu(axionField, dz, delta2, nQcd, Lx, V, S, precision);
}

void	PropTheta::runXeon	()
{
#ifdef	USE_XEON
	propThetaXeon(axionField, dz, delta2, nQcd, Lx, V, S, precision);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	PropTheta::lowXeon	()
{
#ifdef	USE_XEON
	propThetaLowMemXeon(axionField, dz, delta2, nQcd, Lx, V, S, precision);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	propTheta	(Scalar *field, const double dz, const double nQcd, const double delta, DeviceType dev, FlopCounter *fCount)
{
	PropTheta *prop = new PropTheta(field, nQcd, delta, dz);

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

	fCount->addFlops(32.*4.*field->Size()*1.e-9, 10.*4.*field->DataSize()*field->Size()*1.e-9);

	return;
}
