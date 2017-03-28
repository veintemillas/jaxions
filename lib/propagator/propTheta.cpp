#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "enum-field.h"
#include "propagator/RKParms.h"

#include "propagator/propThetaXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "propagator/propThetaGpu.h"
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

void	PropTheta::runCpu	()
{
	propThetaCpu(axionField, dz, delta2, nQcd, Lx, V, S, precision);
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

void	propTheta	(Scalar *field, const double dz, const double nQcd, const double delta, DeviceType dev, FlopCounter *fCount)
{
	PropTheta *prop = new PropTheta(field, nQcd, delta, dz);

	switch (dev)
	{
		case DEV_CPU:
			prop->runCpu ();
			break;

		case DEV_GPU:
			prop->runGpu ();
			break;

		case	DEV_XEON:
			prop->runXeon();
			break;

		default:
			printf ("Not a valid device\n");
			break;
	}

	delete	prop;

	fCount->addFlops(16.*4.*field->Size()*1.e-9, 10.*4.*field->DataSize()*field->Size()*1.e-9);

	return;
}
