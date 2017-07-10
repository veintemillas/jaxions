#include <cstdio>
#include <cstdlib>
#include <string>
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

#include "utils/utils.h"

class	Propagator : public Tunable
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

	void	propLowGpu	(const double c, const double d);

	public:

		 Propagator(Scalar *field, const double LL, const double nQcd, const double delta, const double dz, VqcdType pot);
		~Propagator() {};

	void	sRunCpu	();	// Saxion propagator
	void	sRunGpu	();
	void	sRunXeon();

	void	tRunCpu	();	// Axion propagator
	void	tRunGpu	();
	void	tRunXeon();

	void	lowCpu	();	// Lowmem only available for saxion
	void	lowGpu	();
	void	lowXeon	();
};

	Propagator::Propagator(Scalar *field, const double LL, const double nQcd, const double delta, const double dz, VqcdType pot) : axionField(field), dz(dz), Lx(field->Length()), Lz(field->eDepth()),
		V(field->Size()), S(field->Surf()), c1(C1), d1(D1), c2(C2), d2(D2), c3(C3), d3(D3), c4(C4), d4(D4), delta2(delta*delta), precision(field->Precision()), LL(LL), nQcd(nQcd), pot(pot), lType(field->Lambda())
{
}

void	Propagator::sRunGpu	()
{
#ifdef	USE_GPU
	const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
	const uint ext = uV + uS;
	double *z = axionField->zV();
	double lambda = LL;

	if (lType != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, lambda, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, lambda, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, lambda, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
	*z += dz*d1;


	if (lType != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, lambda, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M2);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, lambda, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, lambda, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
	*z += dz*d2;


	if (lType != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, lambda, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, lambda, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, lambda, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
	*z += dz*d3;


	if (lType != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, lambda, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M2);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, lambda, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, lambda, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

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
	double lambda = LL;

	if (lType != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, lambda, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	updateMGpu(axionField->mGpu(), axionField->vGpu(), dz, d, Lx, 3*S, V-S, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, lambda, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, lambda, nQcd, uLx, uLz, uV,  ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
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

void	Propagator::sRunCpu	()
{
	propagateCpu	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision, pot);
	//propOmelyanCpu	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision, pot);
}

void	Propagator::lowCpu	()
{
	propLowMemCpu	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision, pot);
}

void	Propagator::sRunXeon	()
{
#ifdef	USE_XEON
	propagateXeon	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision, pot);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	Propagator::lowXeon	()
{
#ifdef	USE_XEON
	propLowMemXeon	(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision, pot);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void    Propagator::tRunGpu	()
{
#ifdef  USE_GPU
	const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
	const uint ext = uV + uS;
	double *z = axionField->zV();

	propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
	propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
	propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();        // This is not strictly necessary, but simplifies things a lot
	*z += dz*d1;

	propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M2);
	propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
	propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();        // This is not strictly necessary, but simplifies things a lot
	*z += dz*d2;

	propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
	propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
	propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();        // This is not strictly necessary, but simplifies things a lot
	*z += dz*d3;

	propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M2);
	propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
	propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	cudaDeviceSynchronize();        // This is not strictly necessary, but simplifies things a lot
	*z += dz*d4;
#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void    Propagator::tRunCpu	()
{
	propThetaCpu(axionField, dz, delta2, nQcd, Lx, V, S, precision);
}

void    Propagator::tRunXeon	()
{
#ifdef  USE_XEON
	propThetaXeon(axionField, dz, delta2, nQcd, Lx, V, S, precision);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

using	namespace profiler;

//void	propagate	(Scalar *field, FlopCounter *fCount, const double dz, const double delta, const double nQcd, const double LL, VqcdType pot)
void	propagate	(Scalar *field, const double dz, const double delta, const double nQcd, const double LL, VqcdType pot)
{
	LogMsg	(VERB_HIGH, "Called propagator");
	profiler::Profiler &prof = getProfiler(PROF_PROP);

	Propagator *prop = new Propagator(field, LL, nQcd, delta, dz, pot);

	if	(!field->Folded())
	{
		Folder	munge(field);
		munge(FOLD_ALL);
	}

	prof.start();

	switch (field->Field()) {
		case FIELD_AXION:
			prop->setName("RKN4 Axion");

			switch (field->Device()) {
				case DEV_GPU:
					prop->tRunGpu ();
					break;
				case DEV_CPU:
					prop->tRunCpu ();
					break;
				case DEV_XEON:
					prop->tRunXeon ();
					break;
				default:
					LogError ("Not a valid device");
					prof.stop();
					delete prop;
					return;
			}

			prop->add(16.*4.*field->Size()*1.e-9, 10.*4.*field->DataSize()*field->Size()*1.e-9);

			break;

		case FIELD_SAXION:
			if (field->LowMem()) {
				prop->setName("RKN4 Saxion Lowmem");

				switch (field->Device()) {
					case DEV_GPU:
						prop->lowGpu ();
						break;
					case DEV_CPU:
						prop->lowCpu ();
						break;
					case DEV_XEON:
						prop->lowXeon ();
						break;
					default:
						LogError ("Not a valid device");
						prof.stop();
						delete prop;
						return;
				}
			} else {
				prop->setName("RKN4 Saxion");

				switch (field->Device()) {
					case DEV_GPU:
						prop->sRunGpu ();
						break;
					case DEV_CPU:
						prop->sRunCpu ();
						break;
					case DEV_XEON:
						prop->sRunXeon ();
						break;
					default:
						LogError ("Not a valid device");
						prof.stop();
						delete prop;
						return;
				}

			}

			switch (pot)
			{
				case VQCD_1:
					prop->add((32.*4.+30.)*field->Size()*1.e-9, (10.*4.+9.)*field->DataSize()*field->Size()*1.e-9);
					break;

				case VQCD_2:
					prop->add((35.*4.+33.)*field->Size()*1.e-9, (10.*4.+9.)*field->DataSize()*field->Size()*1.e-9);
					break;
			}
			break;

		default:
			LogError ("Invalid field type");
			prof.stop();
			delete prop;
			return;
	}

	prof.stop();

	prof.add(prop->Name(), prop->GFlops(), prop->GBytes());

	LogMsg	(VERB_HIGH, "Propagator %s reporting %lf GFlops %lf GBytes", prop->Name().c_str(), prof.Prof()[prop->Name()].GFlops(), prof.Prof()[prop->Name()].GBytes());

	delete	prop;

	return;
}
