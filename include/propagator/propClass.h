#ifndef	_PROPCLASS_
	#define	_PROPCLASS_

	#include <cmath>
	#include <string>
	#include <cstring>
	#include <functional>
	#include "propagator/propBase.h"
	#include "scalar/scalarField.h"
	#include "utils/utils.h"

	#include "propagator/propXeon.h"
	#include "propagator/propThetaXeon.h"
	#include "propagator/laplacian.h"
	#include "propagator/sPropXeon.h"
	#include "propagator/sPropThetaXeon.h"

	#ifdef	USE_GPU
		#include <cuda.h>
		#include <cuda_runtime.h>
		#include <cuda_device_runtime_api.h>
		#include "propagator/propGpu.h"
		#include "propagator/propThetaGpu.h"
	#endif

	#ifdef	__GNUG__
		#pragma GCC optimize ("unroll-loops")
	#endif

	template<const int nStages, const bool lastStage, VqcdType VQcd = false>
	class	PropClass : public PropBase
	{
		protected:

		double	c[nStages + (lastStage == true ? 1 : 0)];
		double	d[nStages];
		const double delta2;
		const double nQcd, LL;
		const size_t Lx, Lz, V, S;

		FieldPrecision precision;
		LambdaType lType;

		Scalar	*axionField;

		public:

		inline	 PropClass(Scalar *field, const double LL, const double nQcd, const double delta, const bool spec);
		inline	~PropClass() override {};

		inline void	setCoeff(const double * __restrict__ nC, const double * __restrict__ nD) {
			for(int i=0; i<nStages; i++) { c[i] = nC[i]; d[i] = nD[i]; } if (lastStage) { c[nStages] = nC[nStages]; }
		}

		inline void	sRunCpu	(const double)	override;	// Saxion propagator
		inline void	sRunGpu	(const double)	override;

		inline void	sSpecCpu(const double)	override;	// Saxion spectral propagator

		inline void	tRunCpu	(const double)	override;	// Axion propagator
		inline void	tRunGpu	(const double)	override;

		inline void	tSpecCpu(const double)	override;	// Axion spectral propagator

		inline void	lowCpu	(const double)	override;	// Lowmem only available for saxion non-spectral
		inline void	lowGpu	(const double)	override;

		inline double	cFlops	(const bool)	override;
		inline double	cBytes	(const bool)	override;
	};

	template<const int nStages, const bool lastStage, VqcdType VQcd>
		PropClass<nStages, lastStage, VQcd>::PropClass(Scalar *field, const double LL, const double nQcd, const double delta, const bool spec) : axionField(field),
		Lx(field->Length()), Lz(field->eDepth()), V(field->Size()), S(field->Surf()), delta2(delta*delta), precision(field->Precision()), LL(LL), nQcd(nQcd), lType(field->Lambda())
	{
		if (spec) {
			switch (field->Device()) {
				case	DEV_CPU:
					if (field->LowMem()) {
						LogError ("Error: Lowmem not supported with spectral propagators");
						exit(1);
					}

					propSaxion = [this](const double dz) { this->sSpecCpu(dz); };
					propAxion  = [this](const double dz) { this->tSpecCpu(dz); };
					break;

				case	DEV_GPU:
					LogMsg	(VERB_HIGH, "Warning: spectral propagators not supported in gpus, will call standard propagator");
					if (field->LowMem()) {
						propSaxion = [this](const double dz) { this->lowGpu(dz); };
						propAxion  = [this](const double dz) { this->tRunGpu(dz); };
					} else {
						propSaxion = [this](const double dz) { this->sRunGpu(dz); };
						propAxion  = [this](const double dz) { this->tRunGpu(dz); };
					}
					break;

				default:
					LogError ("Error: not a valid device");
					return;
			}
		} else {
			switch (field->Device()) {
				case	DEV_CPU:
					if (field->LowMem()) {
						propSaxion = [this](const double dz) { this->lowCpu(dz); };
						propAxion  = [this](const double dz) { this->tRunCpu(dz); };
					} else {
						propSaxion = [this](const double dz) { this->sRunCpu(dz); };
						propAxion  = [this](const double dz) { this->tRunCpu(dz); };
					}
					break;

				case	DEV_GPU:
					if (field->LowMem()) {
						propSaxion = [this](const double dz) { this->lowGpu(dz); };
						propAxion  = [this](const double dz) { this->tRunGpu(dz); };
					} else {
						propSaxion = [this](const double dz) { this->sRunGpu(dz); };
						propAxion  = [this](const double dz) { this->tRunGpu(dz); };
					}
					break;

				default:
					LogError ("Error: not a valid device");
					return;
			}
		}
	}

	/*		GPU PROPAGATORS			*/

	// Generic axion propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::tRunGpu	(const double dz) {
	#ifdef  USE_GPU
		const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
		const uint ext = uV + uS;
		double *z = axionField->zV();

		const bool wMod = (axionField->Field() == FIELD_AXION_MOD) ? true : false;

		#pragma unroll
		for (int s = 0; s<nStages; s+=2) {
			const double	c1 = c[s], c2 = c[s+1], d1 = d[s], d2 = d[s+1];

			propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, nQcd, uLx, uLz, 2*uS, uV, precision,
				    ((cudaStream_t *)axionField->Streams())[2], wMod);
			axionField->exchangeGhosts(FIELD_M);
			propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, nQcd, uLx, uLz, uS, 2*uS, precision,
				    ((cudaStream_t *)axionField->Streams())[0], wMod);
			propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, nQcd, uLx, uLz, uV,  ext, precision,
				    ((cudaStream_t *)axionField->Streams())[1], wMod);
			cudaDeviceSynchronize();        // This is not strictly necessary, but simplifies things a lot
			*z += dz*d1;

			propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, nQcd, uLx, uLz, 2*uS, uV, precision,
				    ((cudaStream_t *)axionField->Streams())[2], wMod);
			axionField->exchangeGhosts(FIELD_M2);
			propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, nQcd, uLx, uLz, uS, 2*uS, precision,
				    ((cudaStream_t *)axionField->Streams())[0], wMod);
			propThetaGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, nQcd, uLx, uLz, uV,  ext, precision,
				    ((cudaStream_t *)axionField->Streams())[1], wMod);
			cudaDeviceSynchronize();        // This is not strictly necessary, but simplifies things a lot
			*z += dz*d2;
		}

		if (lastStage) {
			LogMsg (VERB_HIGH, "Warning: axion propagator not optimized yet for odd propagators, performance might be reduced");

			const double	c0 = c[nStages];

			propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c0, 0., delta2, nQcd, uLx, uLz, 2*uS, uV, precision,
				    ((cudaStream_t *)axionField->Streams())[2], wMod);
			axionField->exchangeGhosts(FIELD_M);
			propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c0, 0., delta2, nQcd, uLx, uLz, uS, 2*uS, precision,
				    ((cudaStream_t *)axionField->Streams())[0], wMod);
			propThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c0, 0., delta2, nQcd, uLx, uLz, uV,  ext, precision,
				    ((cudaStream_t *)axionField->Streams())[1], wMod);
			cudaDeviceSynchronize();        // This is not strictly necessary, but simplifies things a lot
		}
	#else
		LogError ("Error: gpu support not built");
		exit(1);
	#endif
	}

	// Generic saxion propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::sRunGpu	(const double dz) {
	#ifdef	USE_GPU
		const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
		const uint ext = uV + uS;
		double *z = axionField->zV();
		double lambda = LL;

		#pragma unroll
		for (int s = 0; s<nStages; s+=2) {
			if (lType != LAMBDA_FIXED)
				lambda = LL/((*z)*(*z));

			const double	c1 = c[s], c2 = c[s+1], d1 = d[s], d2 = d[s+1];

		        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, lambda, nQcd, uLx, uLz, 2*uS, uV, precision,
				    ((cudaStream_t *)axionField->Streams())[2]);
			axionField->exchangeGhosts(FIELD_M);
	        	propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, lambda, nQcd, uLx, uLz, uS, 2*uS, precision,
				    ((cudaStream_t *)axionField->Streams())[0]);
	        	propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, lambda, nQcd, uLx, uLz, uV,  ext, precision,
				    ((cudaStream_t *)axionField->Streams())[1]);

			cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
			*z += dz*d1;

			if (lType != LAMBDA_FIXED)
				lambda = LL/((*z)*(*z));

		        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, lambda, nQcd, uLx, uLz, 2*uS, uV, precision,
				    ((cudaStream_t *)axionField->Streams())[2]);
			axionField->exchangeGhosts(FIELD_M2);
	        	propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, lambda, nQcd, uLx, uLz, uS, 2*uS, precision,
				    ((cudaStream_t *)axionField->Streams())[0]);
	        	propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, lambda, nQcd, uLx, uLz, uV,  ext, precision,
				    ((cudaStream_t *)axionField->Streams())[1]);

			cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
			*z += dz*d2;
		}

		if (lastStage) {
			if (lType != LAMBDA_FIXED)
				lambda = LL/((*z)*(*z));

			const double	c0 = c[nStages];

			updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c0, delta2, lambda, nQcd, uLx, uLz, uS*2, uV, precision,
				  ((cudaStream_t *)axionField->Streams())[2]);
			axionField->exchangeGhosts(FIELD_M);
			updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c0, delta2, lambda, nQcd, uLx, uLz, uS, uS*2, precision,
				  ((cudaStream_t *)axionField->Streams())[0]);
			updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c0, delta2, lambda, nQcd, uLx, uLz, uV,  ext, precision,
				  ((cudaStream_t *)axionField->Streams())[1]);
		}
	#else
		LogError ("Error: gpu support not built");
		exit(1);
	#endif
	}

	// Generic saxion lowmem propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::lowGpu	(const double dz) {
	#ifdef	USE_GPU
		const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
		const uint ext = V + S;
		double *z = axionField->zV();
		double lambda = LL;

		#pragma unroll
		for (int s = 0; s<nStages; s++) {

			if (lType != LAMBDA_FIXED)
				lambda = LL/((*z)*(*z));

			const double c0 = c[s], d0 = d[s];

			updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c0, delta2, lambda, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
			axionField->exchangeGhosts(FIELD_M);
			updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c0, delta2, lambda, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
			updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c0, delta2, lambda, nQcd, uLx, uLz, uV,  ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
			cudaStreamSynchronize(((cudaStream_t *)axionField->Streams())[0]);
			cudaStreamSynchronize(((cudaStream_t *)axionField->Streams())[1]);
			updateMGpu(axionField->mGpu(), axionField->vGpu(), dz, d0, Lx, uS, ext, precision, ((cudaStream_t *)axionField->Streams())[2]);

			*z += dz*d0;

			cudaStreamSynchronize(((cudaStream_t *)axionField->Streams())[2]);
		}

		if (lastStage) {
			const double c0 = c[nStages];

			if (lType != LAMBDA_FIXED)
				lambda = LL/((*z)*(*z));

			updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c0, delta2, lambda, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[2]);
			axionField->exchangeGhosts(FIELD_M);
			updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c0, delta2, lambda, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
			updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c0, delta2, lambda, nQcd, uLx, uLz, uV,  ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
			cudaDeviceSynchronize();
		}
	#else
		LogError ("Error: gpu support not built");
		exit(1);
	#endif
	}

	/*		CPU PROPAGATORS			*/

	// Generic axion propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::tRunCpu	(const double dz) {
		const double ood2 = 1./delta2;
		double *z = axionField->zV();

		#pragma unroll
		for (int s = 0; s<nStages; s+=2) {
			const double	c1 = c[s], c2 = c[s+1], d1 = d[s], d2 = d[s+1];

			axionField->sendGhosts(FIELD_M, COMM_SDRV);
			propThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, c1, d1, ood2, nQcd, Lx, 2*S, V, precision);
			axionField->sendGhosts(FIELD_M, COMM_WAIT);
			propThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, c1, d1, ood2, nQcd, Lx, S, 2*S, precision);
			propThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, c1, d1, ood2, nQcd, Lx, V, V+S, precision);
			*z += dz*d1;

			axionField->sendGhosts(FIELD_M2, COMM_SDRV);
			propThetaKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, c2, d2, ood2, nQcd, Lx, 2*S, V, precision);
			axionField->sendGhosts(FIELD_M2, COMM_WAIT);
			propThetaKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, c2, d2, ood2, nQcd, Lx, S, 2*S, precision);
			propThetaKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, c2, d2, ood2, nQcd, Lx, V, V+S, precision);
			*z += dz*d2;
		}

		if (lastStage) {
			LogMsg (VERB_HIGH, "Warning: axion propagator not optimized yet for odd propagators, performance might be reduced");

			const double	c0 = c[nStages];

			axionField->sendGhosts(FIELD_M, COMM_SDRV);
			propThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, c0, 0., ood2, nQcd, Lx, 2*S, V, precision);
			axionField->sendGhosts(FIELD_M, COMM_WAIT);
			propThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, c0, 0., ood2, nQcd, Lx, S, 2*S, precision);
			propThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, c0, 0., ood2, nQcd, Lx, V, V+S, precision);
		}
	}

	// Generic axion spectral propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::tSpecCpu	(const double dz) {

		double *z = axionField->zV();
		auto	Lz = axionField->Depth();
		auto	Lx = axionField->Length();

		auto	Sf = Lz*Lx;
		auto	dataLine = axionField->DataSize()*Lx;

		char	*mO = static_cast<char *>(axionField->mCpu())  + axionField->Surf()*axionField->DataSize();
		char	*mF = static_cast<char *>(axionField->m2Cpu()) + axionField->Surf()*axionField->DataSize();

		const double fMom = -(4.*M_PI*M_PI)/(sizeL*sizeL*((double) axionField->Size()));

		if	(axionField->Folded())
		{
			Folder	munge(axionField);
			munge(UNFOLD_ALL);
		}

		#pragma unroll
		for (int s = 0; s<nStages; s++) {
			const double	c0 = c[s], d0 = d[s];

			#pragma omp parallel for schedule(static)
			for (int sl=0; sl<Sf; sl++) {
				auto	oOff = sl*axionField->DataSize()*Lx;
				auto	fOff = sl*axionField->DataSize()*(Lx+2);

				memcpy (mF+fOff, mO+oOff, dataLine);
			}

			applyLaplacian(axionField);
			sPropThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, c0, d0, nQcd, fMom, Lx, S, V+S, precision);
			*z += dz*d[s];
		}

		if (lastStage) {
			LogMsg (VERB_HIGH, "Warning: spectral propagator not optimized yet for odd propagators, performance might be reduced");

			const double	c0 = c[nStages];

			#pragma omp parallel for schedule(static)
			for (int sl=0; sl<Sf; sl++) {
				auto	oOff = sl*axionField->DataSize()*Lx;
				auto	fOff = sl*axionField->DataSize()*(Lx+2);
				memcpy (mF+fOff, mO+oOff, dataLine);
			}

			applyLaplacian(axionField);
			sPropThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, c0, 0.0, nQcd, fMom, Lx, S, V+S, precision);
		}
	}
	// Generic saxion propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::sRunCpu	(const double dz) {
		const double ood2 = 1./delta2;
		double *z = axionField->zV();
		double lambda = LL;

		#pragma unroll
		for (int s = 0; s<nStages; s+=2) {

			axionField->sendGhosts(FIELD_M, COMM_SDRV);

			if (lType != LAMBDA_FIXED)
				lambda = LL/((*z)*(*z));

			const double	c1 = c[s], c2 = c[s+1], d1 = d[s], d2 = d[s+1];

			propagateKernelXeon<VQcd>(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, c1, d1, ood2, lambda, nQcd, Lx, 2*S, V, precision);
			axionField->sendGhosts(FIELD_M, COMM_WAIT);
			propagateKernelXeon<VQcd>(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, c1, d1, ood2, lambda, nQcd, Lx, S, 2*S, precision);
			propagateKernelXeon<VQcd>(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, c1, d1, ood2, lambda, nQcd, Lx, V, V+S, precision);
			*z += dz*d1;

			axionField->sendGhosts(FIELD_M2, COMM_SDRV);

			if (lType != LAMBDA_FIXED)
				lambda = LL/((*z)*(*z));

			propagateKernelXeon<VQcd>(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, c2, d2, ood2, lambda, nQcd, Lx, 2*S, V, precision);
			axionField->sendGhosts(FIELD_M2, COMM_WAIT);
			propagateKernelXeon<VQcd>(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, c2, d2, ood2, lambda, nQcd, Lx, S, 2*S, precision);
			propagateKernelXeon<VQcd>(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, c2, d2, ood2, lambda, nQcd, Lx, V, V+S, precision);
			*z += dz*d2;
		}

		if (lastStage) {
			axionField->sendGhosts(FIELD_M, COMM_SDRV);

			if (lType != LAMBDA_FIXED)
				lambda = LL/((*z)*(*z));

			const double	c0 = c[nStages];

			updateVXeon<VQcd>(axionField->mCpu(), axionField->vCpu(), z, dz, c0, ood2, lambda, nQcd, Lx, 2*S, V, S, precision);
			axionField->sendGhosts(FIELD_M, COMM_WAIT);
			updateVXeon<VQcd>(axionField->mCpu(), axionField->vCpu(), z, dz, c0, ood2, lambda, nQcd, Lx, S, 2*S, S, precision);
			updateVXeon<VQcd>(axionField->mCpu(), axionField->vCpu(), z, dz, c0, ood2, lambda, nQcd, Lx, V, V+S, S, precision);
		}
	}

	// Generic saxion lowmem propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::lowCpu	(const double dz) {
		const double ood2 = 1./delta2;
		double *z = axionField->zV();
		double lambda = LL;

		#pragma unroll
		for (int s = 0; s<nStages; s++) {
			axionField->sendGhosts(FIELD_M, COMM_SDRV);

			if (lType != LAMBDA_FIXED)
				lambda = LL/((*z)*(*z));

			const double c0 = c[s], d0 = d[s];

			updateVXeon<VQcd>(axionField->mCpu(), axionField->vCpu(), z, dz, c0, ood2, lambda, nQcd, Lx, 2*S, V, S, precision);
			axionField->sendGhosts(FIELD_M, COMM_WAIT);
			updateVXeon<VQcd>(axionField->mCpu(), axionField->vCpu(), z, dz, c0, ood2, lambda, nQcd, Lx, S, 2*S, S, precision);
			updateVXeon<VQcd>(axionField->mCpu(), axionField->vCpu(), z, dz, c0, ood2, lambda, nQcd, Lx, V, V+S, S, precision);
			updateMXeon(axionField->mCpu(), axionField->vCpu(), dz, d0, S, V + S, S, precision);
			*z += dz*d0;
		}

		if (lastStage) {
			axionField->sendGhosts(FIELD_M, COMM_SDRV);

			if (lType != LAMBDA_FIXED)
				lambda = LL/((*z)*(*z));

			const double c0 = c[nStages];

			updateVXeon<VQcd>(axionField->mCpu(), axionField->vCpu(), z, dz, c0, ood2, lambda, nQcd, Lx, 2*S, V, S, precision);
			axionField->sendGhosts(FIELD_M, COMM_WAIT);
			updateVXeon<VQcd>(axionField->mCpu(), axionField->vCpu(), z, dz, c0, ood2, lambda, nQcd, Lx, S, 2*S, S, precision);
			updateVXeon<VQcd>(axionField->mCpu(), axionField->vCpu(), z, dz, c0, ood2, lambda, nQcd, Lx, V, V+S, S, precision);
		}
	}

	// Generic saxion spectral propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::sSpecCpu	(const double dz) {

		double *z = axionField->zV();
		double lambda = LL;

		const double fMom = -(4.*M_PI*M_PI)/(sizeL*sizeL*((double) axionField->Size()));

		#pragma unroll
		for (int s = 0; s<nStages; s++) {
			const double	c0 = c[s], d0 = d[s];

			applyLaplacian(axionField);

			if (lType != LAMBDA_FIXED)
				lambda = LL/((*z)*(*z));

			sPropKernelXeon<VQcd>(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, c0, d0, lambda, nQcd, fMom, Lx, S, V+S, precision);
			*z += dz*d0;
		}

		if (lastStage) {
			LogMsg (VERB_HIGH, "Warning: spectral propagator not optimized yet for odd propagators, performance might be reduced");

			const double	c0 = c[nStages];

			applyLaplacian(axionField);

			if (lType != LAMBDA_FIXED)
				lambda = LL/((*z)*(*z));

			sPropKernelXeon<VQcd>(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, c0, 0.0, lambda, nQcd, fMom, Lx, S, V+S, precision);
		}
	}

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	double	PropClass<nStages, lastStage, VQcd>::cFlops	(const bool spec) {
		if (!spec) {
			switch (axionField->Field()) {

				case FIELD_SAXION:
					switch (VQcd) {
						case VQCD_1:
							return	(1e-9 * ((double) axionField->Size()) * (42. * ((double) nStages) + (lastStage ? 38. : 0.)));
							break;

						case VQCD_2:
							return	(1e-9 * ((double) axionField->Size()) * (45. * ((double) nStages) + (lastStage ? 41. : 0.)));
							break;
					}
					break;

				case FIELD_AXION:
				case FIELD_AXION_MOD:	// Seguro??
					return	(1e-9 * ((double) axionField->Size()) * (23. * ((double) nStages) + (lastStage ? 15. : 0.)));
					break;

				case FIELD_WKB:
					return	0.;
					break;
			}
		} else {
			switch (axionField->Field()) {

				case FIELD_SAXION:
					switch (VQcd) {
						case VQCD_1:
							return	(1e-9 * ((double) axionField->Size()) * ((26. + 1.) * ((double) nStages) + (lastStage ? 22. + 1. : 0.) 
								+ 5.*1.44695*log(((double) axionField->Size()))));
							break;

						case VQCD_2:
							return	(1e-9 * ((double) axionField->Size()) * ((29. + 1.) * ((double) nStages) + (lastStage ? 25. + 1. : 0.)
								+ 5.*1.44695*log(((double) axionField->Size()))));
							break;
					}
					break;

				case FIELD_AXION:
				case FIELD_AXION_MOD:	// Seguro??
					return	(1e-9 * ((double) axionField->Size()) * (21. * ((double) nStages) + (lastStage ? 13. : 0.)
						+ 2.5*1.44695*log(((double) axionField->Size()))));
					break;

				case FIELD_WKB:
					return	0.;
					break;
			}
		}
	}

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	double	PropClass<nStages, lastStage, VQcd>::cBytes	(const bool spec) {
		if (!spec) {
			return	(1e-9 * ((double) (axionField->Size()*axionField->DataSize())) * (   10.    * ((double) nStages) + (lastStage ? 9. : 0.)));
		} else {
			return	(1e-9 * ((double) (axionField->Size()*axionField->DataSize())) * ((6. + 4.) * ((double) nStages) + (lastStage ? 6. + 3. : 0.) + 2.));
		}
	}
#endif
