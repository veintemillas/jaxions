#ifndef	_PROPCLASS_
	#define	_PROPCLASS_

	#include <cmath>
	#include <string>
	#include <cstring>
	#include <functional>
	#include "propagator/propBase.h"
	#include "scalar/scalarField.h"
	#include "scalar/fourier.h"
	#include "utils/utils.h"


	#include "propagator/propXeon.h"
	#include "propagator/propThetaXeon.h"
	#include "propagator/laplacian.h"
	#include "propagator/sPropXeon.h"
	#include "propagator/fsPropXeon.h"
	#include "propagator/sPropThetaXeon.h"
	#include "propagator/propNaxXeon.h"
	#include "propagator/propPaxXeon.h"

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

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	class	PropClass : public PropBase
	{
		protected:

		Scalar		*axion;
		const size_t	Lx, Lz, V, S;
		const double	ood2;
		double		&lambda;
		FieldPrecision	precision;
		double		&gamma;
		LambdaType	lType;

		double	c[nStages + (lastStage == true ? 1 : 0)];
		double	d[nStages];

		public:

		inline	 PropClass(Scalar *field, const PropcType spec);
		inline	~PropClass() override {};

		inline void	setCoeff(const double * __restrict__ nC, const double * __restrict__ nD) {
			for(int i=0; i<nStages; i++) { c[i] = nC[i]; d[i] = nD[i]; } if (lastStage) { c[nStages] = nC[nStages]; }
		}


		inline void	sRunCpu	(const double)	override;	// Saxion propagator
		inline void	sRunGpu	(const double)	override;
		inline void	sSpecCpu(const double)	override;	// Saxion spectral propagator
		inline void	sFpecCpu(const double)	override;	// Saxion spectral propagator


		inline void	tRunCpu	(const double)	override;	// Axion propagator
		inline void	tRunGpu	(const double)	override;
		inline void	tSpecCpu(const double)	override;	// Axion spectral propagator
		// inline void	tfsSpecCpu(const double)	override;	// Axion spectral propagator FIX IT!


		inline void	nRunCpu	(const double)	override;			// Naxion propagator

		inline void	pRunCpu	(const double)	override;			// Paxion propagator

		inline void	lowCpu	(const double)	override;	// Lowmem only available for saxion non-spectral
		inline void	lowGpu	(const double)	override;



		inline double	cFlops	(const PropcType)	override;
		inline double	cBytes	(const PropcType)	override;
	};

	template<const int nStages, const bool lastStage, VqcdType VQcd>
		PropClass<nStages, lastStage, VQcd>::PropClass(Scalar *field, PropcType spec) : axion(field), Lx(field->Length()), Lz(field->eDepth()), V(field->Size()), S(field->Surf()),
		ood2(1./(field->Delta()*field->Delta())), lambda(field->BckGnd()->Lambda()), precision(field->Precision()), gamma(field->BckGnd()->Gamma()), lType(field->LambdaT()) {

		/*	Default block size gives just one block	*/
		int tmp   = field->DataAlign()/field->DataSize();
		int shift = 0;

		while (tmp != 1) {
			shift++;
			tmp >>= 1;
		}

		xBest = xBlock = Lx << shift;
		yBest = yBlock = Lx >> shift;
		zBest = zBlock = Lz;

		switch (spec){
			case PROPC_SPEC:
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
					LogMsg	(VERB_HIGH, "Warning: spectral propagators not supported in gpus, Exit");
					exit(1);
					break;

					default:
						LogError ("Error: not a valid device");
						return;
					}
					break;

			case PROPC_FSPEC:
				switch (field->Device()) {
					case	DEV_CPU:
						if (field->LowMem()) {
							LogError ("Error: Lowmem not supported with fpectral propagators");
							exit(1);
						}

						propSaxion = [this](const double dz) { this->sFpecCpu(dz); };
						propAxion  = [this](const double dz) { this->tSpecCpu(dz); }; // include new full spectral propagator!
						break;

						case	DEV_GPU:
						LogMsg	(VERB_HIGH, "Warning: spectral propagators not supported in gpus, Exit");
						exit(1);
						break;

					default:
						LogError ("Error: not a valid device");
						return;
					}
					break;

			default:
			case PROPC_BASE:
				switch (field->Device()) {
					case	DEV_CPU:
						if (field->LowMem()) {
							propSaxion = [this](const double dz) { this->lowCpu(dz); };
							propAxion  = [this](const double dz) { this->tRunCpu(dz); };
						} else {
							propSaxion = [this](const double dz) { this->sRunCpu(dz); };
							propAxion  = [this](const double dz) { this->tRunCpu(dz); };
							propNaxion = [this](const double dz) { this->nRunCpu(dz); };
							propPaxion = [this](const double dz) { this->pRunCpu(dz); };
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
				break;
		}

	}

	/*		GPU PROPAGATORS			*/

	// Generic axion propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::tRunGpu	(const double dz) {
	#ifdef  USE_GPU
		const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
		const uint ext = uV + uS;
		// eom only depend on R
		double *z = axion->zV();
		double *R = axion->RV();

		const bool wMod = (axion->Field() == FIELD_AXION_MOD) ? true : false;

		#pragma unroll
		for (int s = 0; s<nStages; s+=2) {
			const double	c1 = c[s], c2 = c[s+1], d1 = d[s], d2 = d[s+1];

			auto maa = axion->AxionMassSq();

			propThetaGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), R, dz, c1, d1, ood2, maa, uLx, uLz, 2*uS, uV, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[2], wMod);
			axion->exchangeGhosts(FIELD_M);
			propThetaGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), R, dz, c1, d1, ood2, maa, uLx, uLz, uS, 2*uS, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[0], wMod);
			propThetaGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), R, dz, c1, d1, ood2, maa, uLx, uLz, uV,  ext, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[1], wMod);
			cudaDeviceSynchronize();        // This is not strictly necessary, but simplifies things a lot

			*z += dz*d1;
			axion->updateR();
			maa = axion->AxionMassSq();

			propThetaGpu(axion->m2Gpu(), axion->vGpu(), axion->mGpu(), R, dz, c2, d2, ood2, maa, uLx, uLz, 2*uS, uV, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[2], wMod);
			axion->exchangeGhosts(FIELD_M2);
			propThetaGpu(axion->m2Gpu(), axion->vGpu(), axion->mGpu(), R, dz, c2, d2, ood2, maa, uLx, uLz, uS, 2*uS, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[0], wMod);
			propThetaGpu(axion->m2Gpu(), axion->vGpu(), axion->mGpu(), R, dz, c2, d2, ood2, maa, uLx, uLz, uV,  ext, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[1], wMod);
			cudaDeviceSynchronize();        // This is not strictly necessary, but simplifies things a lot
			*z += dz*d2;
			axion->updateR();
		}

		if (lastStage) {
			LogMsg (VERB_HIGH, "Warning: axion propagator not optimized yet for odd propagators, performance might be reduced");

			const double	c0 = c[nStages], maa = axion->AxionMassSq();

			propThetaGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), R, dz, c0, 0., ood2, maa, uLx, uLz, 2*uS, uV, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[2], wMod);
			axion->exchangeGhosts(FIELD_M);
			propThetaGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), R, dz, c0, 0., ood2, maa, uLx, uLz, uS, 2*uS, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[0], wMod);
			propThetaGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), R, dz, c0, 0., ood2, maa, uLx, uLz, uV,  ext, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[1], wMod);
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

		// eom only depend on R
		double *z = axion->zV();
		double *R = axion->RV();
		double cLmbda = axion->LambdaP();

		#pragma unroll
		for (int s = 0; s<nStages; s+=2) {

			cLmbda = axion->LambdaP();

			const double	c1 = c[s], c2 = c[s+1], d1 = d[s], d2 = d[s+1];

			auto maa = axion->AxionMassSq();

			propagateGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), R, dz, c1, d1, ood2, cLmbda, maa, gamma, uLx, uLz, 2*uS, uV, VQcd, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[2]);
			axion->exchangeGhosts(FIELD_M);
			propagateGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), R, dz, c1, d1, ood2, cLmbda, maa, gamma, uLx, uLz, uS, 2*uS, VQcd, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[0]);
			propagateGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), R, dz, c1, d1, ood2, cLmbda, maa, gamma, uLx, uLz, uV,  ext, VQcd, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[1]);

			cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot

			*z += dz*d1;
			axion->updateR();
			cLmbda = axion->LambdaP();

			maa = axion->AxionMassSq();

			propagateGpu(axion->m2Gpu(), axion->vGpu(), axion->mGpu(), R, dz, c2, d2, ood2, cLmbda, maa, gamma, uLx, uLz, 2*uS, uV, VQcd, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[2]);
			axion->exchangeGhosts(FIELD_M2);
			propagateGpu(axion->m2Gpu(), axion->vGpu(), axion->mGpu(), R, dz, c2, d2, ood2, cLmbda, maa, gamma, uLx, uLz, uS, 2*uS, VQcd, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[0]);
			propagateGpu(axion->m2Gpu(), axion->vGpu(), axion->mGpu(), R, dz, c2, d2, ood2, cLmbda, maa, gamma, uLx, uLz, uV,  ext, VQcd, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[1]);

			cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
			*z += dz*d2;
			axion->updateR();
		}

		if (lastStage) {
		cLmbda = axion->LambdaP();

			const double	c0 = c[nStages], maa = axion->AxionMassSq();

			updateVGpu(axion->mGpu(), axion->vGpu(), R, dz, c0, ood2, cLmbda, maa, gamma, uLx, uLz, uS*2, uV, VQcd, precision, xBlock, yBlock, zBlock,
				  ((cudaStream_t *)axion->Streams())[2]);
			axion->exchangeGhosts(FIELD_M);
			updateVGpu(axion->mGpu(), axion->vGpu(), R, dz, c0, ood2, cLmbda, maa, gamma, uLx, uLz, uS, uS*2, VQcd, precision, xBlock, yBlock, zBlock,
				  ((cudaStream_t *)axion->Streams())[0]);
			updateVGpu(axion->mGpu(), axion->vGpu(), R, dz, c0, ood2, cLmbda, maa, gamma, uLx, uLz, uV,  ext, VQcd, precision, xBlock, yBlock, zBlock,
				  ((cudaStream_t *)axion->Streams())[1]);
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
		double *z = axion->zV();
		double *R = axion->RV();
		double cLmbda ;

		#pragma unroll
		for (int s = 0; s<nStages; s++) {

			axion->updateR();
			cLmbda = axion->LambdaP();

			const double c0 = c[s], d0 = d[s], maa = axion->AxionMassSq();

			updateVGpu(axion->mGpu(), axion->vGpu(), R, dz, c0, ood2, cLmbda, maa, gamma, uLx, uLz, 2*uS, uV, VQcd, precision, xBlock, yBlock, zBlock,
				  ((cudaStream_t *)axion->Streams())[2]);
			axion->exchangeGhosts(FIELD_M);
			updateVGpu(axion->mGpu(), axion->vGpu(), R, dz, c0, ood2, cLmbda, maa, gamma, uLx, uLz, uS, 2*uS, VQcd, precision, xBlock, yBlock, zBlock,
				  ((cudaStream_t *)axion->Streams())[0]);
			updateVGpu(axion->mGpu(), axion->vGpu(), R, dz, c0, ood2, cLmbda, maa, gamma, uLx, uLz, uV,  ext, VQcd, precision, xBlock, yBlock, zBlock,
				  ((cudaStream_t *)axion->Streams())[1]);
			cudaStreamSynchronize(((cudaStream_t *)axion->Streams())[0]);
			cudaStreamSynchronize(((cudaStream_t *)axion->Streams())[1]);
			updateMGpu(axion->mGpu(), axion->vGpu(), dz, d0, Lx, uS, ext, precision, xBlock, yBlock, zBlock, ((cudaStream_t *)axion->Streams())[2]);

			*z += dz*d0;
			axion->updateR();
			cudaStreamSynchronize(((cudaStream_t *)axion->Streams())[2]);
		}

		if (lastStage) {
			const double c0 = c[nStages], maa = axion->AxionMassSq();

			cLmbda = axion->LambdaP();

			updateVGpu(axion->mGpu(), axion->vGpu(), R, dz, c0, ood2, cLmbda, maa, gamma, uLx, uLz, 2*uS, uV, VQcd, precision, xBlock, yBlock, zBlock,
				  ((cudaStream_t *)axion->Streams())[2]);
			axion->exchangeGhosts(FIELD_M);
			updateVGpu(axion->mGpu(), axion->vGpu(), R, dz, c0, ood2, cLmbda, maa, gamma, uLx, uLz, uS, 2*uS, VQcd, precision, xBlock, yBlock, zBlock,
				  ((cudaStream_t *)axion->Streams())[0]);
			updateVGpu(axion->mGpu(), axion->vGpu(), R, dz, c0, ood2, cLmbda, maa, gamma, uLx, uLz, uV,  ext, VQcd, precision, xBlock, yBlock, zBlock,
				  ((cudaStream_t *)axion->Streams())[1]);
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
		double *z = axion->zV();
		// double *R = axion->RV();

		/* Returns ghost size region in slices */
		// size_t NG   = axion->getNg();
		// LogMsg(VERB_DEBUG,"[propAx] Ng %d",NG);
		// /* Size of Boundary */
		// size_t BO = NG*S;
		// /* Size of Core  */
		// // size_t CO = V-2*NG*S;
		// double *PC = axion->getCO();
		const bool wMod = (axion->Field() == FIELD_AXION_MOD) ? true : false;

		PropParms ppar;
		ppar.Ng     = axion->getNg();
		ppar.ood2a  = ood2;
		ppar.PC     = axion->getCO();
		ppar.Lx     = Lx;
		ppar.frw    = axion->BckGnd()->Frw();

		ppar.massA2 = axion->AxionMassSq();
		ppar.R      = *axion->RV();
		ppar.Rpp    = axion->Rpp();
		/* Returns ghost size region in slices */
		size_t BO = ppar.Ng*S;


		#pragma unroll
		for (int s = 0; s<nStages; s+=2) {
			axion->sendGhosts(FIELD_M, COMM_SDRV);

			const double c1 = c[s], c2 = c[s+1], d1 = d[s], d2 = d[s+1];

			propThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, 2*BO, V , precision, xBlock, yBlock, zBlock, wMod);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, BO, 2*BO, precision, xBlock, yBlock, zBlock, wMod);
			propThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, V , V+BO, precision, xBlock, yBlock, zBlock, wMod);
			*z += dz*d1;
			axion->updateR();
			axion->sendGhosts(FIELD_M2, COMM_SDRV);

			ppar.massA2 = axion->AxionMassSq();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();

			propThetaKernelXeon(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, 2*BO, V   , precision, xBlock, yBlock, zBlock, wMod);
			axion->sendGhosts(FIELD_M2, COMM_WAIT);
			propThetaKernelXeon(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, BO  , 2*BO, precision, xBlock, yBlock, zBlock, wMod);
			propThetaKernelXeon(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, V   , V+BO, precision, xBlock, yBlock, zBlock, wMod);
			*z += dz*d2;
			axion->updateR();
		}

		if (lastStage) {
			axion->sendGhosts(FIELD_M, COMM_SDRV);
			/* Last kick */
			LogMsg (VERB_HIGH, "Warning: axion propagator not optimized yet for odd propagators, performance might be reduced");

			const double	c0 = c[nStages], maa = axion->AxionMassSq();

			ppar.massA  = axion->AxionMassSq();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();

			propThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0., 2*BO, V   , precision, xBlock, yBlock, zBlock, wMod);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0., BO  , 2*BO, precision, xBlock, yBlock, zBlock, wMod);
			propThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0., V   , V+BO, precision, xBlock, yBlock, zBlock, wMod);
		}

	}

	// Generic axion spectral propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::tSpecCpu	(const double dz) {

		double *z     = axion->zV();
		double *R     = axion->RV();
		auto	Lz    = axion->Depth();
		auto	Lx    = axion->Length();
		auto	lSize = axion->BckGnd()->PhysSize();

		auto	Sf = Lz*Lx;
		auto	dataLine = axion->DataSize()*Lx;

		char	*mO = static_cast<char *>(axion->mCpu())  + axion->Surf()*axion->DataSize();
		char	*mF = static_cast<char *>(axion->m2Cpu());

		const double fMom = -(4.*M_PI*M_PI)/(lSize*lSize*((double) axion->TotalSize()));

		if	(axion->Folded())
		{
			Folder	munge(axion);
			munge(UNFOLD_ALL);
		}

		#pragma unroll
		for (int s = 0; s<nStages; s++) {
			const double	c0 = c[s], d0 = d[s], maa = axion->AxionMassSq();

			#pragma omp parallel for schedule(static)
			for (size_t sl=0; sl<Sf; sl++) {
				auto	oOff = sl*axion->DataSize()*Lx;
				auto	fOff = sl*axion->DataSize()*(Lx+2);

				memcpy (mF+fOff, mO+oOff, dataLine);
			}

			applyLaplacian(axion);
			sPropThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), R, dz, c0, d0, maa, fMom, Lx, S, V+S, precision);
			*z += dz*d[s];
			axion->updateR();
		}

		if (lastStage) {
			LogMsg (VERB_HIGH, "Warning: spectral propagator not optimized yet for odd propagators, performance might be reduced");

			const double	c0 = c[nStages], maa = axion->AxionMassSq();

			#pragma omp parallel for schedule(static)
			for (size_t sl=0; sl<Sf; sl++) {
				auto	oOff = sl*axion->DataSize()*Lx;
				auto	fOff = sl*axion->DataSize()*(Lx+2);
				memcpy (mF+fOff, mO+oOff, dataLine);
			}

			applyLaplacian(axion);
			sPropThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), R, dz, c0, 0.0, maa, fMom, Lx, S, V+S, precision);
		}
		axion->setM2     (M2_DIRTY);
	}




	// Generic saxion propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::sRunCpu	(const double dz) {
		double *z = axion->zV();
		// double *R = axion->RV();
		// double cLmbda ;

		/* Returns ghost size region in slices */
		// size_t NG   = axion->getNg();
		// LogMsg(VERB_DEBUG,"[propSax] Ng %d",NG);
		// /* Size of Boundary */
		// size_t BO = NG*S;
		// /* Size of Core  */
		// size_t CO = V-2*NG*S;

		PropParms ppar;
		ppar.Ng     = axion->getNg();
		ppar.Lx     = Lx;
		ppar.PC     = axion->getCO();
		ppar.ood2a  = ood2;
		ppar.gamma  = axion->BckGnd()->Gamma();
		ppar.frw    = axion->BckGnd()->Frw();

		ppar.lambda = axion->LambdaP();
		ppar.massA2 = axion->AxionMassSq();
		ppar.R      = *axion->RV();
		ppar.Rpp    = axion->Rpp();
		/* Returns ghost size region in slices */
		size_t BO = ppar.Ng*S;


		#pragma unroll
		for (int s = 0; s<nStages; s+=2) {

			axion->sendGhosts(FIELD_M, COMM_SDRV);

			const double	c1 = c[s], c2 = c[s+1], d1 = d[s], d2 = d[s+1];

			// cLmbda = axion->LambdaP();
			// auto maa = axion->AxionMassSq();

			// propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), NG, R, dz, c1, d1, ood2, cLmbda, maa, gamma, Lx, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			// axion->sendGhosts(FIELD_M, COMM_WAIT);
			// propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), NG, R, dz, c1, d1, ood2, cLmbda, maa, gamma, Lx, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			// propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), NG, R, dz, c1, d1, ood2, cLmbda, maa, gamma, Lx, V   , V+BO, precision, xBlock, yBlock, zBlock);
			propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, V   , V+BO, precision, xBlock, yBlock, zBlock);
			*z += dz*d1;
			axion->updateR();

			axion->sendGhosts(FIELD_M2, COMM_SDRV);

			// cLmbda = axion->LambdaP();
			// maa = axion->AxionMassSq();
			ppar.lambda = axion->LambdaP();
			ppar.massA2 = axion->AxionMassSq();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();

			propagateKernelXeon<VQcd>(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M2, COMM_WAIT);
			propagateKernelXeon<VQcd>(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			propagateKernelXeon<VQcd>(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, V   , V+BO, precision, xBlock, yBlock, zBlock);
			*z += dz*d2;
			axion->updateR();
		}

		if (lastStage) {
			axion->sendGhosts(FIELD_M, COMM_SDRV);

			// cLmbda = axion->LambdaP();
			const double	c0 = c[nStages], maa = axion->AxionMassSq();
			/* Last kick but not drift d = 0 */

			ppar.lambda = axion->LambdaP();
			ppar.massA2 = axion->AxionMassSq();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();

			propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0.0, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0.0, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0.0, V   , V+BO, precision, xBlock, yBlock, zBlock);

		}
		axion->setM2     (M2_DIRTY);
	}




	// Generic saxion lowmem propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::lowCpu	(const double dz) {
		double *z = axion->zV();
		double *R = axion->RV();
		double cLmbda ;

		size_t NG = axion->getNg();

		#pragma unroll
		for (int s = 0; s<nStages; s++) {
			axion->sendGhosts(FIELD_M, COMM_SDRV);

			cLmbda = axion->LambdaP();

			const double c0 = c[s], d0 = d[s], maa = axion->AxionMassSq();

			updateVXeon<VQcd>(axion->mCpu(), axion->vCpu(), NG, R, dz, c0, ood2, cLmbda, maa, gamma, Lx, 2*S, V, S, precision);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			updateVXeon<VQcd>(axion->mCpu(), axion->vCpu(), NG, R, dz, c0, ood2, cLmbda, maa, gamma, Lx, S, 2*S, S, precision);
			updateVXeon<VQcd>(axion->mCpu(), axion->vCpu(), NG, R, dz, c0, ood2, cLmbda, maa, gamma, Lx, V, V+S, S, precision);
			updateMXeon(axion->mCpu(), axion->vCpu(), dz, d0, S, V + S, precision);
			*z += dz*d0;
			axion->updateR();
		}

		if (lastStage) {
			axion->sendGhosts(FIELD_M, COMM_SDRV);

			cLmbda = axion->LambdaP();


			const double c0 = c[nStages], maa = axion->AxionMassSq();

			updateVXeon<VQcd>(axion->mCpu(), axion->vCpu(), NG, R, dz, c0, ood2, cLmbda, maa, gamma, Lx, 2*S, V, S, precision);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			updateVXeon<VQcd>(axion->mCpu(), axion->vCpu(), NG, R, dz, c0, ood2, cLmbda, maa, gamma, Lx, S, 2*S, S, precision);
			updateVXeon<VQcd>(axion->mCpu(), axion->vCpu(), NG, R, dz, c0, ood2, cLmbda, maa, gamma, Lx, V, V+S, S, precision);
		}
	}

	// Generic saxion spectral propagator
	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::sSpecCpu	(const double dz) {

		double *z = axion->zV();
		double *R = axion->RV();
		double cLmbda;
		auto   lSize  = axion->BckGnd()->PhysSize();

		const double fMom = -(4.*M_PI*M_PI)/(lSize*lSize*((double) axion->TotalSize()));

		//debug
		if	(axion->Folded())
		{
			Folder	munge(axion);
			munge(UNFOLD_ALL);
		}

		#pragma unroll
		for (int s = 0; s<nStages; s++) {
			const double	c0 = c[s], d0 = d[s], maa = axion->AxionMassSq();

			applyLaplacian(axion);

			cLmbda = axion->LambdaP();

			sPropKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), R, dz, c0, d0, ood2, cLmbda, maa, gamma, fMom, Lx, S, V+S, precision);
			*z += dz*d0;
			axion->updateR();
		}

		if (lastStage) {
			LogMsg (VERB_HIGH, "Warning: spectral propagator not optimized yet for odd propagators, performance might be reduced");

			const double	c0 = c[nStages], maa = axion->AxionMassSq();

			applyLaplacian(axion);

			cLmbda = axion->LambdaP();

			sPropKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), R, dz, c0, 0.0, ood2, cLmbda, maa, gamma, fMom, Lx, S, V+S, precision);
		}
		axion->setM2     (M2_DIRTY);
	}

	// Generic saxion full spectral propagator (in Fourier space)
	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::sFpecCpu	(const double dz) {

		double *z = axion->zV();
		double *R = axion->RV();
		double cLmbda ;
		auto   lSize  = axion->BckGnd()->PhysSize();
		size_t Tz = axion->TotalDepth();

		const double fMom1 = (2.*M_PI)/(lSize);

		if (axion->Folded())
		{
			Folder munge(axion);
			munge(UNFOLD_ALL);
		}

		// If field is in configuration space transform to momentum space
		if	( !axion->MMomSpace() || !axion->VMomSpace() )
		{
			if (debug) LogOut("[fs] FT!\n");
			FTfield pelotas(axion);
			pelotas(FIELD_MV, FFT_FWD); // FWD is to send to momentum space transposed out
		}

		float *mmm = static_cast<float *>(axion->mStart());
		float *vvv = static_cast<float *>(axion->vCpu());
		float *mm2 = static_cast<float *>(axion->m2Cpu());

		#pragma unroll
		for (int s = 0; s<nStages; s++) {
			const double	c0 = c[s], d0 = d[s], maa = axion->AxionMassSq();

			// computes m into m2 in configuration space
			FTfield pelota(axion);
			pelota(FIELD_MTOM2, FFT_BCK); // BCK is to send to conf space

			// computes acceleration
			cLmbda = axion->LambdaP();

			/* computes the acceleration in configuration space */

			// LogOut("[fs] m  values %f %f %f %f \n",mmm[0],mmm[1],mmm[2],mmm[3]);
			// LogOut("[fs] v  values %f %f %f %f \n",vvv[0],vvv[1],vvv[2],vvv[3]);
			// LogOut("[fs] m2 values %f %f %f %f \n",mm2[0],mm2[1],mm2[2],mm2[3]);
			// LogFlush();
			fsAccKernelXeon<VQcd>(axion->vCpu(), axion->m2Cpu(), R, dz, c0, d0, ood2, cLmbda, maa, gamma, fMom1, Lx, S, V+S, precision);
			// LogOut("[fs] ac values %f %f %f %f \n",mm2[0],mm2[1],mm2[2],mm2[3]);

			if (debug) LogOut("[fs] accelerated \n");
			pelota(FIELD_M2TOM2, FFT_FWD); // FWD sends M2 to mom space
			if (debug) LogOut("[fs] fff \n");
			/* kicks in momentum space */
			// LogOut("[fs] ac momspace %f %f %f %f \n",mm2[0],mm2[1],mm2[2],mm2[3]);
			const double intemas3  =  axion->IAxionMassSqn(*z,*z + dz*d0,3);
			const double iintemas3 = axion->IIAxionMassSqn(*z,*z + dz*d0,3);
			const double shift = axion->Saskia();
			fsPropKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), R, dz, c0, d0, intemas3, iintemas3, shift, fMom1, Lx, Tz, precision);

			*z += dz*d0;
			axion->updateR();

		}

		if (lastStage) {
			LogMsg (VERB_HIGH, "Warning: fspectral propagator not working for odd propagators");
		}

		axion->setM2     (M2_DIRTY);
	}




	// Generic Naxion propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::nRunCpu	(const double dz) {

		PropParms ppar;
		ppar.Ng    = axion->getNg();
		ppar.ood2a = ood2;
		ppar.PC    = axion->getCO();
		ppar.Lx    = Lx;
		ppar.frw   = axion->BckGnd()->Frw();
		ppar.beta  = axion->BckGnd()->ICData().beta;
		/* Returns ghost size region in slices */
		size_t BO = ppar.Ng*S;
		size_t CO = V-2*ppar.Ng*S;

		LogMsg(VERB_DEBUG,"[propNax] Ng %d",ppar.Ng);

		#pragma unroll
		for (int s = 0; s<nStages; s+=2) {

			axion->sendGhosts(FIELD_M, COMM_SDRV);

			const double	c1 = c[s], c2 = c[s+1], d1 = d[s], d2 = d[s+1];

			ppar.ct     = *axion->zV();
			ppar.R      = *axion->RV();
			ppar.massA  = axion->AxionMass();
			propagateNaxKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propagateNaxKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			propagateNaxKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, V   , V+BO, precision, xBlock, yBlock, zBlock);
			*axion->zV() += dz*d1;
			axion->updateR();

			axion->sendGhosts(FIELD_M2, COMM_SDRV);

			ppar.ct     = *axion->zV();
			ppar.R      = *axion->RV();
			ppar.massA  = axion->AxionMass();
			propagateNaxKernelXeon<VQcd>(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M2, COMM_WAIT);
			propagateNaxKernelXeon<VQcd>(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			propagateNaxKernelXeon<VQcd>(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, V   , V+BO, precision, xBlock, yBlock, zBlock);
			*axion->zV() += dz*d2;
			axion->updateR();
		}

		if (lastStage) {
			axion->sendGhosts(FIELD_M, COMM_SDRV);

			const double	c0 = c[nStages];
			/* Last kick but not drift d = 0 */
			ppar.ct     = *axion->zV();
			ppar.R      = *axion->RV();
			ppar.massA  = axion->AxionMass();
			propagateNaxKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0.0, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propagateNaxKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0.0, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			propagateNaxKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0.0, V   , V+BO, precision, xBlock, yBlock, zBlock);

		}
		axion->setM2     (M2_DIRTY);
	}




	// Generic Paxion propagator

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::pRunCpu	(const double dz) {

		PropParms ppar;
		ppar.Ng    = axion->getNg();
		ppar.ood2a = ood2;
		ppar.PC    = axion->getCO();
		ppar.Lx    = Lx;
		ppar.beta  = axion->BckGnd()->ICData().beta;
		ppar.frw   = axion->BckGnd()->Frw();

		/* Returns ghost size region in slices */
		size_t BO = ppar.Ng*S;

		LogMsg(VERB_DEBUG,"[propPax] Ng %d ood2 %e beta %f PC %f %f %f ",ppar.Ng,ppar.ood2a,ppar.beta,ppar.PC[0],ppar.PC[1],ppar.PC[2]);


		ppar.ct     = *axion->zV();
		ppar.R      = *axion->RV();
		ppar.massA  = axion->AxionMass();
		ppar.sign   = 1;

		propagatePaxKernelXeon<KIDI_POT>(axion->mCpu(), axion->vCpu(), ppar, 0.5*dz, BO,   V+BO, precision, xBlock, yBlock, zBlock);

		#pragma unroll
		for (int s = 0; s<nStages; s++) {

			axion->sendGhosts(FIELD_M, COMM_SDRV);

			const double	c1 = c[s], d1 = d[s];

			ppar.ct     = *axion->zV();
			ppar.R      = *axion->RV();
			ppar.massA  = axion->AxionMass();
			ppar.sign   = 1;
			/*updates v(2) into m2(3) with m(1) lap data and NL function ; also (copies 3 into 2) */
			propagatePaxKernelXeon<KIDI_LAP>(axion->mCpu(), axion->vCpu(), ppar, dz*c1, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propagatePaxKernelXeon<KIDI_LAP>(axion->mCpu(), axion->vCpu(), ppar, dz*c1, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			propagatePaxKernelXeon<KIDI_LAP>(axion->mCpu(), axion->vCpu(), ppar, dz*c1, V   , V+BO, precision, xBlock, yBlock, zBlock);

			axion->sendGhosts(FIELD_V, COMM_SDRV);

			ppar.ct     = *axion->zV();
			ppar.R      = *axion->RV();
			ppar.massA  = axion->AxionMass();
			ppar.sign   = -1;
			propagatePaxKernelXeon<KIDI_LAP>(axion->vCpu(), axion->mCpu(), ppar, dz*d1, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_V, COMM_WAIT);
			propagatePaxKernelXeon<KIDI_LAP>(axion->vCpu(), axion->mCpu(), ppar, dz*d1, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			propagatePaxKernelXeon<KIDI_LAP>(axion->vCpu(), axion->mCpu(), ppar, dz*d1, V   , V+BO, precision, xBlock, yBlock, zBlock);
			*axion->zV() += dz*d1;
			axion->updateR();
		}

		if (lastStage) {
			axion->sendGhosts(FIELD_M, COMM_SDRV);

			const double	c0 = c[nStages];
			/* Last kick but not drift d = 0 */
			ppar.ct     = *axion->zV();
			ppar.R      = *axion->RV();
			ppar.massA  = axion->AxionMass();
			ppar.sign   = 1;
			propagatePaxKernelXeon<KIDI_LAP>(axion->mCpu(), axion->vCpu(), ppar, dz*c0, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propagatePaxKernelXeon<KIDI_LAP>(axion->mCpu(), axion->vCpu(), ppar, dz*c0, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			propagatePaxKernelXeon<KIDI_LAP>(axion->mCpu(), axion->vCpu(), ppar, dz*c0, V   , V+BO, precision, xBlock, yBlock, zBlock);

		}

		ppar.ct     = *axion->zV();
		ppar.R      = *axion->RV();
		ppar.massA  = axion->AxionMass();
		ppar.sign   = 1;
		propagatePaxKernelXeon<KIDI_POT>(axion->mCpu(), axion->vCpu(), ppar, 0.5*dz, BO,   V+BO , precision, xBlock, yBlock, zBlock);
	}






	template<const int nStages, const bool lastStage, VqcdType VQcd>
	double	PropClass<nStages, lastStage, VQcd>::cFlops	(const PropcType spec) {
		switch (spec)
		{
			case PROPC_BASE:
			{
				double lapla = 18.0 * axion->getNg();

				switch (axion->Field()) {
					case FIELD_SAXION:
						switch (VQcd & VQCD_TYPE) {	//FIXME Wrong for damping/only rho

							default:
							case VQCD_1:
								return	(1e-9 * ((double) axion->Size()) * ((24.+lapla) * ((double) nStages) + (lastStage ? (20.+lapla) : 0.)));
								break;

							case VQCD_PQ_ONLY:
								return	(1e-9 * ((double) axion->Size()) * ((22.+lapla) * ((double) nStages) + (lastStage ? (18.+lapla) : 0.)));
								break;

							case VQCD_2:
								return	(1e-9 * ((double) axion->Size()) * ((27.+lapla) * ((double) nStages) + (lastStage ? (23.+lapla) : 0.)));
								break;

							case VQCD_1_PQ_2:
								return	(1e-9 * ((double) axion->Size()) * ((26.+lapla) * ((double) nStages) + (lastStage ? (22.+lapla) : 0.)));
								break;

							case VQCD_1_PQ_2_RHO:
								return	(1e-9 * ((double) axion->Size()) * ((32.+lapla) * ((double) nStages) + (lastStage ? (28.+lapla) : 0.)));
								break;

							case VQCD_1N2:
								return	(1e-9 * ((double) axion->Size()) * ((25.+lapla) * ((double) nStages) + (lastStage ? (23.+lapla) : 0.))); //check the laststage?
								break;

							case VQCD_QUAD:
								return	(1e-9 * ((double) axion->Size()) * ((25.+lapla) * ((double) nStages) + (lastStage ? (25.+lapla) : 0.))); //check the laststage?
								break;
						}
						break;

					case FIELD_AXION:
					case FIELD_AXION_MOD:	// Seguro??
						return	(1e-9 * ((double) axion->Size()) * (23. * ((double) nStages) + (lastStage ? 15. : 0.)));
						break;

					default:
					case FIELD_WKB:
						return	0.;
						break;
				}
			}
			break;

			case PROPC_SPEC:
			case PROPC_FSPEC:
		 	{
				switch (axion->Field()) {
					case FIELD_SAXION: {
						auto &planFFT   = AxionFFT::fetchPlan("SpSx");
						double fftFlops = planFFT.GFlops(FFT_FWDBCK) * (((double) nStages) + (lastStage ? 1. : 0.));
						switch (VQcd & VQCD_TYPE) {	//FIXME Wrong for damping/only rho
							default:
							case VQCD_1:
								return	(1e-9 * ((double) axion->Size()) * ((26. + 1.) * ((double) nStages) + (lastStage ? 22. + 1. : 0.)
									) + fftFlops);//+ 5.*1.44695*log(((double) axion->Size()))));
								break;

							case VQCD_2:
								return	(1e-9 * ((double) axion->Size()) * ((29. + 1.) * ((double) nStages) + (lastStage ? 25. + 1. : 0.)
									) + fftFlops);//+ 5.*1.44695*log(((double) axion->Size()))));
								break;

							case VQCD_1_PQ_2:
								return	(1e-9 * ((double) axion->Size()) * ((26. + 1.) * ((double) nStages) + (lastStage ? 22. + 1. : 0.)
									) + fftFlops);//+ 5.*1.44695*log(((double) axion->Size()))));
								break;
						}
					}
					break;

					case FIELD_AXION:
					case FIELD_AXION_MOD: {	// Seguro??
						auto &planFFT   = AxionFFT::fetchPlan("SpSx");
						double fftFlops = planFFT.GFlops(FFT_FWDBCK) * (((double) nStages) + (lastStage ? 1. : 0.));
						return	(1e-9 * ((double) axion->Size()) * (21. * ((double) nStages) + (lastStage ? 13. : 0.)
							) + fftFlops);//+ 2.5*1.44695*log(((double) axion->Size()))));
					}
					break;

					default:
					case FIELD_WKB:
					return	0.;
				}
				}
				break;
		} // end final switch

		return	0.;
	}

	template<const int nStages, const bool lastStage, VqcdType VQcd>
	double	PropClass<nStages, lastStage, VQcd>::cBytes	(const PropcType spec) {

		double lapla = 1.0 + 6.0 * axion->getNg();

		switch (spec)
		{
			case PROPC_BASE:
				return	(1e-9 * ((double) (axion->Size()*axion->DataSize())) * (   (3. + lapla)    * ((double) nStages) + (lastStage ? (2. + lapla) : 0.)));
			break;

			case PROPC_SPEC:
			case PROPC_FSPEC:
				return	(1e-9 * ((double) (axion->Size()*axion->DataSize())) * ((6. + 4.) * ((double) nStages) + (lastStage ? 6. + 3. : 0.) + 2.));
			break;
		}
	}

#endif
