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
	#include "propagator/fsPropThetaXeon.h"
	#include "propagator/sPropThetaXeon.h"
	#include "propagator/propNaxXeon.h"
	#include "propagator/propPaxXeon.h"
	#include "gravity/potential.h"

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

	void loadparms(PropParms *pipar, Scalar *field);

	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
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

		double	c[nStages + (lastStage == PROP_LAST  ? 1 : 0)];
		double	d[nStages + (lastStage == PROP_FIRST ? 1 : 0)];

		public:

		inline	 PropClass(Scalar *field, const PropcType spec);
		inline	~PropClass() override {};

		inline void	setCoeff(const double * __restrict__ nC, const double * __restrict__ nD) {
			for(int i=0; i<nStages; i++) { c[i] = nC[i]; d[i] = nD[i]; } if (lastStage == PROP_FIRST) { d[nStages] = nD[nStages]; } if (lastStage == PROP_LAST) { c[nStages] = nC[nStages]; }
		}


		inline void	sRunCpu	(const double)	override;	// Saxion propagator
		inline void	sRunGpu	(const double)	override;
		inline void	sSpecCpu(const double)	override;	// Saxion spectral propagator
		inline void	sFpecCpu(const double)	override;	// Saxion spectral propagator


		inline void	tRunCpu	(const double)	override;	// Axion propagator
		inline void	tRunGpu	(const double)	override;
		inline void	tSpecCpu(const double)	override;	// Axion spectral propagator
		inline void	tFpecCpu(const double)	override;	// Axion spectral propagator exeriment


		inline void	nRunCpu	(const double)	override;			// Naxion propagator

		inline void	pRunCpu	(const double)	override;			// Paxion propagator

		inline void	lowCpu	(const double)	override;	// Lowmem only available for saxion non-spectral
		inline void	lowGpu	(const double)	override;

		inline double	cFlops	(const PropcType)	override;
		inline double	cBytes	(const PropcType)	override;

	};

	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
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

		gravity = false;

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
						propAxion  = [this](const double dz) { this->tFpecCpu(dz); }; // preliminar
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

	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::tRunGpu	(const double dz) {
	#ifdef  USE_GPU
		const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
		const uint ext = uV + uS;
		// eom only depend on R
		double *z  = axion->zV();
		double *R  = axion->RV();

		double *cD = d;

		const bool wMod = (axion->Field() == FIELD_AXION_MOD) ? true : false;

		if (lastStage == PROP_FIRST) {
			const double	d0 = d[0];

			updateMGpu(axion->mGpu(), axion->vGpu(), dz, d0, uLx, uS, ext, precision, xBlock, yBlock, zBlock, ((cudaStream_t *)axion->Streams())[2], FIELD_AXION);
			*z += dz*d0;
			axion->updateR();
			cD = &(d[1]);
			cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
		}

		#pragma unroll
		for (int s = 0; s<nStages; s+=2) {
			const double	c1 = c[s], c2 = c[s+1], d1 = cD[s], d2 = cD[s+1];

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

		if (lastStage == PROP_LAST) {
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

	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::sRunGpu	(const double dz) {
	#ifdef	USE_GPU
		PropParms ppar;
		loadparms(&ppar, axion);
		const uint uLx = Lx, uLz = Lz, uS = ppar.Ng*S, uV = V;
		const uint ext = uV + uS;


		double *z = axion->zV();

		auto *cD = d;

		if (lastStage == PROP_FIRST) {
			const double	d0 = d[0];

			updateMGpu(axion->mGpu(), axion->vGpu(), dz, d0, uLx, uS, ext, precision, xBlock, yBlock, zBlock, ((cudaStream_t *)axion->Streams())[2]);
			*z += dz*d0;
			axion->updateR();
			cD = &(d[1]);
			cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
		}

		#pragma unroll
		for (int s = 0; s<nStages; s+=2) {

			loadparms(&ppar, axion);

			const double	c1 = c[s], c2 = c[s+1], d1 = cD[s], d2 = cD[s+1];

			propagateGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), ppar, dz, c1, d1, 2*uS, uV, VQcd, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[2]);
			axion->exchangeGhosts(FIELD_M);
			propagateGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), ppar, dz, c1, d1, uS, 2*uS, VQcd, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[0]);
			if (uV>uS)
			propagateGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), ppar, dz, c1, d1, uV,  ext, VQcd, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[1]);

			cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot

			*z += dz*d1;
			axion->updateR();
			loadparms(&ppar, axion);

			propagateGpu(axion->m2Gpu(), axion->vGpu(), axion->mGpu(), ppar, dz, c2, d2, 2*uS, uV, VQcd, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[2]);
			axion->exchangeGhosts(FIELD_M2);
			propagateGpu(axion->m2Gpu(), axion->vGpu(), axion->mGpu(), ppar, dz, c2, d2, uS, 2*uS, VQcd, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[0]);
			if (uV>uS)
			propagateGpu(axion->m2Gpu(), axion->vGpu(), axion->mGpu(), ppar, dz, c2, d2, uV,  ext, VQcd, precision, xBlock, yBlock, zBlock,
				    ((cudaStream_t *)axion->Streams())[1]);

			cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
			*z += dz*d2;
			axion->updateR();
		}

		if (lastStage == PROP_LAST) {
			double cLmbda = axion->LambdaP();

			const double    c0 = c[nStages], maa = axion->AxionMassSq();

			loadparms(&ppar, axion);

			updateVGpu(axion->mGpu(), axion->vGpu(), ppar, dz, c0, uS*2, uV, VQcd, precision, xBlock, yBlock, zBlock,
				  ((cudaStream_t *)axion->Streams())[2]);
			axion->exchangeGhosts(FIELD_M);
			updateVGpu(axion->mGpu(), axion->vGpu(), ppar, dz, c0, uS, uS*2, VQcd, precision, xBlock, yBlock, zBlock,
				  ((cudaStream_t *)axion->Streams())[0]);
			if (uV>uS)
			updateVGpu(axion->mGpu(), axion->vGpu(), ppar, dz, c0, uV,  ext, VQcd, precision, xBlock, yBlock, zBlock,
				  ((cudaStream_t *)axion->Streams())[1]);
		}

	#else
		LogError ("Error: gpu support not built");
		exit(1);
	#endif
	}

	// Generic saxion lowmem propagator

	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
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

	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::tRunCpu	(const double dz) {
		double *z  = axion->zV();
		double *cD = d;

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

		ppar.beta  = axion->BckGnd()->ICData().beta;
		/* Returns ghost size region in slices */
		size_t BO = ppar.Ng*S;

		if (lastStage == PROP_FIRST) {
			/* Last kick */
			LogMsg (VERB_PARANOID, "Warning: axion propagator not optimized yet for odd propagators, performance might be reduced");
			const double	d0 = d[0];
			/* First drift no kick c = 0 */

			updateMThetaXeon(axion->mCpu(), axion->vCpu(), dz, d0, Lx, BO, V+BO, precision, xBlock, yBlock, zBlock);
			*z += dz*d0;
			axion->updateR();
			ppar.R   = *axion->RV();
			ppar.Rpp = axion->Rpp();
			cD = &(d[1]);
		}

		#pragma unroll
		for (int s = 0; s<nStages; s+=2) {
			axion->sendGhosts(FIELD_M, COMM_SDRV);

			const double c1 = c[s], c2 = c[s+1], d1 = cD[s], d2 = cD[s+1];

			propThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, 2*BO, V , precision, xBlock, yBlock, zBlock, wMod, VQcd);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, BO, 2*BO, precision, xBlock, yBlock, zBlock, wMod, VQcd);
			if (V>BO)
			propThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, V , V+BO, precision, xBlock, yBlock, zBlock, wMod, VQcd);
			*z += dz*d1;
			axion->updateR();
			axion->sendGhosts(FIELD_M2, COMM_SDRV);

			ppar.massA2 = axion->AxionMassSq();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();

			propThetaKernelXeon(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, 2*BO, V   , precision, xBlock, yBlock, zBlock, wMod, VQcd);
			axion->sendGhosts(FIELD_M2, COMM_WAIT);
			propThetaKernelXeon(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, BO  , 2*BO, precision, xBlock, yBlock, zBlock, wMod, VQcd);
			if (V>BO)
			propThetaKernelXeon(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, V   , V+BO, precision, xBlock, yBlock, zBlock, wMod, VQcd);
			*z += dz*d2;
			axion->updateR();
		}

		if (lastStage == PROP_LAST) {
			axion->sendGhosts(FIELD_M, COMM_SDRV);
			/* Last kick */
			LogMsg (VERB_PARANOID, "Warning: axion propagator not optimized yet for odd propagators, performance might be reduced");

			const double	c0 = c[nStages], maa = axion->AxionMassSq();

			ppar.massA  = axion->AxionMassSq();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();

			propThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0., 2*BO, V   , precision, xBlock, yBlock, zBlock, wMod, VQcd);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0., BO  , 2*BO, precision, xBlock, yBlock, zBlock, wMod, VQcd);
			if (V>BO)
			propThetaKernelXeon(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0., V   , V+BO, precision, xBlock, yBlock, zBlock, wMod, VQcd);
		}

	}

	// Generic axion spectral propagator

	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
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

	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::sRunCpu	(const double dz) {
		double *z = axion->zV();

		PropParms ppar;
		ppar.Ng     = axion->getNg();
		ppar.Lx     = Lx;
		ppar.PC     = axion->getCO();
		ppar.ood2a  = ood2;
		ppar.gamma  = axion->BckGnd()->Gamma();
		ppar.frw    = axion->BckGnd()->Frw();
		ppar.dectime= axion->BckGnd()->DecTime();

		/* Returns ghost size region in slices */
		size_t BO = ppar.Ng*S;

		auto *cD  = d;

		if (lastStage == PROP_FIRST) {
			const double d0 = d[0];
			/* First drift no kick c = 0 */

			updateMXeon(axion->mCpu(), axion->vCpu(), dz, d0, Lx, BO, V+BO, precision, xBlock, yBlock, zBlock);
			*z += dz*d0;
			axion->updateR();
			cD = &(d[1]);
		}

		#pragma unroll
		for (int s = 0; s<nStages; s+=2) {

			ppar.lambda = axion->LambdaP();
			ppar.massA2 = axion->AxionMassSq();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();
			ppar.Rp     = axion->BckGnd()->Rp(*axion->zV());

			axion->sendGhosts(FIELD_M, COMM_SDRV);

			const double	c1 = c[s], c2 = c[s+1], d1 = cD[s], d2 = cD[s+1];

			propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			if (V>BO)
			propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c1, d1, V   , V+BO, precision, xBlock, yBlock, zBlock);
			*z += dz*d1;
			axion->updateR();

			axion->sendGhosts(FIELD_M2, COMM_SDRV);

			ppar.lambda = axion->LambdaP();
			ppar.massA2 = axion->AxionMassSq();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();
			ppar.Rp     = axion->BckGnd()->Rp(*axion->zV());

			propagateKernelXeon<VQcd>(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M2, COMM_WAIT);
			propagateKernelXeon<VQcd>(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			if (V>BO)
			propagateKernelXeon<VQcd>(axion->m2Cpu(), axion->vCpu(), axion->mCpu(), ppar, dz, c2, d2, V   , V+BO, precision, xBlock, yBlock, zBlock);
			*z += dz*d2;
			axion->updateR();
		}

		if (lastStage == PROP_LAST) {
			axion->sendGhosts(FIELD_M, COMM_SDRV);

			const double    c0 = c[nStages], maa = axion->AxionMassSq();
			/* Last kick but not drift d = 0 */

			ppar.lambda = axion->LambdaP();
			ppar.massA2 = axion->AxionMassSq();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();
			ppar.Rp     = axion->BckGnd()->Rp(*axion->zV());

			propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0.0, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0.0, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			if (V>BO)
			propagateKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0.0, V   , V+BO, precision, xBlock, yBlock, zBlock);
		}

		axion->setM2     (M2_DIRTY);
	}




	// Generic saxion lowmem propagator

	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::lowCpu	(const double dz) {
		double *z = axion->zV();

		PropParms ppar;
		ppar.Ng     = axion->getNg();
		ppar.Lx     = Lx;
		ppar.PC     = axion->getCO();
		ppar.ood2a  = ood2;
		ppar.gamma  = axion->BckGnd()->Gamma();
		ppar.frw    = axion->BckGnd()->Frw();
		ppar.dectime= axion->BckGnd()->DecTime();


		size_t BO = ppar.Ng*S;

		#pragma unroll
		for (int s = 0; s<nStages; s++) {

			ppar.lambda = axion->LambdaP();
			ppar.massA2 = axion->AxionMassSq();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();

			axion->sendGhosts(FIELD_M, COMM_SDRV);

			const double c0 = c[s], d0 = d[s];

			updateVXeon<VQcd>(axion->mCpu(), axion->vCpu(), ppar, dz, c0, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			updateVXeon<VQcd>(axion->mCpu(), axion->vCpu(), ppar, dz, c0, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			updateVXeon<VQcd>(axion->mCpu(), axion->vCpu(), ppar, dz, c0, V   , V+BO, precision, xBlock, yBlock, zBlock);

			/*missing update M?*/
			*z += dz*d0;
			axion->updateR();
		}

		if (lastStage) {
			axion->sendGhosts(FIELD_M, COMM_SDRV);

			ppar.lambda = axion->LambdaP();
			ppar.massA2 = axion->AxionMassSq();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();

			const double c0 = c[nStages];

			updateVXeon<VQcd>(axion->mCpu(), axion->vCpu(), ppar, dz, c0, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			updateVXeon<VQcd>(axion->mCpu(), axion->vCpu(), ppar, dz, c0, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			updateVXeon<VQcd>(axion->mCpu(), axion->vCpu(), ppar, dz, c0, V   , V+BO, precision, xBlock, yBlock, zBlock);

		}
	}





	// Generic saxion spectral propagator
	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::sSpecCpu	(const double dz) {

		double *z = axion->zV();

		PropParms ppar;
		ppar.Ng     = axion->getNg();
		ppar.Lx     = Lx;
		ppar.PC     = axion->getCO();
		ppar.ood2a  = ood2;
		ppar.gamma  = axion->BckGnd()->Gamma();
		ppar.frw    = axion->BckGnd()->Frw();
		ppar.fMom1 = -(4.*M_PI*M_PI)/(axion->BckGnd()->PhysSize()*axion->BckGnd()->PhysSize()*((double) axion->TotalSize()));

		/* Returns ghost size region in slices */
		size_t BO = ppar.Ng*S;

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

			ppar.lambda = axion->LambdaP();
			ppar.massA2 = axion->AxionMassSq();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();

			sPropKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, d0, S, V+S, precision);
			*z += dz*d0;
			axion->updateR();
		}

		if (lastStage) {
			LogMsg (VERB_HIGH, "Warning: spectral propagator not optimized yet for odd propagators, performance might be reduced");

			const double	c0 = c[nStages], maa = axion->AxionMassSq();

			applyLaplacian(axion);

			ppar.lambda = axion->LambdaP();
			ppar.massA2 = axion->AxionMassSq();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();

			sPropKernelXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, 0.0, S, V+S, precision);
		}
		axion->setM2     (M2_DIRTY);
	}





	// Generic saxion full spectral propagator (in Fourier space)
	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
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

			// LogOut("[fs] m  values %e %e %e %e \n",mmm[0],mmm[1],mmm[2],mmm[3]);
			// LogOut("[fs] v  values %e %e %e %e \n",vvv[0],vvv[1],vvv[2],vvv[3]);
			// LogOut("[fs] m2 values %e %e %e %e \n",mm2[0],mm2[1],mm2[2],mm2[3]);
			LogFlush();
			fsAccKernelXeon<VQcd>(axion->vCpu(), axion->m2Cpu(), R, dz, c0, d0, ood2, cLmbda, maa, gamma, fMom1, Lx, S, V+S, precision);
			// LogOut("[fs] ac values %e %e %e %e \n",mm2[0],mm2[1],mm2[2],mm2[3]);

			if (debug) LogOut("[fs] accelerated \n");
			pelota(FIELD_M2TOM2, FFT_FWD); // FWD sends M2 to mom space
			if (debug) LogOut("[fs] fff \n");
			/* kicks in momentum space */
			// LogOut("[fs] ac momspace %e %e %e %e \n",mm2[0],mm2[1],mm2[2],mm2[3]);
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



	// // Generic Axion full spectral propagator (in Fourier space)
	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
	void	PropClass<nStages, lastStage, VQcd>::tFpecCpu	(const double dz) {

		PropParms ppar;
		ppar.ood2a  = ood2;
		ppar.Lx     = axion->Length();
		ppar.Tz     = axion->TotalDepth();
		ppar.frw    = axion->BckGnd()->Frw();
		ppar.fMom1  = (2.*M_PI)/(axion->BckGnd()->PhysSize());

		/* Returns volume size region including padding! */
		size_t VD = (axion->Length()+2)*axion->Length()*axion->Depth();

		if (axion->Folded())
		{
			Folder munge(axion);
			munge(UNFOLD_ALL);
		}

		// If field is in configuration space, pad, transform to momentum space, unghost
		if	( !axion->MMomSpace() || !axion->VMomSpace() )
		{
			if (debug) LogOut("[fs] FT!\n");
			FTfield pelotas(axion);
			pelotas(FIELD_MV, FFT_FWD); // FWD is to send to momentum space transposed out
		}

		float *mmm = static_cast<float *>(axion->mCpu());
		float *vvv = static_cast<float *>(axion->vCpu());
		float *mm2 = static_cast<float *>(axion->m2Cpu());

		#pragma unroll
		for (int s = 0; s<nStages; s++) {
			const double	c0 = c[s], d0 = d[s];
			ppar.massA2 = axion->AxionMassSq();
			ppar.ct     = *axion->zV();
			ppar.R      = *axion->RV();
			ppar.Rpp    = axion->Rpp();

			// LogOut("[fs] m  values %.2e %.2e %.2e %.2e \n",mmm[0],mmm[1],mmm[2],mmm[3]);
			// LogOut("[fs] v  values %.2e %.2e %.2e %.2e \n",vvv[0],vvv[1],vvv[2],vvv[3]);
			// LogOut("[fs] m2 values %.2e %.2e %.2e %.2e \n",mm2[0],mm2[1],mm2[2],mm2[3]);

			// computes m into m2 in configuration space
			FTfield pelota(axion);
			pelota(FIELD_MTOM2, FFT_BCK); // BCK is to send to conf space

			// LogOut("[fs] m2 values %.2e %.2e %.2e %.2e \n",mm2[0],mm2[1],mm2[2],mm2[3]);

			/* computes the acceleration in configuration space */
			fsAccKernelThetaXeon<VQcd>(axion->m2Cpu(), ppar, 0, VD, precision);
			// LogOut("[fs] ac values %.2e %.2e %.2e %.2e \n",mm2[0],mm2[1],mm2[2],mm2[3]);

			pelota(FIELD_M2TOM2, FFT_FWD); // FWD sends M2 to mom space
			/* kicks in momentum space */
			// LogOut("[fs] ac momspace %.2e %.2e %.2e %.2e \n",mm2[0],mm2[1],mm2[2],mm2[3]);
			fsPropKernelThetaXeon<VQcd>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, dz, c0, d0, precision);

			*axion->zV() += dz*d0;
			axion->updateR();

		}

		if (lastStage) {
			LogMsg (VERB_HIGH, "Warning: fspectral propagator not working for odd propagators");
		}

		axion->setM2     (M2_DIRTY);
	}




	// Generic Naxion propagator

	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
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

		LogMsg(VERB_PARANOID,"[propNax] Ng %d",ppar.Ng);

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

	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
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

		LogMsg(VERB_PARANOID,"[propPax] Ng %d ood2 %e beta %f PC %f %f %f ",ppar.Ng,ppar.ood2a,ppar.beta,ppar.PC[0],ppar.PC[1],ppar.PC[2]);

		void *nada;
		ppar.ct     = *axion->zV();
		ppar.R      = *axion->RV();
		ppar.n      = axion->BckGnd()->DlogMARDlogct(ppar.ct);
		ppar.massA  = axion->AxionMass();
		ppar.sign   = 1;
		ppar.grav   = axion->BckGnd()->ICData().grav; /*TODO*/
		if (gravity){
			calculateGraviPotential	();
			propagatePaxKernelXeon<KIDI_POT_GRAV>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, 0.5*dz, BO,   V+BO, precision, xBlock, yBlock, zBlock);
		} else
			propagatePaxKernelXeon<KIDI_POT>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, 0.5*dz, BO,   V+BO, precision, xBlock, yBlock, zBlock);

		#pragma unroll
		for (int s = 0; s<nStages; s++) {

			axion->sendGhosts(FIELD_M, COMM_SDRV);

			const double	c1 = c[s], d1 = d[s];

			ppar.ct     = *axion->zV();
			ppar.R      = *axion->RV();
			ppar.n      = axion->BckGnd()->DlogMARDlogct(ppar.ct);
			ppar.massA  = axion->AxionMass();
			ppar.sign   = 1;
			/*updates v(2) with m(1) lap data and NL function */
			propagatePaxKernelXeon<KIDI_LAP>(axion->mCpu(), axion->vCpu(), nada, ppar, dz*c1, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propagatePaxKernelXeon<KIDI_LAP>(axion->mCpu(), axion->vCpu(), nada, ppar, dz*c1, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			propagatePaxKernelXeon<KIDI_LAP>(axion->mCpu(), axion->vCpu(), nada, ppar, dz*c1, V   , V+BO, precision, xBlock, yBlock, zBlock);

			axion->sendGhosts(FIELD_V, COMM_SDRV);

			ppar.ct     = *axion->zV();
			ppar.R      = *axion->RV();
			ppar.n      = axion->BckGnd()->DlogMARDlogct(ppar.ct);
			ppar.massA  = axion->AxionMass();
			ppar.sign   = -1;
			propagatePaxKernelXeon<KIDI_LAP>(axion->vCpu(), axion->mCpu(), nada, ppar, dz*d1, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_V, COMM_WAIT);
			propagatePaxKernelXeon<KIDI_LAP>(axion->vCpu(), axion->mCpu(), nada, ppar, dz*d1, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			propagatePaxKernelXeon<KIDI_LAP>(axion->vCpu(), axion->mCpu(), nada, ppar, dz*d1, V   , V+BO, precision, xBlock, yBlock, zBlock);
			*axion->zV() += dz*d1;
			axion->updateR();
		}

		if (lastStage) {
			axion->sendGhosts(FIELD_M, COMM_SDRV);

			const double	c0 = c[nStages];
			/* Last kick but not drift d = 0 */
			ppar.ct     = *axion->zV();
			ppar.R      = *axion->RV();
			ppar.n      = axion->BckGnd()->DlogMARDlogct(ppar.ct);
			ppar.massA  = axion->AxionMass();
			ppar.sign   = 1;
			propagatePaxKernelXeon<KIDI_LAP>(axion->mCpu(), axion->vCpu(), nada, ppar, dz*c0, 2*BO, V   , precision, xBlock, yBlock, zBlock);
			axion->sendGhosts(FIELD_M, COMM_WAIT);
			propagatePaxKernelXeon<KIDI_LAP>(axion->mCpu(), axion->vCpu(), nada, ppar, dz*c0, BO  , 2*BO, precision, xBlock, yBlock, zBlock);
			propagatePaxKernelXeon<KIDI_LAP>(axion->mCpu(), axion->vCpu(), nada, ppar, dz*c0, V   , V+BO, precision, xBlock, yBlock, zBlock);

		}

		ppar.ct     = *axion->zV();
		ppar.R      = *axion->RV();
		ppar.n      = axion->BckGnd()->DlogMARDlogct(ppar.ct);
		ppar.massA  = axion->AxionMass();
		ppar.sign   = 1;

		if (gravity){
			calculateGraviPotential	();
			propagatePaxKernelXeon<KIDI_POT_GRAV>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, 0.5*dz, BO,   V+BO, precision, xBlock, yBlock, zBlock);
		} else
			propagatePaxKernelXeon<KIDI_POT>(axion->mCpu(), axion->vCpu(), axion->m2Cpu(), ppar, 0.5*dz, BO,   V+BO, precision, xBlock, yBlock, zBlock);

	}






	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
	double	PropClass<nStages, lastStage, VQcd>::cFlops	(const PropcType spec) {
		switch (spec)
		{
			case PROPC_BASE:
			{
				double lapla = 18.0 * axion->getNg();

				/* TODO GENERALISE */
				switch (axion->Field()) {
					case FIELD_SAXION:
						switch (VQcd & V_TYPE) {

							default:
							case V_QCD1_PQ1:
								return	(1e-9 * ((double) axion->Size()) * ((24.+lapla) * ((double) nStages) + (lastStage ? (20.+lapla) : 0.)));
								break;

							case V_QCD0_PQ1:
								return	(1e-9 * ((double) axion->Size()) * ((22.+lapla) * ((double) nStages) + (lastStage ? (18.+lapla) : 0.)));
								break;

							case V_QCDV_PQ1:
								return	(1e-9 * ((double) axion->Size()) * ((27.+lapla) * ((double) nStages) + (lastStage ? (23.+lapla) : 0.)));
								break;

							case V_QCD1_PQ2:
								return	(1e-9 * ((double) axion->Size()) * ((26.+lapla) * ((double) nStages) + (lastStage ? (22.+lapla) : 0.)));
								break;

							case V_QCD1_PQ2_RHO:
								return	(1e-9 * ((double) axion->Size()) * ((32.+lapla) * ((double) nStages) + (lastStage ? (28.+lapla) : 0.)));
								break;

							case V_QCD2_PQ1:
								return	(1e-9 * ((double) axion->Size()) * ((25.+lapla) * ((double) nStages) + (lastStage ? (23.+lapla) : 0.))); //check the laststage?
								break;

							case V_QCDL_PQ1:
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
						switch (VQcd & V_TYPE) {	//FIXME Wrong for most propagators
							default:
							case V_QCD1_PQ1:
								return	(1e-9 * ((double) axion->Size()) * ((26. + 1.) * ((double) nStages) + (lastStage ? 22. + 1. : 0.)
									) + fftFlops);//+ 5.*1.44695*log(((double) axion->Size()))));
								break;

							case V_QCDV_PQ1:
								return	(1e-9 * ((double) axion->Size()) * ((29. + 1.) * ((double) nStages) + (lastStage ? 25. + 1. : 0.)
									) + fftFlops);//+ 5.*1.44695*log(((double) axion->Size()))));
								break;

							case V_QCD1_PQ2:
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

	template<const int nStages, const PropStage lastStage, VqcdType VQcd>
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

	void loadparms(PropParms *pipar, Scalar *axion)
	{
		(*pipar).lambda = axion->LambdaP();
		(*pipar).massA2 = axion->AxionMassSq();
		(*pipar).R      = *axion->RV();
		(*pipar).Rpp    = axion->Rpp();
		(*pipar).Rp     = axion->BckGnd()->Rp(*axion->zV());

		(*pipar).Ng     = axion->getNg();
		(*pipar).Lx     = axion->Length();;
		(*pipar).PC     = axion->getCO();
		(*pipar).ood2a  = 1./(axion->Delta()*axion->Delta());
		(*pipar).gamma  = axion->BckGnd()->Gamma();
		(*pipar).frw    = axion->BckGnd()->Frw();
		(*pipar).dectime= axion->BckGnd()->DecTime();

	}
#endif
