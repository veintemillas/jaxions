#ifndef	_POTCLASS_
	#define	_POTCLASS_

	#include <string>
	#include <complex>
	#include <memory>
	#include "scalar/scalarField.h"
	#include "scalar/folder.h"
	#include "enum-field.h"
	#include "energy/energy.h"
	#include "scalar/scaleField.h"

	#ifdef	USE_GPU
		#include <cuda.h>
		#include <cuda_runtime.h>
		#include <cuda_device_runtime_api.h>
	#endif

	#include "utils/utils.h"
	#include "fft/fftCode.h"
	#include "comms/comms.h"

	#include "gravity/gravityPaxionXeon.h"

	#include "io/readWrite.h"

	class	GraVi : public Tunable
	{
		private:

		Scalar			*afield;

		const FieldPrecision	precision;
		const size_t		Lx;
		const size_t		Lz;
		const size_t		Tz;
		const size_t		S;
		const size_t		V;
		const size_t		Ng;
		const size_t		BO;

		const double k0, norm;
		const size_t hLx;
		const size_t hLy;
		const size_t hTz;

		const size_t nModeshc;
		const size_t dl;
		const size_t pl;
		const size_t ss;

		size_t zBase;
		size_t index;

		/* still do not know how to define those input?*/
		size_t redfft ;
		size_t smsteps;
		size_t smstep2;

		double rnorm ;

		size_t fSize;
		size_t shift;

		bool hybrid;

		PropParms ppar;

		void	rfuap (Scalar *field, const size_t ref);
		template<typename Float>
		void	rfuap (Scalar *field, const size_t ref);

		void	ilap (Scalar *field, const size_t red, const size_t smit);
		template<typename Float>
		void	ilap (Scalar *field, const size_t red, const size_t smit);

		void	eguf (Scalar *field, const size_t red);
		template<typename Float>
		void	eguf (Scalar *field, const size_t ref);

		double	mean (Scalar *field, void *punt, size_t size, size_t tsize);
		template<typename Float>
		double	mean (void *punt, size_t size, size_t tsize);



		template<typename Float>
		void	normaliseFields ();

		template<class Float>
		void	GraHybridCpu	();
		template<class Float>
		void	GraCpu	();

		void	RunCpu	();
		void	RunGpu	();

		void	report	(int index, int red, int pad, int fft, int slice, char *name);

		public:

			/* Constructor */

			GraVi (Scalar *field) : afield(field), precision(afield->Precision()), Lx(afield->Length()), Lz(afield->Depth()), Tz(afield->TotalDepth()),
								S(afield->Surf()), V(afield->Size()), Ng(afield->getNg()), BO(Ng*S),
								k0((2.0*M_PI/ (double) afield->BckGnd()->PhysSize())),  hLx((Lx >> 1)+1), hLy(Lx >> 1), hTz(Tz >> 1), nModeshc(Lx*hLx*Lz),
								norm( -1./(k0*k0*((double) afield->TotalSize()))), dl(precision*Lx), pl(precision*(Lx+2)), ss(Lz*Lx),
								fSize(afield->DataSize()), shift(afield->DataAlign()/fSize) {

					zBase = Lx/commSize()*commRank();


					if (afield->LowMem()) {
						LogError ("Error: potential not supported in lowmem runs");
						exit(0);
					}

					/* better use the tuned one */
					/*	Default block size gives just one block	*/

					ppar.Ng    = Ng;
					ppar.ood2a = 1.0;
					ppar.PC    = afield->getCO();
					ppar.Lx    = Lx;
					ppar.Lz    = Lz;

					/* in principle we only need one of those and to init it once */
					AxionFFT::initPlan (afield, FFT_RtoC_M2StoM2S, FFT_FWDBCK, "m2Sr2c");

					redfft  = 2;
					smsteps = 2*redfft;
					smstep2 = 2*redfft;

					AxionFFT::initPlan (afield, FFT_PSPEC_AX,  FFT_FWDBCK, "m2redFFT",redfft);

					hybrid = false;

					index  = 0;
			}

		 	~GraVi()  {};

			void	SetHybrid	(bool la) {hybrid = la;};
			void	tunehybrid	();
			void	normaliseFields ();

			void	Run	();
	};




	void  GraVi::Run	()
	{
		switch (afield->Device()) {
			case DEV_GPU:
				this->RunGpu ();
				break;
			case DEV_CPU:
				this->RunCpu ();
				break;
			default:
				LogError ("Error: invalid device");
				return;
		}
	}

	void  GraVi::RunGpu	()
	{
	#ifdef  USE_GPU
	#else
		LogError ("Error: gpu support not built");
		exit(1);
	#endif
	}


	void	GraVi::RunCpu	()
	{
		if (precision == FIELD_DOUBLE){
			if (hybrid)
				GraHybridCpu<double>() ;
			else
				GraCpu<double>();
		} else if (precision == FIELD_SINGLE) {
			if (hybrid)
				GraHybridCpu<float>();
			else
				GraCpu<float>();
		}
		afield->setM2(M2_POT);
	}



	void	GraVi::tunehybrid	()
	{

		index = 0;

LogMsg(VERB_PARANOID,"Full test (index %d)",index);

		double t0 = (double) Timer()*1.0e-6;
		this->Run();
		double te = (double) Timer()*1.0e-6;
		double delta = te-t0;

LogMsg(VERB_PARANOID,"Full delta %e",delta);


			size_t best = 1;

			hybrid = true;

			/* adjust dynamically ? */
			for (int fo = 2; fo<32; fo *=2)
			{
LogMsg(VERB_PARANOID,"[GVt] Hybrid test red=%d (index %d)\n",fo, index);
				AxionFFT::removePlan("m2redFFT");
				redfft  = fo;
				AxionFFT::initPlan (afield, FFT_PSPEC_AX,  FFT_FWDBCK, "m2redFFT",redfft);
				smsteps = 2*redfft;
				smstep2 = 2*redfft;
				t0 = (double) Timer()*1.0e-6;
				this->Run();
				te = (double) Timer()*1.0e-6;
LogMsg(VERB_PARANOID,"Hybrid delta %e",te-t0);
				if (te-t0 < delta){
LogMsg(VERB_PARANOID,"best");
					delta = te-t0;
					best = fo;
					}

			}
			AxionFFT::removePlan("m2redFFT");
LogMsg(VERB_PARANOID,"best was redfft=%d",best);

			if (best == 1)
				hybrid = false;
			else {
				redfft  = best;
				smsteps = 2*redfft;
				smstep2 = 2*redfft;
			}
			AxionFFT::initPlan (afield, FFT_PSPEC_AX,  FFT_FWDBCK, "m2redFFT",redfft);

	}
	/* Template calls */


	/* main operator does the full FTT potential */



	template<class Float>
	void	GraVi::GraCpu	()
	{
			LogMsg(VERB_PARANOID,"[GV] begin GravCpu()");
			void *nada;

// int index =0 ;

			Folder munge(afield);

			/* Energy map in m2Start (folded or unfolded) */
			/* (assumes paxion, otherwise use energy map) */
			if (afield->Field() == FIELD_PAXION) 
				graviPaxKernelXeon<KIDI_ENE>(afield->mCpu(), afield->vCpu(), nada, afield->m2Cpu(), ppar, BO, V+BO, afield->Precision(), xBlock, yBlock, zBlock);
			else if (afield->Field() == FIELD_AXION)
			{
				void *eRes;
				trackAlloc(&eRes, 256);
				energy(afield,eRes,EN_MAP,0); // energy unfolded starting in m2
				double factor = 1/(((double *) eRes)[TH_KIN] + ((double *) eRes)[TH_POT] + ((double *) eRes)[TH_GRX] + ((double *) eRes)[TH_GRY] + ((double *) eRes)[TH_GRZ]);  
				scaleField(afield,FIELD_M2,factor);
				memmove(afield->m2Start(),afield->m2Cpu(),afield->Size()*afield->Precision());
			}
				
int red =1;
int pad =0;
int fft =0;
report (index, red, pad, fft, 1, "After energy");


			/* If field was folded energy in m2 it is too and we need it unfolded */

			if (afield->Folded() && afield->Field() == FIELD_PAXION){
// LogMsg(VERB_PARANOID,"[GV] Unfold M2");
				afield->setM2Folded(true);
				munge(UNFOLD_M2);
			}

report (index, red, pad, fft, 1, "After unfolding");


			/* pad in place m2S
				note that this exceeds the first half of m2 so things in m2h will be erased */
			LogMsg(VERB_PARANOID,"[GV] pad in place m2s");
			char *M2  = static_cast<char *> ((void *) afield->m2Cpu());
			char *MS  = static_cast<char *> ((void *) afield->m2Start());
			for (int l = ss-1; l>0; l--)
				memmove(MS + l*pl, MS + l*dl, dl);


LogMsg(VERB_PARANOID,"[GV] Retrieve plan");
			auto &myPlan = AxionFFT::fetchPlan("m2Sr2c");
			/* fft */
			myPlan.run(FFT_FWD);

pad = 2;
fft = 1;
report (index, red, pad, fft, 1, "FFT");


			/* ilap */
			Float fnorm   = (Float) norm;
// LogMsg(VERB_PARANOID,"[GV] inverse lap (norm %e %e)",norm,fnorm);
// LogMsg(VERB_PARANOID,"[GV] hLx %d",hLx);
// LogMsg(VERB_PARANOID,"[GV] Tz %d",Tz);
// LogMsg(VERB_PARANOID,"[GV] zB %d",zBase);
			std::complex<Float> *m2   = static_cast<std::complex<Float> *> (afield->m2Start());

			#pragma omp parallel for schedule(static)
			for (size_t idx=0; idx < nModeshc; idx++) {
				size_t tmp = idx/hLx; /* kz + ky*rTz */
				int    kx  = idx - tmp*hLx;
				int    ky  = tmp/Tz;
				int    kz  = tmp - Tz*((size_t) ky);
				ky += (int) zBase;
				if (ky > static_cast<int>(hLy)) ky -= static_cast<int>(Lx);
				if (kz > static_cast<int>(hTz)) kz -= static_cast<int>(Tz);
				Float k2    = (Float) (kx*kx + ky*ky + kz*kz);
				m2[idx]	*= fnorm/k2;
			}
			/* mode 0 to zero */
			if (commRank() == 0)
				m2[0] = (0,0);

report (index, red, pad, fft, 1, "FFT after ilap");


// LogMsg(VERB_PARANOID,"[GV] iFFT");
			myPlan.run(FFT_BCK);

pad = 2;
fft = 0;
report (index, red, pad, fft, 1, "Potential Padded");


LogMsg(VERB_PARANOID,"[GV] unpad in place m2s");
			/* unpad note potential ends up in m2Start*/
			for (size_t l =0; l<ss; l++)
				memmove(MS + l*dl, MS + l*pl, dl);

pad = 0;
report (index, red, pad, fft, 1, "Potential");

			/* If field was folded, we want gravi folded */
			if (afield->Folded()){
				// LogMsg(VERB_PARANOID,"[GV] Fold m2");
				munge(FOLD_M2);
			}
// LogMsg(VERB_PARANOID,"[GV] end!");


report (index, red, pad, fft, 1, "Potential Folded");

LogMsg(VERB_PARANOID,"[GV] end GravCpu()");
	}





/* hybrid procedure */


	template<class Float>
	void	GraVi::GraHybridCpu	()
	{
		LogMsg(VERB_PARANOID,"[GV] begin GraHybridCpu()");
		void *nada;

		Folder munge(afield);

		/* Energy map in m2Start */
		if (afield->Field() == FIELD_PAXION)
			graviPaxKernelXeon<KIDI_ENE>(afield->mCpu(), afield->vCpu(), nada, afield->m2Cpu(), ppar, BO, V+BO, afield->Precision(), xBlock, yBlock, zBlock);
		else if (afield->Field() == FIELD_AXION)
		{
			void *eRes;
			trackAlloc(&eRes, 256);
			energy(afield,eRes,EN_MAP,0); // energy unfolded starting in m2
			double factor = 1/(((double *) eRes)[TH_KIN] + ((double *) eRes)[TH_POT] + ((double *) eRes)[TH_GRX] + ((double *) eRes)[TH_GRY] + ((double *) eRes)[TH_GRZ]);  
			scaleField(afield,FIELD_M2,factor);
			memmove(afield->m2Start(),afield->m2Cpu(),afield->Size()*afield->Precision());
		}


int red =1;
int pad =0;
report (index, red, pad, pad, 1, "After energy");

		// LogOut("> Calculate and substract average value (index %d)\n",index);
		double eA = mean(afield, afield->m2Start(), V, afield->TotalSize());
		ppar.beta  = -eA;
		// LogOut("> energy Average %f\n",eA);
		graviPaxKernelXeon<KIDI_ADD>(nada, nada, nada, afield->m2Cpu(), ppar, BO, V+BO, afield->Precision(), xBlock, yBlock, zBlock);

LogMsg(VERB_PARANOID,"[GV] eA is %e",eA);
report (index, red, pad, pad, 1, "After contrast");

		/* If field was unfolded, energy in m2 it is as well but we need it folded */
		/* Requires modification for afield */
		if (!afield->Folded()){
			afield->setM2Folded(false);
			munge(FOLD_M2);
		} else {
			afield->setM2Folded(true);
		}

report (index, red, pad, pad, 1, "After folding?");
LogMsg(VERB_PARANOID,"[GV] reserve contrast in m2h folded");

		/* reserve ENERGY-eA to m2half (ghosted)*/
		// LogOut("> copy to m2h \n");
		memmove(afield->m2half(),afield->m2Cpu(),afield->eSize()*afield->Precision());

		/* SOR rho in m2 to smooth ENERGY, diffussion mode */
		// LogOut("> SOR %d times\n",smsteps);
LogMsg(VERB_PARANOID,"[GV] SOR %d steps",smsteps);
		ppar.beta  = 0.0;
		for (int iz = 0; iz < smsteps; iz++)
		{
			afield->sendGhosts(FIELD_M2, COMM_SDRV);
			graviPaxKernelXeon<KIDI_SOR>(nada, nada, afield->m2half(), afield->m2Cpu(), ppar, 2*BO, V   , afield->Precision(), xBlock, yBlock, zBlock);
			afield->sendGhosts(FIELD_M2, COMM_WAIT);
			graviPaxKernelXeon<KIDI_SOR>(nada, nada, afield->m2half(), afield->m2Cpu(), ppar, BO  , 2*BO, afield->Precision(), xBlock, yBlock, zBlock);
			graviPaxKernelXeon<KIDI_SOR>(nada, nada, afield->m2half(), afield->m2Cpu(), ppar, V   , V+BO, afield->Precision(), xBlock, yBlock, zBlock);

if (debug){
munge(UNFOLD_SLICEM2);
report (index, red, pad, pad, 0, "SOR iteration");
}

		}

LogMsg(VERB_PARANOID,"[GV] build reduced grid unfolded and padded");
		/* build reduced grid unfolded and padded */
		// LogOut("> build reduced grid unfolded and padded (index %d)\n",index);
		rfuap(afield, redfft);

red = redfft;
pad = 2;
report (index, red, pad, 0, 0, "After Red+Pad");

LogMsg(VERB_PARANOID,"[GV] pick plan and fft");
	auto &myPlanR = AxionFFT::fetchPlan("m2redFFT");
		/* FFT */
		myPlanR.run(FFT_FWD);

report (index, red, pad, 1, 0, "Reduced FFT");
		/* divide by -k^2 */
LogMsg(VERB_PARANOID,"[GV] ilap");
		ilap(afield,redfft,smsteps);

report (index, red, pad, 1, 0, "FFT after ilap");

		/* Inverse FFT */
LogMsg(VERB_PARANOID,"[GV] iFFT");
		myPlanR.run(FFT_BCK);

report (index, red, pad, 0, 0, "Potential Red+Pad");

		/* expand smoothed-potential grid folded and unpadded */
		// LogOut("> expand grid (index %d)\n",index);
LogMsg(VERB_PARANOID,"[GV] expand grid folded and unpadded");
		eguf(afield,redfft);

red = 1;
pad = 0;
if (debug){
munge(UNFOLD_SLICEM2);
report (index, red, pad, 0, 0, "After Exp+Fold+uPad");
}

		/* SOR rho in m2 to build PHI, Poisson mode*/
		// LogOut("> SOR full grid %d steps (index %d)\n",smstep2, index);
LogMsg(VERB_PARANOID,"[GV] SOR full grid %d steps (index %d)",smstep2, index);
		ppar.beta  = afield->Delta()*afield->Delta();
		// LogOut("> external normalisation %lf\n",ppar.beta);
		for (int iz = 0; iz < smstep2; iz++)
		{
			afield->sendGhosts(FIELD_M2, COMM_SDRV);
			graviPaxKernelXeon<KIDI_SOR>(nada, nada, afield->m2half(), afield->m2Cpu(), ppar, 2*BO, V   , afield->Precision(), xBlock, yBlock, zBlock);
			afield->sendGhosts(FIELD_M2, COMM_WAIT);
			graviPaxKernelXeon<KIDI_SOR>(nada, nada, afield->m2half(), afield->m2Cpu(), ppar, BO  , 2*BO, afield->Precision(), xBlock, yBlock, zBlock);
			graviPaxKernelXeon<KIDI_SOR>(nada, nada, afield->m2half(), afield->m2Cpu(), ppar, V   , V+BO, afield->Precision(), xBlock, yBlock, zBlock);

if (debug){
munge(UNFOLD_SLICEM2);
report (index, red, pad, 0, 0, "SOR smooth");
}
		}

LogMsg(VERB_PARANOID,"[GV] end GraHybridCpu()");

	}




	/* External Function */











	/* Auxiliary functions */



	void	GraVi::rfuap (Scalar *field, const size_t ref)
	{
		switch(field->Precision())
		{
			case FIELD_SINGLE:
			rfuap<float> (field, ref);
			break;
			case FIELD_DOUBLE:
			rfuap<double> (field, ref);
			break;
		}
	}

	template<typename Float>
	void	GraVi::rfuap (Scalar *field, const size_t ref)
	{
		LogMsg (VERB_PARANOID, "[rfuap] Called reducing factor %d ",ref);
		if (field->Device() == DEV_GPU)
			return;
		/* Assumes Field Folded in m2Start
				we want it reduced by factor ref in m2 */

		/* We take values in the vertex of the
				cubes that are reduced to a point
				fundmental cube 0-L,0-L,0-L take point 0,0,0 */
		Float *m2   = static_cast<Float *> ((void *) field->m2Cpu());
		Float *m2S  = static_cast<Float *> ((void *) field->m2Start());

		size_t rLx    = Lx/ref;
		size_t rLz    = Lz/ref;
		if ((ref*rLx != Lx) || (ref*rLz != Lz)) {
			LogError("Non integer reduction factor!");
			exit(0);
		}
		// LogOut(VERB_PARANOID,"")
		/* fidx = iZ*n2 + iiy*shift*Lx +ix*shift +sy
			 idx  = iZ*n2 + [iy+sy*(n1/shift)]*Lx +ix
			 ix   = rix*ref , etc... */

		for (size_t riz=0; riz < rLz; riz++){
			#pragma omp parallel for schedule(static)
			for (size_t riy=0; riy < rLx; riy++)
				for (size_t rix=0; rix < rLx; rix++)
				{
					/* need iy and sy */
					size_t sy  = (ref*riy)/(Lx/shift);
					size_t iiy = ref*riy - (sy*Lx/shift);
					size_t fIdx = ref*(riz*S + rix*shift) + iiy*Lx*shift + sy;
					/* Padded box copy padded */
					size_t uIdx = (rLx+2)*(riz*rLx + riy) + rix;
					// LogOut("%d %d %d\n",rix, riy, riz);
					m2[uIdx]	= m2S[fIdx];
				}
			}

		field->setM2Folded(false);
		LogMsg (VERB_PARANOID, "[rfuap] reduced, unfolded and padded ");

		return;
	}


	void	GraVi::ilap (Scalar *field, const size_t red, const size_t smit)
	{
		switch(field->Precision())
		{
			case FIELD_SINGLE:
			ilap<float> (field, red, smit);
			break;
			case FIELD_DOUBLE:
			ilap<double> (field, red, smit);
			break;
		}
	}

	template<typename Float>
	void	GraVi::ilap (Scalar *field, const size_t ref, const size_t smit)
	{
		LogMsg (VERB_PARANOID, "[ilap] Called reducing factor %d smoothing steps %d",ref, smit);
		/* multiplies modes by 1/k^2 1/N
			and sets mode 0 to 0*/

		std::complex<Float> *m2   = static_cast<std::complex<Float> *> (field->m2Cpu());
		std::complex<Float> im(0,1);

		size_t rLx   = Lx/ref;
		size_t rLz   = Lz/ref;
		size_t rTz   = Tz/ref;

		Float rnorm   = (Float) (-1./(k0*k0*((double) rLx*rLx*rTz)));
LogMsg (VERB_PARANOID, "[ilap] rnorm %e ",rnorm);

		if ((ref*rLx != Lx) || (ref*rLz != Lz)) {
			LogError("Non integer reduction factor!");
			exit(0);
		}
		size_t hrLx      = (rLx >> 1)+1;
		size_t hrLy      = (rLx >> 1);
		size_t hrTz      = (rTz >> 1);
		size_t nModesrhc = rLx*hrLx*rLz;

		// LogOut("ilap k0 %e norma %e zBase %d\n",k0,norm,zBase);

		/* Compute smoothing factor
				the same that appears in propPaxXeon.h */
		double *PC = field->getCO();
		double sum=0.;
		for (int in=0; in < field->getNg(); in++)
			sum += PC[in];
		sum /= 6.;

		/* modes are stored as Idx = kx + kz*hrLx * ky*hrLx*rTz */

		#pragma omp parallel for schedule(static)
		for (size_t idx=0; idx<nModesrhc; idx++) {
			size_t tmp = idx/hrLx; /* kz + ky*rTz */
			int    kx  = idx - tmp*hrLx;
			int    ky  = tmp/rTz;
			int    kz  = tmp - ((size_t) ky)*rTz;
			ky += (int) zBase;

			if (ky > static_cast<int>(hrLy)) ky -= static_cast<int>(rLx);
			if (kz > static_cast<int>(hrTz)) kz -= static_cast<int>(rTz);

			Float k2    = (Float) (kx*kx + ky*ky + kz*kz);
			/* We also shift it by delta/2*/
			// Float phi = (Float) M_PI*( (((double) kx) / ((double) rLx)) + (((double) ky) / ((double) rLx)) + (((double) kz) / ((double) rTz))  ) ;
			Float phix = (Float) M_PI* ( ((Float) (kx)) / ((Float) rLx)) ;
			Float phiy = (Float) M_PI* ( ((Float) (ky)) / ((Float) rLx)) ;
			Float phiz = (Float) M_PI* ( ((Float) (kz)) / ((Float) rTz)) ;
			Float phi = phix+phiy+phiz ;
			/* We can correct for the decrease of long modes by the averager
				cop = (sum) * exp(-p^2 it) in the continuum
				cop = (1 - (sum)p^2)^it in the discrete
				where p^2 = 2(1-cos(kx) + 1-cos(ky) + 1-cos(kz))
				cop correction is avoided for the moment (diverges for smit=2)
			 */
			// Float cop = pow(1 - 4*sum*(pow(sin(phix/ref),2) + pow(sin(phiy/ref),2) + pow(sin(phiz/ref),2)),smit);
			// m2[idx]	*= rnorm/k2/cop *displa;
			std::complex<Float> displa = std::exp(im*phi);
			m2[idx]	*= rnorm/k2*displa;
		}
		/* mode 0 to zero */
		if (commRank() == 0)
			m2[0] = (0,0);


		LogMsg (VERB_PARANOID, "[ilap] prepared ");

		return;
	}


	/* Expand grid unpadded and unfolded */
	void	GraVi::eguf (Scalar *field, const size_t red)
	{
		switch(field->Precision())
		{
			case FIELD_SINGLE:
			eguf<float> (field, red);
			break;
			case FIELD_DOUBLE:
			eguf<double> (field, red);
			break;
		}
	}

	template<typename Float>
	void	GraVi::eguf (Scalar *field, const size_t ref)
	{
		LogMsg (VERB_PARANOID, "[eguf] Called expansion factor %d ",ref);
		if (field->Device() == DEV_GPU)
			return;
		/* Assumes reduced Field (paded) in m2
				we want it expanded, unpadded and folded!
				we undo the approximations done before */

		Float *m2   = static_cast<Float *> ((void *) field->m2Cpu());
		Float *m2S  = static_cast<Float *> ((void *) field->m2Start());

		size_t rLx    = Lx/ref;
		size_t rLz    = Lz/ref;
		if ((ref*rLx != Lx) || (ref*rLz != Lz)) {
			LogError("Non integer reduction factor!");
			exit(0);
		}
		// Float norm = (Float) ref*ref;

		/* each point in a box of ref*ref*ref
		takes the value of the lowest vertex (0,0,0)
		of the reduced box; which has been shifted
		to contain the average of the box */
		if (ref>1){
			for (int iz = Lz-1; iz>=0; iz--){
				size_t su  = iz*Lx*Lx;
				size_t rsu = (iz/ref)*(rLx+2)*rLx;
				#pragma omp parallel for schedule(static) collapse(2)
		 		for (size_t riy=0; riy < rLx; riy++){
		 			for (size_t rix=0; rix < rLx; rix++){
						size_t oIdx = rsu + riy*(rLx+2) + rix;
						Float wui = m2[oIdx];
						for (size_t yy = 0; yy< ref ; yy++ ){
							size_t sy  = (ref*riy + yy)/(Lx/shift);
							size_t iiy = ref*riy + yy - sy*(Lx/shift);
							for (size_t xx = 0; xx< ref ; xx++ ){
								size_t fIdx = su + iiy*Lx*shift + (rix*ref+xx)*shift + sy;
		// if(iz == 0)
		// 	LogOut("%d %d %d (syiiyix %d %d %d)> %lu %lu\n",iz, rix, riy, sy, iiy, rix*ref+xx, oIdx, fIdx);
								m2S[fIdx] = wui;
							}
						}
		 			}
				}
			} // end loop z
		} else {
			/* unpad */
			char *M2  = static_cast<char *> ((void *) field->m2Cpu());
			char *MS  = static_cast<char *> ((void *) field->m2Start());
			size_t dl = field->Precision()*Lx;
			size_t pl = field->Precision()*(Lx+2);
			size_t ss = Lz*Lx;
			for (size_t l =1; l<ss; l++)
				memmove(M2 + l*dl, M2 + l*pl, dl);
			/* ghost */
			memmove(MS, M2, field->Precision()*field->Size());
			/* ghost */
			field->setM2Folded(false);
			Folder munge(field);
			munge(FOLD_M2);
		}

		field->setM2Folded(true);
		LogMsg (VERB_PARANOID, "[eguf] grid expanded, unpadded, and folded ");

		return;
	}

	/* Builds contrast */
	double	GraVi::mean (Scalar *field, void *punt, size_t size, size_t tsize)
	{
		if (field->Device() == DEV_GPU)
			return 0.0;

		double eA =0;
		switch(field->Precision())
		{
			case FIELD_SINGLE:
			eA = mean<float> (punt, size, tsize);
			break;
			case FIELD_DOUBLE:
			eA = mean<double> (punt, size, tsize);
			break;
		}
		return eA;
	}

	template<typename Float>
	double	GraVi::mean (void *punt, size_t size, size_t tsize)
	{
			double eA = 0, eAl = 0;

			#pragma omp parallel for schedule(static) reduction(+:eA)
			for (size_t idx=0; idx < size; idx++)
				eA += (double) static_cast<Float*>(punt)[idx];

			MPI_Allreduce (&eA, &eAl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			eA /= (double) tsize;

		LogMsg (VERB_PARANOID, "[bcon] contrast build, eA = %e ",eA);

		return eA;
	}

	/* Normalise fields to have <|cpax|^2>=1 */
	void	GraVi::normaliseFields ()
	{
		if (afield->Device() == DEV_GPU)
			return ;

		double eA =0;
		switch(afield->Precision())
		{
			case FIELD_SINGLE:
		 	normaliseFields<float> ();
			break;
			case FIELD_DOUBLE:
			normaliseFields<double> ();
			break;
		}
		return ;
	}

	template<typename Float>
	void	GraVi::normaliseFields ()
	{
			double eA = 0, eAl = 0;

			#pragma omp parallel for schedule(static) reduction(+:eA)
			for (size_t idx=0; idx < V; idx++){
				eA += pow((double) static_cast<Float*>(afield->mStart())[idx],2);
				eA += pow((double) static_cast<Float*>(afield->vStart())[idx],2);
			}


			MPI_Allreduce (&eA, &eAl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			eA /= (double) (afield->TotalSize());

			double nn = 1./std::sqrt(eA);

			#pragma omp parallel for schedule(static)
			for (size_t idx=0; idx < V; idx++){
				static_cast<Float*>(afield->mStart())[idx] *= nn ;
				static_cast<Float*>(afield->vStart())[idx] *= nn ;
			}

		LogMsg (VERB_PARANOID, "[Norm] normalised Field to have <|cpax|^2>=1, eA = %e ",eA);
		IcData ics = afield->BckGnd()->ICData();
		ics.beta *= eA;
		afield->BckGnd()->SetICData(ics);
		LogMsg (VERB_PARANOID, "[Norm] Self-coupling beta renormalised to beta*eA^2 = %e ",afield->BckGnd()->ICData().beta);
		return ;
	}


	void	GraVi::report (int inde, int red, int pad, int fft, int slice, char *name)
	{
		if (debug){
LogMsg (VERB_PARANOID, "[GV] Report index %d red %d pad %d fft %d slice %d Message %s",
																		inde, red, pad, fft, slice, name);
		createMeas(afield, inde);
			writeAttribute(&red, "Red", H5T_NATIVE_INT);
			writeAttribute(&pad, "Pad", H5T_NATIVE_INT);
			writeAttribute(&fft, "fft", H5T_NATIVE_INT);
			writeAttribute(name, "Message", H5T_C_S1);
			writeEMapHdf5s (afield,slice);
			writeEDens (afield,MAP_M2S);
		destroyMeas();
		index++;
		}
	}

#endif
