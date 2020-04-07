#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "propagator/allProp.h"
#include "energy/energy.h"
#include "utils/utils.h"
#include "io/readWrite.h"
#include "comms/comms.h"
#include "map/map.h"
#include "strings/strings.h"
#include "powerCpu.h"
#include "scalar/scalar.h"
#include "spectrum/spectrum.h"
#include "meas/measa.h"

#include"fft/fftCode.h"
#include "gravity/gravityPaxionXeon.h"

using namespace std;

// void	ufFm2 (Scalar *field, const size_t sZ);
// template<typename Float>
// void	ufFm2 (Scalar *field, const size_t sZ);

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

int	main (int argc, char *argv[])
{

	double zendWKB = 10. ;
	Cosmos myCosmos = initAxions(argc, argv);

	if (nSteps==0)
	return 0 ;

	//--------------------------------------------------
	//       AUX STUFF
	//--------------------------------------------------


	double  *binarray	 ;
	trackAlloc((void**) (&binarray),  10000*sizeof(size_t));
	double *bA = static_cast<double *> (binarray);
	size_t sliceprint = 0 ; // sizeN/2;



	commSync();
	LogOut("\n-------------------------------------------------\n");
	LogOut("\n Worksheet for gravitational potential solver    \n");
	LogOut("\n-------------------------------------------------\n");

	LogOut("\n-------------------------------------------------\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	Scalar *axion;

	LogOut ("reading conf %d ...", fIndex);
	readConf(&myCosmos, &axion, fIndex);
	if (axion == NULL)
	{
		LogOut ("Error reading HDF5 file\n");
		exit (0);
	}
	LogOut ("\n");


	double z_now = (*axion->zV())	;
	LogOut("--------------------------------------------------\n");
	LogOut("           READ CONDITIONS                     \n\n");

	LogOut("Length =  %2.2f\n", myCosmos.PhysSize());
	LogOut("nQCD   =  %2.2f\n", myCosmos.QcdExp());
	LogOut("N      =  %ld\n",   axion->Length());
	LogOut("Nz     =  %ld\n",   axion->Depth());
	LogOut("zGrid  =  %ld\n",   zGrid);
	LogOut("z      =  %2.2f\n", z_now);
	LogOut("zthr   =  %3.3f\n", myCosmos.ZThRes());
	LogOut("zres   =  %3.3f\n", myCosmos.ZRestore());
	LogOut("mass   =  %3.3f\n\n", axion->AxionMass());

	if (axion->Precision() == FIELD_SINGLE)
		LogOut("precis = SINGLE(%d)\n",FIELD_SINGLE);
	else
		LogOut("precis = DOUBLE(%d)\n",FIELD_DOUBLE);

	LogOut("--------------------------------------------------\n");

	//--------------------------------------------------
	//       MEASUREMENT
	//--------------------------------------------------
	//- Measurement
	MeasData lm;
	//- number of plaquetes pierced by strings
	lm.str.strDen = 0 ;
	//- Info to measurement
	MeasInfo ninfa = deninfa;
	ninfa.index = fIndex;
	ninfa.redmap = endredmap;


	int counter = 0;
	int index ;
	double dzaux;
	int i_meas = 0;
	bool measrightnow = false;

	ninfa.index=index;
	// ninfa.measdata |= MEAS_3DMAP;
	// lm = Measureme (axion, ninfa);
	// ninfa.measdata ^= MEAS_3DMAP;

	/* Works in PAXION (and perhaps in AXION) */
	LogOut("> Loading Paxion \n");
	if (axion->Field() == FIELD_AXION)
		thetaToPaxion(axion);

	LogOut("> Paxion loaded \n");

	//--------------------------------------------------
	//       MEASUREMENT
	//--------------------------------------------------

	size_t redfft  = axion->BckGnd()->ICData().kMax;
	size_t smsteps = redfft*redfft;
	size_t smstep2 = axion->BckGnd()->ICData().siter;

	FILE *cacheFile = nullptr;
	if (((cacheFile  = fopen("./red.dat", "r")) == nullptr)){
		LogMsg(VERB_NORMAL,"No red.dat file use defaults kmax, kmax**2, siter");
	} else {
		fscanf (cacheFile ,"%lu ", &redfft);
		fscanf (cacheFile ,"%lu ", &smsteps);
		fscanf (cacheFile ,"%lu ", &smstep2);
		LogOut("[gravi] red.dat file used \n");
		LogOut("        redfft %d \n", redfft);
		LogOut("        smsteps %d \n",smsteps);
		LogOut("        smstep2 %d \n",smstep2);
	}

	LogOut("> Plan in m2 reduced %d\n",redfft);
	AxionFFT::initPlan (axion, FFT_PSPEC_AX,  FFT_FWDBCK, "m2redFFT",redfft);
	auto &myPlan = AxionFFT::fetchPlan("m2redFFT");

	/* 1 - compute energy folded in m2h
		 PAXION is simply M^2+V^2 times some constant which we add later
		 SAXION, AXION, PROBLEM -> ONLY NON-RELATIVISTIC IS TRIVIAL! */
	LogOut("> Compute energy in m2_Start \n");

	PropParms ppar;
		ppar.Ng    = axion->getNg();
		ppar.ood2a = 1.0;
		ppar.PC    = axion->getCO();
		ppar.Lx    = axion->Length();
		ppar.Lz    = axion->Depth();
		size_t BO  = ppar.Ng*ppar.Lx*ppar.Lx;
		size_t V   = axion->Size();

	size_t xBlock, yBlock, zBlock;
	int tmp   = axion->DataAlign()/axion->DataSize();
	int shift = 0;
		while (tmp != 1) {
		shift++;
		tmp >>= 1;
	}

	xBlock = ppar.Lx << shift;
	yBlock = ppar.Lx >> shift;
	zBlock = ppar.Lz;

	Folder munge(axion);

	if (axion->Folded())
		munge(UNFOLD_ALL);

	void *nada;
	/*Energy map in m2Start*/
	graviPaxKernelXeon<KIDI_ENE>(axion->mCpu(), axion->vCpu(), nada, axion->m2Cpu(), ppar, BO, V+BO, axion->Precision(), xBlock, yBlock, zBlock);

	LogOut("> Calculate and substract average value (index %d)\n",index);
	double eA = mean(axion, axion->m2Start(), V, axion->TotalSize());
	ppar.beta  = -eA;
	LogOut("> energy Average %f\n",eA);
	graviPaxKernelXeon<KIDI_ADD>(nada, nada, nada, axion->m2Cpu(), ppar, BO, V+BO, axion->Precision(), xBlock, yBlock, zBlock);

	axion->setM2(M2_ENERGY);
	createMeas(axion, index);
		/* First slice of energy */
		writeEMapHdf5s (axion,ppar.Ng);
	destroyMeas();
	index++;

	LogOut("> Fold \n");
	munge(FOLD_M2);

	/* reserve ENERGY to m2half (ghosted)*/
	LogOut("> copy to m2h \n");
	memmove(axion->m2half(),axion->m2Cpu(),axion->eSize()*axion->Precision());


	/* SOR rho in m2 to smooth ENERGY, diffussion mode */
	LogOut("> SOR %d times\n",smsteps);
	ppar.beta  = 0.0;
	for (int iz = 0; iz < smsteps; iz++)
	{
		axion->sendGhosts(FIELD_M2, COMM_SDRV);
		graviPaxKernelXeon<KIDI_SOR>(nada, nada, axion->m2half(), axion->m2Cpu(), ppar, 2*BO, V   , axion->Precision(), xBlock, yBlock, zBlock);
		axion->sendGhosts(FIELD_M2, COMM_WAIT);
		graviPaxKernelXeon<KIDI_SOR>(nada, nada, axion->m2half(), axion->m2Cpu(), ppar, BO  , 2*BO, axion->Precision(), xBlock, yBlock, zBlock);
		graviPaxKernelXeon<KIDI_SOR>(nada, nada, axion->m2half(), axion->m2Cpu(), ppar, V   , V+BO, axion->Precision(), xBlock, yBlock, zBlock);

		munge(UNFOLD_SLICEM2,0);
		createMeas(axion, index);
			writeEMapHdf5s (axion,0);
		destroyMeas();
		index++;
	}

	/* build reduced grid unfolded and padded */
	LogOut("> build reduced grid unfolded and padded (index %d)\n",index);
	rfuap(axion, redfft);
	createMeas(axion, index);
		writeEMapHdf5s (axion,0);
	destroyMeas();
	index++;

	/* FFT */
	LogOut("> fft \n");
	myPlan.run(FFT_FWD);

	/* divide by k*2 */
	LogOut("> inverse laplacian \n");
	ilap(axion,redfft,smsteps);

	/* Inverse FFT */
	LogOut("> ifft (index %d)\n",index);
	myPlan.run(FFT_BCK);

	createMeas(axion, index);
		writeEMapHdf5s (axion,0);
	destroyMeas();
	index++;


	/* expand smoothed-potential grid folded and unpadded
	 	also renormalise by the gradient factor redfft**2 > no*/
	LogOut("> expand grid (index %d)\n",index);
	eguf(axion,redfft);
	munge(UNFOLD_SLICEM2,0);
	createMeas(axion, index);
		writeEMapHdf5s (axion,0); //5
	destroyMeas();
	index++;

	/* SOR rho in m2 to build PHI, Poisson mode*/
	LogOut("> SOR full grid %d steps (index %d)\n",smstep2, index);
	ppar.beta  = axion->Delta()*axion->Delta();
	LogOut("> external normalisation %lf\n",ppar.beta);
	// ppar.beta  = 1;
	for (int iz = 0; iz < smstep2; iz++)
	{
		axion->sendGhosts(FIELD_M2, COMM_SDRV);
		graviPaxKernelXeon<KIDI_SOR>(nada, nada, axion->m2half(), axion->m2Cpu(), ppar, 2*BO, V   , axion->Precision(), xBlock, yBlock, zBlock);
		axion->sendGhosts(FIELD_M2, COMM_WAIT);
		graviPaxKernelXeon<KIDI_SOR>(nada, nada, axion->m2half(), axion->m2Cpu(), ppar, BO  , 2*BO, axion->Precision(), xBlock, yBlock, zBlock);
		graviPaxKernelXeon<KIDI_SOR>(nada, nada, axion->m2half(), axion->m2Cpu(), ppar, V   , V+BO, axion->Precision(), xBlock, yBlock, zBlock);

		munge(UNFOLD_SLICEM2,0);
		createMeas(axion, index);
			writeEMapHdf5s (axion,0);
		destroyMeas();
		index++;
		/* condition to stop? */
	}


	/* Finally builds full solution to compare*/

	{
		if (axion->Folded())
			munge(UNFOLD_ALL);
		/*Energy map in m2Start*/
		graviPaxKernelXeon<KIDI_ENE>(axion->mCpu(), axion->vCpu(), nada, axion->m2half(), ppar, BO, V+BO, axion->Precision(), xBlock, yBlock, zBlock);
		LogOut("> Ralculate full thing again (index %d)\n",index);
		char *MS  = static_cast<char *> ((void *) axion->m2half()) + axion->getNg()*axion->Surf()*axion->Precision();
		double eA = mean(axion, static_cast<void *>(MS), V, axion->TotalSize());
		ppar.beta  = -eA;
		LogOut("> energy Average %f\n",eA);
		graviPaxKernelXeon<KIDI_ADD>(nada, nada, nada, axion->m2half(), ppar, BO, V+BO, axion->Precision(), xBlock, yBlock, zBlock);
		/*pad into m2*/
		size_t Lx    = axion->Length();
		size_t Lz    = axion->Depth();
		char *M2  = static_cast<char *> ((void *) axion->m2Cpu());
		// char *MS  = static_cast<char *> ((void *) axion->m2half())+axion->getNg()*axion->Surf()*axion->Precision();
		size_t dl = axion->Precision()*Lx;
		size_t pl = axion->Precision()*(Lx+2);
		size_t ss = Lz*Lx;
		for (size_t l =0; l<ss; l++)
			memmove(M2 + l*pl, MS + l*dl, dl);

		auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
		/* fft */
		myPlan.run(FFT_FWD);
		/* ilap */
		{
			std::complex<float> *m2   = static_cast<complex<float> *> (axion->m2Cpu());
			size_t Lx    = axion->Length();
			size_t Lz    = axion->Depth();
			size_t Tz    = axion->TotalDepth();
			double k0     = (double) (2*M_PI/axion->BckGnd()->PhysSize());
			double norm   = (double) (-1./(k0*k0*((double) Lx*Lx*Tz)));
			size_t hLx   = (Lx >> 1)+1;
			size_t hLy   = (Lx >> 1);
			size_t hTz   = (Tz >> 1);
			size_t nModeshc = Lx*hLx*Lz;
			size_t	zBase = (Lx/commSize())*commRank();
			#pragma omp parallel for schedule(static)
			for (size_t idx=0; idx<nModeshc; idx++) {
				size_t tmp = idx/hLx; /* kz + ky*rTz */
				int    kx  = idx - tmp*hLx;
				int    ky  = tmp/Tz;
				int    kz  = tmp - ((size_t) ky)*Tz;
				kz += (int) zBase;
				if (ky > static_cast<int>(hLy)) ky -= static_cast<int>(Lx);
				if (kz > static_cast<int>(hTz)) kz -= static_cast<int>(Tz);
				float k2    = (float) (kx*kx + ky*ky + kz*kz);
				m2[idx]	*= norm/k2;
			}
			/* mode 0 to zero */
			if (commRank() == 0)
				m2[0] = (0,0);
		} //end ilap
		myPlan.run(FFT_BCK);
		/* unpad */
		MS  = static_cast<char *> ((void *) axion->m2Start());
		for (size_t l =0; l<ss; l++)
			memmove(M2 + l*dl, M2 + l*pl, dl);

		createMeas(axion, index);
			writeEMapHdf5s (axion,0);
		destroyMeas();
		index++;
	}




	//--------------------------------------------------
	//       SAVE DATA
	//--------------------------------------------------

	LogOut ("Done!\n\n");

	endAxions();

	return 0;
}


//--------------------------------------------------
//       AUX FUNCTIONS
//--------------------------------------------------




void	rfuap (Scalar *field, const size_t ref)
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
void	rfuap (Scalar *field, const size_t ref)
{
	if (field->Device() == DEV_GPU)
		return;
	/* Assumes Field Folded in m2Start
			we want it reduced to newN in m2
			only works for n1/newN = integer */

	/* We take values in the vertex of the
			cubes that are reduced to a point
			example red 2
			fundmental cube 0,0,0 take point 0,0,0 */
	Float *m2   = static_cast<Float *> ((void *) field->m2Cpu());
	Float *m2S  = static_cast<Float *> ((void *) field->m2Start());

	size_t fSize = field->DataSize();
	size_t shift = field->DataAlign()/fSize;
	size_t Lx    = field->Length();
	size_t Lz    = field->Depth();
	size_t S     = field->Surf();

	size_t rLx    = Lx/ref;
	size_t rLz    = Lz/ref;
	if ((ref*rLx != Lx) || (ref*rLz != Lz)) {
		LogError("Non integer reduction factor!");
		exit(0);
	}
	// LogOut(VERB_HIGH,"")
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
				size_t iiy = ref*riy - sy*Lx/shift;
				size_t fIdx = ref*(riz*S + rix*shift) + iiy*Lx*shift + sy;
				/* Padded box copy padded */
				size_t uIdx = (rLx+2)*(riz*rLx + riy) + rix;
				// LogOut("%d %d %d\n",rix, riy, riz);
				m2[uIdx]	= m2S[fIdx];
			}
		}

	LogMsg (VERB_HIGH, "[rfuap] unfolded, reduced and padded ");

	return;
}


void	ilap (Scalar *field, const size_t red, const size_t smit)
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
void	ilap (Scalar *field, const size_t ref, const size_t smit)
{

	/* multiplies modes by 1/k^2 1/N
		and sets mode 0 to 0*/

	std::complex<Float> *m2   = static_cast<complex<Float> *> (field->m2Cpu());
	std::complex<Float> im(0,1);

	size_t Lx    = field->Length();
	size_t Lz    = field->Depth();
	size_t rLx   = Lx/ref;
	size_t rLz   = Lz/ref;
	size_t rTz   = field->TotalDepth()/ref;

	Float k0     = (Float) (2*M_PI/field->BckGnd()->PhysSize());
	Float norm   = (Float) (-1./(k0*k0*((Float) rLx*rLx*rTz)));



	if ((ref*rLx != Lx) || (ref*rLz != Lz)) {
		LogError("Non integer reduction factor!");
		exit(0);
	}
	size_t hrLx   = (rLx >> 1)+1;
	size_t hrLy   = (rLx >> 1);
	size_t hrTz   = (rTz >> 1);
	size_t nModeshc = rLx*hrLx*rLz;
	size_t	zBase = (rLx/commSize())*commRank();


	// LogOut("ilap k0 %e norma %e zBase %d\n",k0,norm,zBase);

	/* Compute smoothing factor; FIX ME assumes N=1 */
	double *PC = field->getCO();
	double sum=0.;
	for (int in=0; in < field->getNg(); in++)
		sum += PC[in];
	sum /= 6.;

	/* modes are stored as
	idx = kx + ky*hrLx * kz*hrLx*rTz */
	#pragma omp parallel for schedule(static)
	for (size_t idx=0; idx<nModeshc; idx++) {
		size_t tmp = idx/hrLx; /* kz + ky*rTz */
		int    kx  = idx - tmp*hrLx;
		int    ky  = tmp/rTz;
		int    kz  = tmp - ((size_t) ky)*rTz;
		kz += (int) zBase;

		if (ky > static_cast<int>(hrLy)) ky -= static_cast<int>(rLx);
		if (kz > static_cast<int>(hrTz)) kz -= static_cast<int>(rTz);

		Float k2    = (Float) (kx*kx + ky*ky + kz*kz);
		/* We also shift it by delta/2*/
		// Float phi = (Float) M_PI*( (((double) kx) / ((double) rLx)) + (((double) ky) / ((double) rLx)) + (((double) kz) / ((double) rTz))  ) ;
		Float phix = (Float) M_PI* ( ((Float) (kx)) / ((Float) rLx)) ;
		Float phiy = (Float) M_PI* ( ((Float) (ky)) / ((Float) rLx)) ;
		Float phiz = (Float) M_PI* ( ((Float) (kz)) / ((Float) rTz)) ;
		Float phi = phix+phiy+phiz ;
		/* We also correct for the too strong averaging of the smoother
			assuming each more is damped like exp(-p^2 it)
			where p2 = 1-cos(kx) + 1-cos(ky) + 1-cos(kz) or so*/
		Float cop = pow(1 - 4*sum*(pow(sin(phix/ref),2) + pow(sin(phiy/ref),2) + pow(sin(phiz/ref),2)),smit);
		std::complex<Float> displa = std::exp(im*phi);
		m2[idx]	*= norm/k2/cop *displa;
		// LogOut("%d %d %d m2 %e %e displa %e %e > ",kx, ky, kz, m2[idx].real(),m2[idx].imag(),displa.real(), displa.imag());
		// m2[idx]	*= norm *displa;
		// LogOut(" %e %e\n",m2[idx].real(),m2[idx].imag());

	}

	/* mode 0 to zero */
	if (commRank() == 0)
		m2[0] = (0,0);


	LogMsg (VERB_HIGH, "[ilap] prepared ");

	return;
}


/* Expand grid unpadded and unfolded */
void	eguf (Scalar *field, const size_t red)
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
void	eguf (Scalar *field, const size_t ref)
{

	if (field->Device() == DEV_GPU)
		return;
	/* Assumes reduced Field (paded) in m2
			we want it expanded, unpadded and folded!
			we undo the approximations done before */

	Float *m2   = static_cast<Float *> ((void *) field->m2Cpu());
	Float *m2S  = static_cast<Float *> ((void *) field->m2Start());

	size_t fSize = field->DataSize();
	size_t shift = field->DataAlign()/fSize;
	size_t Lx    = field->Length();
	size_t Lz    = field->Depth();

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
						size_t iiy = ref*riy + yy - sy*Lx/shift;
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
		LogOut("1");
		char *M2  = static_cast<char *> ((void *) field->m2Cpu());
		char *MS  = static_cast<char *> ((void *) field->m2Start());
		size_t dl = field->Precision()*Lx;
		size_t pl = field->Precision()*(Lx+2);
		size_t ss = Lz*Lx;
		for (size_t l =1; l<ss; l++)
			memmove(M2 + l*dl, M2 + l*pl, dl);
		/* ghost */
		LogOut("2");
		memmove(MS, M2, field->Precision()*field->Size());
		/* ghost */
		LogOut("3");
		field->setM2Folded(false);
		Folder munge(field);
		munge(FOLD_M2);


		// Float *m2h  = static_cast<Float *> ((void *) field->m2half());
		// char *m2c  = static_cast<char *> ((void *) field->m2Start());
		// for (int iz = Lz-1; iz>=0; iz--){
		// 	size_t su  = iz*Lx*Lx;
		// 	size_t rsu = (iz/ref)*(rLx+2)*rLx;
		// 	#pragma omp parallel for schedule(static) collapse(2)
	 	// 	for (size_t riy=0; riy < rLx; riy++){
	 	// 		for (size_t rix=0; rix < rLx; rix++){
		// 			size_t oIdx = rsu + riy*(rLx+2) + rix;
		// 			Float wui = m2[oIdx];
		// 			for (size_t yy = 0; yy< ref ; yy++ ){
		// 				size_t sy  = (ref*riy + yy)/(Lx/shift);
		// 				size_t iiy = ref*riy + yy - sy*Lx/shift;
		// 				for (size_t xx = 0; xx< ref ; xx++ ){
		// 					// size_t fIdx = su + iiy*Lx*shift + (rix*ref+xx)*shift + sy;
		// 					size_t fIdx = iiy*Lx*shift + (rix*ref+xx)*shift + sy; // no z
		// 					m2h[fIdx] = wui;
		// 				}
		// 			}
	 	// 		}
		// 	}
		// 	// move slice
		// 	memmove(m2c + su*field->Precision(), m2h, field->Surf()*field->Precision());
		// } // end loop z

	}
	field->setM2Folded(true);
	LogMsg (VERB_HIGH, "[rfuap] unfolded, reduced and padded ");

	return;
}

/* Builds contrast */
double	mean (Scalar *field, void *punt, size_t size, size_t tsize)
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
double	mean (void *punt, size_t size, size_t tsize)
{
		double eA = 0, eAl = 0;

		#pragma omp parallel for schedule(static) reduction(+:eA)
		for (size_t idx=0; idx < size; idx++)
			eA += (double) static_cast<Float*>(punt)[idx];

		MPI_Allreduce (&eA, &eAl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		eA /= (double) tsize;

		// #pragma omp parallel for schedule(static)
		// for (size_t idx=0; idx < axion->Size(); idx++)
		// 	static_cast<Float*>(axion->m2_Start())[idx] -= (Float) eA;

	LogMsg (VERB_HIGH, "[bcon] contrast build, eA = %e ",eA);

	return eA;
}
