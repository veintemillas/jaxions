#include<cstdlib>
#include<cstring>
#include<complex>
#include "comms/comms.h"

#ifdef	USE_GPU
	#include<cuda.h>
	#include<cuda_runtime.h>
	#include "cudaErrors.h"
#endif

#include"scalar/folder.h"
#include"scalar/fourier.h"
#include "fft/fftCode.h"
#include "scalar/scaleField.h"
#include"utils/utils.h"

using namespace std;

	FTfield::FTfield(Scalar *scalar) : field(scalar), Lz(scalar->Depth()), n1(scalar->Length()), n2(scalar->Surf()), n3(scalar->Size()), N(scalar->TotalSize())
{
}

/* used to convert configuration into Fourier space (transposed)*/
/* in case of axions it also pads or unpads and shifts by ghosts */
void	FTfield::ftField(FieldIndex mvomv)
{
	if (field->Device() == DEV_GPU )
		return;

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}
	double scale = 1.0/((double) N);

	LogMsg (VERB_NORMAL, "[ftField] (type=%d) (prec=%d) (scale=%e)", field->Field(), field->Precision(),scale);

	switch(field->Field()){
		case FIELD_SAXION:
		{
			if (mvomv & FIELD_M)
				{
					if (field->MMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("C2CM2M");
					myPlan.run(FFT_FWD);
					scaleField	(field, FIELD_M, scale );
					field->setMMomSpace(true);
					LogMsg  (VERB_HIGH, "[Ftfield] complex M transformed into momentum space ");
				}
			if (mvomv & FIELD_V)
				{
					if (field->VMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("C2CV2V");
					myPlan.run(FFT_FWD);
					scaleField	(field, FIELD_V, scale);
					field->setVMomSpace(true);
					LogMsg  (VERB_HIGH, "[Ftfield] complex V transformed into momentum space ");
				}
			if (mvomv & FIELD_M2TOM2)
				{
					auto &myPlan = AxionFFT::fetchPlan("C2CM22M2");
					myPlan.run(FFT_FWD);
					scaleField	(field, FIELD_M2, scale);
					LogMsg  (VERB_HIGH, "[Ftfield] complex M2 transformed into momentum space ");
				}

		} break;
		case FIELD_AXION:
		{
			if (mvomv & FIELD_M)
				{
					if (field->MMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("R2CM22M");
					/* we assume the field is ghosted and unpadded
					and we want its FT at m (unghosted)
					first we pad into M2, then we FT into M */
					padtom2(field->mStart());
					myPlan.run(FFT_FWD);
					scaleField	(field, FIELD_M, scale);
					field->setMMomSpace(true);
					LogMsg  (VERB_HIGH, "[Ftfield] real M transformed into momentum space ");
				}
			if (mvomv & FIELD_V)
				{
					if (field->VMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("R2CM22V");
					/* we assume the field is unghosted unpadded
					and we want its FT at v (unghosted)
					first we pad into M2, then we FT into V*/
					padtom2(field->vCpu());
					myPlan.run(FFT_FWD);
					scaleField	(field, FIELD_V, scale);
					field->setVMomSpace(true);
					LogMsg  (VERB_HIGH, "[Ftfield] real V transformed into momentum space ");
				}
			if (mvomv & FIELD_M2TOM2)
				{
					/* we assume the field is unghosted but padded in m2!!
					and we want its FT at m2 (unghosted)
					so we can directly use the FFT*/
					auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
					myPlan.run(FFT_FWD);
					scaleField	(field, FIELD_M2, scale);
					LogMsg  (VERB_HIGH, "[Ftfield] real M2 transformed into momentum space ");
				}

		}
		default:
		return;
		break;
	}

	LogMsg (VERB_HIGH, "[FTField] Field transformed to momentum space and divided by 1/N");

	return;
}

/* used to convert Fourier space into configuration space (transposed in) */
void	FTfield::iftField(FieldIndex mvomv)
{
	if (field->Device() == DEV_GPU)
		return;

	LogMsg (VERB_NORMAL, "Calling inverse-ftField (type=%d) (prec=%d)", field->Field(), field->Precision());

	switch(field->Field()){
		case FIELD_SAXION:
		{
			if (mvomv & FIELD_M)
				{
					if (!field->MMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("C2CM2M");
					myPlan.run(FFT_BCK);
					field->setMMomSpace(false);
					LogMsg  (VERB_HIGH, "[Ftfield] complex M transformed into position space ");
				}
			if (mvomv & FIELD_V)
				{
					if (!field->VMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("C2CV2V");
					myPlan.run(FFT_BCK);
					field->setVMomSpace(false);
					LogMsg  (VERB_HIGH, "[Ftfield] complex V transformed into position space ");
				}
			if (mvomv & FIELD_MTOM2)
				{
					if (!field->MMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("C2CM22M");
					myPlan.run(FFT_BCK);
					// field->setM2(false);
					LogMsg  (VERB_HIGH, "[Ftfield] complex M transformed into position space >> in M2 !!");
				}
		} break;
		case FIELD_AXION:
		{
			if (mvomv & FIELD_M)
				{
					if (!field->MMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("R2CM22M");
					/* Here we asume that FT(m) is unghosted in m
					and we want m ghosted and unpadded in m
					first we FFT into m2 and then we unpad into mStart*/
					myPlan.run(FFT_BCK);
					unpadfromm2( field->mStart() );
					field->setMMomSpace(false);
					LogMsg  (VERB_HIGH, "[Ftfield] real M transformed into position space");
				}
			if (mvomv & FIELD_V)
				{
					if (!field->VMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("R2CM22V");
					/* Here we asume that FT(m) is unghosted in m
					and we want v ghosted and unpadded in v
					first we FFT into m2 and then we unpad into vCpu*/
					myPlan.run(FFT_BCK);
					unpadfromm2( field->vCpu() );
					field->setVMomSpace(false);
					LogMsg  (VERB_HIGH, "[Ftfield] real V transformed into position space");
				}
			if (mvomv & FIELD_MTOM2)
				{
					/* Here we asume that FT(m) is unghosted in m
					and we want m in m2 (to compute accelerations)
					so we do it directly */
					if (!field->MMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("R2CM22M");
					myPlan.run(FFT_BCK);
					// field->setM2(false);
					LogMsg  (VERB_HIGH, "[Ftfield] real M transformed into position space >> in M2 !! ");
				}

		}
		default:
		return;
		break;
	}

	LogMsg (VERB_HIGH, "[FTField] Field (inverse) transformed to configuration space");
	field->setFolded(false);

	return;
}

void	FTfield::padtom2(void* point)
{
	switch(field->Field()){
		case FIELD_SAXION:
		LogMsg  (VERB_NORMAL, "[p2m2] case not coded! ");
		return;

		case FIELD_AXION:
		char *de = static_cast<char *> (point);
		char *m2 = static_cast<char *>(field->m2Cpu());
		size_t dl = field->Length()*field->Precision();
		size_t pl = (field->Length()+2)*field->Precision();
		size_t ss	= field->Length()*field->Depth();

// float *mmm = static_cast<float *>(point);
// float *mm2 = static_cast<float *>(field->m2Cpu());
// LogOut("[p2m2] m  values %.2e %.2e %.2e %.2e \n",mmm[0],mmm[1],mmm[2],mmm[3]);
// LogOut("[p2m2] 2  values %.2e %.2e %.2e %.2e \n",mm2[0],mm2[1],mm2[2],mm2[3]);
		for (size_t sl=0; sl<ss; sl++) {
			size_t	oOff = sl*dl;
			size_t	fOff = sl*pl;
			memmove	(m2+fOff, de+oOff, dl);
			}
		LogMsg  (VERB_HIGH, "[p2m2] m or v padded into m2! ");
// LogOut("[p2m2] m  values %.2e %.2e %.2e %.2e \n",mmm[0],mmm[1],mmm[2],mmm[3]);
// LogOut("[p2m2] 2  values %.2e %.2e %.2e %.2e \n",mm2[0],mm2[1],mm2[2],mm2[3]);
		return;

	}
}

void	FTfield::unpadfromm2(void* point)
{
	switch(field->Field()){
		case FIELD_SAXION:
		LogMsg  (VERB_NORMAL, "[m22p] case not coded! ");
		return;

		case FIELD_AXION:
		char *de = static_cast<char *> (point);
		char *m2 = static_cast<char *>(field->m2Cpu());
		size_t dl = field->Length()*field->Precision();
		size_t pl = (field->Length()+2)*field->Precision();
		size_t ss	= field->Length()*field->Depth();
// float *mmm = static_cast<float *>(point);
// float *mm2 = static_cast<float *>(field->m2Cpu());
// LogOut("[m22p] m  values %.2e %.2e %.2e %.2e \n",mmm[0],mmm[1],mmm[2],mmm[3]);
// LogOut("[m22p] 2  values %.2e %.2e %.2e %.2e \n",mm2[0],mm2[1],mm2[2],mm2[3]);

		for (size_t sl=0; sl<ss; sl++) {
			size_t	oOff = sl*dl;
			size_t	fOff = sl*pl;
			memmove	(de+oOff, m2+fOff, dl);
			}
		LogMsg  (VERB_HIGH, "[m22p] m or v unpadded from m2! ");
// LogOut("[m22p] m  values %.2e %.2e %.2e %.2e \n",mmm[0],mmm[1],mmm[2],mmm[3]);
// LogOut("[m22p] 2  values %.2e %.2e %.2e %.2e \n",mm2[0],mm2[1],mm2[2],mm2[3]);
		return;

	}
}

void	FTfield::operator()(FieldIndex mvomv, FFTdir dir)
{
	// Careful here, GPUS might want to call CPU routines
	if (field->Device() == DEV_GPU)
		return;

	LogMsg  (VERB_HIGH, "[Ftfield] Called ");
	profiler::Profiler &prof = profiler::getProfiler(PROF_FTFIELD);

	prof.start();

	switch(dir)
	{
		case FFT_FWD:
			LogMsg  (VERB_HIGH, "[Ftfield] ... to momentum space ");
			ftField(mvomv);
		break;
		case FFT_BCK:
			LogMsg  (VERB_HIGH, "[Ftfield] ... to position space ");
			iftField(mvomv);
		break;
		default:
		return;
	}

	prof.stop();

	// prof.add(Name(), GFlops(), GBytes());	// In truth is x4 because we move data to the ghost slices before folding/unfolding
	//
	// LogMsg  (VERB_HIGH, "Folder %s reporting %lf GFlops %lf GBytes", Name().c_str(), prof.Prof()[Name()].GFlops(), prof.Prof()[Name()].GBytes());
	//
	// reset();
}
