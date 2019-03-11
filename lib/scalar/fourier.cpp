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

/* used to convert configuration into Fourier space */
void	FTfield::ftField(FieldIndex mvomv)
{
	if (field->Device() == DEV_GPU )
		return;

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	LogMsg (VERB_NORMAL, "Calling ftField (type=%f) (prec=%d)", field->Field(), field->Precision());

	switch(field->Field()){
		case FIELD_SAXION:
		{
			if (mvomv & FIELD_M)
				{
					if (field->MMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("C2CM2M");
					myPlan.run(FFT_BCK);
					scaleField	(field, FIELD_M, 1/ ((double) N) );
					field->setMMomSpace(true);
				}
			if (mvomv & FIELD_V)
				{
					if (field->VMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("C2CV2V");
					myPlan.run(FFT_BCK);
					scaleField	(field, FIELD_V, 1/ ((double) N));
					field->setVMomSpace(true);
				}
			if (mvomv & FIELD_M2TOM2)
				{
					auto &myPlan = AxionFFT::fetchPlan("C2CM22M2");
					myPlan.run(FFT_BCK);
					scaleField	(field, FIELD_M2, 1/ ((double) N));
				}

		} break;
		case FIELD_AXION:
		{
			if (mvomv & FIELD_M)
				{
					if (field->MMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("R2CM2M");
					myPlan.run(FFT_BCK);
					scaleField	(field, FIELD_M, 1/ ((double) N));
					field->setMMomSpace(true);
				}
			if (mvomv & FIELD_V)
				{
					if (field->VMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("R2CV2V");
					myPlan.run(FFT_BCK);
					scaleField	(field, FIELD_V, 1/ ((double) N));
					field->setVMomSpace(true);
				}
		}
		default:
		return;
		break;
	}

	LogMsg (VERB_HIGH, "[FTField] Field transformed to momentum space and divided by 1/N");

	return;
}

/* used to convert Fourier space into configuration space  */
void	FTfield::iftField(FieldIndex mvomv)
{
	if (field->Device() == DEV_GPU)
		return;

	LogMsg (VERB_NORMAL, "Calling inverse-ftField (type=%f) (prec=%d)", field->Field(), field->Precision());

	switch(field->Field()){
		case FIELD_SAXION:
		{
			if (mvomv & FIELD_M)
				{
					if (!field->MMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("C2CM2M");
					myPlan.run(FFT_FWD);
					field->setMMomSpace(false);
				}
			if (mvomv & FIELD_V)
				{
					if (!field->VMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("C2CV2V");
					myPlan.run(FFT_FWD);
					field->setVMomSpace(false);
				}
			if (mvomv & FIELD_MTOM2)
				{
					if (!field->MMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("nSpecSxM");
					myPlan.run(FFT_FWD);
					// field->setM2(false);
				}
		} break;
		case FIELD_AXION:
		{
			if (mvomv & FIELD_M)
				{
					if (!field->MMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("R2CM2M");
					myPlan.run(FFT_FWD);
					field->setMMomSpace(false);
				}
			if (mvomv & FIELD_V)
				{
					if (!field->VMomSpace()) return;
					auto &myPlan = AxionFFT::fetchPlan("R2CV2V");
					myPlan.run(FFT_FWD);
					field->setVMomSpace(false);
				}
			if (mvomv & FIELD_MTOM2)
				{
					/* finish! */
					/* I need a complex to real transform m>m2 */
					// if (!field->MMomSpace()) return;
					// auto &myPlan = AxionFFT::fetchPlan("nSpecSxM");
					// myPlan.run(FFT_FWD);
					// field->setM2(false);
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

void	FTfield::operator()(FieldIndex mvomv, FFTdir dir)
{
	// Careful here, GPUS might want to call CPU routines
	if (field->Device() == DEV_GPU)
		return;

	LogMsg  (VERB_HIGH, "Called FTfield");
	profiler::Profiler &prof = profiler::getProfiler(PROF_FTFIELD);

	prof.start();

	switch(dir)
	{
		case FFT_BCK:
			ftField(mvomv);
		break;
		case FFT_FWD:
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
