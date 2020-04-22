#ifndef	__SIMPLOPS__
	#define	__SIMPLOPS__

#include "enum-field.h"
#include "scalar/scalarField.h"
#include <math.h>

// double anymean(FieldIndex ftipo);
// void   susum(FieldIndex from, FieldIndex to);
// void   mulmul(FieldIndex from, FieldIndex to);
// void   axby(FieldIndex from, FieldIndex to, double a, double b);

char*	chosechar	(Scalar *field, PadIndex start){
switch (start){
	case PFIELD_M:
		return static_cast<char*>(field->mCpu());
		break;
	case PFIELD_MS:
		return static_cast<char*>(field->mStart());
		break;
	case PFIELD_V:
		return static_cast<char*>(field->vCpu());
		break;
	case PFIELD_VS:
		return static_cast<char*>(field->vStart());
		break;
	case PFIELD_M2:
		return static_cast<char*>(field->m2Cpu());
		break;
	case PFIELD_M2S:
		return static_cast<char*>(field->m2Start());
		break;
	case PFIELD_M2H:
		return static_cast<char*>(field->m2half());
		break;
	case PFIELD_M2HS:
		return static_cast<char*>(field->m2hStart());
		break;
	}
}

void	mulmul(Scalar *axion, PadIndex ftipo1, PadIndex ftipo2)
{
	/* Multiplies something 1 into something 2*/
	LogMsg(VERB_NORMAL,"[SOP] multiply field%d times field%d ",ftipo2,ftipo1);
	void* meandro  = static_cast<void *> (chosechar(axion,ftipo1));
  void* meandro2 = static_cast<void *> (chosechar(axion,ftipo2));
  size_t nData = axion->DataSize()/axion->Precision();

	switch(axion->Precision()){
		case FIELD_DOUBLE:{
			double *m = static_cast<double*>(meandro);
			double *d = static_cast<double*>(meandro2);
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)
				for (size_t idx = 0; idx < axion->Size()*nData; idx++)
				d[idx] *= m[idx];
			}
		} break;
		case FIELD_SINGLE:{
			float *m = static_cast<float*>(meandro);
			float *d = static_cast<float*>(meandro2);
			#pragma omp parallel
			{
				#pragma omp for schedule(static)
				for (size_t idx = 0; idx < axion->Size()*nData; idx++)
				d[idx] *= m[idx];
			}
		} break;
		}

	return;
} //end mulmul

void	axby(Scalar *axion, PadIndex ftipo1, PadIndex ftipo2, double a, double b)
{
	/* Multiplies something 1 into something 2
    could be very nicely vectorised */
	LogMsg(VERB_NORMAL,"[SOP] axby (f%d) = (f%d)x%.e + (f%d)x%.e",ftipo2,ftipo2,b,ftipo1,a);
	void* meandro  = static_cast<void *> (chosechar(axion,ftipo1));
  void* meandro2 = static_cast<void *> (chosechar(axion,ftipo2));
  size_t nData = axion->DataSize()/axion->Precision();

	switch(axion->Precision()){
		case FIELD_DOUBLE:{
			double *m = static_cast<double*>(meandro);
			double *d = static_cast<double*>(meandro2);
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)
				for (size_t idx = 0; idx < axion->Size()*nData; idx++)
				d[idx] = b*d[idx] + a*m[idx];
			}
		} break;
		case FIELD_SINGLE:{
			float *m = static_cast<float*>(meandro);
			float *d = static_cast<float*>(meandro2);
			#pragma omp parallel
			{
				#pragma omp for schedule(static)
				for (size_t idx = 0; idx < axion->Size()*nData; idx++)
				d[idx] = ((float) b)*d[idx] + ((float) a)*m[idx];
			}
		} break;
		}

	return;
} //end mulmul


void	unMoor(Scalar *axion, PadIndex ftipo1)
{
	/* Converts Moore theta into standard (unmended) */
	LogMsg(VERB_NORMAL,"[SOP] unMoor (f%d)",ftipo1);
	void* meandro  = static_cast<void *> (chosechar(axion,ftipo1));
  size_t nData = axion->DataSize()/axion->Precision();

  double R = *axion->RV();
	switch(axion->Precision()){
		case FIELD_DOUBLE:{
			double *m = static_cast<double*>(meandro);
      double twoPi = 2.0 * M_PI;
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)
				for (size_t idx = 0; idx < axion->Size()*nData; idx++)
          m[idx]= ( m[idx]+M_PI - twoPi * floor((m[idx] + M_PI)/twoPi) - M_PI)*R;
			}
		} break;
		case FIELD_SINGLE:{
			float *m = static_cast<float*>(meandro);
      float twoPi = 2.0 * M_PI;
			#pragma omp parallel
			{
				#pragma omp for schedule(static)
				for (size_t idx = 0; idx < axion->Size()*nData; idx++)
			     m[idx]= ( m[idx]+M_PI - twoPi * floor((m[idx] + M_PI)/twoPi) - M_PI)*R;
			}
		} break;
		}

	return;
} //end unMoor

#endif
