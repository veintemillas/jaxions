#include <string>
#include <complex>
#include <memory>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
#endif

#include "utils/utils.h"
#include "fft/fftCode.h"
#include "comms/comms.h"

class	NLaplacian : public Tunable
{
	private:

	const size_t		Ng;

	const FieldPrecision	precision;
	const size_t		Lx;
	const size_t		Ly;
	const size_t		Lz;
	const size_t		Tz;
	const size_t		Sf;

	const double		ood2;
	Scalar			*field;

	template<class cFloat>
	void			lapCpu();

	template<class cFloat>
	void			lapGpu();

	public:

		NLaplacian (Scalar *field, size_t Norder) : precision(field->Precision()), Lx(field->Length()), Ly(field->Length()/commSize()), Lz(field->Depth()), Tz(field->TotalDepth()),
					    Sf(field->Surf()), field(field), Ng(Norder), ood2(field->Length()*field->Length()/field->BckGnd()->PhysSize()/field->BckGnd()->PhysSize()) {
		if (field->LowMem()) {
			LogError ("Error: laplacian not supported in lowmem runs");
			exit(0);
		}

	}

	void	sRunCpu	();	// Saxion laplacian
	void	sRunGpu	();

	void	tRunCpu	();	// Axion laplacian
	void	tRunGpu	();
};

void	NLaplacian::sRunGpu	()
{
#ifdef	USE_GPU
#else
	LogError ("Error: gpu support not built");
	exit(1);
#endif
}


template<class Float>
void	NLaplacian::lapCpu	()
{

	double CO[4] = {0.0, 0.0, 0.0, 0.0} ;
	if (Ng == 2) {
		CO[0] = 16/12; CO[1] = -1/12;
	} else if (Ng == 3) {
		CO[0] = 3/2; CO[1]   = -3/20; CO[2] = 1/90 ;
	} else if (Ng == 4) {
		CO[0] = 8/5; CO[1]   = -1/5;  CO[2] = 8/315 ; CO[3] = (Float) -1/560 ;
	} else if (Ng == 0) {
		CO[0] = 0 ;
	} else {
		CO[0] = 1 ;
	}

	int lin = 2*Ng+1;

	Float *m = static_cast<Float*> (field->mStart());
	Float *mData = static_cast<Float*> (field->m2Cpu());

	#pragma omp parallel for default(shared) schedule(static)
	for (size_t iz = 0; iz < Lx; iz++)
	{
		size_t auxi;
		size_t Xol[lin], Yol[lin], Zol[lin];
		size_t aYol[lin], aZol[lin], bZol[lin];
		for (size_t l = -Ng; l < Ng+1; l++)
		{
			bZol[Ng+l] = ((iz + l + Lx)%Lx)*Sf ; //no ghost
		}
		for (size_t iy = 0; iy < Lx; iy++)
		{
			for (size_t l = -Ng; l < Ng+1; l++)
			{
				Yol[Ng+l] = ((iy + l + Lx )%Lx)*Lx ;
			}
			//set the Y, Z, in position
			for (size_t l = 0; l<lin; l++)
			{
				aZol[l]  = bZol[l]+Yol[Ng];
				aYol[l]  = bZol[Ng]+Yol[l];
			}

			for (size_t ix = 0; ix < Lx; ix++)
			{
				for (size_t l = 0; l < lin; l++)
				{
					Xol[l] = aZol[Ng] + (ix + l-Ng + Lx)%Lx ;
				}

				Float lap = 0;
				for (size_t l = 1; l < Ng+1; l++)
				{
					lap = lap + (m[Xol[Ng+l]] + m[Xol[Ng-l]] +
												m[Yol[Ng+l]+ix] + m[Yol[Ng-l]+ix] +
												m[Zol[Ng+l]+ix] + m[Zol[Ng-l]+ix] - m[Xol[Ng]]*6)*CO[l-1] ;
				}
				mData[Xol[Ng]] = lap*ood2;
			}
		 }
		}
}

void	Laplacian::sRunCpu	()
{
	switch (precision) {
		case FIELD_SINGLE:
			lapCpu<std::complex<float>, false>(std::string("SpSx"));
			break;

		case FIELD_DOUBLE:
			lapCpu<std::complex<double>,false>(std::string("SpSx"));
			break;

		default:
			LogError ("Couldn't calculate laplacian: Wrong precision");
			break;
	}
}

void    Laplacian::tRunGpu	()
{
#ifdef  USE_GPU
#else
	LogError ("Error: gpu support not built");
	exit(1);
#endif
}

void    Laplacian::tRunCpu	()
{
	switch (precision) {
		case FIELD_SINGLE:
			lapCpu<std::complex<float>, true>(std::string("pSpecAx"));
			break;

		case FIELD_DOUBLE:
			lapCpu<std::complex<double>,true>(std::string("pSpecAx"));
			break;

		default:
			LogError ("Couldn't calculate laplacian: Wrong precision");
			break;
	}
}

using	namespace profiler;

void	applyLaplacian	(Scalar *field)
{
	LogMsg	(VERB_HIGH, "Called laplacian");
	profiler::Profiler &prof = getProfiler(PROF_PROP);

	auto lap = std::make_unique<Laplacian>(field);

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	//prof.start();

	switch (field->Field()) {
		case FIELD_AXION_MOD:
		case FIELD_AXION:
			//lap->setName("Laplacian Axion");

			switch (field->Device()) {
				case DEV_GPU:
					lap->tRunGpu ();
					break;
				case DEV_CPU:
					lap->tRunCpu ();
					break;
				default:
					LogError ("Error: invalid device");
					prof.stop();
					return;
			}

			//lap->add(16.*4.*field->Size()*1.e-9, 10.*4.*field->DataSize()*field->Size()*1.e-9);

			break;

		case FIELD_SAXION:
			//lap->setName("Laplacian Saxion");

			switch (field->Device()) {
				case DEV_GPU:
					lap->sRunGpu ();
					break;
				case DEV_CPU:
					lap->sRunCpu ();
					break;
				default:
					LogError ("Error: invalid device");
					//prof.stop();
					return;
			}
			break;

		default:
			LogError ("Error: invalid field type");
			//prof.stop();
			return;
	}

	//prof.stop();

	//prof.add(lap->Name(), lap->GFlops(), lap->GBytes());

	//LogMsg	(VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", lap->Name().c_str(), prof.Prof()[lap->Name()].GFlops(), prof.Prof()[lap->Name()].GBytes());

	return;
}
