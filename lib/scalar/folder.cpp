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
#include"utils/utils.h"

using namespace std;

	Folder::Folder(Scalar *scalar) : field(scalar), Lz(scalar->Depth()), n1(scalar->Length()), n2(scalar->Surf()), n3(scalar->Size())
{
}

template<typename cFloat>
void	Folder::foldField()
{
	if (field->Folded() || field->Device() == DEV_GPU)
		return;

	cFloat *m = static_cast<cFloat *> ((void *) field->mCpu());
 	cFloat *v = static_cast<cFloat *> ((void *) field->vCpu());

	fSize = field->DataSize();
	shift = field->DataAlign()/fSize;

	LogMsg (VERB_NORMAL, "Calling foldField mAlign=%d, fSize=%d, shift=%d", field->DataAlign(), fSize, shift);

	for (size_t iz=0; iz < Lz; iz++)
	{
		memcpy (m,           m + n2*(1+iz), fSize*n2);
		memcpy (m + (n3+n2), v + n2*iz,     fSize*n2);

		#pragma omp parallel for schedule(static)
		for (size_t iy=0; iy < n1/shift; iy++)
			for (size_t ix=0; ix < n1; ix++)
				for (size_t sy=0; sy<shift; sy++)
				{
					size_t oIdx = (iy+sy*(n1/shift))*n1 + ix;
					size_t dIdx = iz*n2 + ((size_t) (iy*n1*shift + ix*shift + sy));

					m[dIdx+n2] = m[oIdx];
					v[dIdx]    = m[oIdx+n2+n3];
				}
	}

	field->setFolded(true);
	LogMsg (VERB_HIGH, "[Folder] Field folded");

	return;
}

template<typename cFloat>
void	Folder::unfoldField()
{
	if (!field->Folded() || field->Device() == DEV_GPU)
		return;

	cFloat *m = static_cast<cFloat *> ((void *) field->mCpu());
	cFloat *v = static_cast<cFloat *> ((void *) field->vCpu());

	fSize = field->DataSize();
	shift = field->DataAlign()/fSize;

	LogMsg (VERB_NORMAL, "Calling unfoldField mAlign=%d, fSize=%d, shift=%d", field->DataAlign(), fSize, shift);

	for (size_t iz=0; iz < Lz; iz++)
	{
		memcpy (m,           m + n2*(1+iz), fSize*n2);
		memcpy (m + (n3+n2), v + n2*iz,     fSize*n2);

		#pragma omp parallel for schedule(static)
		for (size_t iy=0; iy < n1/shift; iy++)
			for (size_t ix=0; ix < n1; ix++)
				for (size_t sy=0; sy<shift; sy++)
				{
					size_t oIdx = iy*n1*shift + ix*shift + sy;
					size_t dIdx = iz*n2 + (iy+sy*(n1/shift))*n1 + ix;

					m[dIdx+n2] = m[oIdx];
					v[dIdx]    = m[oIdx+n2+n3];
				}
	}

	field->setFolded(false);
	LogMsg (VERB_HIGH, "Field unfolded");

	return;
}

template<typename cFloat>	// Only rank 0 can do this, and currently we quietly exist for any other rank. This can generate bugs if sZ > local Lz
void	Folder::unfoldField2D (const size_t sZ)
{
	if ((sZ < 0) || (sZ > field->Depth()) || field->Device() == DEV_GPU)
		return;

	cFloat *m = static_cast<cFloat *> (field->mCpu());
	cFloat *v = static_cast<cFloat *> (field->vCpu());

	if (!field->Folded())
	{
		LogMsg (VERB_HIGH, "unfoldField2D called in an unfolded configuration, copying data to ghost zones");
		memcpy ( m,        &m[(sZ+1)*n2], sizeof(cFloat)*n2);
		memcpy (&m[n2+n3], &v[sZ*n2],     sizeof(cFloat)*n2);
		return;
	}

	fSize = field->DataSize();
	shift = field->DataAlign()/fSize;

	LogMsg (VERB_HIGH, "Calling unfoldField2D mAlign=%d, fSize=%d, shift=%d", field->DataAlign(), fSize, shift);

	#pragma omp parallel for schedule(static)
	for (size_t iy=0; iy < n1/shift; iy++)
		for (size_t ix=0; ix < n1; ix++)
			for (size_t sy=0; sy<shift; sy++)
			{
				size_t oIdx = (sZ)*n2 + iy*n1*shift + ix*shift + sy;
				size_t dIdx = (iy+sy*(n1/shift))*n1 + ix;
				//this copies m into buffer 1
				m[dIdx]		= m[oIdx+n2];
				//this copies v into buffer last
				m[dIdx+n3+n2]	= v[oIdx];
			}

	LogMsg (VERB_HIGH, "Slice unfolded");

	return;
}

void	Folder::operator()(FoldType fType, size_t cZ)
{
	// Careful here, GPUS might want to call CPU routines
	if (field->Device() == DEV_GPU)
		return;

	LogMsg  (VERB_HIGH, "Called folder");
	profiler::Profiler &prof = profiler::getProfiler(PROF_FOLD);

	prof.start();

	switch (fType)
	{
		case	FOLD_ALL:

			setName("Fold");
			add(0., field->Size()*field->DataSize()*2.e-9);

			switch(field->Precision())
			{
				case	FIELD_DOUBLE:

					switch (field->Field())
					{
						case	FIELD_SAXION:
							foldField<complex<double>>();
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
							foldField<double>();
							break;

						default:
							break;
					}

					break;

				case	FIELD_SINGLE:

					switch (field->Field())
					{
						case	FIELD_SAXION:
							foldField<complex<float>>();
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
							foldField<float>();
							break;

						default:
							break;
					}

					break;

				default:
					break;
			}

			break;

		case	UNFOLD_ALL:

			setName("Unfold");
			add(0., field->Size()*field->DataSize()*2.e-9);

			switch(field->Precision())
			{
				case	FIELD_DOUBLE:

					switch (field->Field())
					{
						case	FIELD_SAXION:
							unfoldField<complex<double>>();
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
							unfoldField<double>();
							break;

						default:
							break;
					}

					break;

				case	FIELD_SINGLE:

					switch (field->Field())
					{
						case	FIELD_SAXION:
							unfoldField<complex<float>>();
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
							unfoldField<float>();
							break;

						default:
							break;
					}

					break;

				default:
					break;
			}

			break;

		case	UNFOLD_SLICE:

			setName("Unfold slice");
			add(0., field->Surf()*field->DataSize()*2.e-9);

			switch(field->Precision())
			{
				case	FIELD_DOUBLE:

					switch (field->Field())
					{
						case	FIELD_SAXION:
							unfoldField2D<complex<double>>(cZ);
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
							unfoldField2D<double>(cZ);
							break;

						default:
							break;
					}

					break;

				case	FIELD_SINGLE:

					switch (field->Field())
					{
						case	FIELD_SAXION:
							unfoldField2D<complex<float>>(cZ);
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
							unfoldField2D<float>(cZ);
							break;

						default:
							break;
					}

					break;

				default:
					break;
			}

			break;

		default:
			LogError ("Unrecognized folding option");
			break;
	}

	prof.stop();

	prof.add(Name(), GFlops(), GBytes());	// In truth is x4 because we move data to the ghost slices before folding/unfolding

	LogMsg  (VERB_HIGH, "Folder %s reporting %lf GFlops %lf GBytes", Name().c_str(), prof.Prof()[Name()].GFlops(), prof.Prof()[Name()].GBytes());

	reset();
}
