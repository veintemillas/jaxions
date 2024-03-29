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

	cFloat *mg1 = static_cast<cFloat *> ((void *) field->mFrontGhost());
	cFloat *mg2 = static_cast<cFloat *> ((void *) field->mBackGhost());
	cFloat *m   = static_cast<cFloat *> ((void *) field->mStart());
 	cFloat *v   = static_cast<cFloat *> ((void *) field->vStart());

	fSize = field->DataSize();
	shift = field->DataAlign()/fSize;

	LogMsg (VERB_NORMAL, "Calling foldField mAlign=%d, fSize=%d, shift=%d", field->DataAlign(), fSize, shift);

	for (size_t iz=0; iz < Lz; iz++)
	{
		memcpy (mg1, &m[n2*iz], sizeof(cFloat)*n2);
		memcpy (mg2, &v[n2*iz], sizeof(cFloat)*n2);

		#pragma omp parallel for schedule(static)
		for (size_t iy=0; iy < n1/shift; iy++)
			for (size_t ix=0; ix < n1; ix++)
				for (size_t sy=0; sy<shift; sy++)
				{
					size_t oIdx = (iy+sy*(n1/shift))*n1 + ix;
					size_t dIdx = iz*n2 + ((size_t) (iy*n1*shift + ix*shift + sy));

					m[dIdx] = mg1[oIdx];
					v[dIdx] = mg2[oIdx];
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

	cFloat *mg1 = static_cast<cFloat *> ((void *) field->mFrontGhost());
	cFloat *mg2 = static_cast<cFloat *> ((void *) field->mBackGhost());
	cFloat *m   = static_cast<cFloat *> ((void *) field->mStart());
 	cFloat *v   = static_cast<cFloat *> ((void *) field->vStart());

	fSize = field->DataSize();
	shift = field->DataAlign()/fSize;

	LogMsg (VERB_NORMAL, "Calling unfoldField mAlign=%d, fSize=%d, shift=%d", field->DataAlign(), fSize, shift);

	for (size_t iz=0; iz < Lz; iz++)
	{
		memcpy (mg1, m + n2*iz, fSize*n2);
		memcpy (mg2, v + n2*iz, fSize*n2);

		#pragma omp parallel for schedule(static)
		for (size_t iy=0; iy < n1/shift; iy++)
			for (size_t ix=0; ix < n1; ix++)
				for (size_t sy=0; sy<shift; sy++)
				{
					size_t oIdx = iy*n1*shift + ix*shift + sy;
					size_t dIdx = iz*n2 + (iy+sy*(n1/shift))*n1 + ix;

					m[dIdx]    = mg1[oIdx];
					v[dIdx]    = mg2[oIdx];
				}
	}

	field->setFolded(false);
 	LogMsg (VERB_HIGH, "[Folder] Field unfolded");

	return;
}





template<typename cFloat>	// Only rank 0 can do this, and currently we quietly exist for any other rank. This can generate bugs if sZ > local Lz
void	Folder::unfoldField2D (const size_t sZ)
{
	if ((sZ < 0) || (sZ > field->Depth()) || field->Device() == DEV_GPU)
		return;

	cFloat *mg1 = static_cast<cFloat *> ((void *) field->mFrontGhost());
	cFloat *mg2 = static_cast<cFloat *> ((void *) field->mBackGhost());
	cFloat *m   = static_cast<cFloat *> ((void *) field->mStart());
 	cFloat *v   = static_cast<cFloat *> ((void *) field->vStart());

	if (!field->Folded())
	{
		LogMsg (VERB_HIGH, "unfoldField2D called in an unfolded configuration, copying data to ghost zones");LogFlush();
		memcpy (mg1, &m[n2*sZ], sizeof(cFloat)*n2);
		memcpy (mg2, &v[n2*sZ], sizeof(cFloat)*n2);
		return;
	}

	fSize = field->DataSize();
	shift = field->DataAlign()/fSize;

	LogMsg (VERB_HIGH, "Calling unfoldField2D mAlign=%d, fSize=%d, shift=%d", field->DataAlign(), fSize, shift);LogFlush();

	#pragma omp parallel for schedule(static)
	for (size_t iy=0; iy < n1/shift; iy++)
		for (size_t ix=0; ix < n1; ix++)
			for (size_t sy=0; sy<shift; sy++)
			{
				size_t oIdx = (sZ)*n2 + iy*n1*shift + ix*shift + sy;
				size_t dIdx = (iy+sy*(n1/shift))*n1 + ix;
				//this copies m into buffer 1
				mg1[dIdx]	= m[oIdx];
				//this copies v into buffer last
				mg2[dIdx]	= v[oIdx];
			}

	LogMsg (VERB_HIGH, "Slice unfolded");

	return;
}





// unfolds a X=constant slice
template<typename cFloat>
void	Folder::unfoldField2DYZ (const size_t sX)
{
	if ((sX < 0) || (sX > field->Length()) || field->Device() == DEV_GPU)
		return;

	cFloat *mg1 = static_cast<cFloat *> (field->mFrontGhost());
	cFloat *mg2 = static_cast<cFloat *> (field->mBackGhost());
	cFloat *m  = static_cast<cFloat *> (field->mStart());
	cFloat *v  = static_cast<cFloat *> (field->vStart());

	int z0 = 0;
	size_t zT = field->Depth();

	if (!field->Folded())
	{
		LogMsg (VERB_HIGH, "[uf2X] unfoldField2D called in an unfolded configuration, copying data to ghost zones");
		LogFlush();
		// memcpy ( m,        &m[(sZ+1)*n2], sizeof(cFloat)*n2);
		// memcpy (&m[n2+n3], &v[sZ*n2],     sizeof(cFloat)*n2);
		// in m ghost zone, y is permuted for x and z for y to ease the block saving
		#pragma omp parallel for schedule(static)
		for (size_t iy=0; iy < n1; iy++)
			for (size_t iz=z0; iz < zT ; iz++)
				{
					size_t oIdx = iz*n2 + iy*n1 + sX;
					size_t dIdx =         iz*n1 + iy ;
					//this copies m into buffer 1
					mg1[dIdx] = m[oIdx];
					//this copies v into buffer last
					mg2[dIdx]	= v[oIdx];
				}
				LogMsg (VERB_PARANOID, "[uf2X] done");
				LogFlush();
		return;
	}

	fSize = field->DataSize();
	shift = field->DataAlign()/fSize;

	LogMsg (VERB_HIGH, "[uf2X] Calling unfoldField2D mAlign=%d, fSize=%d, shift=%d", field->DataAlign(), fSize, shift);
	LogFlush();

	#pragma omp parallel for schedule(static)
	for (size_t iy=0; iy < n1/shift; iy++)
		for (size_t sy=0; sy<shift; sy++)
			for (size_t iz=z0; iz < zT ; iz++)
			{
				size_t oIdx = (iz)*n2 + iy*n1*shift + sX*shift + sy;
				size_t dIdx = iz*n1 + (iy+sy*(n1/shift));
				//this copies m into buffer 1
				mg1[dIdx] = m[oIdx];
				//this copies v into buffer last
				mg2[dIdx] = v[oIdx];
			}
			return;

	LogMsg (VERB_HIGH, "[uf2X] Slice unfolded");
	LogFlush();
	return;
}


	/* m2 folding experimental */

	template<typename cFloat>
	void	Folder::foldM2()
	{
		if (field->M2Folded() || field->Device() == DEV_GPU)
			return;

		cFloat *mg1 = static_cast<cFloat *> ((void *) field->mFrontGhost());
		cFloat *m   = static_cast<cFloat *> ((void *) field->m2Start());

		fSize = field->DataSize();
		shift = field->DataAlign()/fSize;

		LogMsg (VERB_NORMAL, "Calling foldField mAlign=%d, fSize=%d, shift=%d", field->DataAlign(), fSize, shift);

		for (size_t iz=0; iz < Lz; iz++)
		{
			memcpy (mg1, &m[n2*iz], sizeof(cFloat)*n2);

			#pragma omp parallel for schedule(static)
			for (size_t iy=0; iy < n1/shift; iy++)
				for (size_t ix=0; ix < n1; ix++)
					for (size_t sy=0; sy<shift; sy++)
					{
						size_t oIdx = (iy+sy*(n1/shift))*n1 + ix;
						size_t dIdx = iz*n2 + ((size_t) (iy*n1*shift + ix*shift + sy));

						m[dIdx] = mg1[oIdx];
					}
		}

		field->setM2Folded(true);
		LogMsg (VERB_HIGH, "[Folder] Field M2 folded (from M2Start)");

		return;
	}

	template<typename cFloat>
	void	Folder::unfoldM2()
	{
		if (!field->M2Folded() || field->Device() == DEV_GPU)
			return;

		cFloat *mg1 = static_cast<cFloat *> ((void *) field->mFrontGhost());
		cFloat *m   = static_cast<cFloat *> ((void *) field->m2Start());

		fSize = field->DataSize();
		shift = field->DataAlign()/fSize;

		LogMsg (VERB_NORMAL, "Calling unfoldField mAlign=%d, fSize=%d, shift=%d", field->DataAlign(), fSize, shift);

		for (size_t iz=0; iz < Lz; iz++)
		{
			memcpy (mg1, m + n2*iz, fSize*n2);

			#pragma omp parallel for schedule(static)
			for (size_t iy=0; iy < n1/shift; iy++)
				for (size_t ix=0; ix < n1; ix++)
					for (size_t sy=0; sy<shift; sy++)
					{
						size_t oIdx = iy*n1*shift + ix*shift + sy;
						size_t dIdx = iz*n2 + (iy+sy*(n1/shift))*n1 + ix;

						m[dIdx]    = mg1[oIdx];
					}
		}

		field->setM2Folded(false);
	 	LogMsg (VERB_HIGH, "[Folder] Field M2 unfolded (m2Start)");

		return;
	}





template<typename cFloat>	// Only rank 0 can do this, and currently we quietly exist for any other rank. This can generate bugs if sZ > local Lz
void	Folder::unfoldM22D (const size_t sZ)
{
	if ((sZ < 0) || (sZ > field->Depth()) || field->Device() == DEV_GPU)
		return;

	cFloat *mg1 = static_cast<cFloat *> ((void *) field->m2FrontGhost());
	cFloat *m   = static_cast<cFloat *> ((void *) field->m2Start());

	if (!field->M2Folded())
	{
		LogMsg (VERB_HIGH, "unfoldM22D called in an unfolded configuration, copying data to ghost zones");LogFlush();
		memcpy (mg1, &m[n2*sZ], sizeof(cFloat)*n2);
		return;
	}

	fSize = field->DataSize();
	shift = field->DataAlign()/fSize;

	LogMsg (VERB_HIGH, "Calling unfoldM22D mAlign=%d, fSize=%d, shift=%d", field->DataAlign(), fSize, shift);LogFlush();

	#pragma omp parallel for schedule(static)
	for (size_t iy=0; iy < n1/shift; iy++)
		for (size_t ix=0; ix < n1; ix++)
			for (size_t sy=0; sy<shift; sy++)
			{
				size_t oIdx = (sZ)*n2 + iy*n1*shift + ix*shift + sy;
				size_t dIdx = (iy+sy*(n1/shift))*n1 + ix;
				//this copies m into buffer 1
				mg1[dIdx]	= m[oIdx];
			}

	LogMsg (VERB_HIGH, "Slice unfolded");

	return;
}




/* basic operator */

void	Folder::operator()(FoldType fType, size_t cZ)
{
	// Careful here, GPUS might want to call CPU routines
	if (field->Device() == DEV_GPU)
		return;

	LogMsg  (VERB_HIGH, "[Fold] Called with m/v field %d (folded/unfolded 1/0)",field->Folded());
	LogMsg  (VERB_HIGH, "[Fold] Called with m2  field %d (folded/unfolded 1/0)",field->M2Folded());

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
						case	FIELD_NAXION:
							foldField<complex<double>>();
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
						case	FIELD_PAXION:
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
						case	FIELD_NAXION:
							foldField<complex<float>>();
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
						case  FIELD_PAXION:
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
						case	FIELD_NAXION:
							unfoldField<complex<double>>();
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
						case  FIELD_PAXION:
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
						case	FIELD_NAXION:
							unfoldField<complex<float>>();
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
						case  FIELD_PAXION:
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
						case	FIELD_NAXION:
							unfoldField2D<complex<double>>(cZ);
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
						case  FIELD_PAXION:
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
						case	FIELD_NAXION:
							unfoldField2D<complex<float>>(cZ);
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
						case  FIELD_PAXION:
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

		case	UNFOLD_SLICEYZ:

			setName("Unfold slice YZ");
			add(0., field->Surf()*field->DataSize()*2.e-9);

			switch(field->Precision())
			{
				case	FIELD_DOUBLE:

					switch (field->Field())
					{
						case	FIELD_SAXION:
						case	FIELD_NAXION:
							unfoldField2DYZ<complex<double>>(cZ);
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
						case  FIELD_PAXION:
							unfoldField2DYZ<double>(cZ);
							break;

						default:
							break;
					}

					break;

				case	FIELD_SINGLE:

					switch (field->Field())
					{
						case	FIELD_SAXION:
						case	FIELD_NAXION:
							unfoldField2DYZ<complex<float>>(cZ);
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
						case  FIELD_PAXION:
							unfoldField2DYZ<float>(cZ);
							break;

						default:
							break;
					}

					break;

				default:
					break;
			}

			break;

			case	FOLD_M2:

				setName("Fold M2");
				add(0., field->Size()*field->DataSize()*2.e-9);

				switch(field->Precision())
				{
					case	FIELD_DOUBLE:

						switch (field->Field())
						{
							case	FIELD_SAXION:
							case	FIELD_NAXION:
								foldM2<complex<double>>();
								break;

							case	FIELD_AXION_MOD:
							case	FIELD_AXION:
							case	FIELD_WKB:
							case  FIELD_PAXION:
								foldM2<double>();
								break;

							default:
								break;
						}

						break;

					case	FIELD_SINGLE:

						switch (field->Field())
						{
							case	FIELD_SAXION:
							case	FIELD_NAXION:
								foldM2<complex<float>>();
								break;

							case	FIELD_AXION_MOD:
							case	FIELD_AXION:
							case	FIELD_WKB:
							case  FIELD_PAXION:
								foldM2<float>();
								break;

							default:
								break;
						}

						break;

					default:
						break;
				}

				break;

			case	UNFOLD_M2:

				setName("Unfold M2");
				add(0., field->Size()*field->DataSize()*2.e-9);

				switch(field->Precision())
				{
					case	FIELD_DOUBLE:

						switch (field->Field())
						{
							case	FIELD_SAXION:
							case	FIELD_NAXION:
								unfoldM2<complex<double>>();
								break;

							case	FIELD_AXION_MOD:
							case	FIELD_AXION:
							case	FIELD_WKB:
							case  FIELD_PAXION:
								unfoldM2<double>();
								break;

							default:
								break;
						}

						break;

					case	FIELD_SINGLE:

						switch (field->Field())
						{
							case	FIELD_SAXION:
							case	FIELD_NAXION:
								unfoldM2<complex<float>>();
								break;

							case	FIELD_AXION_MOD:
							case	FIELD_AXION:
							case	FIELD_WKB:
							case  FIELD_PAXION:
								unfoldM2<float>();
								break;

							default:
								break;
						}

						break;

					default:
						break;
				}

				break;


			case	UNFOLD_SLICEM2:

				setName("Unfold slice M2");
				add(0., field->Surf()*field->DataSize()*2.e-9);

				switch(field->Precision())
				{
					case	FIELD_DOUBLE:

						switch (field->Field())
						{
							case	FIELD_SAXION:
							case	FIELD_NAXION:
								unfoldM22D<complex<double>>(cZ);
								break;

							case	FIELD_AXION_MOD:
							case	FIELD_AXION:
							case	FIELD_WKB:
							case  FIELD_PAXION:
								unfoldM22D<double>(cZ);
								break;

							default:
								break;
						}

						break;

					case	FIELD_SINGLE:

						switch (field->Field())
						{
							case	FIELD_SAXION:
							case	FIELD_NAXION:
								unfoldM22D<complex<float>>(cZ);
								break;

							case	FIELD_AXION_MOD:
							case	FIELD_AXION:
							case	FIELD_WKB:
							case  FIELD_PAXION:
								unfoldM22D<float>(cZ);
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
