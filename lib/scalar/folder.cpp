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

	Folder::Folder(Scalar *scalar) : field(scalar), Nx(scalar->NX()), Ny(scalar->NY()), Nz(scalar->NZ()),  Nxy(scalar->NXY()), Nxyz(scalar->NXYZ())
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
	shift = field->DataAlign()/fSize; // number of cFloat elements in a vector

	size_t N1=Nx , N2=Ny; // we vectorise the 2 direction
	bool foldX = (Ny == 1)? true:false;
	if (foldX)
		{N1=1;N2=Nx;}
	size_t Nv = N2/shift; // number of vectors

	if (N2 % shift != 0) {
		LogError("[fold] ERROR: fold dimension %zu not divisible by shift=%zu",
		       N2, shift);
		return;
	}

	LogMsg (VERB_NORMAL, "[fold] Calling foldField (%c) mAlign=%d, fSize=%d, shift=%d ", foldX? 'X':'Y', field->DataAlign(), fSize, shift);
	LogFlush();
	for (size_t iz=0; iz < Nz; iz++)
	{
		memcpy (mg1, &m[Nxy*iz], sizeof(cFloat)*Nxy);
		memcpy (mg2, &v[Nxy*iz], sizeof(cFloat)*Nxy);

		#pragma omp parallel for schedule(static)
		for (size_t i_v=0; i_v < Nv; i_v++)
			for (size_t ix=0; ix < N1; ix++)
				for (size_t s=0; s<shift; s++)
				{
					size_t oIdx = (i_v+s*Nv)*N1 + ix;
					size_t dIdx = iz*Nxy + i_v*N1*shift + ix*shift + s;

					m[dIdx] = mg1[oIdx];
					v[dIdx] = mg2[oIdx];
				}

	}

	field->setFolded(true);
	somethingdone = true;
	LogMsg (VERB_HIGH, "[fold] Field folded");


	return;
}

template<typename cFloat>
void Folder::unfoldField()
{
	if (!field->Folded() || field->Device() == DEV_GPU)
		return;

	cFloat *mg1 = static_cast<cFloat *>((void *) field->mFrontGhost());
	cFloat *mg2 = static_cast<cFloat *>((void *) field->mBackGhost());
	cFloat *m   = static_cast<cFloat *>((void *) field->mStart());
	cFloat *v   = static_cast<cFloat *>((void *) field->vStart());

	fSize = field->DataSize();
	shift = field->DataAlign()/fSize;   // number of cFloat elements in a vector

	size_t N1 = Nx, N2 = Ny;            // N2 = folded direction, N1 = non-folded direction
	bool foldX = (Ny == 1);

	if (foldX) {
		N1 = 1;
		N2 = Nx;
	}

	if (N2 % shift != 0) {
		LogError("[unfold] ERROR: folded dimension %zu not divisible by shift=%zu",
		       N2, shift);
		return;
	}

	size_t Nv = N2/shift;               // number of vectors along folded direction

	LogMsg(VERB_NORMAL,
	       "[unfold] Calling unfoldField (%s) mAlign=%d, fSize=%d, shift=%zu",
	       foldX ? "X" : "Y", field->DataAlign(), fSize, shift);

	for (size_t iz = 0; iz < Nz; iz++)
	{
		memcpy(mg1, m + Nxy*iz, fSize*Nxy);
		memcpy(mg2, v + Nxy*iz, fSize*Nxy);

		#pragma omp parallel for schedule(static)
		for (size_t i_v = 0; i_v < Nv; i_v++)
			for (size_t ix = 0; ix < N1; ix++)
				for (size_t s = 0; s < shift; s++)
				{
					// source index inside folded z-slice: [i_v][ix][s]
					size_t oIdx = i_v*N1*shift + ix*shift + s;

					// destination index inside original z-slice
					// original layout is [(i_v + s*Nv)][ix]
					size_t dIdx = (i_v + s*Nv)*N1 + ix;

					m[Nxy*iz + dIdx] = mg1[oIdx];
					v[Nxy*iz + dIdx] = mg2[oIdx];
				}
	}

	field->setFolded(false);
	LogMsg(VERB_HIGH, "[unfold] Field unfolded");

	return;
}


template<typename cFloat>   // Only rank 0 can do this; currently we quietly exit for any other rank.
void Folder::unfoldField2D(const size_t sZ)
{
	if (sZ >= Nz || field->Device() == DEV_GPU)
		return;

	cFloat *mg1 = static_cast<cFloat *>((void *) field->mFrontGhost());
	cFloat *mg2 = static_cast<cFloat *>((void *) field->mBackGhost());
	cFloat *m   = static_cast<cFloat *>((void *) field->mStart());
	cFloat *v   = static_cast<cFloat *>((void *) field->vStart());

	if (!field->Folded())
	{
		LogMsg(VERB_HIGH, "[ufXY] unfoldField2D called in an unfolded configuration, copying %zu slice to ghost zone 1", sZ);
		LogFlush();
		memcpy(mg1, &m[Nxy*sZ], sizeof(cFloat)*Nxy);
		memcpy(mg2, &v[Nxy*sZ], sizeof(cFloat)*Nxy);
		LogMsg(VERB_HIGH, "[ufXY] copying Done");
		LogFlush();
		somethingdone = true;
		return;
	}

	fSize = field->DataSize();
	shift = field->DataAlign()/fSize;

	const bool foldX = (Ny == 1);
	const size_t N1  = foldX ? 1  : Nx;
	const size_t N2  = foldX ? Nx : Ny;

	if (N2 % shift != 0)
		return;

	const size_t Nv = N2/shift;

	LogMsg(VERB_HIGH, "[ufXY] Calling unfoldField2D (%s) mAlign=%d, fSize=%d, shift=%d",
	       foldX ? "X" : "Y", field->DataAlign(), fSize, shift);
	LogFlush();

	if (!foldX)
	{
		// Y-fold: folded layout [i_v][x][s], with y = i_v + s*Nv
		#pragma omp parallel for schedule(static)
		for (size_t i_v = 0; i_v < Nv; i_v++)
			for (size_t ix = 0; ix < Nx; ix++)
				for (size_t s = 0; s < shift; s++)
				{
					size_t iy   = i_v + s*Nv;
					size_t oIdx = sZ*Nxy + i_v*N1*shift + ix*shift + s;
					size_t dIdx = iy*Nx + ix;

					mg1[dIdx] = m[oIdx];
					mg2[dIdx] = v[oIdx];
				}
	}
	else
	{
		// X-fold with Ny == 1: folded layout [i_v][s], with x = i_v + s*Nv
		#pragma omp parallel for schedule(static)
		for (size_t i_v = 0; i_v < Nv; i_v++)
			for (size_t s = 0; s < shift; s++)
			{
				size_t ix   = i_v + s*Nv;
				size_t oIdx = sZ*Nxy + i_v*N1*shift + s;   // N1 = 1
				size_t dIdx = ix;                          // Ny = 1 -> y = 0

				mg1[dIdx] = m[oIdx];
				mg2[dIdx] = v[oIdx];
			}
	}

	LogMsg(VERB_HIGH, "[ufXY] Slice unfolded");
	LogFlush();
	somethingdone = true;
}





/* unfolds a X = constant slice */

template<typename cFloat>
void Folder::unfoldField2DYZ(const size_t sX)
{
	if ((sX >= Nx) || field->Device() == DEV_GPU)
		return;

	cFloat *mg1 = static_cast<cFloat *>(field->mFrontGhost());
	cFloat *mg2 = static_cast<cFloat *>(field->mBackGhost());
	cFloat *m   = static_cast<cFloat *>(field->mStart());
	cFloat *v   = static_cast<cFloat *>(field->vStart());

	const size_t z0 = 0;
	const size_t zT = Nz;

	if (!field->Folded())
	{
		LogMsg(VERB_HIGH, "[ufYZ] unfoldField2D called in an unfolded configuration, copying %zu slice to ghost zone 1", sX);
		#pragma omp parallel for schedule(static)
		for (size_t iy = 0; iy < Ny; iy++)
			for (size_t iz = z0; iz < zT; iz++)
			{
				size_t oIdx = iz*Nxy + iy*Nx + sX;
				size_t dIdx = iz*Ny + iy;
				mg1[dIdx] = m[oIdx];
				mg2[dIdx] = v[oIdx];
			}
		LogMsg(VERB_HIGH, "[ufYZ] copying Done");
		somethingdone = true;
		return;
	}

	fSize = field->DataSize();
	shift = field->DataAlign()/fSize;

	const bool foldX = (Ny == 1);

	size_t N1 = foldX ? 1  : Nx;
	size_t N2 = foldX ? Nx : Ny;

	if (N2 % shift != 0)
		return;

	LogMsg(VERB_HIGH, "[ufYZ] Calling unfoldField2D (%s) mAlign=%d, fSize=%d, shift=%d",
	       foldX ? "X" : "Y", field->DataAlign(), fSize, shift);
	LogFlush();

	const size_t Nv     = N2/shift;
	const size_t ixBase = foldX ? 0 : sX;

	#pragma omp parallel for schedule(static)
	for (size_t i_v = 0; i_v < Nv; i_v++)
		for (size_t s = 0; s < shift; s++)
			for (size_t iz = z0; iz < zT; iz++)
			{
				size_t c = i_v + s*Nv;

				if (foldX && c != sX)
					continue;

				size_t iy   = foldX ? 0 : c;
				size_t oIdx = iz*Nxy + i_v*N1*shift + ixBase*shift + s;
				size_t dIdx = iz*Ny + iy;

				mg1[dIdx] = m[oIdx];
				mg2[dIdx] = v[oIdx];
			}

	LogMsg(VERB_HIGH, "[ufYZ] Slice unfolded");
	somethingdone = true;
}


template<typename cFloat>
void Folder::unfoldField2DXZ(const size_t sY)
{
	if (sY >= Ny || field->Device() == DEV_GPU)
		return;

	cFloat *mg1 = static_cast<cFloat *>(field->mFrontGhost());
	cFloat *mg2 = static_cast<cFloat *>(field->mBackGhost());
	cFloat *m   = static_cast<cFloat *>(field->mStart());
	cFloat *v   = static_cast<cFloat *>(field->vStart());

	// ---- Unfolded case ----
	if (!field->Folded())
	{
		LogMsg(VERB_HIGH, "[ufXZ] unfoldField2D called in an unfolded configuration, copying %zu slice to ghost zone 1", sY);
		#pragma omp parallel for schedule(static)
		for (size_t iz = 0; iz < Nz; iz++)
			for (size_t ix = 0; ix < Nx; ix++)
			{
				size_t oIdx = iz*Nxy + sY*Nx + ix;
				size_t dIdx = iz*Nx + ix;

				mg1[dIdx] = m[oIdx];
				mg2[dIdx] = v[oIdx];
			}
		LogMsg(VERB_HIGH, "[ufXZ] copying Done");
		somethingdone = true;
		return;
	}

	// ---- Folded case ----
	fSize = field->DataSize();
	shift = field->DataAlign()/fSize;

	const bool foldX = (Ny == 1);
	const size_t N1  = foldX ? 1  : Nx;
	const size_t N2  = foldX ? Nx : Ny;

	if (N2 % shift != 0)
		return;

	const size_t Nv = N2/shift;

	LogMsg(VERB_HIGH, "[ufXZ] Calling unfoldField2D (%s) mAlign=%d, fSize=%d, shift=%d",
	       foldX ? "X" : "Y", field->DataAlign(), fSize, shift);
	LogFlush();

	if (!foldX)
	{
		// Y-fold: folded layout [i_v][x][s], y = i_v + s*Nv
		// We fix y = sY → need (i_v, s) such that:
		//   sY = i_v + s*Nv
		const size_t i_v = sY % Nv;
		const size_t s   = sY / Nv;

		#pragma omp parallel for schedule(static)
		for (size_t iz = 0; iz < Nz; iz++)
			for (size_t ix = 0; ix < Nx; ix++)
			{
				size_t oIdx = iz*Nxy + i_v*N1*shift + ix*shift + s;
				size_t dIdx = iz*Nx + ix;

				mg1[dIdx] = m[oIdx];
				mg2[dIdx] = v[oIdx];
			}
	}
	else
	{
		// X-fold (Ny == 1): trivial, y = 0 only
		// so sY must be 0; just decode x = i_v + s*Nv
		#pragma omp parallel for schedule(static)
		for (size_t i_v = 0; i_v < Nv; i_v++)
			for (size_t s = 0; s < shift; s++)
				for (size_t iz = 0; iz < Nz; iz++)
				{
					size_t ix   = i_v + s*Nv;
					size_t oIdx = iz*Nxy + i_v*N1*shift + s;
					size_t dIdx = iz*Nx + ix;

					mg1[dIdx] = m[oIdx];
					mg2[dIdx] = v[oIdx];
				}
	}

	somethingdone = true;
}


	/* m2 folding experimental */

	template<typename cFloat>
	void	Folder::foldM2()
	{
		if (field->M2Folded() || field->Device() == DEV_GPU || Nx == Nxy)
			return;

		cFloat *mg1 = static_cast<cFloat *> ((void *) field->mFrontGhost());
		cFloat *m   = static_cast<cFloat *> ((void *) field->m2Start());

		fSize = field->DataSize();
		shift = field->DataAlign()/fSize;

		LogMsg (VERB_NORMAL, "Calling foldField mAlign=%d, fSize=%d, shift=%d", field->DataAlign(), fSize, shift);

		for (size_t iz=0; iz < Nz; iz++)
		{
			memcpy (mg1, &m[Nxy*iz], sizeof(cFloat)*Nxy);

			#pragma omp parallel for schedule(static)
			for (size_t iy=0; iy < Nx/shift; iy++)
				for (size_t ix=0; ix < Nx; ix++)
					for (size_t sy=0; sy<shift; sy++)
					{
						size_t oIdx = (iy+sy*(Nx/shift))*Nx + ix;
						size_t dIdx = iz*Nxy + ((size_t) (iy*Nx*shift + ix*shift + sy));

						m[dIdx] = mg1[oIdx];
					}
		}

		field->setM2Folded(true);
		LogMsg (VERB_HIGH, "[Folder] Field M2 folded (from M2Start)");
		somethingdone = true;
		return;
	}

	template<typename cFloat>
	void	Folder::unfoldM2()
	{
		if (!field->M2Folded() || field->Device() == DEV_GPU || Nx == Nxy)
			return;

		cFloat *mg1 = static_cast<cFloat *> ((void *) field->mFrontGhost());
		cFloat *m   = static_cast<cFloat *> ((void *) field->m2Start());

		fSize = field->DataSize();
		shift = field->DataAlign()/fSize;

		LogMsg (VERB_NORMAL, "Calling unfoldField mAlign=%d, fSize=%d, shift=%d", field->DataAlign(), fSize, shift);

		for (size_t iz=0; iz < Nz; iz++)
		{
			memcpy (mg1, m + Nxy*iz, fSize*Nxy);

			#pragma omp parallel for schedule(static)
			for (size_t iy=0; iy < Nx/shift; iy++)
				for (size_t ix=0; ix < Nx; ix++)
					for (size_t sy=0; sy<shift; sy++)
					{
						size_t oIdx = iy*Nx*shift + ix*shift + sy;
						size_t dIdx = iz*Nxy + (iy+sy*(Nx/shift))*Nx + ix;

						m[dIdx]    = mg1[oIdx];
					}
		}

		field->setM2Folded(false);
	 	LogMsg (VERB_HIGH, "[Folder] Field M2 unfolded (m2Start)");
		somethingdone = true;
		return;
	}

	template<typename Float>
	void Folder::foldM2AsComplex()
	{
		if (field->M2Folded() || field->Device() == DEV_GPU)
			return;

		using cFloat = complex<Float>;
		cFloat *mg = static_cast<cFloat *>(field->m2Cpu());
		cFloat *m  = mg + Nxy*field->getNg();

		fSize = 2*field->DataSize();
		shift = field->DataAlign()/fSize;

		const bool foldX = (Ny == 1);
		const size_t N1 = foldX ? 1 : Nx;
		const size_t N2 = foldX ? Nx : Ny;
		if (N2 % shift != 0) {
			LogError("[foldM2AsComplex] ERROR: fold dimension %zu not divisible by shift=%zu", N2, shift);
			return;
		}
		const size_t Nv = N2/shift;

		LogMsg(VERB_NORMAL, "[foldM2AsComplex] Folding M2 (%c) mAlign=%d, fSize=%d, shift=%d",
		       foldX ? 'X' : 'Y', field->DataAlign(), fSize, shift);

		for (size_t iz = 0; iz < Nz; ++iz) {
			memcpy(mg, m + Nxy*iz, sizeof(cFloat)*Nxy);
			#pragma omp parallel for schedule(static)
			for (size_t i_v = 0; i_v < Nv; ++i_v)
				for (size_t ix = 0; ix < N1; ++ix)
					for (size_t s = 0; s < shift; ++s) {
						const size_t oIdx = (i_v + s*Nv)*N1 + ix;
						const size_t dIdx = iz*Nxy + i_v*N1*shift + ix*shift + s;
						m[dIdx] = mg[oIdx];
					}
		}

		field->setM2Folded(true);
		somethingdone = true;
	}

	template<typename Float>
	void Folder::unfoldM2AsComplex()
	{
		if (!field->M2Folded() || field->Device() == DEV_GPU)
			return;

		using cFloat = complex<Float>;
		cFloat *mg = static_cast<cFloat *>(field->m2Cpu());
		cFloat *m  = mg + Nxy*field->getNg();

		fSize = 2*field->DataSize();
		shift = field->DataAlign()/fSize;

		const bool foldX = (Ny == 1);
		const size_t N1 = foldX ? 1 : Nx;
		const size_t N2 = foldX ? Nx : Ny;
		if (N2 % shift != 0) {
			LogError("[unfoldM2AsComplex] ERROR: fold dimension %zu not divisible by shift=%zu", N2, shift);
			return;
		}
		const size_t Nv = N2/shift;

		LogMsg(VERB_NORMAL, "[unfoldM2AsComplex] Unfolding M2 (%c) mAlign=%d, fSize=%d, shift=%d",
		       foldX ? 'X' : 'Y', field->DataAlign(), fSize, shift);

		for (size_t iz = 0; iz < Nz; ++iz) {
			memcpy(mg, m + Nxy*iz, sizeof(cFloat)*Nxy);
			#pragma omp parallel for schedule(static)
			for (size_t i_v = 0; i_v < Nv; ++i_v)
				for (size_t ix = 0; ix < N1; ++ix)
					for (size_t s = 0; s < shift; ++s) {
						const size_t oIdx = i_v*N1*shift + ix*shift + s;
						const size_t dIdx = (i_v + s*Nv)*N1 + ix;
						m[iz*Nxy + dIdx] = mg[oIdx];
					}
		}

		field->setM2Folded(false);
		somethingdone = true;
	}




// CHECK IF NEEDED
template<typename cFloat>	// Only rank 0 can do this, and currently we quietly exist for any other rank. This can generate bugs if sZ > local Nz
void	Folder::unfoldM22D (const size_t sZ)
{
	if ((sZ < 0) || (sZ > field->Depth()) || field->Device() == DEV_GPU)
		return;

	cFloat *mg1 = static_cast<cFloat *> ((void *) field->m2FrontGhost());
	cFloat *m   = static_cast<cFloat *> ((void *) field->m2Start());

	if (!field->M2Folded())
	{
		LogMsg (VERB_HIGH, "unfoldM22D called in an unfolded configuration, copying data to ghost zones");LogFlush();
		memcpy (mg1, &m[Nxy*sZ], sizeof(cFloat)*Nxy);
		return;
	}

	fSize = field->DataSize();
	shift = field->DataAlign()/fSize;

	LogMsg (VERB_HIGH, "Calling unfoldM22D mAlign=%d, fSize=%d, shift=%d", field->DataAlign(), fSize, shift);LogFlush();

	#pragma omp parallel for schedule(static)
	for (size_t iy=0; iy < Nx/shift; iy++)
		for (size_t ix=0; ix < Nx; ix++)
			for (size_t sy=0; sy<shift; sy++)
			{
				size_t oIdx = (sZ)*Nxy + iy*Nx*shift + ix*shift + sy;
				size_t dIdx = (iy+sy*(Nx/shift))*Nx + ix;
				//this copies m into buffer 1
				mg1[dIdx]	= m[oIdx];
			}

	LogMsg (VERB_HIGH, "Slice unfolded");
	somethingdone = true;
	return;
}




/* basic operator */

void	Folder::operator()(FoldType fType, size_t cZ)
{
	somethingdone = false;
	// Careful here, GPUS might want to call CPU routines
	if (field->Device() == DEV_GPU)
		return;

	LogMsg  (VERB_NORMAL, "[Fold] Called with m/v %s and m2 %s",field->Folded()? "FOLDED":"UNFOLDED",field->M2Folded()? "FOLDED":"UNFOLDED");
	LogFlush();

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

		case	UNFOLD_SLICEXZ:

			setName("Unfold slice XZ");
			add(0., field->NX()*field->NZ()*field->DataSize()*2.e-9);

			switch(field->Precision())
			{
				case	FIELD_DOUBLE:

					switch (field->Field())
					{
						case	FIELD_SAXION:
						case	FIELD_NAXION:
							unfoldField2DXZ<complex<double>>(cZ);
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
						case  FIELD_PAXION:
							unfoldField2DXZ<double>(cZ);
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
							unfoldField2DXZ<complex<float>>(cZ);
							break;

						case	FIELD_AXION_MOD:
						case	FIELD_AXION:
						case	FIELD_WKB:
						case  FIELD_PAXION:
							unfoldField2DXZ<float>(cZ);
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

		case FOLD_M2_AS_CMPLX:
			setName("Fold M2 as complex");
			add(0., field->Size()*field->DataSize()*4.e-9);
			if (field->Precision() == FIELD_DOUBLE)
				foldM2AsComplex<double>();
			else if (field->Precision() == FIELD_SINGLE)
				foldM2AsComplex<float>();
			break;

		case UNFOLD_M2_AS_CMPLX:
			setName("Unfold M2 as complex");
			add(0., field->Size()*field->DataSize()*4.e-9);
			if (field->Precision() == FIELD_DOUBLE)
				unfoldM2AsComplex<double>();
			else if (field->Precision() == FIELD_SINGLE)
				unfoldM2AsComplex<float>();
			break;

		default:
			LogError ("Unrecognized folding option");
			break;
	}

	prof.stop();

	prof.add(Name(), GFlops(), GBytes());	// In truth is x4 because we move data to the ghost slices before folding/unfolding

	if (somethingdone)
		LogMsg  (VERB_HIGH, "Folder %s reporting %lf GFlops %lf GBytes", Name().c_str(), prof.Prof()[Name()].GFlops(), prof.Prof()[Name()].GBytes());

	reset();
}
