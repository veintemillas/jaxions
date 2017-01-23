#include<cstdlib>
#include<cstring>
#include<complex>

#include"scalar/scalarField.h"
#include"enum-field.h"

#ifdef	USE_GPU
	#include<cuda.h>
	#include<cuda_runtime.h>
	#include "cudaErrors.h"
#endif

#define	_FOLDER_CLASS_

using namespace std;

class	Folder
{
	private:

	int shift;
	int fSize;
	const int Lz;
	const int n1;
	const int n2;
	const int n3;

	Scalar *field;

	template<typename cFloat>
	void	foldField();

	template<typename cFloat>
	void	unfoldField();

	template<typename cFloat>
	void	unfoldField2D(const size_t cZ);

	public:

		 Folder(Scalar *scalar);
		~Folder() {};

	void	operator()(FoldType fType, size_t Cz=0);
};

	Folder:: Folder(Scalar *scalar) : field(scalar), Lz(scalar->Depth()), n1(scalar->Length()), n2(scalar->Surf()), n3(scalar->Size())
{
}

template<typename cFloat>
void	Folder::foldField()
{
	if (field->Folded())
		return;

	const int fSize = field->DataSize();
	const int shift = field->DataAlign()/fSize;
	printf("Foldfield mAlign=%d, fSize=%d, shift=%d, n2=%d ... \n", field->DataAlign(), field->DataSize(), shift, n2);

	cFloat *m = static_cast<cFloat *> ((void *) field->mCpu());
	cFloat *v = static_cast<cFloat *> ((void *) field->vCpu());

	for (size_t iz=0; iz < Lz; iz++)
	{
		//printf("slice %d ",iz);fflush(stdout);
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

	printf("Done from inside Folder!\n");
	return;
}

template<typename cFloat>
void	Folder::unfoldField()
{
	if (!field->Folded())
		return;

	cFloat *m = static_cast<cFloat *> ((void *) field->mCpu());
	cFloat *v = static_cast<cFloat *> ((void *) field->vCpu());

	const int fSize = field->DataSize();
	const int shift = field->DataAlign()/fSize;

	//printf("Unfoldfield mAlign=%d, fSize=%d, shift=%d, n2=%d ... \n", field->DataAlign(), field->DataSize(),shift,n2);

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
	//printf("Done!\n");
	return;
}

template<typename cFloat>	// Only rank 0 can do this, and currently we quietly exist for any other rank. This can generate bugs if sZ > local Lz
void	Folder::unfoldField2D (const size_t sZ)
{
	if ((sZ < 0) || (sZ > field->Depth()))
		return;

	cFloat *m = static_cast<cFloat *> (field->mCpu());
	cFloat *v = static_cast<cFloat *> (field->vCpu());

	if (!field->Folded())
	{
		memcpy ( m,        &m[(sZ+1)*n2], sizeof(cFloat)*n2);
		memcpy (&m[n2+n3], &v[sZ*n2],     sizeof(cFloat)*n2);
		return;
	}

	const int fSize = field->DataSize();
	const int shift = field->DataAlign()/fSize;

	//unfolds m(slice[sZ]]) into buffer 1 and v(slice[sZ]) into buffer2
	//printf("MAP: Unfold-2D mAlign=%d, fSize=%d, shift=%d \n", field->DataAlign(), field->DataSize(),shift);
	//fflush(stdout);

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
	return;
}

void	Folder::operator()(FoldType fType, size_t cZ)
{
	switch (fType)
	{
		case	FOLD_ALL:

			switch(field->Precision())
			{
				case	FIELD_DOUBLE:

					switch (field->Field())
					{
						case	FIELD_SAXION:
							foldField<complex<double>>();
							break;

						case	FIELD_AXION:
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

						case	FIELD_AXION:
							foldField<float>();
							break;

						default:
							break;
					}

					break;
			}

			break;

		case	UNFOLD_ALL:

			switch(field->Precision())
			{
				case	FIELD_DOUBLE:

					switch (field->Field())
					{
						case	FIELD_SAXION:
							unfoldField<complex<double>>();
							break;

						case	FIELD_AXION:
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

						case	FIELD_AXION:
							unfoldField<float>();
							break;

						default:
							break;
					}

					break;
			}

			break;

		case	UNFOLD_SLICE:

			switch(field->Precision())
			{
				case	FIELD_DOUBLE:

					switch (field->Field())
					{
						case	FIELD_SAXION:
							unfoldField2D<complex<double>>(cZ);
							break;

						case	FIELD_AXION:
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

						case	FIELD_AXION:
							unfoldField2D<float>(cZ);
							break;

						default:
							break;
					}

					break;
			}

			break;

		default:
			printf ("Unrecognized option\n");
			break;
	}
}
