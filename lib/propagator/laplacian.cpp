#include <string>
#include <complex>
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

class	Laplacian : public Tunable
{
	private:

	const FieldPrecision	precision;
	const size_t		Lx;
	const size_t		Lz;
	const size_t		Sf;

	Scalar			*field;

	template<class cFloat>
	void			lapCpu(std::string name);

	template<class cFloat>
	void			lapGpu(std::string name);

	template<class cFloat>
	void			lapXeon(std::string name);

	public:

		Laplacian (Scalar *field) : precision(field->Precision()), Lx(field->Length()), Lz(field->Depth()), Sf(field->Surf()), field(field) {
		if (field->LowMem()) {
			LogError ("Error: laplacian not supported in lowmem runs");
			exit(0);
		}

		if (commSize() > 1) {
			LogError ("Error: laplacian not supported in MPI runs");
			exit(0);
		}
	}

	void	sRunCpu	();	// Saxion laplacian
	void	sRunGpu	();
	void	sRunXeon();

	void	tRunCpu	();	// Axion laplacian
	void	tRunGpu	();
	void	tRunXeon();
};

void	Laplacian::sRunGpu	()
{
#ifdef	USE_GPU
#else
	LogError ("Error: gpu support not built");
	exit(1);
#endif
}


template<class cFloat>
void	Laplacian::lapCpu	(std::string name)
{
	auto &planFFT = AxionFFT::fetchPlan(name);
	planFFT.run(FFT_FWD);

	cFloat *mData = static_cast<cFloat*> (field->m2Cpu());
	const int hLx = Lx>>1;
	const int hLz = Lz>>1;

	#pragma omp parallel for schedule(static) default(shared)
	for (int oz = 0; oz < Lz; oz++)
	{
        	int pz = oz;

		if (oz > hLz)
        		pz = oz - Lz;

		size_t pz2 = pz*pz;
		size_t idz = (oz+1)*Sf;

		for (int oy = 0; oy < Lx; oy++)
        	{
			int py = oy ;
			if (oy > hLx)
				py = oy - Lx;

			size_t py2 = py*py;
			size_t idy = py*Lx;

			for (int ox = 0; ox < Lx; ox++)
			{
				size_t idx = ox + idy + idz;

				int px = ox;
				if (ox > hLx)
					px = ox - Lx;

				size_t p2 = pz2 + py2 + px*px;

				mData[idx] *= (cFloat) (p2);
			}
	        }
	}

	planFFT.run(FFT_BCK);
}

void	Laplacian::sRunCpu	()
{
	switch (precision) {
		case FIELD_SINGLE:
			lapCpu<std::complex<float>> (std::string("SpSx"));
			break;

		case FIELD_DOUBLE:
			lapCpu<std::complex<double>>(std::string("SpSx"));
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
			//lapCpu<float> (std::string("SpAx"));
			break;

		case FIELD_DOUBLE:
			//lapCpu<double>(std::string("SpAx"));
			break;
	}
}

using	namespace profiler;

void	applyLaplacian	(Scalar *field)
{
	LogMsg	(VERB_HIGH, "Called laplacian");
	profiler::Profiler &prof = getProfiler(PROF_PROP);

	Laplacian *lap = new Laplacian(field);

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	prof.start();

	switch (field->Field()) {
		case FIELD_AXION:
			lap->setName("Laplacian Axion");

			switch (field->Device()) {
				case DEV_GPU:
					lap->tRunGpu ();
					break;
				case DEV_CPU:
					lap->tRunCpu ();
					break;
				default:
					LogError ("Not a valid device");
					prof.stop();
					delete lap;
					return;
			}

			lap->add(16.*4.*field->Size()*1.e-9, 10.*4.*field->DataSize()*field->Size()*1.e-9);

			break;

		case FIELD_SAXION:
			lap->setName("Laplacian Saxion");

			switch (field->Device()) {
				case DEV_GPU:
					lap->sRunGpu ();
					break;
				case DEV_CPU:
					lap->sRunCpu ();
					break;
				default:
					LogError ("Not a valid device");
					prof.stop();
					delete lap;
					return;
			}
			break;

		default:
			LogError ("Invalid field type");
			prof.stop();
			delete lap;
			return;
	}

	prof.stop();

	prof.add(lap->Name(), lap->GFlops(), lap->GBytes());

	LogMsg	(VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", lap->Name().c_str(), prof.Prof()[lap->Name()].GFlops(), prof.Prof()[lap->Name()].GBytes());

	delete	lap;

	return;
}
