#include <complex>
#include <vector>
#include <fftw3-mpi.h>
#include <omp.h>
#include "enum-field.h"

using namespace std;

fftw_plan p, pb;
fftwf_plan pf, pfb;

static bool iFFT = false, single = false, useThreads = true;

void	initFFT	(void *m, void *m2, const size_t n1, const size_t Lz, FieldPrecision prec, bool lowmem)
{
	printf ("Initializing FFT...\n");
	fflush (stdout);

	if (iFFT == true)
	{
		printf ("Already initialized!!\n");
		fflush (stdout);
	}

	if (!fftw_init_threads())
	{
		printf ("Error initializing FFT with threads\n");
		fflush (stdout);
		useThreads = false;
	} else {
		int nThreads = omp_get_max_threads();
		printf ("Using %d threads for the FFTW\n", nThreads);
		fflush (stdout);
		fftw_plan_with_nthreads(nThreads);
	}

	fftw_mpi_init();

	printf ("  MPI Ok\n");
	fflush (stdout);

	printf ("  Plan 3d (%lld x %lld x %lld)\n", (ptrdiff_t) n1, (ptrdiff_t) n1, (ptrdiff_t) Lz);
	fflush (stdout);

	switch (prec)
	{
		case FIELD_DOUBLE:

		single = false;
		if (lowmem) {
			p  = fftw_mpi_plan_dft_3d(Lz, n1, n1, static_cast<fftw_complex*>(m), static_cast<fftw_complex*>(m), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE);
			pb = fftw_mpi_plan_dft_3d(Lz, n1, n1, static_cast<fftw_complex*>(m), static_cast<fftw_complex*>(m), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
		} else {
			p  = fftw_mpi_plan_dft_3d(Lz, n1, n1, static_cast<fftw_complex*>(m), static_cast<fftw_complex*>(m2), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE);
			pb = fftw_mpi_plan_dft_3d(Lz, n1, n1, static_cast<fftw_complex*>(m2), static_cast<fftw_complex*>(m), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
		}
//		p  = fftw_plan_many_dft(2, nD, Lz, static_cast<fftw_complex*>(m), NULL, 1, dist, static_cast<fftw_complex*>(m), NULL, 1, dist, FFTW_FORWARD,  FFTW_MEASURE);
//		pb = fftw_plan_many_dft(2, nD, Lz, static_cast<fftw_complex*>(m), NULL, 1, dist, static_cast<fftw_complex*>(m), NULL, 1, dist, FFTW_BACKWARD, FFTW_MEASURE);
		break;

		case FIELD_SINGLE:

		single = true;
		if (lowmem) {
			pf  = fftwf_mpi_plan_dft_3d(Lz, n1, n1, static_cast<fftwf_complex*>(m), static_cast<fftwf_complex*>(m), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE);
			pfb = fftwf_mpi_plan_dft_3d(Lz, n1, n1, static_cast<fftwf_complex*>(m), static_cast<fftwf_complex*>(m), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
		} else {
			pf  = fftwf_mpi_plan_dft_3d(Lz, n1, n1, static_cast<fftwf_complex*>(m), static_cast<fftwf_complex*>(m2), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE);
			pfb = fftwf_mpi_plan_dft_3d(Lz, n1, n1, static_cast<fftwf_complex*>(m2), static_cast<fftwf_complex*>(m), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
		}
//		pf  = fftwf_plan_many_dft(2, nD, Lz, static_cast<fftwf_complex*>(m), NULL, 1, dist, static_cast<fftwf_complex*>(m), NULL, 1, dist, FFTW_FORWARD,  FFTW_MEASURE);
//		pfb = fftwf_plan_many_dft(2, nD, Lz, static_cast<fftwf_complex*>(m), NULL, 1, dist, static_cast<fftwf_complex*>(m), NULL, 1, dist, FFTW_BACKWARD, FFTW_MEASURE);
		break;

		default:

		break;
	}

	printf ("  Plans Ok\n");
	printf ("Done!\n");
	fflush (stdout);

	iFFT = true;
}

void	runFFT(int sign)
{
	printf ("Executing FFT...\n");
	fflush (stdout);

	switch (sign)
	{
		case FFTW_FORWARD:

		if (single)
			fftwf_execute(pf);
		else
			fftw_execute(p);
		break;

		case FFTW_BACKWARD:

		if (single)
			fftwf_execute(pfb);
		else
			fftw_execute(pb);
		break;
	}

	printf ("Done!\n");
	fflush (stdout);
}

void	closeFFT	()
{
	if (!iFFT)
		return;

	if (single)
	{
		fftwf_destroy_plan(pf);
		fftwf_destroy_plan(pfb);

		if (useThreads)
			fftwf_cleanup_threads();
		else
			fftwf_cleanup();
	}
	else
	{
		fftw_destroy_plan(p);
		fftw_destroy_plan(pb);

		if (useThreads)
			fftw_cleanup_threads();
		else
			fftw_cleanup();
	}
}



//----------------------------------------------------------------------------------------------------------
// 			FFT Spectrum
//----------------------------------------------------------------------------------------------------------

fftw_plan p2;
fftwf_plan pf2;


static bool iFFTSpectrum = false;

void	initFFTSpectrum	(void *m2, const size_t n1, const size_t Lz, FieldPrecision prec, bool lowmem)
{

	printf ("Initializing FFTSpectrum...\n");
	fflush (stdout);

	if (iFFTSpectrum == true)
	{
		printf ("Already initialized!!\n");
		fflush (stdout);
	}

	//fftw_mpi_init();

	//printf ("  MPI Ok\n");
	//fflush (stdout);

	printf ("  Plan 3d (%lld x %lld x %lld)\n", (ptrdiff_t) n1, (ptrdiff_t) n1, (ptrdiff_t) Lz);
	fflush (stdout);

	switch (prec)
	{
		case FIELD_DOUBLE:

		single = false;
		if (lowmem) {
			printf("Spectrum not available in lowmem until the end");
		} else {
			p2  = fftw_mpi_plan_dft_3d(Lz, n1, n1, static_cast<fftw_complex*>(m2), static_cast<fftw_complex*>(m2), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE);
		}
//		p  = fftw_plan_many_dft(2, nD, Lz, static_cast<fftw_complex*>(m), NULL, 1, dist, static_cast<fftw_complex*>(m), NULL, 1, dist, FFTW_FORWARD,  FFTW_MEASURE);
//		pb = fftw_plan_many_dft(2, nD, Lz, static_cast<fftw_complex*>(m), NULL, 1, dist, static_cast<fftw_complex*>(m), NULL, 1, dist, FFTW_BACKWARD, FFTW_MEASURE);
		break;

		case FIELD_SINGLE:

		single = true;
		if (lowmem) {
			printf("Spectrum not available in lowmem until the end");
		} else {
			pf2  = fftwf_mpi_plan_dft_3d(Lz, n1, n1, static_cast<fftwf_complex*>(m2), static_cast<fftwf_complex*>(m2), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE);
		}
//		pf  = fftwf_plan_many_dft(2, nD, Lz, static_cast<fftwf_complex*>(m), NULL, 1, dist, static_cast<fftwf_complex*>(m), NULL, 1, dist, FFTW_FORWARD,  FFTW_MEASURE);
//		pfb = fftwf_plan_many_dft(2, nD, Lz, static_cast<fftwf_complex*>(m), NULL, 1, dist, static_cast<fftwf_complex*>(m), NULL, 1, dist, FFTW_BACKWARD, FFTW_MEASURE);
		break;

		default:

		break;
	}

	printf ("  Plan_Spectrum Ok\n");
	fflush (stdout);

	iFFTSpectrum = true;
}

void	runFFTSpectrum(int sign)
{
	printf ("Spectrum FFT... ");
	fflush (stdout);

	if (single) {
		fftwf_execute(pf2);
	}
	else
	{
		fftw_execute(p2);
	}

	printf ("Done! ");
	fflush (stdout);
}

void	closeFFTSpectrum	()
{
	if (!iFFTSpectrum)
		return;

	if (single)
	{
		fftwf_destroy_plan(pf2);
		void fftwf_cleanup_threads(void);
	}
	else
	{
		fftw_destroy_plan(p2);
		void fftw_cleanup_threads(void);
	}
}
