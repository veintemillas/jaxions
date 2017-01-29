#include <complex>
#include <vector>
#include <fftw3-mpi.h>
#include <omp.h>
#include "enum-field.h"

using namespace std;

fftw_plan p, pb;
fftwf_plan pf, pfb;

static bool iFFT = false, iFFTPlans = false, single = false, useThreads = true;

void	initFFT	()
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
		fftw_mpi_init();
	} else {
		int nThreads = omp_get_max_threads();
		printf ("Using %d threads for the FFTW\n", nThreads);
		fflush (stdout);
		fftw_mpi_init();
		fftw_plan_with_nthreads(nThreads);
	}

	iFFT = true;
}

void	initFFTPlans	(void *m, void *m2, const size_t n1, const size_t Lz, FieldPrecision prec, bool lowmem)
{
	if (!iFFT)
		initFFT();

	int rank;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
	if (rank == 0)
	{
		if(!fftw_import_wisdom_from_filename("wisdomsave.txt"))
		{
			printf("  Warning: could not import wisdom\n");
		}
		else
		{
			printf("  Wisdom file loaded\n\n");
		}
	}

	printf ("  Planning 3d (%lld x %lld x %lld)\n", (ptrdiff_t) n1, (ptrdiff_t) n1, (ptrdiff_t) Lz);
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

	fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) fftw_export_wisdom_to_filename("wisdomsave.txt");

	printf ("  Wisdom saved\n");
	printf ("Done!\n");
	fflush (stdout);

	iFFTPlans = true;
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
	int rank;
	fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) fftw_export_wisdom_to_filename("wisdomsave");

	printf ("  Wisdom saved once more in rank %d\n",rank);

	if (!iFFT)
		return;

	if (useThreads)
		fftwf_cleanup_threads();
	else
		fftwf_cleanup();
}

void	closeFFTPlans	()
{
	int rank;
	fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) fftw_export_wisdom_to_filename("wisdomsave");

	printf ("  Wisdom saved once more in rank %d\n",rank);

	if (!iFFTPlans)
		return;

	if (single)
	{
		fftwf_destroy_plan(pf);
		fftwf_destroy_plan(pfb);
	}
	else
	{
		fftw_destroy_plan(p);
		fftw_destroy_plan(pb);
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
	if (!iFFT)
		initFFT();

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
			p2  = fftw_mpi_plan_dft_3d(Lz, n1, n1, static_cast<fftw_complex*>(m2), static_cast<fftw_complex*>(m2), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE);
		}
//		p  = fftw_plan_many_dft(2, nD, Lz, static_cast<fftw_complex*>(m), NULL, 1, dist, static_cast<fftw_complex*>(m), NULL, 1, dist, FFTW_FORWARD,  FFTW_MEASURE);
//		pb = fftw_plan_many_dft(2, nD, Lz, static_cast<fftw_complex*>(m), NULL, 1, dist, static_cast<fftw_complex*>(m), NULL, 1, dist, FFTW_BACKWARD, FFTW_MEASURE);
		break;

		case FIELD_SINGLE:

		single = true;
		if (lowmem) {
			printf("Spectrum not available in lowmem until the end");
		} else {
			pf2  = fftwf_mpi_plan_dft_3d(Lz, n1, n1, static_cast<fftwf_complex*>(m2), static_cast<fftwf_complex*>(m2), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE);
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
		fftwf_destroy_plan(pf2);
	else
		void fftw_cleanup_threads(void);
}

//----------------------------------------------------------------------------------------------------------
// 			FFT HALO SMOOTHING M->V (DENS->V AND BACK)
//----------------------------------------------------------------------------------------------------------

fftw_plan p3,p3b;
fftwf_plan pf3, pf3b;


static bool iFFThalo = false;

void	initFFThalo	(void *m, void *v, const size_t n1, const size_t Lz, FieldPrecision prec)
{
	if (!iFFT)
		initFFT();

	printf ("Initializing FFTSpectrum halo...\n");
	fflush (stdout);

	if (iFFThalo == true)
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
			p3   = fftw_mpi_plan_dft_r2c_3d(Lz, n1, n1, static_cast<double*>(m), static_cast<fftw_complex*>(v), MPI_COMM_WORLD, FFTW_ESTIMATE );
			p3b  = fftw_mpi_plan_dft_c2r_3d(Lz, n1, n1, static_cast<fftw_complex*>(v), static_cast<double*>(m), MPI_COMM_WORLD, FFTW_ESTIMATE );

		break;

		case FIELD_SINGLE:

		single = true;
			pf3  = fftwf_mpi_plan_dft_r2c_3d(Lz, n1, n1, static_cast<float*>(m), static_cast<fftwf_complex*>(v), MPI_COMM_WORLD, FFTW_ESTIMATE );
			pf3b = fftwf_mpi_plan_dft_c2r_3d(Lz, n1, n1, static_cast<fftwf_complex*>(v), static_cast<float*>(m), MPI_COMM_WORLD, FFTW_ESTIMATE );
		break;

		default:

		break;
	}

	printf ("  Plan_Spectrum Ok\n");
	fflush (stdout);

	iFFThalo = true;
}


void	runFFThalo(int sign)
{
	printf ("Halo FFT...");
	fflush (stdout);

	switch (sign)
	{
		case FFTW_FORWARD:

		if (single)
			fftwf_execute(pf3);
		else
			fftw_execute(p3);
		break;

		case FFTW_BACKWARD:

		if (single)
			fftwf_execute(pf3b);
		else
			fftw_execute(p3b);
		break;
	}
	printf ("Done!\n");
	fflush (stdout);
}

void	closeFFThalo()
{
	if (!iFFThalo)
		return;

	if (single)
	{
		fftwf_destroy_plan(pf3);
		fftwf_destroy_plan(pf3b);
	}
	else
	{
		fftw_destroy_plan(p3);
		fftw_destroy_plan(p3b);
	}
}
