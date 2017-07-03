#include <complex>
#include <vector>
#include <fftw3-mpi.h>
#include <omp.h>
#include "enum-field.h"
#include "comms/comms.h"

using namespace std;

#define printMpi(...) do {		\
	if (!commRank()) {		\
	  printf(__VA_ARGS__);  	\
	  fflush(stdout); }		\
}	while (0)

fftw_plan p, pb;
fftwf_plan pf, pfb, pfb2;

static bool iFFT = false, iFFTPlans = false, single = false, useThreads = true;

void	initFFT	(const FieldPrecision &prec)
{
	printMpi ("  Initializing FFT (#MPI=%d)...\n",commSize());
	fflush (stdout);

	if (iFFT == true)
	{
		printMpi ("  Already initialized!!\n");
		fflush (stdout);
	}

	switch (prec)
	{
		case FIELD_DOUBLE:

		if (!fftw_init_threads())
		{
			printf ("  Error initializing FFT with threads\n");
			fflush (stdout);
			useThreads = false;
			fftw_mpi_init();
		} else {
			int nThreads = omp_get_max_threads();
			printMpi ("  Using %d threads for the FFTW\n", nThreads);
			fflush (stdout);
			fftw_mpi_init();
			fftw_plan_with_nthreads(nThreads);
		}

		break;

		case FIELD_SINGLE:

		if (!fftwf_init_threads())
		{
			printf ("  Error initializing FFT with threads\n");
			fflush (stdout);
			useThreads = false;
			fftwf_mpi_init();
		} else {
			int nThreads = omp_get_max_threads();
			printMpi ("  Using %d threads for the FFTW\n", nThreads);
			fflush (stdout);
			fftwf_mpi_init();
			fftwf_plan_with_nthreads(nThreads);
		}

		break;

		default:

		printf ("Unrecognized precision\n");
		return;
		break;
	}

	iFFT = true;
}

void	initFFTPlans	(void *m, void *m2, const size_t n1, const size_t Tz, FieldPrecision prec, bool lowmem)
{
	if (!iFFT)
		initFFT(prec);

	int rank;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printMpi ("  Planning 3d (%lld x %lld x %lld)\n", (ptrdiff_t) n1, (ptrdiff_t) n1, (ptrdiff_t) Tz);
	fflush (stdout);

	switch (prec)
	{
		case FIELD_DOUBLE:

		if (rank == 0) {
			if (fftw_import_wisdom_from_filename("fftWisdom.double") == 0)
				printf ("  Warning: could not import wisdom from fftWisdom.double\n");
		}

		fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);

		single = false;
		if (lowmem) {
			p  = fftw_mpi_plan_dft_3d(Tz, n1, n1, static_cast<fftw_complex*>(m), static_cast<fftw_complex*>(m), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE);
			pb = fftw_mpi_plan_dft_3d(Tz, n1, n1, static_cast<fftw_complex*>(m), static_cast<fftw_complex*>(m), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
		} else {
			p  = fftw_mpi_plan_dft_3d(Tz, n1, n1, static_cast<fftw_complex*>(m), static_cast<fftw_complex*>(m2), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE);
			pb = fftw_mpi_plan_dft_3d(Tz, n1, n1, static_cast<fftw_complex*>(m2), static_cast<fftw_complex*>(m), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
		}

		fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
		if (rank == 0) { fftw_export_wisdom_to_filename("fftWisdom.double"); }
		printMpi ("  Wisdom saved\n");

		break;

		case FIELD_SINGLE:

		single = true;

		if (rank == 0) {
			if (fftwf_import_wisdom_from_filename("../fftWisdom.single") == 0)
				printf ("  Warning: could not import wisdom from fftWisdom.single\n");
		}

		fftwf_mpi_broadcast_wisdom(MPI_COMM_WORLD);

		if (lowmem) {
			pf  = fftwf_mpi_plan_dft_3d(Tz, n1, n1, static_cast<fftwf_complex*>(m), static_cast<fftwf_complex*>(m), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE);
			pfb = fftwf_mpi_plan_dft_3d(Tz, n1, n1, static_cast<fftwf_complex*>(m), static_cast<fftwf_complex*>(m), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
		} else {
			pf  = fftwf_mpi_plan_dft_3d(Tz, n1, n1, static_cast<fftwf_complex*>(m), static_cast<fftwf_complex*>(m2), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_PATIENT);
			pfb = fftwf_mpi_plan_dft_3d(Tz, n1, n1, static_cast<fftwf_complex*>(m2), static_cast<fftwf_complex*>(m), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_PATIENT);
			pfb2 = fftwf_mpi_plan_dft_3d(Tz, n1, n1, static_cast<fftwf_complex*>(m2), static_cast<fftwf_complex*>(m), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_PATIENT);
		}
//		pf  = fftwf_plan_many_dft(2, nD, Lz, static_cast<fftwf_complex*>(m), NULL, 1, dist, static_cast<fftwf_complex*>(m), NULL, 1, dist, FFTW_FORWARD,  FFTW_MEASURE);
//		pfb = fftwf_plan_many_dft(2, nD, Lz, static_cast<fftwf_complex*>(m), NULL, 1, dist, static_cast<fftwf_complex*>(m), NULL, 1, dist, FFTW_BACKWARD, FFTW_MEASURE);

		fftwf_mpi_gather_wisdom(MPI_COMM_WORLD);
		if (rank == 0) fftwf_export_wisdom_to_filename("../fftWisdom.single");

		break;

		default:

		break;
	}

	printMpi ("  Plans Ok\n");

	iFFTPlans = true;
}

void	runFFT(int sign)
{
	printMpi ("Executing FFT...\n");
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

		case 2:

		if (single)
			fftwf_execute(pfb2);
		else
			//fftw_execute(pb);
		break;


	}

	printMpi ("Done!\n");
	fflush (stdout);
}

void	closeFFT	()
{
	if (!iFFT)
		return;

	if (useThreads)
		fftwf_cleanup_threads();
	else
		fftwf_cleanup();
}

void	closeFFTPlans	()
{
	if (!iFFTPlans)
		return;

	if (single)
	{
		fftwf_destroy_plan(pf);
		fftwf_destroy_plan(pfb);
		fftwf_destroy_plan(pfb2);
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

void	initFFTSpectrum	(void *m2, const size_t n1, const size_t Tz, FieldPrecision prec, bool lowmem)
{
	if (!iFFT)
		initFFT(prec);

	printMpi ("Initializing FFTSpectrum...\n");
	fflush (stdout);

	if (iFFTSpectrum == true)
	{
		printMpi ("Already initialized!!\n");
		fflush (stdout);
	}

	//fftw_mpi_init();
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//printf ("  MPI Ok\n");
	//fflush (stdout);


	ptrdiff_t alloc_local, local_n0, local_0_start, local_n1, local_1_start;

	// alloc_local = fftw_mpi_local_size_3d(n1, n1, n1, MPI_COMM_WORLD, &local_n0, &local_0_start);
	// alloc_local = fftw_mpi_local_size_3d_transposed(
	// 						 Tz, n1, n1, MPI_COMM_WORLD,
	// 						 &local_n0, &local_0_start,
	// 						 &local_n1, &local_1_start);
	//
	// printf ("	 rank=%d - local_nz=%lld - local_nz_start=%lld - alloc_need =%lld*n2 - Lz=%lld\n", rank, local_n0, local_0_start, alloc_local/(n1*n1), Tz);
	// printf ("	 transpo - local_ny=%lld - local_ny_start=%lld \n", local_n1, local_1_start);

	printMpi ("  Plan 3d (%lld x %lld x %lld)\n", (ptrdiff_t) n1, (ptrdiff_t) n1, (ptrdiff_t) Tz);
	fflush (stdout);

	switch (prec)
	{
		case FIELD_DOUBLE:

		single = false;
		if (lowmem) {
			printMpi("  Spectrum not available in lowmem until the end");
		} else {
			p2  = fftw_mpi_plan_dft_3d(Tz, n1, n1, static_cast<fftw_complex*>(m2), static_cast<fftw_complex*>(m2), MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);
		}
//		p  = fftw_plan_many_dft(2, nD, Lz, static_cast<fftw_complex*>(m), NULL, 1, dist, static_cast<fftw_complex*>(m), NULL, 1, dist, FFTW_FORWARD,  FFTW_MEASURE);
//		pb = fftw_plan_many_dft(2, nD, Lz, static_cast<fftw_complex*>(m), NULL, 1, dist, static_cast<fftw_complex*>(m), NULL, 1, dist, FFTW_BACKWARD, FFTW_MEASURE);
		break;

		case FIELD_SINGLE:

		single = true;
/*
		if(!fftwf_import_wisdom_from_filename("wisdomsavef.txt"))
		{
			printMpi("  Warning: could not import wisdom-f\n");
		}
		else
		{
			printMpi("  Wisdom-f file loaded\n\n");
		}
*/
		if (lowmem) {
			printMpi("Spectrum not available in lowmem until the end");
		} else {
			pf2  = fftwf_mpi_plan_dft_3d(Tz, n1, n1, static_cast<fftwf_complex*>(m2), static_cast<fftwf_complex*>(m2), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);
		}

		fftwf_mpi_gather_wisdom(MPI_COMM_WORLD);
		if (rank == 0) fftwf_export_wisdom_to_filename("../fftWisdom.single");

		// if (rank == 0) fftwf_export_wisdom_to_filename("wisdomsavef.txt");
		// printMpi ("  f-Wisdom saved\n");

		break;

		default:

		break;
	}

	printMpi ("  Plan_Spectrum Ok\n");


	iFFTSpectrum = true;
}

void	runFFTSpectrum(int sign)
{
	printMpi ("Spectrum FFT... ");


	if (single)
		fftwf_execute(pf2);
	else
		fftw_execute(p2);

	printMpi ("Done! ");

}

void	closeFFTSpectrum	()
{
	if (!iFFTSpectrum)
		return;

	if (single)
		fftwf_destroy_plan(pf2);
	else
		fftw_destroy_plan(p2);
		//void fftw_cleanup_threads(void);
}

//----------------------------------------------------------------------------------------------------------
// 			FFT HALO SMOOTHING M->V (DENS->V AND BACK)
//----------------------------------------------------------------------------------------------------------

fftw_plan p3,p3b;
fftwf_plan pf3, pf3b;


static bool iFFThalo = false;

void	initFFThalo	(void *m, void *v, const size_t n1, const size_t Tz, FieldPrecision prec)
{
	if (!iFFT)
		initFFT(prec);

	printf ("Initializing FFTSpectrum halo...\n");
	fflush (stdout);

	if (iFFThalo == true)
	{
		printf ("Already initialized!!\n");
		fflush (stdout);
	}
	int rank;
	//fftw_mpi_init();

	//printf ("  MPI Ok\n");
	//fflush (stdout);

	printf ("  Plan 3d (%lld x %lld x %lld)\n", (ptrdiff_t) n1, (ptrdiff_t) n1, (ptrdiff_t) Tz);
	fflush (stdout);

	switch (prec)
	{
		case FIELD_DOUBLE:

		single = false;
			p3   = fftw_mpi_plan_dft_r2c_3d(Tz, n1, n1, static_cast<double*>(m), static_cast<fftw_complex*>(v), MPI_COMM_WORLD, FFTW_MEASURE );
			p3b  = fftw_mpi_plan_dft_c2r_3d(Tz, n1, n1, static_cast<fftw_complex*>(v), static_cast<double*>(m), MPI_COMM_WORLD, FFTW_MEASURE );

		break;

		case FIELD_SINGLE:

		single = true;
			pf3  = fftwf_mpi_plan_dft_r2c_3d(Tz, n1, n1, static_cast<float*>(m), static_cast<fftwf_complex*>(v), MPI_COMM_WORLD, FFTW_MEASURE );
			pf3b = fftwf_mpi_plan_dft_c2r_3d(Tz, n1, n1, static_cast<fftwf_complex*>(v), static_cast<float*>(m), MPI_COMM_WORLD, FFTW_MEASURE );
/*
			fftwf_mpi_gather_wisdom(MPI_COMM_WORLD);
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			if (rank == 0) fftwf_export_wisdom_to_filename("wisdomsavef.txt");
			printf ("  f-Wisdom saved\n");
*/
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
