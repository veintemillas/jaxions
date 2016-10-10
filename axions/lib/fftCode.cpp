#include<complex>
#include<vector>
#include<fftw3-mpi.h>
#include"enum-field.h"

using namespace std;

fftw_plan p, pb;
fftwf_plan pf, pfb;

bool single;

void	initFFT	(void *m, void *m2, const int n1, const int Lz, FieldPrecision prec)
{
	const int nD[2] = { n1, n1 };
	const int dist  = n1*n1;

	fftw_mpi_init();

	switch (prec)
	{
		case FIELD_DOUBLE:

		single = false;
		p  = fftw_mpi_plan_dft_3d(Lz, n1, n1, reinterpret_cast<fftw_complex*>(m), reinterpret_cast<fftw_complex*>(m2), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE);
		pb = fftw_mpi_plan_dft_3d(Lz, n1, n1, reinterpret_cast<fftw_complex*>(m2), reinterpret_cast<fftw_complex*>(m), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
//		p  = fftw_plan_many_dft(2, nD, Lz, reinterpret_cast<fftw_complex*>(m), NULL, 1, dist, reinterpret_cast<fftw_complex*>(m), NULL, 1, dist, FFTW_FORWARD,  FFTW_MEASURE);
//		pb = fftw_plan_many_dft(2, nD, Lz, reinterpret_cast<fftw_complex*>(m), NULL, 1, dist, reinterpret_cast<fftw_complex*>(m), NULL, 1, dist, FFTW_BACKWARD, FFTW_MEASURE);
		break;

		case FIELD_SINGLE:

		single = true;
		pf  = fftwf_mpi_plan_dft_3d(Lz, n1, n1, reinterpret_cast<fftwf_complex*>(m), reinterpret_cast<fftwf_complex*>(m2), MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE);
		pfb = fftwf_mpi_plan_dft_3d(Lz, n1, n1, reinterpret_cast<fftwf_complex*>(m2), reinterpret_cast<fftwf_complex*>(m), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
//		pf  = fftwf_plan_many_dft(2, nD, Lz, reinterpret_cast<fftwf_complex*>(m), NULL, 1, dist, reinterpret_cast<fftwf_complex*>(m), NULL, 1, dist, FFTW_FORWARD,  FFTW_MEASURE);
//		pfb = fftwf_plan_many_dft(2, nD, Lz, reinterpret_cast<fftwf_complex*>(m), NULL, 1, dist, reinterpret_cast<fftwf_complex*>(m), NULL, 1, dist, FFTW_BACKWARD, FFTW_MEASURE);
		break;

		default:

		break;
	}
}

void	runFFT(int sign)
{
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
}
            
void	closeFFT	()
{
	if (single)
	{
		fftwf_destroy_plan(pf);
		fftwf_destroy_plan(pfb);
		void fftwf_cleanup_threads(void);
	}
	else	
	{
		fftw_destroy_plan(p);
		fftw_destroy_plan(pb);
		void fftw_cleanup_threads(void);
	}
}

