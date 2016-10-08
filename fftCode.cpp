#include<complex>
#include<vector>
#include<fftw3.h>

using namespace std;

fftw_plan p;

void	initFFT	(vector<complex<double> > &m, const int n1)
{
	p = fftw_plan_dft_3d(n1, n1, n1, reinterpret_cast<fftw_complex*>(m.data()), reinterpret_cast<fftw_complex*>(m.data()), FFTW_FORWARD, FFTW_MEASURE);
}


void	runFFT()
{
	fftw_execute(p);         
}
            
void	closeFFT	()
{
	void fftw_cleanup_threads(void);
}
