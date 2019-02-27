#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "propagator/allProp.h"
#include "energy/energy.h"
#include "utils/utils.h"
#include "io/readWrite.h"
#include "comms/comms.h"
#include "map/map.h"
#include "strings/strings.h"
#include "powerCpu.h"
#include "scalar/scalar.h"
#include "scalar/scaleField.h"
#include "spectrum/spectrum.h"
#include "scalar/mendTheta.h"
#include "projector/projector.h"

#include "meas/measa.h"
#include "WKB/WKB.h"

#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>

#include <fstream>
#include <iostream>

using namespace std;

int	main ()
{


	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	FILE *data_file;
	data_file = fopen("./data.txt", "r");

	double zEnd 			;
	double nQcd 			;
	fscanf (data_file, "%lf %lf\n", &zEnd, &nQcd);
	fclose(data_file);

	FILE *file_samp ;
	file_samp = NULL;
	file_samp = fopen("./result.txt","w+");
	// std::ifstream ou("result.txt");

	double res, z, w2 ;
	double m2 				= pow(zEnd,nQcd+2.0);
	double nn2  			= 1./(2.0+nQcd)+1.0;
	double n2p1       = 1.0+ nQcd/2.;
	double phiBase2	  = zEnd/(2.0+nQcd/2.0);

	printf("Check precisions\n");
	printf("zEnd = %f nQcd = %f\n",zEnd,nQcd);
	for (int k = 1; k<4096; k++)
	{
	w2 = m2+k*k;
	z = k*k/w2;
	old = std::chrono::high_resolution_clock::now();
	// printf("%f %f \n",z,nn2);
	for (int j =0; j<100;j++)
		res = gsl_sf_hyperg_2F1(0.5, 1.0, nn2, 1.0-z);
	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);

	res = phiBase2*sqrt(w2)*(1.+n2p1*z*res);

	fprintf(file_samp,"%d %lf %e \n", k, res, elapsed);
	printf("%d %lf %e \n", k, res, elapsed);
	// ou << k << res << elapsed << "\n";
	}
	fclose(file_samp);
	return 0;
}
