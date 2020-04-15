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
#include "spectrum/spectrum.h"
#include "meas/measa.h"


#include "WKB/WKB.h"

using namespace std;
using namespace AxionWKB;

int	main (int argc, char *argv[])
{

	double zendWKB = 10. ;
	Cosmos myCosmos = initAxions(argc, argv);

	if (nSteps==0)
	return 0 ;

	//--------------------------------------------------
	//       AUX STUFF
	//--------------------------------------------------


	double  *binarray	 ;
	trackAlloc((void**) (&binarray),  10000*sizeof(size_t));
	double *bA = static_cast<double *> (binarray);
	size_t sliceprint = 0 ; // sizeN/2;



	commSync();
	LogOut("\n-------------------------------------------------\n");
	LogOut("\n Calculation of conbin for different smoothings  \n");
	LogOut("\n-------------------------------------------------\n");

	LogOut("\n-------------------------------------------------\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	Scalar *axion;

	LogOut ("reading conf %d ...", fIndex);
	readConf(&myCosmos, &axion, fIndex);
	if (axion == NULL)
	{
		LogOut ("Error reading HDF5 file\n");
		exit (0);
	}
	LogOut ("\n");


	double z_now = (*axion->zV())	;
	LogOut("--------------------------------------------------\n");
	LogOut("           READ CONDITIONS                     \n\n");

	LogOut("Length =  %2.2f\n", myCosmos.PhysSize());
	LogOut("nQCD   =  %2.2f\n", myCosmos.QcdExp());
	LogOut("N      =  %ld\n",   axion->Length());
	LogOut("Nz     =  %ld\n",   axion->Depth());
	LogOut("zGrid  =  %ld\n",   zGrid);
	LogOut("z      =  %2.2f\n", z_now);
	LogOut("zthr   =  %3.3f\n", myCosmos.ZThRes());
	LogOut("zres   =  %3.3f\n", myCosmos.ZRestore());
	LogOut("mass   =  %3.3f\n\n", axion->AxionMass());

	if (axion->Precision() == FIELD_SINGLE)
		LogOut("precis = SINGLE(%d)\n",FIELD_SINGLE);
	else
		LogOut("precis = DOUBLE(%d)\n",FIELD_DOUBLE);

	LogOut("--------------------------------------------------\n");

	//--------------------------------------------------
	//       MEASUREMENT
	//--------------------------------------------------
	//- Measurement
	MeasData lm;
	//- number of plaquetes pierced by strings
	lm.str.strDen = 0 ;
	//- Info to measurement
	MeasInfo ninfa = deninfa;
	ninfa.index = fIndex;
	ninfa.redmap = endredmap;


	int counter = 0;
	int index ;
	double dzaux;
	int i_meas = 0;
	bool measrightnow = false;

	ninfa.index=index;
	// ninfa.measdata |= MEAS_3DMAP;
	// lm = Measureme (axion, ninfa);
	// ninfa.measdata ^= MEAS_3DMAP;
	LogOut("Measure energy \n");

	void *eRes;
	trackAlloc(&eRes, 512);
	memset(eRes, 0, 512);
	double *eR = static_cast<double *> (eRes);
	energy(axion, eRes, EN_MAP, 0.0);

	float eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);

	LogOut("Compute FFT \n");
	SpecBin specAna(axion, false);
	specAna.pRun();

	auto &myPlan = AxionFFT::fetchPlan("pSpecAx");

	LogOut("Save to m2h \n");
	char *m2  = static_cast<char*>(axion->m2Cpu());
	char *m2h = static_cast<char*>(axion->m2half());
	memmove(m2h,m2,axion->eSize()*axion->Precision());

	// LogOut("Open file\n");

	index = 0;
	double delta = axion->BckGnd()->PhysSize()/axion->Length();
	size_t LyLz = axion->Length()*axion->Depth();
	size_t dl = axion->Length()*axion->Precision();
	size_t pl = (axion->Length()+2)*axion->Precision();

	LogOut ("Start smoothing loop\n\n");
	for (int i = 0; i < nSteps; i	++)
	{
		/* filter with exp(-0.5 (k (delta/2) x)**2 )
			i.e. the argument is in units of delta/2 */
		double smthi = 0.5*std::exp( std::log(10*sizeN/0.5)*i/nSteps )	;
		double smth = smthi*delta/2;
		// double smth = pow(1.2,i)*delta/2.;
		LogOut("SmOOthing with sigma-length %.2e\n",smth);
		specAna.filterFFT<float>	(smthi);
		myPlan.run(FFT_BCK);
		/*uppad*/
		/* unpad m2 in place */
		for (size_t sl=1; sl<LyLz; sl++) {
			size_t	oOff = sl*dl;
			size_t	fOff = sl*pl;
			memmove	(m2+oOff, m2+fOff, dl);
		}

		{
			createMeas(axion, index);
			LogMsg(VERB_NORMAL, "bin contrast");
			Binner<3000,float> contBin(static_cast<float *>(axion->m2Cpu()), axion->Size(),
							[eMean = eMean] (float x) -> float { return (double) (log10(x/eMean)) ;});
			contBin.run();
			char PRELABEL[256];
			// sprintf(PRELABEL, "%s_%.3f", "contB",smth*delta/2.);
			writeBinner(contBin, "/bins", "contB");
			writeAttribute(&smth,   "sigma");
			destroyMeas();
			index++;
		}
		LogMsg(VERB_NORMAL, "copying FFT to me");
		memmove(m2,m2h,axion->eSize()*axion->Precision());
		axion->setM2h(M2_ENERGY_FFT);
	}


	//--------------------------------------------------
	//       SAVE DATA
	//--------------------------------------------------

	LogOut ("Done!\n\n");

	endAxions();

	return 0;
}
