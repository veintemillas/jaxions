#include <cmath>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>


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
#include "axiton/tracker.h"

#include <iostream>
#include <fstream>
#include <string>
#include <map>

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
#endif

#include "gen/genConf.h"

using namespace std;
using namespace AxionWKB;

void genSpecKineticMis(size_t N, double L,double theta1, double vtheta1,bool hom = false);
void printIcData(const IcData &ic);
double fluctuationTheta3D_radial(const double *amp, const std::vector<size_t> &mult,size_t N);
std::vector<size_t> makeShellMultiplicity(size_t N, size_t kMax);
void writeMultiplicity(const std::vector<size_t> &mult);
bool readIC(const std::string &fname,double &theta1,double &vheta1);
double meanThetaPeriodic(std::vector<double> &history, double thetaNew, size_t maxHistory = 100);

int	main (int argc, char *argv[])
{
	
    Cosmos myCosmos = initAxions(argc, argv);

    double theta1 = 1.0 , vheta1 = 0.0;

    if (!readIC("ic.dat", theta1, vheta1))
    {
        std::cerr << "Error reading ic.dat\n";
        return 1;
    }

    LogOut("\n-------------------------------------------------\n");
	LogOut("\n               Kinetic misalignment 3D!      \n");
    LogOut("\n               theta1 = %.5e                 \n",theta1);
    LogOut("\n               vheta1 = %.5e                 \n",vheta1);
	LogOut("\n-------------------------------------------------\n\n");


// Choose optimum parameters?

// base measurement type from parse

MeasInfo meas_parsed = deninfa;

// mycosmos has the parsed IC info

IcData icd_save =   myCosmos.ICData();

uwDz = true;
wDz = 0.1;

// 1st we use linear evolution
myCosmos.ICData().nSteps = 10000000;

myCosmos.ICData().fType = FIELD_AXION;
myCosmos.ICData().linmodevol = true;
myCosmos.ICData().lme_no_rhs = false;
myCosmos.ICData().cType      = CONF_KM;
myCosmos.ICData().zi         = 1.0;

myCosmos.SetQcdPot  ( V_PQ1 | V_QCDC );
myCosmos.SetUeC  (true);


myCosmos.ICData().Nx = 256;
myCosmos.ICData().Ny = 256;
myCosmos.ICData().Nz = 256/zGrid;
// myCosmos.ICData().Nghost = 4;

// Create mode list
genSpecKineticMis(256, myCosmos.PhysSize(), theta1, vheta1);

// printIcData(myCosmos.ICData());

MeasInfo meas_lin; 
meas_lin.measdata = MEAS_LINMODES;
MeasData lm;

pType = PROP_MODES | PROP_RKN4;

Scalar *axion;
double taui = 1.0;
sPrec = FIELD_DOUBLE;
axion = new Scalar (&myCosmos, 1, 1, sPrec, cDev, taui, lowmem, zGrid, myCosmos.ICData().fType, lType, myCosmos.ICData().Nghost);

initPropagator (pType, axion, myCosmos.QcdPot(),myCosmos.ICData().Nghost);
tunePropagator (axion);

int counter = 0;
int index   = 0;

auto mult = makeShellMultiplicity(axion->NX(), axion->NModes ());
writeMultiplicity(mult);
// during evolution

double var = fluctuationTheta3D_radial(static_cast<double*> (axion->m_aCpu()), mult, axion->NX());
double rms = std::sqrt(var);
int i_tau = 0;

LogOut("Sigma at start %.3e \n",i_tau, rms/(*axion->RV()));

std::vector<double> thetaHist;

LogOut("Linear evolution %d steps \n ",myCosmos.ICData().nSteps);


while (i_tau < myCosmos.ICData().nSteps)
{

    double dtau_ = (uwDz) ? axion->dct_Adaptive() : (zFinl-zInit)/myCosmos.ICData().nSteps ;
    propagate (axion, dtau_);
    counter++;

    double R = *axion->RV();
    // double thetaMean = meanThetaPeriodic(thetaHist, static_cast<double*> (axion->m_aCpu())[0]/R, 1000);
    double m = axion->AxionMass()*R;
    double theta_ref = pow( sin(static_cast<double*> (axion->m_aCpu())[0]/R) ,2) + pow(static_cast<double*> (axion->v_aCpu())[0]/(m*R),2);
    theta_ref =sqrt(theta_ref);
    int binOld = int(std::floor(100.0*((*axion->zV())-dtau_ )));
    int binNew = int(std::floor(100.0*(*axion->zV())));

    if (binNew > binOld){
                // LogOut("Meas %d \n ",i_tau);
        		meas_lin.index=index;
				meas_lin.cTimesec = (double) Timer()*1.0e-6;
				meas_lin.propstep = i_tau;
				lm = Measureme (axion, meas_lin);
				index++;
				// i_meas++ ;
                double var = fluctuationTheta3D_radial(static_cast<double*> (axion->m_aCpu()), mult, axion->NX());
                double rms = std::sqrt(var)/R;
                LogOut("Meas %d Tau %.6e dtau %.3e thetamean %.3e sigma %.3e \n",
                    index, *axion->zV(), dtau_, theta_ref, rms);
                if (rms > 0.1 * theta_ref)
                    break;
    }
    i_tau++;
}

meas_lin.index   = index;
meas_lin.cTimesec = (double) Timer()*1.0e-6;
meas_lin.propstep = i_tau;
lm = Measureme (axion, meas_lin);
index++;
LogOut("Linear evolution ended with %d steps \n ",i_tau);


// Some checks 

LogOut("Creating 3D configuration  \n ");
myCosmos.ICData().km_use_mv_a_data = true;
genConf (&myCosmos, axion);

MeasInfo meas_NL; 
meas_NL.index    = index;
meas_NL.propstep = i_tau;
meas_NL.nbinsspec = -1;  
meas_NL.measdata = MEAS_3DMAP | MEAS_BINTHETA | MEAS_2DMAP | MEAS_PSP_A | MEAS_NSP_A | MEAS_NNSPEC ;
meas_NL.mask     = SPMASK_FLAT;
meas_NL.nrt      = NRUN_K | NRUN_G | NRUN_V | NRUN_S;
meas_NL.maty    = MAPT_XYPE;
lm = Measureme (axion, meas_NL);

// retune propagator

resetPropagator(axion);
pType = PROP_BASE | PROP_MLEAP;
initPropagator (pType, axion, myCosmos.QcdPot(),myCosmos.ICData().Nghost);
tunePropagator (axion);

// Continue evolution

while (i_tau < myCosmos.ICData().nSteps)
{
    double dtau_ = (uwDz) ? axion->dct_Adaptive() : (zFinl-zInit)/myCosmos.ICData().nSteps ;
    propagate (axion, dtau_);
    counter++;

    double R = *axion->RV();
    // double thetaMean = meanThetaPeriodic(thetaHist, static_cast<double*> (axion->m_aCpu())[0]/R, 1000);
    double m = axion->AxionMass()*R;
    double theta_ref = pow( sin(static_cast<double*> (axion->m_aCpu())[0]/R) ,2) + pow(static_cast<double*> (axion->v_aCpu())[0]/(m*R),2);
    theta_ref =sqrt(theta_ref);
    int binOld = int(std::floor(100.0*((*axion->zV())-dtau_ )));
    int binNew = int(std::floor(100.0*(*axion->zV())));

    if (binNew > binOld){
                // LogOut("Meas %d \n ",i_tau);
        		meas_NL.index=index;
				meas_NL.cTimesec = (double) Timer()*1.0e-6;
				meas_NL.propstep = i_tau;
				lm = Measureme (axion, meas_NL);
				index++;
				// i_meas++ ;
                double var = fluctuationTheta3D_radial(static_cast<double*> (axion->m_aCpu()), mult, axion->NX());
                double rms = std::sqrt(var)/R;
                LogOut("Meas %d Tau %.6e dtau %.3e thetamean %.3e sigma %.3e \n",
                    index, *axion->zV(), dtau_, theta_ref, rms);
                if (rms > 0.1 * theta_ref)
                    break;
    }
    i_tau++;
}



delete axion;

endAxions();

return 0;
}




















void printIcData(const IcData &ic)
{
	std::cout
	<< "fType       = " << int(ic.fType)      << '\n'
	<< "Nghost      = " << ic.Nghost          << '\n'
	<< "icdrule     = " << ic.icdrule         << '\n'
	<< "preprop     = " << ic.preprop         << '\n'
	<< "icstudy     = " << ic.icstudy         << '\n'
	<< "prepstL     = " << ic.prepstL         << '\n'
	<< "prepcoe     = " << ic.prepcoe         << '\n'
	<< "pregammo    = " << ic.pregammo        << '\n'
	<< "prelZ2e     = " << ic.prelZ2e         << '\n'
	<< "prevtype    = " << int(ic.prevtype)   << '\n'
	<< "normcore    = " << ic.normcore        << '\n'
	<< "alpha       = " << ic.alpha           << '\n'
	<< "siter       = " << ic.siter           << '\n'
	<< "kMax        = " << ic.kMax            << '\n'
	<< "kcr         = " << ic.kcr             << '\n'
	<< "mode0       = " << ic.mode0           << '\n'
	<< "beta        = " << ic.beta            << '\n'
	<< "zi          = " << ic.zi              << '\n'
	<< "logi        = " << ic.logi            << '\n'
	<< "kickalpha   = " << ic.kickalpha       << '\n'
	<< "extrav      = " << ic.extrav          << '\n'
	<< "cType       = " << int(ic.cType)      << '\n'
	<< "smvarType   = " << int(ic.smvarType)  << '\n'
	<< "mocoty      = " << int(ic.mocoty)     << '\n'
	<< "fieldindex  = " << int(ic.fieldindex) << '\n'
	<< "grav        = " << ic.grav            << '\n'
	<< "L1_pc       = " << ic.L1_pc           << '\n'
	<< "grav_hyb    = " << ic.grav_hyb        << '\n'
	<< "grav_sat    = " << ic.grav_sat        << '\n'
	<< "part_vel    = " << ic.part_vel        << '\n'
	<< "sm_vel      = " << ic.sm_vel          << '\n'
	<< "part_disp   = " << ic.part_disp       << '\n'
	<< "randommom   = " << ic.randommom       << '\n'
	<< "uEvolAll    = " << ic.uEvolAll        << '\n'
	<< "linmodevol  = " << ic.linmodevol      << '\n'
	<< "lme_no_rhs  = " << ic.lme_no_rhs      << '\n';
}

void genSpecKineticMis(size_t N, double L,
                       double theta1, double vtheta1,
                       bool hom )
{
    constexpr double pi = 3.14159265358979323846;

    std::ofstream out("initialspectrum.dat");

    const double k0 = 2.0*pi/L;

    // Zero mode
    out << theta1 << " "
        << (theta1 + vtheta1) << "\n";

    // Higher modes
    for (size_t i = 1; i < 2*N; i++)
    {
        double k = k0*i;

        double m = hom ? 0.0
                       : 4.5e-5*std::pow(L*k,-1.5)*vtheta1;

        out << m << " 0.0\n";
    }
}

std::vector<size_t> makeShellMultiplicity(size_t N, size_t kMax)
{
	std::vector<size_t> mult(kMax, 0);

	const int NN = static_cast<int>(N);

	#pragma omp parallel
	{
		std::vector<size_t> local(kMax, 0);

		#pragma omp for collapse(3) nowait
		for (int ix = 0; ix < NN; ix++)
		for (int iy = 0; iy < NN; iy++)
		for (int iz = 0; iz < NN; iz++)
		{
			int nx = (ix <= NN/2) ? ix : ix-NN;
			int ny = (iy <= NN/2) ? iy : iy-NN;
			int nz = (iz <= NN/2) ? iz : iz-NN;

			size_t k = static_cast<size_t>(
				std::llround(std::sqrt(double(nx*nx + ny*ny + nz*nz)))
			);

			if (k < kMax)
				local[k]++;
		}

		#pragma omp critical
		{
			for (size_t k = 0; k < kMax; k++)
				mult[k] += local[k];
		}
	}

	return mult;
}

void writeMultiplicity(const std::vector<size_t> &mult)
{
	std::ofstream out("out/nn.txt");

	for (size_t k = 0; k < mult.size(); k++)
		out << k << " " << mult[k] << "\n";
}

double fluctuationTheta3D_radial(
	const double *amp,
	const std::vector<size_t> &mult,
	size_t N)
{
	double sum = 0.0;

	#pragma omp parallel for reduction(+:sum)
	for (size_t k = 1; k < mult.size(); k++)
		sum += double(mult[k])*amp[k]*amp[k];

	// const double norm = double(N)*double(N)*double(N);

	return sum;
}

bool readIC(const std::string &fname,
            double &theta1,
            double &vheta1)
{
	std::ifstream fin(fname);

	if (!fin)
		return false;

	if (!(fin >> theta1 >> vheta1))
		return false;

	return true;
}

double meanThetaPeriodic(std::vector<double> &history,
                         double thetaNew,
                         size_t maxHistory )
{
    constexpr double twopi = 2.0*M_PI;

    // Add newest value
    history.push_back(thetaNew);

    // Keep buffer bounded
    if (history.size() > maxHistory)
        history.erase(history.begin());

    // Average after wrapping all values near thetaNew
    double sum = 0.0;

    for (double th : history)
    {
        while (th - thetaNew >  M_PI) th -= twopi;
        while (th - thetaNew < -M_PI) th += twopi;

        sum += th;
    }

    return sum/history.size();
}