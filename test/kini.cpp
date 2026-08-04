#include <cmath>
#include <cstring>
#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

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

bool genSpecKineticMis(size_t N, double L,double theta1, double vtheta1,bool hom = false);
void printIcData(const IcData &ic);
double fluctuationTheta3D_radial(const double *amp, const std::vector<size_t> &mult,size_t N);
double fluctuationTheta3D_interpolated(const double *amp, size_t nModes, size_t N);
double spatialThetaSigma(Scalar *axion);
std::vector<size_t> makeShellMultiplicity(size_t N, size_t kMax);
bool writeMultiplicity(const std::vector<size_t> &mult);
bool readIC(const std::string &fname, double &theta1, double &vtheta1,
            double &epsilon1, double &vepsilon1);
double meanThetaPeriodic(std::vector<double> &history, double thetaNew, size_t maxHistory = 100);
int readMeasurementSchedule(Scalar *axion, MeasFileParms &schedule);
void loadMeasurement(const MeasFileParms &schedule, MeasInfo &info, size_t measurement);
size_t selectTrackedPoint(Scalar *axion, bool useRequestedPoint, size_t requestedPoint,
                          double &selectedDensity);
size_t kiniUnfoldIndex(size_t idx, Scalar *axion);
void printSpatialSample(FILE *file, Scalar *axion, size_t globalIndex,
                        double transitionDensity);
int kiniRuntimeControl();

int	main (int argc, char *argv[])
{
    /*
     * Kinetic-misalignment evolution in two stages:
     *
     *  1. Evolve the isotropic Fourier modes with the linear mode propagator
     *     until their fluctuations reach 10% of the homogeneous amplitude.
     *  2. Transform those modes into a three-dimensional axion field and
     *     continue the nonlinear spatial evolution with RKN4.
     *
     * Spatial measurements follow measfile.dat when it is present; otherwise
     * they use the interval selected with --dump. A final measurement is
     * always written with the same exceptional flags used by vaxions.
     */
	
    Cosmos myCosmos = initAxions(argc, argv);

    /*
     * The inexpensive radial-mode stage has its own conservative adaptive
     * accuracy. Preserve --wDz for the spatial RKN4 stage, where the user may
     * want a different performance/accuracy tradeoff.
     */
    const bool userSpatialWDz = uwDz;
    const double spatialWDz = userSpatialWDz ? wDz : 0.1;
    wDz = 0.1;
    uwDz = true;
    LogOut("Linear-mode adaptive step uses wDz = %.6g\n", wDz);

    // kini uses adaptive stepping; these are generous caps for a minimal run.
    if (!uZin)
    {
        zInit = 1.0;
        myCosmos.ICData().zi = 1.0;
        LogOut("No --zi supplied; using kini IC reference time 1\n");
    }
    if (myCosmos.ICData().nSteps == 0)
    {
        myCosmos.ICData().nSteps = 10000000;
        LogOut("No --steps supplied; using kini default 10000000\n");
    }
    if (!uZfn)
    {
        zFinl = 10.0;
        LogOut("No --zf supplied; using kini default 10\n");
    }
    if (!userSpatialWDz)
    {
        LogOut("No --wDz supplied; spatial evolution will use default wDz = %.6g\n",
               spatialWDz);
    }

    // The homogeneous initial angle and velocity are supplied separately.
    double theta1 = 1.0, vheta1 = 0.0;
    double epsilon1 = 0.0, vepsilon1 = 0.0;

    if (!readIC("ic.dat", theta1, vheta1, epsilon1, vepsilon1))
    {
        std::cerr << "Error reading ic.dat\n";
        endAxions();
        return 1;
    }

    LogOut("\n-------------------------------------------------\n");
	LogOut("\n               Kinetic misalignment 3D!      \n");
    LogOut("\n               theta1 = %.5e                 \n",theta1);
    LogOut("\n               vheta1 = %.5e                 \n",vheta1);
	LogOut("\n-------------------------------------------------\n\n");


// 1st we use linear evolution
myCosmos.ICData().fType = FIELD_AXION;
myCosmos.ICData().linmodevol = true;
myCosmos.ICData().lme_no_rhs = false;
myCosmos.ICData().cType      = CONF_KM;
myCosmos.ICData().km_ic_physical = true;

myCosmos.SetQcdPot  ( V_PQ1 | V_QCDC );
myCosmos.SetQcdExp  (8.0);
myCosmos.SetUeC  (false);
LogOut("Using analytic QCD cosmology with susceptibility exponent n = 8\n");

// Rank 0 writes the shared radial spectrum; all ranks wait before reading it.
int spectrumReady = 1;
if (commRank() == 0)
    spectrumReady = genSpecKineticMis(
        myCosmos.ICData().Nx, myCosmos.PhysSize(), theta1, vheta1) ? 1 : 0;
MPI_Bcast(&spectrumReady, 1, MPI_INT, 0, MPI_COMM_WORLD);
commSync();

if (!spectrumReady)
{
    LogOut("Error: could not write initialspectrum.dat\n");
    endAxions();
    return 1;
}

// printIcData(myCosmos.ICData());

MeasInfo meas_lin = deninfa;
// Linear evolution has no spatial field: suppress inherited maps/spectra and
// write only the radial mode data handled by MEAS_LINMODES.
meas_lin.measdata = MEAS_LINMODES;
meas_lin.maty = MAPT_NO;
meas_lin.mask = SPMASK_NONE;
meas_lin.nrt = NRUN_NONE;
meas_lin.redmap = 0;
MeasData lm;

pType = PROP_MODES | PROP_RKN4;

Scalar *axion;
double taui = myCosmos.ICData().zi;
sPrec = FIELD_DOUBLE;
axion = new Scalar (&myCosmos, 1, 1, sPrec, cDev, taui, lowmem, zGrid, myCosmos.ICData().fType, lType, myCosmos.ICData().Nghost);

/*
 * genConf also initializes epsilon from its double-precision theta field.
 * Replace it with the value obtained directly from the high-precision input
 * and synchronize the compatibility psi0 slots.
 */
*axion->epsilonV() = epsilon1;
*axion->epsilonPV() = vepsilon1;
{
    const double pi = std::acos(-1.0);
    const double R = *axion->RV();
    const double H = axion->BckGnd()->Rp(*axion->zV());
    const double theta0 = pi - epsilon1;
    const double theta0p = -vepsilon1;
    static_cast<double*>(axion->m_aCpu())[0] = R*theta0;
    static_cast<double*>(axion->v_aCpu())[0] = R*(theta0p + H*theta0);
}

initPropagator (pType, axion, myCosmos.QcdPot(),myCosmos.ICData().Nghost);
tunePropagator (axion);

int counter = 0;
int index   = 0;

auto mult = makeShellMultiplicity(axion->NX(), axion->NModes ());
if (!writeMultiplicity(mult))
    LogOut("Warning: could not write mode multiplicities to out/nn.txt\n");
// Multiplicity converts one radial amplitude per |k| shell into a 3D variance.
double var = fluctuationTheta3D_radial(static_cast<double*> (axion->m_aCpu()), mult, axion->NX());
double rms = std::sqrt(var);
int i_tau = 0;

LogOut("Sigma at start %.3e \n", rms/(*axion->RV()));

LogOut("Linear evolution %zu steps \n ",myCosmos.ICData().nSteps);


while (i_tau < myCosmos.ICData().nSteps)
{

    double dtau_ = (uwDz) ? axion->dct_Adaptive() : (zFinl-zInit)/myCosmos.ICData().nSteps ;
    propagate (axion, dtau_);
    counter++;

    const double R = *axion->RV();
    // double thetaMean = meanThetaPeriodic(thetaHist, static_cast<double*> (axion->m_aCpu())[0]/R, 1000);
    const double epsilon0 = *axion->epsilonV();
    const double thetaPrimeR = -R*(*axion->epsilonPV());
    const double conformalMass = axion->AxionMass()*R;
    const double theta_ref = std::sqrt(
        std::pow(std::sin(epsilon0), 2) +
        std::pow(thetaPrimeR/(conformalMass*R), 2));
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
                LogOut("Meas %d Tau %.6e dtau %.3e thetaamp %.3e sigma %.3e "
                       "epsilon %.16e epsilonprime %.16e\n",
                    index, *axion->zV(), dtau_, theta_ref, rms,
                    *axion->epsilonV(), *axion->epsilonPV());
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


// Convert the evolved radial modes into a real-space 3D axion configuration.
const double sigmaLinear = std::sqrt(fluctuationTheta3D_interpolated(
    static_cast<double*>(axion->m_aCpu()), axion->NModes(), axion->NX()))
    / *axion->RV();

LogOut("Creating 3D configuration  \n ");
myCosmos.ICData().km_use_mv_a_data = true;
genConf (&myCosmos, axion);
myCosmos.ICData().linmodevol = false;

const double sigmaSpatial = spatialThetaSigma(axion);
const double sigmaRatio = sigmaLinear > 0.0 ? sigmaSpatial/sigmaLinear : 0.0;
const double sigmaRelativeDifference = sigmaLinear > 0.0 ? std::abs(sigmaRatio - 1.0) : 0.0;
LogOut("KM conversion sigma test: linear %.6e spatial %.6e ratio %.6e rel.diff %.3e\n",
       sigmaLinear, sigmaSpatial, sigmaRatio, sigmaRelativeDifference);

MeasInfo meas_NL = deninfa;
meas_NL.measdata = meas_NL.measdata | MEAS_BINTHETA;
meas_NL.index    = index;
meas_NL.propstep = i_tau;

// The field is now spatial: discard the mode propagator and initialize RKN4.
resetPropagator(axion);
wDz = spatialWDz;
uwDz = true;
LogOut("Spatial adaptive step uses wDz = %.6g%s\n", wDz,
       userSpatialWDz ? " (from --wDz)" : " (default)");
if (cDev == DEV_GPU)
{
    LogOut("Transferring converted 3D configuration to GPU\n");
    axion->transferDev(FIELD_MV);
}
pType = PROP_BASE | PROP_RKN4;
initPropagator (pType, axion, myCosmos.QcdPot(),myCosmos.ICData().Nghost);
tunePropagator (axion);

/*
 * Pick a point only after the random Fourier phases have been realised.
 * A nonzero --idxprint remains an explicit override; otherwise track the
 * largest full axion energy density at the linear-to-spatial handoff.
 */
double trackedTransitionDensity = 0.0;
const bool requestedTrackedPoint = deninfa.idxprint != 0;
const size_t trackedPoint = selectTrackedPoint(
    axion, requestedTrackedPoint, deninfa.idxprint, trackedTransitionDensity);
meas_NL.idxprint = trackedPoint;

FILE *sampleFile = nullptr;
const int trackedRank = static_cast<int>(
    (trackedPoint/axion->Surf())/axion->Depth());
if (commRank() == trackedRank)
{
    char sampleName[2048];
    std::snprintf(sampleName, sizeof(sampleName), "%s/../sample.txt", outDir);
    sampleFile = std::fopen(sampleName, "w");
    if (sampleFile == nullptr)
        LogError("Could not open %s for the tracked-point history", sampleName);
    else
    {
        std::fprintf(sampleFile,
            "# tau R mass field_a field_b derived_a derived_b transition_rho "
            "global_index x y z\n");
        std::fprintf(sampleFile,
            "# theta stage: field_a=psi field_b=psi' derived_a=theta "
            "derived_b=theta'\n");
        std::fflush(sampleFile);
    }
}
printSpatialSample(sampleFile, axion, trackedPoint, trackedTransitionDensity);

/*
 * Measure the freshly converted spatial field before evolving it.  Together
 * with the final linear-mode measurement above, this gives two files at the
 * same conformal time and makes the handoff directly comparable.
 */
meas_NL.index = index;
meas_NL.cTimesec = (double) Timer()*1.0e-6;
meas_NL.propstep = i_tau;
LogOut("Initial spatial measurement at linear-to-3D conversion: %05d\n", index);
lm = Measureme(axion, meas_NL);
index++;

bool paxionStage = false;
auto maximumRelativisticRatio = [&]() {
    const double kMaximum =
        std::sqrt(3.0)*M_PI*double(axion->Length())/myCosmos.PhysSize();
    const double conformalMass = axion->AxionMass()*(*axion->RV());
    return conformalMass > 0.0
        ? kMaximum/conformalMass
        : std::numeric_limits<double>::infinity();
};

auto paxionTransitionDue = [&]() {
    if (paxionStage)
        return false;
    const bool ratioDue =
        paxionRatio > 0.0 && maximumRelativisticRatio() <= paxionRatio;
    const bool timeDue =
        paxionTime > 0.0 && *axion->zV() >= paxionTime;
    return ratioDue || timeDue;
};

auto convertToPaxion = [&]() {
    const double ratio = maximumRelativisticRatio();
    LogOut("Theta-to-paxion transition at tau %.9e: "
           "kmax/(m_a R)=%.6e (threshold %.6e)\n",
           *axion->zV(), ratio, paxionRatio);
    LogOut("Paxion UV-collapse control: FAT=%s msa=%.6g\n",
           myCosmos.FAT() ? "on" : "off", myCosmos.MAa());
    if (!myCosmos.FAT())
        LogOut("WARNING: paxion FAT regulation is disabled; attractive "
               "self-interactions may collapse to the lattice cutoff. "
               "Use --FAT --msa 2.0 (or a tested alternative).\n");

    // Preserve the last relativistic state immediately before conversion.
    meas_NL.index = index;
    meas_NL.cTimesec = (double) Timer()*1.0e-6;
    meas_NL.propstep = i_tau;
    LogOut("Final theta measurement before paxion conversion: %05d\n", index);
    lm = Measureme(axion, meas_NL);
    index++;

    resetPropagator(axion);
    thetaToPaxion(axion);
    pType = PROP_BASE | PROP_RKN4;
    initPropagator(pType, axion, myCosmos.QcdPot(),
                   myCosmos.ICData().Nghost);
    tunePropagator(axion);
    paxionStage = true;

    meas_NL.measdata = static_cast<MeasureType>(
        static_cast<unsigned int>(meas_NL.measdata) &
        ~static_cast<unsigned int>(MEAS_BINTHETA));

    if (sampleFile != nullptr)
    {
        std::fprintf(sampleFile,
            "# paxion stage: field_a=Re(P) field_b=Im(P) "
            "derived_a=|P|^2 derived_b=0\n");
        std::fflush(sampleFile);
    }
    printSpatialSample(sampleFile, axion, trackedPoint,
                       trackedTransitionDensity);

    // First paxion file: same time and configuration as the theta file above.
    meas_NL.index = index;
    meas_NL.cTimesec = (double) Timer()*1.0e-6;
    meas_NL.propstep = i_tau;
    LogOut("Initial paxion measurement after conversion: %05d\n", index);
    lm = Measureme(axion, meas_NL);
    index++;
};

LogOut("Paxion transition control: kmax/(m_a R) <= %.6g%s\n",
       paxionRatio, paxionRatio > 0.0 ? "" : " (disabled)");
if (paxionTime > 0.0)
    LogOut("Explicit paxion transition time: %.9e\n", paxionTime);

if (paxionTransitionDue())
    convertToPaxion();

// Select spatial measurement times from measfile.dat or the --dump interval.
MeasFileParms measSchedule;
const int scheduleStatus = readMeasurementSchedule(axion, measSchedule);
if (scheduleStatus < 0)
{
    delete axion;
    endAxions();
    return 1;
}

const bool measurementsFromFile = scheduleStatus > 0;
size_t nextMeasurement = 0;
int spatialStep = 0;
bool runtimeStopped = false;
bool runtimeAborted = false;
if (measurementsFromFile)
{
    zFinl = measSchedule.ct.back();
    LogOut("Spatial measurements read from measfile.dat; final time set to %.6e\n", zFinl);
}
else
{
    LogOut("Spatial measurements every %d propagation steps\n", dump);
}

while (i_tau < myCosmos.ICData().nSteps && *axion->zV() < zFinl)
{
    double dtau_ = (uwDz) ? axion->dct_Adaptive() : (zFinl-zInit)/myCosmos.ICData().nSteps ;
    if (*axion->zV() + dtau_ > zFinl)
        dtau_ = zFinl - *axion->zV();
    if (!paxionStage && paxionTime > *axion->zV() &&
        *axion->zV() + dtau_ > paxionTime)
        dtau_ = paxionTime - *axion->zV();

    bool measureNow = false;
    if (measurementsFromFile && nextMeasurement < measSchedule.ct.size())
    {
        const double measurementTime = measSchedule.ct[nextMeasurement];
        if (*axion->zV() + dtau_ >= measurementTime)
        {
            dtau_ = measurementTime - *axion->zV();
            loadMeasurement(measSchedule, meas_NL, nextMeasurement);
            if (!paxionStage)
                meas_NL.measdata = meas_NL.measdata | MEAS_BINTHETA;
            else
                meas_NL.measdata = static_cast<MeasureType>(
                    static_cast<unsigned int>(meas_NL.measdata) &
                    ~static_cast<unsigned int>(MEAS_BINTHETA));
            measureNow = nextMeasurement + 1 < measSchedule.ct.size();
            nextMeasurement++;
        }
    }

    propagate (axion, dtau_);
    i_tau++;
    spatialStep++;
    counter++;

    printSpatialSample(sampleFile, axion, trackedPoint, trackedTransitionDensity);

    if (!measurementsFromFile && dump > 0 && !(spatialStep % dump))
        measureNow = true;

    if (measureNow)
    {
        meas_NL.index = index;
        meas_NL.cTimesec = (double) Timer()*1.0e-6;
        meas_NL.propstep = i_tau;
        lm = Measureme(axion, meas_NL);
        index++;
    }

    if (paxionTransitionDue())
        convertToPaxion();

    const int runtimeAction = kiniRuntimeControl();
    if (runtimeAction != 0)
    {
        if (runtimeAction == 4)
        {
            LogOut("savejaxconf detected: writing configuration %05d and continuing\n",
                   index);
            if (cDev == DEV_GPU)
                axion->transferCpu(FIELD_MV);
            writeConf(axion, index);
            commSync();
        }
        else if (runtimeAction == 3)
        {
            LogOut("abort detected: terminating without a checkpoint\n");
            runtimeAborted = true;
            break;
        }
        else
        {
            LogOut("%s: writing configuration %05d and stopping\n",
                   runtimeAction == 2 ? "stop detected" : "walltime reached",
                   index);
            if (cDev == DEV_GPU)
                axion->transferCpu(FIELD_MV);
            writeConf(axion, index, 1);
            commSync();
            runtimeStopped = true;
            break;
        }
    }

    if (*axion->zV() >= zFinl || std::abs(*axion->zV() - zFinl) < 1.0e-10)
        break;
}

if (runtimeStopped || runtimeAborted)
{
    if (sampleFile != nullptr)
        std::fclose(sampleFile);
    delete axion;
    endAxions();
    return runtimeAborted ? 1 : 0;
}

// Match vaxions' final-output exceptions independently of the regular cadence.
MeasureType finalMeasurement = paxionStage
    ? static_cast<MeasureType>(
        static_cast<unsigned int>(defaultmeasType) &
        ~static_cast<unsigned int>(MEAS_BINTHETA))
    : defaultmeasType | MEAS_BINTHETA;
if (meas_NL.printconf & PRINTCONF_FINAL)
    finalMeasurement = finalMeasurement | MEAS_3DMAP;
if (pconfinal)
    finalMeasurement = finalMeasurement | MEAS_ENERGY3DMAP;
if (endredmap > 0)
    finalMeasurement = finalMeasurement | MEAS_REDENE3DMAP;

meas_NL.index = index;
meas_NL.measdata = finalMeasurement;
meas_NL.cTimesec = (double) Timer()*1.0e-6;
meas_NL.propstep = i_tau;
LogOut("Final measurement file is: %05d\n", index);
lm = Measureme(axion, meas_NL);
LogOut("Evolution finished after %d propagation steps\n", counter);

if (sampleFile != nullptr)
    std::fclose(sampleFile);
delete axion;

endAxions();

return 0;
}

/*
 * Runtime control files, matching vaxions:
 *   stop         checkpoint and terminate
 *   abort        terminate without a checkpoint
 *   savejaxconf  checkpoint and continue (one-shot; rank 0 removes the file)
 * The wall-time limit set by --wTime behaves like stop.
 */
int kiniRuntimeControl()
{
    int action = 0;
    if (commRank() == 0)
    {
        if (wTime <= Timer())
            action = 1;

        FILE *control = std::fopen("./stop", "r");
        if (control != nullptr)
        {
            std::fclose(control);
            action = 2;
        }

        control = std::fopen("./abort", "r");
        if (control != nullptr)
        {
            std::fclose(control);
            action = 3;
        }

        control = std::fopen("./savejaxconf", "r");
        if (control != nullptr)
        {
            std::fclose(control);
            if (action == 0)
                action = 4;
            if (std::remove("./savejaxconf") != 0)
                LogOut("Warning: savejaxconf was handled but could not be removed\n");
        }
    }

    MPI_Bcast(&action, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return action;
}

size_t kiniUnfoldIndex(size_t idx, Scalar *axion)
{
    if (!axion->Folded())
        return idx;

    size_t x[3];
    indexXeon::idx2Vec(idx, x, axion->Length());
    const size_t vectorLength = axion->DataAlign()/axion->DataSize();
    const size_t xChunk = axion->Length()*vectorLength;
    const size_t yChunk = axion->Length()/vectorLength;
    const size_t iy = x[1]/yChunk;
    const size_t iv = x[1] - iy*yChunk;
    return x[2]*axion->Surf() + iv*xChunk + x[0]*vectorLength + iy;
}

size_t selectTrackedPoint(Scalar *axion, bool useRequestedPoint,
                          size_t requestedPoint, double &selectedDensity)
{
    double energyResult[23] = {};
    energy(axion, energyResult, EN_MAP, 0.0);

    const size_t localSize = axion->Size();
    const size_t globalSize = axion->TotalSize();
    if (useRequestedPoint && requestedPoint >= globalSize)
    {
        LogError("Requested --idxprint %zu is outside the global lattice (%zu sites); "
                 "selecting the maximum-density point instead.",
                 requestedPoint, globalSize);
        useRequestedPoint = false;
    }

    const void *energyMap = axion->m2Start();
    double localMaximum = -std::numeric_limits<double>::infinity();
    size_t localMaximumIndex = 0;

    if (!useRequestedPoint)
    {
        if (axion->Precision() == FIELD_SINGLE)
        {
            const float *rho = static_cast<const float *>(energyMap);
            #pragma omp parallel
            {
                double threadMaximum = -std::numeric_limits<double>::infinity();
                size_t threadIndex = 0;
                #pragma omp for nowait
                for (size_t idx = 0; idx < localSize; ++idx)
                {
                    const double value = rho[kiniUnfoldIndex(idx, axion)];
                    if (value > threadMaximum)
                    {
                        threadMaximum = value;
                        threadIndex = idx;
                    }
                }
                #pragma omp critical
                if (threadMaximum > localMaximum)
                {
                    localMaximum = threadMaximum;
                    localMaximumIndex = threadIndex;
                }
            }
        }
        else
        {
            const double *rho = static_cast<const double *>(energyMap);
            #pragma omp parallel
            {
                double threadMaximum = -std::numeric_limits<double>::infinity();
                size_t threadIndex = 0;
                #pragma omp for nowait
                for (size_t idx = 0; idx < localSize; ++idx)
                {
                    const double value = rho[kiniUnfoldIndex(idx, axion)];
                    if (value > threadMaximum)
                    {
                        threadMaximum = value;
                        threadIndex = idx;
                    }
                }
                #pragma omp critical
                if (threadMaximum > localMaximum)
                {
                    localMaximum = threadMaximum;
                    localMaximumIndex = threadIndex;
                }
            }
        }
    }

    struct {
        double value;
        int rank;
    } localPair, globalPair;

    if (useRequestedPoint)
    {
        const int owner = static_cast<int>(requestedPoint/localSize);
        localPair.value = commRank() == owner ? 0.0
                                              : -std::numeric_limits<double>::infinity();
        localPair.rank = commRank();
        if (commRank() == owner)
        {
            localMaximumIndex = requestedPoint - size_t(owner)*localSize;
            const size_t memoryIndex = kiniUnfoldIndex(localMaximumIndex, axion);
            localPair.value = axion->Precision() == FIELD_SINGLE
                ? static_cast<const float *>(energyMap)[memoryIndex]
                : static_cast<const double *>(energyMap)[memoryIndex];
        }
    }
    else
    {
        localPair.value = localMaximum;
        localPair.rank = commRank();
    }

    MPI_Allreduce(&localPair, &globalPair, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                  MPI_COMM_WORLD);

    unsigned long long winningLocalIndex =
        commRank() == globalPair.rank
            ? static_cast<unsigned long long>(localMaximumIndex) : 0ULL;
    MPI_Bcast(&winningLocalIndex, 1, MPI_UNSIGNED_LONG_LONG, globalPair.rank,
              MPI_COMM_WORLD);

    selectedDensity = globalPair.value;
    const size_t globalIndex =
        size_t(globalPair.rank)*localSize + size_t(winningLocalIndex);
    const size_t x = globalIndex % axion->Length();
    const size_t y = (globalIndex/axion->Length()) % axion->Length();
    const size_t z = globalIndex/axion->Surf();

    LogOut("Tracking %s point: global index %zu, (x,y,z)=(%zu,%zu,%zu), "
           "transition density %.9e, owner rank %d\n",
           useRequestedPoint ? "requested" : "maximum-density",
           globalIndex, x, y, z, selectedDensity, globalPair.rank);
    return globalIndex;
}

void printSpatialSample(FILE *file, Scalar *axion, size_t globalIndex,
                        double transitionDensity)
{
    if (file == nullptr)
        return;

    const size_t localSize = axion->Size();
    const int owner = static_cast<int>(globalIndex/localSize);
    if (commRank() != owner)
        return;

    const size_t localIndex = globalIndex - size_t(owner)*localSize;
    const size_t memoryIndex = kiniUnfoldIndex(localIndex, axion);
    double psi = 0.0;
    double psip = 0.0;

#ifdef USE_GPU
    if (axion->Device() == DEV_GPU)
    {
        if (axion->Precision() == FIELD_SINGLE)
        {
            float fieldValue = 0.0f, velocityValue = 0.0f;
            cudaMemcpy(&fieldValue,
                       &static_cast<float *>(axion->mGpuStart())[localIndex],
                       sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&velocityValue,
                       &static_cast<float *>(axion->vGpu())[localIndex],
                       sizeof(float), cudaMemcpyDeviceToHost);
            psi = fieldValue;
            psip = velocityValue;
        }
        else
        {
            cudaMemcpy(&psi,
                       &static_cast<double *>(axion->mGpuStart())[localIndex],
                       sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&psip,
                       &static_cast<double *>(axion->vGpu())[localIndex],
                       sizeof(double), cudaMemcpyDeviceToHost);
        }
    }
    else
#endif
    {
        if (axion->Precision() == FIELD_SINGLE)
        {
            psi = static_cast<float *>(axion->mStart())[memoryIndex];
            psip = static_cast<float *>(axion->vStart())[memoryIndex];
        }
        else
        {
            psi = static_cast<double *>(axion->mStart())[memoryIndex];
            psip = static_cast<double *>(axion->vStart())[memoryIndex];
        }
    }

    const double tau = *axion->zV();
    const double R = *axion->RV();
    double derivedA = 0.0;
    double derivedB = 0.0;
    if (axion->Field() == FIELD_PAXION)
    {
        derivedA = psi*psi + psip*psip;
    }
    else
    {
        derivedA = psi/R;
        derivedB = psip/R - axion->BckGnd()->Rp(tau)*derivedA;
    }
    const size_t x = globalIndex % axion->Length();
    const size_t y = (globalIndex/axion->Length()) % axion->Length();
    const size_t z = globalIndex/axion->Surf();

    std::fprintf(file,
        "%.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e "
        "%zu %zu %zu %zu\n",
        tau, R, axion->AxionMass(), psi, psip, derivedA, derivedB,
        transitionDensity, globalIndex, x, y, z);
    std::fflush(file);
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

bool genSpecKineticMis(size_t N, double L,
                       double theta1, double vtheta1,
                       bool hom )
{
    constexpr double pi = 3.14159265358979323846;

    std::ofstream out("initialspectrum.dat");
    if (!out)
        return false;

    const double k0 = 2.0*pi/L;

    // kini marks this spectrum as physical: theta_k and theta'_k.
    out << std::setprecision(17);
    out << theta1 << " " << vtheta1 << "\n";

    // Higher modes
    for (size_t i = 1; i < 2*N; i++)
    {
        double k = k0*i;

        double m = hom ? 0.0
                       : 4.5e-5*std::pow(L*k,-1.5)*vtheta1;

        out << m << " 0.0\n";
    }

    return bool(out);
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

bool writeMultiplicity(const std::vector<size_t> &mult)
{
	std::ofstream out("out/nn.txt");
	if (!out)
		return false;

	for (size_t k = 0; k < mult.size(); k++)
		out << k << " " << mult[k] << "\n";

	return bool(out);
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

double fluctuationTheta3D_interpolated(
    const double *amp,
    size_t nModes,
    size_t N)
{
    const int NN = static_cast<int>(N);
    const int kMax = NN/2 - 1;
    const size_t maxModP = 3*size_t(kMax)*size_t(kMax);
    double sum = 0.0;

    // Mirror MOM_KM's cutoff and interpolation on the full Fourier cube.
    #pragma omp parallel for collapse(3) reduction(+:sum)
    for (int ix = 0; ix < NN; ix++)
    for (int iy = 0; iy < NN; iy++)
    for (int iz = 0; iz < NN; iz++)
    {
        const int nx = (ix <= NN/2) ? ix : ix-NN;
        const int ny = (iy <= NN/2) ? iy : iy-NN;
        const int nz = (iz <= NN/2) ? iz : iz-NN;
        const size_t modP = size_t(nx*nx + ny*ny + nz*nz);

        if (modP == 0 || modP > maxModP)
            continue;

        const double shell = std::sqrt(double(modP));
        const size_t lower = static_cast<size_t>(shell);
        if (lower + 1 >= nModes)
            continue;

        const double fraction = shell - double(lower);
        const double mode = amp[lower] + (amp[lower+1] - amp[lower])*fraction;
        sum += mode*mode;
    }

    return sum;
}

double spatialThetaSigma(Scalar *axion)
{
    const double R = *axion->RV();
    const double *field = static_cast<double*>(axion->mStart());
    const size_t localSize = axion->Size();
    double localSum = 0.0;
    double localSumSquare = 0.0;

    #pragma omp parallel for reduction(+:localSum,localSumSquare)
    for (size_t idx = 0; idx < localSize; idx++)
    {
        const double theta = field[idx]/R;
        localSum += theta;
        localSumSquare += theta*theta;
    }

    double globalSum = 0.0;
    double globalSumSquare = 0.0;
    unsigned long long localCount = static_cast<unsigned long long>(localSize);
    unsigned long long globalCount = 0;
    MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localSumSquare, &globalSumSquare, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localCount, &globalCount, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);

    if (globalCount == 0)
        return 0.0;

    const double norm = static_cast<double>(globalCount);
    const double mean = globalSum/norm;
    const double meanSquare = globalSumSquare/norm;
    return std::sqrt(std::max(0.0, meanSquare - mean*mean));
}

bool readIC(const std::string &fname,
            double &theta1,
            double &vtheta1,
            double &epsilon1,
            double &vepsilon1)
{
	std::ifstream fin(fname);

	if (!fin)
		return false;

	std::string thetaText, velocityText, epsilonText;
	if (!(fin >> thetaText >> velocityText))
		return false;
	const bool explicitEpsilon = bool(fin >> epsilonText);

	try
	{
		size_t velocityUsed = 0;
		const long double velocity = std::stold(velocityText, &velocityUsed);
		if (velocityUsed != velocityText.size())
			return false;

		const long double pi = std::acos(static_cast<long double>(-1.0));
		if (explicitEpsilon)
		{
			size_t thetaUsed = 0;
			size_t epsilonUsed = 0;
			const long double theta = std::stold(thetaText, &thetaUsed);
			epsilon1 = std::stod(epsilonText, &epsilonUsed);
			if (thetaUsed != thetaText.size()
			    || epsilonText.empty() || epsilonUsed != epsilonText.size())
				return false;
			theta1 = static_cast<double>(theta);
		}
		else if (thetaText.rfind("pi-", 0) == 0 || thetaText.rfind("PI-", 0) == 0)
		{
			size_t epsilonUsed = 0;
			const std::string deltaText = thetaText.substr(3);
			epsilon1 = std::stod(deltaText, &epsilonUsed);
			if (deltaText.empty() || epsilonUsed != deltaText.size())
				return false;
			theta1 = static_cast<double>(pi - static_cast<long double>(epsilon1));
		}
		else
		{
			size_t thetaUsed = 0;
			const long double theta = std::stold(thetaText, &thetaUsed);
			if (thetaUsed != thetaText.size())
				return false;
			theta1 = static_cast<double>(theta);
			epsilon1 = static_cast<double>(pi - theta);
		}

		vtheta1 = static_cast<double>(velocity);
		vepsilon1 = static_cast<double>(-velocity);
	}
	catch (const std::exception &)
	{
		return false;
	}

	return std::isfinite(theta1) && std::isfinite(vtheta1)
		&& std::isfinite(epsilon1) && std::isfinite(vepsilon1);
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

int readMeasurementSchedule(Scalar *axion, MeasFileParms &schedule)
{
    std::ifstream file("measfile.dat");
    if (!file)
        return 0;

    std::vector<int> columns;
    bool columnsKnown = false;
    double previousTime = *axion->zV();
    std::string line;
    size_t lineNumber = 0;

    while (std::getline(file, line))
    {
        lineNumber++;
        const size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#')
            continue;

        std::istringstream input(line);
        std::vector<std::string> fields;
        for (std::string field; input >> field;)
            fields.push_back(field);

        if (!columnsKnown)
        {
            bool header = false;
            try
            {
                size_t used = 0;
                std::stod(fields.front(), &used);
                header = used != fields.front().size();
            }
            catch (const std::exception &)
            {
                header = true;
            }

            if (header)
            {
                bool hasTimeColumn = false;
                for (const std::string &field : fields)
                {
                    if      (field == "ct")   { columns.push_back(0); hasTimeColumn = true; }
                    else if (field == "meas") columns.push_back(1);
                    else if (field == "map")  columns.push_back(2);
                    else if (field == "mask") columns.push_back(3);
                    else if (field == "kgv")  columns.push_back(4);
                    else
                    {
                        LogOut("Error: unknown measfile.dat column '%s' on line %zu\n",
                               field.c_str(), lineNumber);
                        return -1;
                    }
                }
                if (!hasTimeColumn)
                {
                    LogOut("Error: measfile.dat header must contain a 'ct' column\n");
                    return -1;
                }
                columnsKnown = true;
                continue;
            }

            columns = {0, 1};
            columnsKnown = true;
        }

        if (fields.size() < columns.size())
        {
            LogOut("Error: measfile.dat line %zu has %zu fields; expected %zu\n",
                   lineNumber, fields.size(), columns.size());
            return -1;
        }

        double measurementTime = 0.0;
        int values[4] = {static_cast<int>(defaultmeasType), 0, 0, 0};
        try
        {
            for (size_t column = 0; column < columns.size(); column++)
            {
                if (columns[column] == 0)
                    measurementTime = std::stod(fields[column]);
                else
                    values[columns[column] - 1] = std::stoi(fields[column]);
            }
        }
        catch (const std::exception &error)
        {
            LogOut("Error parsing measfile.dat line %zu: %s\n",
                   lineNumber, error.what());
            return -1;
        }

        if (values[0] < 0)
            values[0] = defaultmeasType;
        values[1] |= deninfa.maty;
        values[2] |= deninfa.mask;
        values[3] |= deninfa.nrt;

        if (measurementTime <= *axion->zV())
        {
            LogOut("Skipping measfile.dat time %.6e at or before spatial start %.6e\n",
                   measurementTime, *axion->zV());
            continue;
        }
        if (measurementTime < previousTime)
        {
            LogOut("Error: measfile.dat times must be ordered (line %zu)\n", lineNumber);
            return -1;
        }

        if (!schedule.ct.empty() && measurementTime == schedule.ct.back())
        {
            schedule.meas.back() |= values[0];
            schedule.map.back()  |= values[1];
            schedule.mask.back() |= values[2];
            schedule.nrt.back()  |= values[3];
            continue;
        }

        schedule.ct.push_back(measurementTime);
        schedule.meas.push_back(values[0]);
        schedule.map.push_back(values[1]);
        schedule.mask.push_back(values[2]);
        schedule.nrt.push_back(values[3]);
        previousTime = measurementTime;
    }

    if (!file.eof())
    {
        LogOut("Error while reading measfile.dat\n");
        return -1;
    }
    if (schedule.ct.empty())
    {
        LogOut("Error: measfile.dat contains no measurement times after the spatial start\n");
        return -1;
    }

    return 1;
}

void loadMeasurement(const MeasFileParms &schedule,
                     MeasInfo &info,
                     size_t measurement)
{
    info.measdata = static_cast<MeasureType>(schedule.meas[measurement]);
    info.maty = static_cast<SliceType>(schedule.map[measurement]);
    info.mask = static_cast<SpectrumMaskType>(schedule.mask[measurement]);
    info.nrt = static_cast<nRunType>(schedule.nrt[measurement]);
}
