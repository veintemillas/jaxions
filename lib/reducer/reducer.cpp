#include <memory>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#include "utils/utils.h"

using namespace profiler;

template<typename Float>
class	Reducer : public Tunable
{
	private:

	Scalar	     *axionField;

	const int  oldLx, oldLz;
	const int  newLx, newLz;
	const bool inPlace;
	FieldIndex fType;

	std::function<complex<Float>(int, int, int, complex<Float>)> filter;

	public:

		 Reducer(Scalar *field, int newLx, int newLz, FieldIndex fType, std::function<complex<Float>(int, int, int, complex<Float>)> myFilter, bool isInPlace=false) :
			axionField(field), fType(fType), filter(myFilter), newLx(newLx), newLz(newLz), oldLx(field->Length()), oldLz(field->TotalDepth()), inPlace(isInPlace) {};
		~Reducer() {};

	void	runCpu	();
	void	runGpu	();
};

void	Reducer::runGpu	()
{
#ifdef	USE_GPU
	LogMsg	 (VERB_NORMAL, "Reducer runs on cpu");

	axionField->transferCpu(FIELD_MV);
	runCpu();
	/*	Restore State to GPU	*/
	//...
	return;
#else
	LogError ("Error: gpu support not built");
	exit(1);
#endif
}

void	Reducer::runCpu	()
{
	if (newLz % commSize() != 0) {
		LogError ("Error: with MPI the new Lz dimension must be divisible by the number of ranks");
		return;
	}

	size_t	splitLz = newLz / commSize();

	if (fType & FIELD_MV) {
		/*	Unfold field	*/
		Folder	munge(axionField);
		munge(UNFOLD_ALL);

		if (field->Field() == FIELD_SAXION) {
			LogError ("Error: reducer not implemented yet for saxion fields");
			return;
		} else {
			auto &myPlan = AxionFFT::fetchPlan("");
		}
	} else {
		int Lz  = field->Depth();
		int hLx = oldLx >> 1;
		int hLz = oldLz >> 1;

		Float	       *mR, *mD;
		complex<Float> *mC;

		if (field->Field() == FIELD_SAXION) {
			mR = static_cast<Float *>(field->m2Cpu());
			mC = static_cast<complex<Float> *>(field->m2Cpu());
			memmove  (mR, mR + field->Surf()*2, sizeof(Float)*oldLx*oldLX*oldLZ);	// Remove ghosts
		} else {
			mR = static_cast<Float *>(field->m2Cpu()) + field->Surf();
			mC = static_cast<complex<Float> *>(field->m2Cpu()) + (field->Surf() >> 1);
		}

		/*	m2 has always the energy, we won't distinguish between	*/
		/*	axion and saxion. This means only theta energy can be	*/
		/*	reduced in the current version of the code. Also the	*/
		/*	FFT with MPI and the padding will spoil the rho energy 	*/

		auto &myPlan = AxionFFT::fetchPlan("pSpecAx");

		size_t Sm       = oldLx*Lz;

		/*	Pad m2. For the saxion, theta energy, we'll need to	*/
		/*	remove the ghosts. Rho energy is all right		*/
		for (size_t sl=Sm-1; sl>0; sl--) {
			auto    oOff = sl*oldLx;
			auto    fOff = sl*(oldLx+2);
			memmove  (mR+fOff, mR+oOff, sizeof(Float)*oldLx);
		}

		myPlan.run(FFT_FWD);

		for (int py = 0; py<Lz; py++)
			for (int pz = 0; pz<oldLx; pz++)
				for (int px = 0; py<oldLx; px++) {
					int kx = px;
					int ky = py + zBase;
					int kz = pz;

					if (kx > hLx) kx -= oldLx;
					if (ky > hLx) ky -= oldLx;
					if (kz > hTz) kz -= oldLz;

					size_t idx = px + oldLx*(pz + oldLz*py);

					auto x = mC[idx];
					mC[idx] = filter(kx, ky, kz, x);
				}

		myPlan.run(FFT_BCK);

		double iX = ((double) oldLx)/((double) newLx);
		double iY = ((double) oldLy)/((double) newLx);
		double iZ = ((double) oldLz)/((double) newLz);

		for (double fZ=0., size_t iz = 0; iz < splitLz; fZ += iZ, iz++) {
			size_t zMax = ceil (fZ);
			size_t zMin = floor(fZ);

			double  rZ = fZ - zMin;

			for (double fY=0., size_t iy=0; iy < newNx; fY += iY, iy++) {
				size_t yMax = ceil (fY);
				size_t yMin = floor(fY);

				double  rY = fY - yMin;

				for (double fX=0., size_t ix=0; ix < newNx; fX += iX, ix++) {
					size_t xMax = ceil (fX);
					size_t xMin = floor(fX);

					double  rX = fX - xMin;

					size_t idx = ix + newLx*(iy + newLx*sz);

					size_t xyz = xMin + oldLx*(yMin + oldLx*zMin);
					size_t Xyz = xMax + oldLx*(yMin + oldLx*zMin);
					size_t xYz = xMin + oldLx*(yMax + oldLx*zMin);
					size_t XYz = xMax + oldLx*(yMax + oldLx*zMin);
					size_t xyZ = xMin + oldLx*(yMin + oldLx*zMax);
					size_t XyZ = xMax + oldLx*(yMin + oldLx*zMax);
					size_t xYZ = xMin + oldLx*(yMax + oldLx*zMax);
					size_t XYZ = xMax + oldLx*(yMax + oldLx*zMax);

					// Averages over the eight points involved in the reduction
					mR[idx] = mR[xyz]*(1.-rX)*(1.-rY)*(1.-rZ) + mR[Xyz]*rX*(1.-rY)*(1.-rZ) +
						  mR[xYz]*(1.-rX)*    rY *(1.-rZ) + mR[XYz]*rX*    rY *(1.-rZ) +
						  mR[xyZ]*(1.-rX)*(1.-rY)*    rZ  + mR[XyZ]*rX*(1.-rY)*    rZ  +
                                                  mR[xYZ]*(1.-rX)*    rY *    rZ  + mR[XYZ]*rX*    rY *    rZ;
				}
			}
		}
	}

	return;
}

void	reduceField	(Scalar *field, size_t newLx, size_t newLz, FieldIndex fType, std::function<double(int, int, int)> myFilter, bool isInPlace=false)
{
	if (!isInPlace && field->Lowmem()) {
		LogError("Error: lowmem requires in place reduction");
		return;
	}

	if ((fType & FIELD_M2) && field->Lowmem()) {
		LogError("Error: can't reduce m2 with lowmem");
		return;
	}

	// If we reduce m or v, we reduce both, for we want to store the configuration

	if ((fType & FIELD_M) || (fType & FIELD_V)) {
		LogError ("Error: reducer implemented only for m2");
		return;
		fType |= FIELD_MV;
		if (fType & FIELD_M2) {
			LogMsg (VERB_HIGH, "Reduction called for m, v and m2");
		} else {
			LogMsg (VERB_HIGH, "Reduction called for m and v");
		}
	} else {
		fType |= FIELD_M2;
		LogMsg (VERB_HIGH, "Reduction called for m2");
	}

	auto	reducer = std::make_unique<Reducer> (field, newLx, newLz, myFilter, isInPlace);
	Profiler &prof = getProfiler(PROF_REDUCER);

	std::stringstream ss;
	ss << "Reduce " << field->Length() << "x" << field->Length() << "x" << field->TotalDepth() << " to " << newLx << "x" << newLx << "x" << newLz; 

	size_t oldVol = field->Size();

	reducer->setName(ss.c_str());
	prof.start();

	switch (field->Device())
	{
		case DEV_CPU:
			reducer->runCpu ();
			break;

		case DEV_GPU:
			reducer->runGpu ();
			break;

		default:
			LogError ("Error: invalid device");
			exit(1);
			break;
	}

	reducer->add(0., field->DataSize()*(field->Size() + oldSize)*7.e-9);

	prof.stop();
	prof.add(reducer->Name(), reducer->GFlops(), reducer->GBytes());

	LogMsg(VERB_NORMAL, "%s finished with %lu jumps", theta->Name().c_str(), nJmps);
	LogMsg  (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", theta->Name().c_str(), prof.Prof()[theta->Name()].GFlops(), prof.Prof()[theta->Name()].GBytes());

	return	nJmps;
}
