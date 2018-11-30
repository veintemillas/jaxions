#include <memory>
#include <complex>
#include <functional>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#include "fft/fftCode.h"
#include "utils/utils.h"

using namespace std;
using namespace profiler;

template<typename Float>
class	Reducer : public Tunable
{
	private:

	Scalar	   *axionField;
	FieldIndex fType;

	std::function<complex<Float>(int, int, int, complex<Float>)> filter;

	const size_t newLx, newLz;
	const bool   inPlace;


	template<typename Field1, typename Field2, typename Field3, const bool pad>
	void	transformField	(Field1 *f1, Field2 *f2, Field3 *f3, const char *plan, FFTdir = FFT_FWDBCK);

	public:

		 Reducer(Scalar *field, int newLx, int newLz, FieldIndex fType, std::function<complex<Float>(int, int, int, complex<Float>)> myFilter, bool isInPlace=false) :
			axionField(field), fType(fType), filter(myFilter), newLx(newLx), newLz(newLz), inPlace(isInPlace) {};
		~Reducer() {};

	Scalar*	runCpu	();
	Scalar*	runGpu	();
};

template<typename Float>
Scalar*	Reducer<Float>::runGpu	()
{
#ifdef	USE_GPU
	LogMsg	 (VERB_NORMAL, "Reducer runs on cpu");

	axionField->transferCpu(FIELD_MV);
	axionField->transferCpu(FIELD_M2);
	Scalar *reduced = runCpu();
	reduced->transferDev(FIELD_MV);
	reduced->transferDev(FIELD_M2);
	return	reduced;
#else
	LogError ("Error: gpu support not built");
	exit(1);
#endif
}

template<typename Float>
template<typename Field1, typename Field2, typename Field3, const bool pad>
void	Reducer<Float>::transformField	(Field1 *f1, Field2 *f2, Field3 *f3, const char *plan, FFTdir fDir) {
	auto &myPlan = AxionFFT::fetchPlan(plan);
	size_t Lx  = axionField->Length();
	size_t Lz  = axionField->Depth();
	size_t Tz  = axionField->TotalDepth();

	size_t Ly  = Lx/commSize();

	size_t hLx = Lx >> 1;
	//size_t hLz = Lz >> 1;
	size_t hTz = Tz >> 1;

	Float  nrm   = 1./((double) (axionField->TotalSize()));
	size_t zBase = Ly*commRank();

	/*	m2 has always the energy, whether it's axion or saxion	*/

	size_t Sm = Lx*Lz;

	if (fDir != FFT_NONE) {
		/*	Pad f1 into f2 for the horrible r2c transform			*/
		if (pad) {
			if (static_cast<void*>(f1) == static_cast<void*>(f2)) {
				// Energy
				for (size_t sl=Sm-1; sl>0; sl--) {
					auto    oOff = sl* Lx;
					auto    fOff = sl*(Lx+2);
					memmove (f1+fOff, f1+oOff, sizeof(Field1)*Lx);
				}
			} else {
				// Axion
				#pragma omp parallel for schedule(static)
				for (size_t sl=0; sl<Sm; sl++) {
					auto    oOff = sl* Lx;
					auto    fOff = sl*(Lx+2);
					memcpy  (f3+fOff, f1+oOff, sizeof(Field1)*Lx);
				}
			}
		}

		myPlan.run(FFT_FWD);
	}

	uint dX = (pad == true) ? hLx+1 : Lx;

	#pragma omp parallel for collapse(3) schedule(static)
	for (uint py = 0; py<Ly; py++)
		for (uint pz = 0; pz<Tz; pz++)
			for (uint px = 0; px<dX; px++) {
				int kx = px;
				int ky = py + zBase;
				int kz = pz;

				if (kx > static_cast<int>(hLx)) kx -= Lx;
				if (ky > static_cast<int>(hLx)) ky -= Lx;
				if (kz > static_cast<int>(hTz)) kz -= Tz;

				size_t idx = px + dX*(pz + Tz*py);

				auto x = f2[idx];
				f2[idx] = filter(kx, ky, kz, x)*nrm;
			}

	myPlan.run(FFT_BCK);

	/*	Unpad f3 if necessary					*/
	if (pad) {
		for (size_t sl=1; sl<Sm; sl++) {
			auto    oOff = sl*(Lx+2);
			auto    fOff = sl* Lx;
			memmove  (f3+fOff, f3+oOff, sizeof(Field1)*Lx);
		}
	}

	double iX = ((double) Lx)/((double) newLx);
	double iY = ((double) Lx)/((double) newLx);
	double iZ = ((double) Lz)/((double) newLz);

	double fX = 0., fY = 0., fZ = 0.;

	for (size_t iz = 0; iz < newLz; fZ += iZ, iz++) {
		size_t zMax = ceil (fZ);
		size_t zMin = floor(fZ);

		if (zMax >= Lz) zMax = Lz-1;

		double  rZ = fZ - zMin;

		for (size_t iy=0; iy < newLx; fY += iY, iy++) {
			size_t yMax = ceil (fY);
			size_t yMin = floor(fY);

			if (yMax >= Lx) yMax = Lx-1;

			double  rY = fY - yMin;

			for (size_t ix=0; ix < newLx; fX += iX, ix++) {
				size_t xMax = ceil (fX);
				size_t xMin = floor(fX);

				if (xMax >= Lx) xMax = Lx-1;

				double  rX = fX - xMin;

				size_t idx = ix + newLx*(iy + newLx*iz);

				size_t xyz = xMin + Lx*(yMin + Lx*zMin);
				size_t Xyz = xMax + Lx*(yMin + Lx*zMin);
				size_t xYz = xMin + Lx*(yMax + Lx*zMin);
				size_t XYz = xMax + Lx*(yMax + Lx*zMin);
				size_t xyZ = xMin + Lx*(yMin + Lx*zMax);
				size_t XyZ = xMax + Lx*(yMin + Lx*zMax);
				size_t xYZ = xMin + Lx*(yMax + Lx*zMax);
				size_t XYZ = xMax + Lx*(yMax + Lx*zMax);

				// Averages over the eight points involved in the reduction
				f3[idx] = f3[xyz]*((Float)((1.-rX)*(1.-rY)*(1.-rZ))) + f3[Xyz]*((Float)(rX*(1.-rY)*(1.-rZ))) +
					  f3[xYz]*((Float)((1.-rX)*    rY *(1.-rZ))) + f3[XYz]*((Float)(rX*    rY *(1.-rZ))) +
					  f3[xyZ]*((Float)((1.-rX)*(1.-rY)*    rZ )) + f3[XyZ]*((Float)(rX*(1.-rY)*    rZ )) +
                                          f3[xYZ]*((Float)((1.-rX)*    rY *    rZ )) + f3[XYZ]*((Float)(rX*    rY *    rZ ));
			}

			fX = 0.;
		}

		fY = 0.;
	}
}	

template<typename Float>
Scalar*	Reducer<Float>::runCpu	()
{
	Scalar	*outField = nullptr;

	if (!inPlace) {
		// Make sure you don't screw up the FFTs!!
		outField = new Scalar(axionField->BckGnd(), newLx, newLz, axionField->Precision(), axionField->Device(), *axionField->zV(), true, commSize(),
				      axionField->Field() | FIELD_REDUCED, axionField->Lambda(), CONF_NONE, 0, 0.);
	} else {
		outField = axionField;
	}

	if (fType & FIELD_MV) {
		/*	Unfold field	*/
		Folder	munge(axionField);
		munge(UNFOLD_ALL);

		if (axionField->Field() == FIELD_SAXION) {
			complex<Float> *mIn  = static_cast<complex<Float>*>(axionField->mCpu()) + axionField->Surf();
			complex<Float> *vIn  = static_cast<complex<Float>*>(axionField->vCpu());
			complex<Float> *fOut = static_cast<complex<Float>*>(axionField->m2Cpu());

			transformField<complex<Float>,complex<Float>,complex<Float>,false>(mIn, fOut, fOut, "SpSx");

			if (inPlace) {
				// Reduce memory of m AND copy new axionField
				// FIXME DO IT!!
			} else {
				complex<Float> *mD = static_cast<complex<Float>*>(outField->mCpu()) + outField->Surf();
				size_t volData     = outField->Size()*sizeof(complex<Float>);

				// Copy m to the second axionField
				memcpy(mD, fOut, volData);
			}

			transformField<complex<Float>,complex<Float>,complex<Float>,false>(vIn, fOut, fOut, "RdSxV");

			if (inPlace) {
				// Reduce memory of v AND copy new axionField
				// Then, reduce the memory of m2
				// FIXME DO IT!!
			} else {
				complex<Float> *vD = static_cast<complex<Float>*>(outField->vCpu());
				size_t volData     = outField->Size()*sizeof(complex<Float>);

				// Copy v to the second axionField
				memcpy(vD, fOut, volData);
			}
		} else {
			Float          *mIn  = static_cast<Float*>         (axionField->mCpu()) + axionField->Surf();
			Float          *vIn  = static_cast<Float*>         (axionField->vCpu());
			Float          *fOut = static_cast<Float*>         (axionField->m2Cpu());
			complex<Float> *fMid = static_cast<complex<Float>*>(axionField->m2Cpu());

			transformField<Float,complex<Float>,Float,true>(mIn, fMid, fOut, "pSpecAx");

			if (inPlace) {
				// Copy new axionField to m2
				// FIXME DO IT!!
			} else {
				Float *mD      = static_cast<Float*>(outField->mCpu()) + outField->Surf();
				size_t volData = outField->Size()*sizeof(Float);

				// Copy m to the second axionField
				memcpy(mD, fOut, volData);
			}

			transformField<Float,complex<Float>,Float,true>(vIn, fMid, fOut, "pSpecAx");

			if (inPlace) {
				// Copy new axionField to last part of m2 without overwriting m
				// Then, reduce the memory of m/v and m2(v)
				// Copy stuff back
				// FIXME DO IT!!
			} else {
				Float *vD      = static_cast<Float*>(outField->vCpu());
				size_t volData = outField->Size()*sizeof(Float);

				// Copy v to the second axionField
				memcpy(vD, fOut, volData);
			}
		}

		axionField->setM2 (M2_DIRTY);
	} else {
		/*	Reduce m2 means reduce energy, so it's always real	*/
		/*	and ALWAYS in place					*/
		Float          *mR = static_cast<Float *>(axionField->m2Cpu());
		complex<Float> *mC = static_cast<complex<Float>*>(axionField->m2Cpu());

		/*	Reduce rho energy as well, needs new FFT		*/
		if (axionField->Field() == FIELD_SAXION) {
			if (axionField->m2Status() != M2_ENERGY) {
				LogError ("Reducer for m2 requires a previous computation of the energy, ignoring request");
				return	outField;
			}
			transformField<Float,complex<Float>,Float,true>(mR, mC, mR, "pSpecSx");
			mR = static_cast<Float *>(axionField->m2Cpu()) + axionField->Size() + axionField->Surf()*2;
			mC = static_cast<complex<Float>*>(static_cast<void*>(mR));
			transformField<Float,complex<Float>,Float,true>(mR, mC, mR, "RhoSx");
		} else {
			if ((axionField->m2Status() != M2_ENERGY) && (axionField->m2Status() != M2_ENERGY_FFT)) {
				LogError ("Reducer for m2 requires a previous computation of the energy, ignoring request");
				return	outField;
			}

			if (axionField->m2Status() == M2_ENERGY_FFT) {
				LogMsg (VERB_HIGH, "Reusing FFT from previous computation");
				transformField<Float,complex<Float>,Float,true>(mR, mC, mR, "pSpecAx", FFT_NONE);
			} else
				transformField<Float,complex<Float>,Float,true>(mR, mC, mR, "pSpecAx");
		}
		// Signal the axionfield that it contains a reduced energy
		axionField->setReduced(true, newLx, newLz);
		axionField->setM2 (M2_ENERGY);
	}


	return	outField;
}

template<typename Float>
Scalar*	redField	(Scalar *field, size_t newLx, size_t newLz, FieldIndex fType, std::function<complex<Float>(int, int, int, complex<Float>)> myFilter, bool isInPlace)
{
	if (field->LowMem()) {
		LogError("Error: lowmem not supported yet");
		return	nullptr;
	}

	if (!isInPlace && field->LowMem()) {
		LogError("Error: lowmem requires in place reduction");
		return	nullptr;
	}

	if ((fType & FIELD_M2) && field->LowMem()) {
		LogError("Error: can't reduce m2 with lowmem");
		return	nullptr;
	}

	// If we reduce m or v, we reduce both, for we want to store the configuration

	if ((fType & FIELD_M) || (fType & FIELD_V)) {
		fType |= FIELD_MV;
		if (fType & FIELD_M2) {
			LogMsg (VERB_HIGH, "Reduction called for m, v and m2");
		} else {
			LogMsg (VERB_HIGH, "Reduction called for m and v");
		}
	} else {
		fType |= FIELD_M2;
		LogMsg (VERB_HIGH, "Reduction called for m2, will be performed in place");
		isInPlace = true;
	}

	auto	reducer = std::make_unique<Reducer<Float>> (field, newLx, newLz, fType, myFilter, isInPlace);
	Profiler &prof = getProfiler(PROF_REDUCER);

	std::stringstream ss;
	ss << "Reduce " << field->Length() << "x" << field->Length() << "x" << field->TotalDepth() << " to " << newLx << "x" << newLx << "x" << newLz*commSize(); 

	//size_t oldVol = field->Size();

	reducer->setName(ss.str().c_str());
	prof.start();

	Scalar	*outField = nullptr;

	switch (field->Device())
	{
		case DEV_CPU:
			outField = reducer->runCpu ();
			break;

		case DEV_GPU:
			outField = reducer->runGpu ();
			break;

		default:
			LogError ("Error: invalid device, reduction ignored");
			prof.stop();
			return	nullptr;
			break;
	}

	reducer->add(0., field->DataSize()*(field->Size() + outField->Size())*7.e-9);

	prof.stop();
	prof.add(reducer->Name(), reducer->GFlops(), reducer->GBytes());

	LogMsg(VERB_NORMAL, "%s finished", reducer->Name().c_str());
	LogMsg  (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", reducer->Name().c_str(), prof.Prof()[reducer->Name()].GFlops(), prof.Prof()[reducer->Name()].GBytes());

	return	outField;
}

Scalar*	reduceField	(Scalar *field, size_t newLx, size_t newLz, FieldIndex fType, std::function<complex<float>(int, int, int, complex<float>)> myFilter, bool isInPlace) {
	if (field->Precision() != FIELD_SINGLE) {
		LogError("Error: double precision filter for single precision configuration");
		return	nullptr;
	}
	
	return	redField<float>(field, newLx, newLz, fType, myFilter, isInPlace);
}

Scalar*	reduceField	(Scalar *field, size_t newLx, size_t newLz, FieldIndex fType, std::function<complex<double>(int, int, int, complex<double>)> myFilter, bool isInPlace) {
	if (field->Precision() != FIELD_DOUBLE) {
		LogError("Error: single precision filter for double precision configuration");
		return	nullptr;
	}
	
	return	redField(field, newLx, newLz, fType, myFilter, isInPlace);
}
