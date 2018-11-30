#include "enum-field.h"
#include "scalar/scalarField.h"
#include "energy/energy.h"
#include "reducer/reducer.h"
#include "utils/binner.h"
#include "io/readWrite.h"

#include <cmath>

using namespace std;

// FIXME Set as variable
constexpr double lowCut = 0.001;

template<typename Float>
void	dContrast	(Scalar *axion, size_t &rSize, bool &rhoMap) {

	if (!axion->Reduced() && rSize != 0) {
		Float  ScaleSize = ((Float) axion->Length())/((Float) rSize);
		Float  eFc = 0.5*M_PI*M_PI*(ScaleSize*ScaleSize)/((Float) axion->Surf());
		size_t nLz = rSize / commSize();

		reduceField(axion, rSize, nLz, FIELD_M2, [eFc = eFc] (int px, int py, int pz, complex<Float> x) -> complex<Float>
		{ return (x*exp(-eFc*(px*px + py*py + pz*pz))); });
	}

	Float *m2 = static_cast<Float*>(axion->m2Cpu());
	char  *rC = static_cast<char *>(axion->sData());

	Float eMax = find<FIND_MAX, Float>(m2, axion->rSize(), [] (Float x) { return (double) x; });
	Float eMin = find<FIND_MIN, Float>(m2, axion->rSize(), [] (Float x) { return (double) x; });

	Float cLow = log(lowCut);
	Float cMax, cMin;

	if (eMin <= lowCut)
		cMin = cLow;
	else
		cMin = log(eMin);

	cMax = log(eMax);

	double dCon = 255./(cMax - cMin);

	// Portar a GPU
	#pragma omp parallel for schedule(static)
	for (size_t idx = 0; idx<axion->rSize(); idx++) {
		auto x = m2[idx];

		if (x <= lowCut)
			rC[idx] = 0;
		else
			rC[idx] = (int) round((log(x) - cMin)*dCon);
	}

	writeDensity(axion, MAP_THETA, eMax, eMin);

	if (axion->Field() == FIELD_SAXION && rhoMap) {
		m2 += axion->Size() + axion->Surf()*2;

		eMax = find<FIND_MAX, Float>(m2, axion->rSize(), [] (Float x) { return (double) x; });
		eMin = find<FIND_MIN, Float>(m2, axion->rSize(), [] (Float x) { return (double) x; });

		cLow = log(lowCut);

		if (eMin <= lowCut)
			cMin = cLow;
		else
			cMin = log(eMin);

		cMax = log(eMax);

		dCon = 255./(cMax - cMin);

		// Portar a GPU
		#pragma omp parallel for schedule(static)
		for (size_t idx = 0; idx<axion->rSize(); idx++) {
			auto x = m2[idx];

			if (x <= lowCut)
				rC[idx] = 0;
			else
				rC[idx] = round((log(x) - cMin)*dCon);
		}

		writeDensity(axion, MAP_RHO, eMax, eMin);
	}

	return;
}

void	cDensityMap	(Scalar *axion, size_t rSize, bool rhoMap) {

	if (axion->m2Status() != M2_ENERGY) {
		double eRes[10];
		energy(axion, eRes, true, 0.);
	}

	if (axion->Device() == DEV_GPU)
		axion->transferCpu(FIELD_M2);

	switch (axion->Precision()) {

		case	FIELD_SINGLE:
			dContrast<float> (axion, rSize, rhoMap);
			break;

		case	FIELD_DOUBLE:
			dContrast<double>(axion, rSize, rhoMap);
			break;

		default:
			LogError ("Error: No precision defined in scalar class.");
			return;
	}

	return;
}

