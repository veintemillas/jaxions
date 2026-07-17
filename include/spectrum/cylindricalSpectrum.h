#ifndef _CYLINDRICAL_SPECTRUM_
#define _CYLINDRICAL_SPECTRUM_

#include "enum-field.h"

class SpecBin;

// Cylindrical implementation behind the public SpecBin facade.  Keeping this
// class independent prevents the Fourier--Bessel data layout from leaking into
// measurement code.
class CylindricalSpectrum {
public:
	static void nRun(SpecBin &spectrum, SpectrumMaskType mask, nRunType nrt);
	static void modeData(SpecBin &spectrum);

private:
	template<typename Float>
	static void runKinetic(SpecBin &spectrum);
};

#endif
