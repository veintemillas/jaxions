#ifndef	__KGVOPS__
	#define	__KGVOPS__

	#include "enum-field.h"
	#include "scalar/scalarField.h"

	/* Define template functions that build the velocity, gradient or potential
	in m2 typically for spectra or dumping.
	It allows masks */

	/* String velocity correction function */

	#define SIGMALUT 0.4

	template<typename Float>
	void	stringcorre	( Float *da, Float *re, int nr);

	void	stringcorre	(void *data, void *result, FieldPrecision fPrec, int nr=1);

	template<typename Float, SpectrumMaskType mask, bool LUTcorr, bool padded>
	size_t buildc_k(Scalar *field, PadIndex pi, Float zaskaFF);

	template <SpectrumMaskType mask, bool LUTcorr>
	void	buildc_k	(Scalar *field, PadIndex pi, double zaskaFF);

	void	buildc_k	(Scalar *field, PadIndex pi, double zaskaFF, SpectrumMaskType mask, bool LUTcorr);

	void	buildc_k_map	(Scalar *field, bool LUTcorr);

	template<typename Float, SpectrumMaskType mask, bool LUTcorr>
	size_t buildc_gx(Scalar *field, PadIndex pi, Float zaskaFF);

	template <SpectrumMaskType mask, bool LUTcorr>
	void	buildc_gx	(Scalar *field, PadIndex pi, double zaskaFF);

	void	buildc_gx	(Scalar *field, PadIndex pi, double zaskaFF, SpectrumMaskType mask, bool LUTcorr);

	template<typename Float, SpectrumMaskType mask, bool LUTcorr>
	size_t buildc_gyz(Scalar *field, PadIndex pi, Float zaskaFF);

	template <SpectrumMaskType mask, bool LUTcorr>
	void	buildc_gyz	(Scalar *field, PadIndex pi, double zaskaFF);

	void	buildc_gyz	(Scalar *field, PadIndex pi, double zaskaFF, SpectrumMaskType mask, bool LUTcorr);

	template<typename Float, SpectrumMaskType mask, bool LUTcorr>
	void buildc_v(Scalar *field, PadIndex pi, Float zaskaFF);

	template <SpectrumMaskType mask, bool LUTcorr>
	void	buildc_v	(Scalar *field, PadIndex pi, double zaskaFF);

	void	buildc_v	(Scalar *field, PadIndex pi, double zaskaFF, SpectrumMaskType mask);
#endif
