#ifndef	_CLASS_SPECTRUM_MASKER_
	#define	_CLASS_SPECTRUM_MASKER_

	void	SpecBin::masker	(int neigh, SpectrumMaskType mask);

	template<typename Float, SpectrumMaskType mask>
	void	SpecBin::masker	(int neigh) ;
#endif
