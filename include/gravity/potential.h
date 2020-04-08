#ifndef	_GRAVI_
	#define	_GRAVI_
	#include "scalar/scalarField.h"

	void    InitGravity  (Scalar *field);
	void    tuneGravity	(unsigned int BlockX, unsigned int BlockY, unsigned int BlockZ);
	void    calculateGraviPotential	();
	void    setHybridMode	(bool ca);
	void    tuneGravityHybrid	();
	void    normaliseFields	();
#endif
