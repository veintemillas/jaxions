#include "scalar/scalarField.h"

void	WKBUNFOLDED(Scalar *axion, void *spectrumK, double zend, const double nnQCD, const double length);
//this function takes evolves theta by an WKB approximation until zend;
//the reult is stored in m2 in the format ma(zend)*pis + i*psi'
//where psi=theta*z
//the number spectrum resulting from the WKB is stored in spectrumK
//it is up to a factor t_1^2/f_a^2
