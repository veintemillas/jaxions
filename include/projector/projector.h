#ifndef	__PROJECTOR_CLASS
	#define	__PROJECTOR_CLASS

	#include<complex>
	#include<functional>
	#include"scalar/scalarField.h"

	void projectField	(Scalar *field, std::function<double(double)> myFilter);
#endif
