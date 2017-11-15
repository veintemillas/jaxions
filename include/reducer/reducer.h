#ifndef	_REDUCER_CLASS_
	#define	_REDUCER_CLASS_

	#include<complex>
	#include<functional>
	#include"scalar/scalarField.h"

	Scalar*	reduceField	(Scalar *field, size_t newLx, size_t newLz, FieldIndex fType,
				 std::function<std::complex<float> (int, int, int, std::complex<float>)>  myFilter, bool isInPlace = false);
	Scalar*	reduceField	(Scalar *field, size_t newLx, size_t newLz, FieldIndex fType,
				 std::function<std::complex<double>(int, int, int, std::complex<double>)> myFilter, bool isInPlace = false);
#endif
