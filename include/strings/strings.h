#ifndef	_STRINGS_
	#define	_STRINGS_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"
	#include <vector>
	#include <complex>

	class	Strings	: public Tunable
	{
		private:

		Scalar	*axionField;
		StringData stringdata;
		std::vector<unsigned short>  pos;

		public:

		Strings	(Scalar *field);

		StringData	runCpu	();
		StringData	runGpu	();

		void SetStrDat (StringData star) {stringdata = star; };
		void resizePos ();
		void resizePos (size_t size);
		// This stores local string data, i.e. in the current rank
		StringData StrDat() { return stringdata; };
		std::vector<unsigned short> &Pos() {return pos;};

	};

	StringData	strings	(Scalar *field);

	std::vector<std::vector<size_t>>  strToCoords     (char *strData, size_t Lx, size_t V);

	void strToCoords     (char *strData, size_t Lx, size_t V, int);

	StringData	strings2	(Scalar *field);

	void setCross (std::complex<double> m, std::complex<double> mu, std::complex<double> mv, std::complex<double> muv, double * dua);

	template<typename Float>
	StringData	stringlength	(Scalar *field, StringData strDen);

	template<typename Float>
	StringData	stringlength2	(Scalar *field, StringData strDen);

	StringData	stringlength	(Scalar *field, StringData strDen);
	StringData	stringlength2	(Scalar *field, StringData strDen);

	int	analyzeStrFolded	(Scalar *axion, const int index);
	int	analyzeStrFoldedNP	(Scalar *axion, const int index);
	int	analyzeStrUNFolded	(Scalar *axion, const int index);
#endif
