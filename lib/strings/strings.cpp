#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#include "strings/stringXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "strings/stringGpu.h"
#endif

//#include "utils/flopCounter.h"
#include "utils/memAlloc.h"
#include "utils/logger.h"
#include "utils/profiler.h"

#include <vector>
#include "utils/index.h"

#include <mpi.h>

class	Strings
{
	private:

	const size_t Lx, V, S;

	FieldPrecision	precision;

	void    *strData;
	Scalar	*axionField;

	public:

		 Strings(Scalar *field, void *str);
		~Strings() {};

	StringData	runCpu	();
	StringData	runGpu	();
	StringData	runXeon	();
};

	Strings::Strings(Scalar *field, void *str) : axionField(field), Lx(field->Length()), V(field->Size()), S(field->Surf()), precision(field->Precision()), strData(str)
{
	memset(strData, 0, V);
}

StringData	Strings::runGpu	()
{
#ifdef	USE_GPU
	const uint uLx = Lx, uS = S, uV = V;
	uint3		tmpData;
	StringData	ret;

	axionField->exchangeGhosts(FIELD_M);
	tmpData = stringGpu(axionField->mGpu(), uLx, uV, uS, precision, strData, ((cudaStream_t *)axionField->Streams())[0]);

	ret.strDen = tmpData.x;
	ret.strChr = tmpData.y;
	ret.wallDn = tmpData.z;

	return	ret;
#else
	LogError("Gpu support not built");
	exit(1);
#endif
}

StringData	Strings::runCpu	()
{
	return	stringCpu(axionField, Lx, V, S, precision, strData);
}

StringData	Strings::runXeon	()
{
#ifdef	USE_XEON
	return	stringXeon(axionField, Lx, V, S, precision, strData);
#else
	LogError("Xeon Phi support not built");
	exit(1);
#endif
}

using namespace profiler;

//StringData	strings	(Scalar *field, void *strData, FlopCounter *fCount)
StringData	strings	(Scalar *field, void *strData)
{
	LogMsg	(VERB_HIGH, "Called strings");
	profiler::Profiler &prof = getProfiler(PROF_STRING);
	const std::string name("Strings and walls");

	prof.start();

	Strings *eStr = new Strings(field, strData);

	StringData	strTmp, strDen;

	if	(!field->Folded())
	{
		Folder	munge(field);
		munge(FOLD_ALL);
	}

	switch (field->Device())
	{
		case DEV_CPU:
			strTmp = eStr->runCpu ();
			break;

		case DEV_GPU:
			strTmp = eStr->runGpu ();
			break;

		case	DEV_XEON:
			strTmp = eStr->runXeon();
			break;

		default:
			LogError ("Not a valid device\n");
			break;
	}

	delete	eStr;

	MPI_Allreduce(&(strTmp.strDen), &(strDen.strDen), 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&(strTmp.strChr), &(strDen.strChr), 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&(strTmp.wallDn), &(strDen.wallDn), 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

	prof.stop();

//	fCount->addFlops((15.*strDen.wallDn + 6.*field->Size())*1.e-9, (7.*field->DataSize() + 1.)*field->Size()*1.e-9);	// Flops are not exact
	prof.add(name, (15.*strDen.wallDn + 6.*field->Size())*1.e-9, (7.*field->DataSize() + 1.)*field->Size()*1.e-9);	// Flops are not exact

	LogMsg	(VERB_HIGH, "Strings reporting %lf GFlops %lf GBytes", prof.Prof()[name].GFlops(), prof.Prof()[name].GBytes());

	return	strDen;
}

std::vector<std::vector<size_t>>	strToCoords	(char *strData, size_t Lx, size_t V)
{
	std::vector<std::vector<size_t>> out(0,std::vector<size_t>(4));

	for (size_t x=0; x<V; x++)
	{
		if (strData[x] != 0)
		{
			std::vector<size_t>	data(4);

			indexXeon::idx2Vec (x, data.data(), Lx);
			data[3] = strData[x];
			out.push_back(data);
		}
	}

	return	out;
}
