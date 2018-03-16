#include <memory>
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

#include "utils/utils.h"

#include <vector>
#include "utils/index.h"

#include <mpi.h>

class	Strings	: public Tunable
{
	private:

	Scalar	*axionField;

	public:

			Strings	(Scalar *field);

	StringData	runCpu	();
	StringData	runGpu	();
};

	Strings::Strings(Scalar *field) : axionField(field)
{
	setName("Strings and walls");
	memset(field->sData(), 0, field->Size());
}

StringData	Strings::runGpu	()
{
#ifdef	USE_GPU
	const uint uLx = axionField->Length(),  uLz = axionField->Depth(),  uS = axionField->Surf();
	const uint rLx = axionField->rLength(), rLz = axionField->rDepth(), uV = axionField->Size();
	uint3		tmpData;
	StringData	ret;

	axionField->exchangeGhosts(FIELD_M);
	tmpData = stringGpu(axionField->mGpu(), uLx, uLz, rLx, rLz, uS, uV, axionField->Precision(), axionField->sData(), ((cudaStream_t *)axionField->Streams())[0]);

	ret.strDen = tmpData.x;
	ret.strChr = tmpData.y;
	ret.wallDn = tmpData.z;

	return	ret;
#else
	LogError("Gpu support not built");
	StringData	ret;
	return	ret;
#endif
}

StringData	Strings::runCpu	()
{
	return	stringCpu(axionField);
}

using namespace profiler;

StringData	strings	(Scalar *field)
{
	LogMsg	(VERB_HIGH, "Called strings");
	profiler::Profiler &prof = getProfiler(PROF_STRING);

	prof.start();

	StringData	strTmp, strDen;

	if ((field->Field() & FIELD_AXION) || (field->Field() == FIELD_WKB)) {
		strDen.strDen = 0;
		strDen.strChr = 0;
		strDen.wallDn = 0;

		prof.stop();

		return strDen;
	}

	auto	eStr = std::make_unique<Strings> (field);

	if	(!field->Folded() && field->Device() == DEV_CPU)
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

		default:
			LogError ("Error: invalid device\n");
			prof.stop();
			return strDen;
	}

	MPI_Allreduce(&(strTmp.strDen), &(strDen.strDen), 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&(strTmp.strChr), &(strDen.strChr), 1, MPI_LONG,          MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&(strTmp.wallDn), &(strDen.wallDn), 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

	prof.stop();

	eStr->add((15.*strDen.wallDn + 6.*field->rSize())*1.e-9, (7.*field->DataSize() + 1.)*field->rSize()*1.e-9);	// Flops are not exact
	prof.add(eStr->Name(), eStr->GFlops(), eStr->GBytes());

	LogMsg	(VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", eStr->Name().c_str(), prof.Prof()[eStr->Name()].GFlops(), prof.Prof()[eStr->Name()].GBytes());

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
