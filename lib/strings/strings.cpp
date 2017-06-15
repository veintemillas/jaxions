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

#include "utils/flopCounter.h"
#include "utils/memAlloc.h"

#include <vector>
#include "utils/index.h"

#include <mpi.h>

class	Strings
{
	private:

	const size_t Lx, V, S;

	FieldPrecision precision;

	void    *strData;
	Scalar	*axionField;

	public:

		 Strings(Scalar *field, void *str);
		~Strings() {};

	size_t	runCpu	();
	size_t	runGpu	();
	size_t	runXeon	();
};

	Strings::Strings(Scalar *field, void *str) : axionField(field), Lx(field->Length()), V(field->Size()), S(field->Surf()), precision(field->Precision()), strData(str)
{
	memset(strData, 0, V);
}

size_t	Strings::runGpu	()
{
#ifdef	USE_GPU
	const uint uLx = Lx, uS = S, uV = V;

	axionField->exchangeGhosts(FIELD_M);
	return	(size_t) stringGpu(axionField->mGpu(), uLx, uV, uS, precision, strData, ((cudaStream_t *)axionField->Streams())[0]);
#else
	printf("Gpu support not built");
	exit(1);
#endif
}

size_t	Strings::runCpu	()
{
	return	stringCpu(axionField, Lx, V, S, precision, strData);
}

size_t	Strings::runXeon	()
{
#ifdef	USE_XEON
	return	stringXeon(axionField, Lx, V, S, precision, strData);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

size_t	strings	(Scalar *field, DeviceType dev, void *strData, FlopCounter *fCount)
{
	Strings *eStr = new Strings(field, strData);

	size_t	strDen = 0, strTmp = 0;

	if	(!field->Folded())
	{
		Folder	munge(field);
		munge(FOLD_ALL);
	}

	switch (dev)
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
			printf ("Not a valid device\n");
			break;
	}

	delete	eStr;
	// int rank ;
	// MPI_Comm_rank( MPI_COMM_WORLD, &rank ) ;
	//
	// printf("rank%d =%d ",rank,strTmp);
	MPI_Allreduce(&strTmp, &strDen, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
//	fCount->addFlops((75.*field->Size() - 10.)*1.e-9, 8.*field->dataSize()*field->Size()*1.e-9);
	

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
