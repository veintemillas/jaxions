#include <memory>
#include <cstring>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#include "strings/strings.h"
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

	ret.strDen       = tmpData.x;
	ret.strChr       = tmpData.y;
	ret.wallDn       = tmpData.z;
	ret.strDen_local = tmpData.x;
	ret.strChr_local = tmpData.y;
	ret.wallDn_local = tmpData.z;

	stringdata = ret;
	return	ret;
#else
	LogError("Gpu support not built");
	StringData	ret;
	return	ret;
#endif
}

StringData	Strings::runCpu	()
{
	stringdata = stringCpu(axionField);
	return	stringdata;
}

void Strings::resizePos ()
{
	LogMsg(VERB_NORMAL," [Strings] Resized to %d",3*stringdata.strDen_local);
	pos.resize(3*stringdata.strDen_local);
	pos.assign(3*stringdata.strDen_local, 0);
}

void Strings::resizePos (size_t size)
{
	LogMsg(VERB_NORMAL," [Strings] Resized to %d",3*stringdata.strDen);
	pos.resize(3*size);
	pos.assign(3*size, 0);
}
// End class


// -----------------------------------------------------
// Function that calls a class element
// -----------------------------------------------------

using namespace profiler;

StringData	strings	(Scalar *field)
{
	LogMsg	(VERB_HIGH, "Called strings");
	profiler::Profiler &prof = getProfiler(PROF_STRING);

	prof.start();

	StringData	strDen;

	if ((field->Field() & FIELD_AXION) || (field->Field() == FIELD_WKB)) {
		strDen.strDen = 0;
		strDen.strChr = 0;
		strDen.wallDn = 0;
		strDen.strDen_local = 0;
		strDen.strChr_local = 0;
		strDen.wallDn_local = 0;
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
			strDen = eStr->runCpu ();
			break;

		case DEV_GPU:
			strDen = eStr->runGpu ();
			break;

		default:
			LogError ("Error: invalid device\n");
			prof.stop();
			return strDen;
	}

	MPI_Allreduce(&(strDen.strDen_local), &(strDen.strDen), 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&(strDen.strChr_local), &(strDen.strChr), 1, MPI_LONG,          MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&(strDen.wallDn_local), &(strDen.wallDn), 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

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

// -----------------------------------------------------
// Alternative Function that saves string positions
// -----------------------------------------------------


StringData	strings2	(Scalar *field)
{
	LogMsg	(VERB_NORMAL, "[st2] Called strings2");
	profiler::Profiler &prof = getProfiler(PROF_STRING);

	prof.start();

	StringData	strDen;

	if ((field->Field() & FIELD_AXION) || (field->Field() == FIELD_WKB)) {
		strDen.strDen = 0;
		strDen.strChr = 0;
		strDen.wallDn = 0;
		strDen.strDen_local = 0;
		strDen.strChr_local = 0;
		strDen.wallDn_local = 0;

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
			strDen = eStr->runCpu ();
			break;

		case DEV_GPU:
			strDen = eStr->runGpu ();
			break;

		default:
			LogError ("Error: invalid device\n");
			prof.stop();
			return strDen;
	}

	int rank = commRank();
	// printf("[Wtrings %d] strDen %d strChr %d wallDn %d\n",rank, strDen.strDen, strDen.strChr, strDen.wallDn);
	// printf("[Dtrings %d] strDen %d strChr %d wallDn %d\n",rank, strDen.strDen_local, strDen.strChr_local, strDen.wallDn_local);


	MPI_Allreduce(&(strDen.strDen_local), &(strDen.strDen), 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&(strDen.strChr_local), &(strDen.strChr), 1, MPI_LONG,          MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&(strDen.wallDn_local), &(strDen.wallDn), 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

	size_t maxstrDen;
	MPI_Allreduce(&(strDen.strDen_local), &(maxstrDen), 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);


	eStr->SetStrDat(strDen);

	prof.stop();

	eStr->add((15.*strDen.wallDn + 6.*field->rSize())*1.e-9, (7.*field->DataSize() + 1.)*field->rSize()*1.e-9);	// Flops are not exact
	prof.add(eStr->Name(), eStr->GFlops(), eStr->GBytes());

	LogMsg	(VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", eStr->Name().c_str(), prof.Prof()[eStr->Name()].GFlops(), prof.Prof()[eStr->Name()].GBytes());

	// Each rank has strTmp.strDen plaquettes with strings
	// loop to convert them into positions
	eStr->resizePos ();

	size_t carde = eStr->StrDat().strDen_local;
	size_t Lx = field->Length();
	size_t Lz = field->Depth();

	// printf("[Strings %d] strDen %d strChr %d wallDn %d\n",rank,eStr->StrDat().strDen, eStr->StrDat().strChr, eStr->StrDat().wallDn);
	// printf("[Ltrings %d] strDen %d strChr %d wallDn %d\n",rank,eStr->StrDat().strDen_local, eStr->StrDat().strChr_local, eStr->StrDat().wallDn_local);

	size_t unilab = 0;
	char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));

	#pragma omp parallel shared(unilab, strdaa)
	{
		size_t x[3];

		#pragma omp for
		for (size_t idx=0; idx < field->Size() ; idx++)
		{
			size_t besidefresi;

			if ( (strdaa[idx] & STRING_ONLY) != 0)
			{

				// printf("besi %d\n",besidefresi);
				// LogOut("%lu %d %d %d \n",idx,strdaa[idx],STRING_ONLY,strdaa[idx] & STRING_ONLY);
				//thread specific
				size_t tmp = idx/Lx;
				x[2] = (tmp/Lx) ;
				x[1] = (tmp - x[2]*Lx) ;
				x[0] = (idx - tmp*Lx) ;
				x[2] += rank*Lz;

				x[2] *= 2 ;
				x[1] *= 2 ;
				x[0] *= 2 ;

				if (strdaa[idx] & (STRING_XY))
				{
					#pragma omp atomic capture
					{ besidefresi = unilab ; unilab += 1 ; }
					eStr->Pos()[besidefresi*3+2] = x[2];
					eStr->Pos()[besidefresi*3+1] = x[1]+1;
					eStr->Pos()[besidefresi*3]   = x[0]+1;
				}

				if (strdaa[idx] & (STRING_YZ))
				{
					#pragma omp atomic capture
					{ besidefresi = unilab ; unilab += 1 ; }
					eStr->Pos()[besidefresi*3+2] = x[2]+1;
					eStr->Pos()[besidefresi*3+1] = x[1]+1;
					eStr->Pos()[besidefresi*3]   = x[0];
				}

				if (strdaa[idx] & (STRING_ZX))
				{ //x[0] += 1; x[2] += 1;
					#pragma omp atomic capture
					{ besidefresi = unilab ; unilab += 1 ; }
					eStr->Pos()[besidefresi*3+2] = x[2]+1;
					eStr->Pos()[besidefresi*3+1] = x[1];
					eStr->Pos()[besidefresi*3]   = x[0]+1;
				}

				// LogOut("[Strings] %lu %lu elements of table %hu %hu %hu \n",	unilab, idx, eStr->Pos()[3*unilab],	eStr->Pos()[3*unilab+1],	eStr->Pos()[3*unilab+2]);
				// LogOut("[Strings]                     XXXXX %lu %lu %lu \n",	x[0], x[1], x[2]);

			}
		}
	}

	// printf("[st2] rank %d proyected %lu, done %lu\n", rank, eStr->StrDat().strDen_local, unilab);
	// LogOut("strda %d, %d\n",  sizeof(strdaa[0]),      field->Size());
	// LogOut("dasa  %d, %d\n\n",sizeof(eStr->Pos()[0]), sizeof(eStr->Pos()[0])*carde);
	commSync();

	double red = (double) sizeof(eStr->Pos()[0])*carde/ (double)(sizeof(strdaa[0])*field->Size());
	LogMsg	(VERB_NORMAL, "[st2] red = %f ", red);
	// LogOut("pos1   %hu, %hu, %hu\n",eStr->Pos()[0], eStr->Pos()[1], eStr->Pos()[2]);
	// LogOut("pos2   %hu, %hu, %hu\n",eStr->Pos()[3], eStr->Pos()[4], eStr->Pos()[5]);
	// LogOut("pos-2   %hu, %hu, %hu\n",eStr->Pos()[3*unilab-3], eStr->Pos()[3*unilab-2], eStr->Pos()[3*unilab-1]);
	// LogOut("pos-1   %hu, %hu, %hu\n",eStr->Pos()[3*unilab], eStr->Pos()[3*unilab+1], eStr->Pos()[3*unilab+2]);


	/* make sure that the sData buffer never explotes in lowmem */
	/* if no lowmem use m2 ! */
	char *dest ;
	size_t charmax ;
	if (field->LowMem()){
		dest = strdaa;
		LogOut("can fail!");
		charmax = field->Size();
	}
	else
		{
			dest = static_cast<char *>(field->m2Cpu());
			charmax = field->Size()*field->DataSize ();
		}

	size_t carde3 =carde*3*sizeof(eStr->Pos()[0]);
	size_t trinitrize = (charmax/3)*3;
	LogMsg	(VERB_NORMAL, "[st2] charmax = %lu (%lu), needed %lu ", charmax,trinitrize,carde3);

	char *orig = static_cast<char*>( static_cast<void*>( &eStr->Pos()[0] ));

	carde3 = std::min(carde3, trinitrize);

	if (field->LowMem()){
		LogMsg(VERB_HIGH,"[Strings2] copyng %d bytes to sData() [%lu bytes]",carde3,field->Size());
		memcpy (dest, orig, carde3);
		field->setSD(SD_STRINGCOORD);
	}
	else
	{
		LogMsg(VERB_HIGH,"[Strings2] copyng %d bytes to m2Cpu() [%lu bytes]",carde3,field->Size());
		memcpy (dest, orig, carde3);
		field->setM2(M2_STRINGCOO);
		// unsigned short *cerda = static_cast<unsigned short *>( static_cast<void*>( &eStr->Pos()[0] ));
		// printf("strings.cpp rank %d   %hu, %hu, %hu\n", rank, cerda[0],cerda[1],cerda[2]);
	}

  commSync();
	return	strDen;
}


void	stMaskToCM2	(Scalar *field)
{
	switch (field->Precision())
	{
		case FIELD_SINGLE :
		stMaskToCM2<float> (field);
		break;

		case FIELD_DOUBLE :
		stMaskToCM2<double> (field);
		break;
	}
}

template<typename Float>
void	stMaskToCM2	(Scalar *field)
{
	LogMsg	(VERB_NORMAL, "[st2] Called stMaskToCM2");
	// profiler::Profiler &prof = getProfiler(PROF_STRING);
	// prof.start();

	if (field->Field() != FIELD_SAXION || !(field->sDStatus() & SD_MAP)){
			LogMsg(VERB_NORMAL,"[st2] stMaskToCM2 called without string map! (Field = %d, sDStatus= %d)\n",field->Field(),field->sDStatus());
			return;
		}

	char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));
	Float *M2 = static_cast<Float *>(static_cast<void *>(field->m2Cpu()));

	#pragma omp parallel for shared(strdaa)
		for (size_t idx=0; idx < field->Size() ; idx++){
			if ( strdaa[idx] & STRING_MASK ){
				M2[2*idx]   = 1.0;
				M2[2*idx+1] = 1.0;
			}
			else {
				M2[2*idx]   = 0.0;
				M2[2*idx+1] = 0.0;
			}
		}
	// printf("[st2] rank %d proyected %lu, done %lu\n", rank, eStr->StrDat().strDen_local, unilab);
	// LogOut("strda %d, %d\n",  sizeof(strdaa[0]),      field->Size());
	// LogOut("dasa  %d, %d\n\n",sizeof(eStr->Pos()[0]), sizeof(eStr->Pos()[0])*carde);
	commSync();

	field->setM2(M2_ANTIMASK);
		// unsigned short *cerda = static_cast<unsigned short *>( static_cast<void*>( &eStr->Pos()[0] ));
		// printf("strings.cpp rank %d   %hu, %hu, %hu\n", rank, cerda[0],cerda[1],cerda[2]);
}
