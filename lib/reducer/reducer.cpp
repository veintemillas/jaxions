#include <memory>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#include "utils/utils.h"

using namespace profiler;

class	Reducer : public Tunable
{
	private:

	Scalar	     *axionField;

	const size_t newLx, newLz;
	const bool   inPlace;

	public:

		 Reducer(Scalar *field, size_t newLx, size_t newLz, bool isInPlace=false) : axionField(field), newLx(newLx), newLz(newLz), inPlace(isInPlace) {};
		~Reducer() {};

	void	runCpu	();
	void	runGpu	();
};

size_t	MendTheta::runGpu	()
{
#ifdef	USE_GPU
	LogMsg	 (VERB_NORMAL, "Reducer runs on cpu");

	axionField->transferCpu(FIELD_MV);
	runCpu();
	/*	Restore State to GPU	*/
	//...
	return;
#else
	LogError ("Error: gpu support not built");
	exit(1);
#endif
}

size_t	MendTheta::runCpu	()
{
	Folder	munge(axionField);

	munge(UNFOLD_ALL);

	return;
}

void	mendTheta	(Scalar *field, size_t newLx, size_t newLz, FieldType fType, bool isInPlace=false)
{
	if (!isInPlace && field->Lowmem()) {
		LogError("Error: lowmem requires in place reduction");
		return;
	}

	if ((fType & FIELD_M2) && field->Lowmem()) {
		LogError("Error: asked to reduce m2 lowmem requires in place reduction");
		return;
	}

	auto	reducer = std::make_unique<Reducer> (field, newLx, newLz, isInPlace);
	Profiler &prof = getProfiler(PROF_REDUCER);

	std::stringstream ss;
	ss << "Reduce " << field->Length() << "x" << field->Length() << "x" << field->TotalDepth() << " to " << newLx << "x" << newLx << "x" << newLz; 

	size_t oldVol = field->Size();

	reducer->setName(ss.c_str());
	prof.start();

	switch (field->Device())
	{
		case DEV_CPU:
			reducer->runCpu ();
			break;

		case DEV_GPU:
			reducer->runGpu ();
			break;

		default:
			LogError ("Error: invalid device");
			exit(1);
			break;
	}

	reducer->add(0., field->DataSize()*(field->Size() + oldSize)*7.e-9);

	prof.stop();
	prof.add(reducer->Name(), reducer->GFlops(), reducer->GBytes());

	LogMsg(VERB_NORMAL, "%s finished with %lu jumps", theta->Name().c_str(), nJmps);
	LogMsg  (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", theta->Name().c_str(), prof.Prof()[theta->Name()].GFlops(), prof.Prof()[theta->Name()].GBytes());

	return	nJmps;
}
