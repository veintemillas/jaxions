#include <memory>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#include "scalar/mendThetaXeon.h"

#include "utils/utils.h"

using namespace profiler;

class	MendTheta : public Tunable
{
	private:

	Scalar	*axionField;

	public:

		 MendTheta(Scalar *field) : axionField(field) {};
		~MendTheta() {};

	void	runCpu	();
	void	runGpu	();
};

void	MendTheta::runGpu	()
{
#ifdef	USE_GPU
	LogMsg	 (VERB_NORMAL, "MendTheta gpu kernel not available, will run on CPU");

	Folder	munge(axionField);

	axionField->transferCpu(FIELD_MV);
	munge(FOLD_ALL);
	runCpu();
	munge(UNFOLD_ALL);
	axionField->transferDev(FIELD_MV);

	return;
#else
	LogError ("Error: gpu support not built");
	exit(1);
#endif
}

void	MendTheta::runCpu	()
{
	mendThetaXeon(axionField);
	return;
}

void	mendTheta	(Scalar *field)
{
	int mIter = 0;

	if (!(field->Field() & FIELD_AXION)) {
		LogError ("Error: mendTheta can only be applied to axion fields");
		return;
	}

	auto	theta = std::make_unique<MendTheta>    (field);
	Profiler &prof = getProfiler(PROF_SCALAR);

	theta->setName("Mend Theta");
	prof.start();

	switch (field->Device())
	{
		case DEV_CPU:
			theta->runCpu ();
			break;

		case DEV_GPU:
			theta->runGpu ();
			break;

		default:
			LogError ("Error: invalid device");
			exit(1);
			break;
	}

	theta->add(field->Size()*6.e-9, field->DataSize()*field->Size()*7.e-9);

	prof.stop();
	prof.add(theta->Name(), theta->GFlops(), theta->GBytes());

	LogMsg(VERB_NORMAL, "%s finished", theta->Name().c_str());
	LogMsg  (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", theta->Name().c_str(), prof.Prof()[theta->Name()].GFlops(), prof.Prof()[theta->Name()].GBytes());

	return;
}
