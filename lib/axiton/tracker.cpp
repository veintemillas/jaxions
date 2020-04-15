#include <string>
#include <complex>
#include <memory>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
#endif

#include "utils/utils.h"
#include "comms/comms.h"

#include "axiton/trackerClass.h"

std::unique_ptr<Tracker> axitrack;

using	namespace profiler;

void	initTracker	(Scalar *field)
{
	LogMsg	(VERB_HIGH, "[AT] Called Init Axiton Tracker");
	profiler::Profiler &prof = getProfiler(PROF_TRACK);

	prof.start();

	axitrack = std::make_unique<Tracker>(field);

	prof.stop();
	prof.add(std::string("Initialisation"), 0.0, 0.0);

	return;
}

void 	searchAxitons()
{
	profiler::Profiler &prof = getProfiler(PROF_TRACK);
	prof.start();
	axitrack->SearchAxitons ();
	axitrack->Update ();
	prof.stop();
	prof.add(std::string("Search Axitons"), 0.0, 0.0);
}

void 	readAxitons()
{
	profiler::Profiler &prof = getProfiler(PROF_TRACK);
	prof.start();
	axitrack->Update ();
	prof.stop();
	prof.add(std::string("Read Axitons"), 0.0, 0.0);
}

void 	printAxitons()
{
	profiler::Profiler &prof = getProfiler(PROF_TRACK);
	prof.start();
	axitrack->PrintAxitons ();
	prof.stop();
	prof.add(std::string("Print Axitons"), 0.0, 0.0);
}
