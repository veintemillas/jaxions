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
	LogMsg	(VERB_NORMAL, "[AT] Called Init Axiton Tracker");

	if (field->Field() != FIELD_AXION && field->Field() != FIELD_PAXION){
		LogMsg(VERB_NORMAL,"[iT] Traker only works in Axion/Paxion mode: Exit.");
		return;
	}

	if (field->BckGnd()->ICData().axtinfo.nMax == -1){
		LogMsg(VERB_NORMAL,"[iT] Traker only works if --axitontracker (int) is parsed");
		return;
	}
	profiler::Profiler &prof = getProfiler(PROF_TRACK);

	prof.start();

	axitrack = std::make_unique<Tracker>(field);

	LogMsg	(VERB_NORMAL, "[AT] Tune by hand, 8, 4, 1");
	axitrack->SetBlockX( field->Length()*8);
	axitrack->SetBlockY( 4);
	axitrack->SetBlockZ( 1);
	axitrack->UpdateBestBlock();

	prof.stop();
	prof.add(std::string("Initialisation"), 0.0, 0.0);

	return;
}

void 	searchAxitons()
{
	LogMsg	(VERB_NORMAL, "[SeA] Searching axitons");
	if (!axitrack)
		return;

	profiler::Profiler &prof = getProfiler(PROF_TRACK);
	prof.start();
	axitrack->SearchAxitons ();
	// axitrack->Update ();
	LogMsg	(VERB_NORMAL, "[SeA] ...");
	prof.stop();
	LogMsg	(VERB_NORMAL, "[SeA] ...");
	prof.add(std::string("Search Axitons"), 0.0, 0.0);
	LogMsg	(VERB_NORMAL, "[SeA] Searching axitons done");
}

void 	readAxitons()
{
	if (!axitrack)
		return;

	profiler::Profiler &prof = getProfiler(PROF_TRACK);
	prof.start();
	axitrack->Update ();
	prof.stop();
	prof.add(std::string("Read Axitons"), 0.0, 0.0);
}

void 	printAxitons()
{
	if (!axitrack)
		return;

	profiler::Profiler &prof = getProfiler(PROF_TRACK);
	prof.start();
	axitrack->PrintAxitons ();
	prof.stop();
	prof.add(std::string("Print Axitons"), 0.0, 0.0);
}

void 	grouptags()
{
	LogMsg	(VERB_NORMAL, "[AT] Called Group Tagged Points");
	if (!axitrack)
		return;

	profiler::Profiler &prof = getProfiler(PROF_TRACK);
	prof.start();
	// axitrack->SetEThreshold(threshold);
	axitrack->GroupTags ();
	axitrack->PatchGroups ();
	prof.stop();
	prof.add(std::string("Group tags"), 0.0, 0.0);
}
