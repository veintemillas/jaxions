#include <string>
#include <complex>
#include <memory>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#include "gravity/potentialClass.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
#endif

#include "utils/utils.h"
#include "fft/fftCode.h"
#include "comms/comms.h"

std::unique_ptr<GraVi> grav;


using	namespace profiler;

void	InitGravity	(Scalar *field)
{
	LogMsg	(VERB_HIGH, "[GV] Called Init Gravi");
	profiler::Profiler &prof = getProfiler(PROF_GRAVI);

	prof.start();

	grav = std::make_unique<GraVi>(field);

	LogOut("Inito blocks %d %d %d\n",grav->BlockX(), grav->BlockY(), grav->BlockZ());
	prof.stop();
	prof.add(std::string("Initialisation"), 0.0, 0.0);

	return;
}



void	tuneGravity	(unsigned int BlockX, unsigned int BlockY, unsigned int BlockZ)
{
	LogMsg	(VERB_HIGH, "[GV] Called tuneGravi Blocks %u %u %u ", BlockX,  BlockY,  BlockZ);


	grav->SetBlockX( BlockX);
	grav->SetBlockY( BlockY);
	grav->SetBlockZ( BlockZ);
	grav->UpdateBestBlock();
	return;
}


void	tuneGravityHybrid	()
{
	profiler::Profiler &prof = getProfiler(PROF_GRAVI);

	prof.start();

	grav->tunehybrid();

	prof.stop();
	prof.add(std::string("Tuning"), 0.0, 0.0);

	return;
}



void	calculateGraviPotential	()
{

	LogMsg	(VERB_HIGH, "[GV] Calculate Gravitational Potential");
	profiler::Profiler &prof = getProfiler(PROF_GRAVI);

	prof.start();

	grav->Run();

	prof.stop();
	prof.add(std::string("Gravi Potential"), 0.0, 0.0);

	return;
}



void	setHybridMode	(bool ca)
{

	LogMsg	(VERB_HIGH, "[GV] Set hybrid mode for Gravitational Potential %d (yes/no)",ca);

	grav->SetHybrid(ca);

	return;
}

void	normaliseFields	()
{
	/* TODO normalise cuadratic coefficient! */
	LogMsg	(VERB_HIGH, "[GV] Called normalisation of cpaxion to have <|cpax|^2> = 1");

	grav->normaliseFields();

	return;
}
