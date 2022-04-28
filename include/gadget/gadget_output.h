#ifndef	_GADGET_O_
	#define	_GADGET_O_

	#include "scalar/scalarField.h"
	#include <hdf5.h>
	#include "utils/binner.h"

	//void createGadget	  (Scalar *axion, double eMean, size_t realN=0, size_t nParts=0, double sigma = 1.0, double L1_pc, bool map_velocity = false);
	void createGadget_Mass (Scalar *axion, size_t realN=0, size_t nParts=0, bool map_velocity = false);
	void createGadget_Grid (Scalar *axion, size_t realN=0, size_t nParts=0, bool map_velocity = false);
	void grad_idx (Scalar *axion, float * grad3, size_t idx);
	float mass_idx(Scalar *axion, size_t idx);
	void grad_interp (float * grad3, float * pos, size_t idx, float x_disp = 0.0, float y_disp = 0.0, float z_disp = 0.0);
	float mass_interp(Scalar *axion, size_t idx, float x_disp = 0.0, float y_disp = 0.0, float z_disp = 0.0);
	//void smooth_vel  ();
#endif
