#ifndef	_GADGET_O_
	#define	_GADGET_O_

	#include "scalar/scalarField.h"
	#include <hdf5.h>
	#include "utils/binner.h"
 
	void grad_idx (Scalar *axion, Scalar *vaxion, float * grad3, size_t idx);
	float mass_idx(Scalar *axion, size_t idx);
        
        void set_velo_fields(Scalar * axion, Scalar *vaxion);      
        void gaussSmooth(Scalar *field, Scalar *vaxion, int vtype, float length);
        void CIC_interp(Scalar *axion, float * grad3, float * mass, size_t idx, float x_disp = 0.0, float y_disp =  0.0, float z_disp = 0, bool mass_flag = false);
	
        void createGadget_Void (Scalar *axion, Scalar *vaxion, size_t realN=0, size_t nParts=0, bool map_velocity = false, bool sm_vel = false, bool disp_flag = false);
	void createGadget_Halo    (Scalar *axion, size_t realN=0, size_t nParts=0, double sigma = 1.0);

#endif
