#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <hdf5.h>

#include <random>
#include <fftw3-mpi.h>

#include "scalar/scalarField.h"
#include "scalar/scaleField.h"
#include "scalar/folder.h"
#include "utils/parse.h"
#include "utils/index.h"
#include "comms/comms.h"
#include "io/readWrite.h"

#include "utils/memAlloc.h"
#include "utils/profiler.h"
#include "utils/logger.h"

#include "fft/fftCode.h"
#include "scalar/fourier.h"

/* In case one reads larger grids */
#include "reducer/reducer.h"

using namespace std;
using namespace profiler;

/*
    Auxiliary functions to calculate phase gradient for velocity
*/
// void create_velocity_field()
// {
// 	float grad[3];
// 	grad_idx(axion,grad,idx);
// }


void grad_idx(Scalar *axion, float * grad3, size_t idx)
{  
    const size_t totlX = axion->Length();
    hsize_t S  = axion->Surf();
    size_t idxPx, idxMx, idxPy, idxMy, idxPz, idxMz, X[3],O[4]; // X = (xC,yC,zC), O = (idxPx,idxMx,idxPy,idxMy) 
    float gr_x,gr_y,gr_z;

	indexXeon::idx2VecNeigh(idx,X,O,totlX);				
	idxPx = O[0];
	idxMx = O[1];
	idxPy = O[2];
	idxMy = O[3];
	idxPz = idx + S;
	idxMz = idx - S;
    
    float   *rea = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->mStart()))); 
    float   *ima = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->vStart())));

    gr_x = 0.5*(  atan2(-rea[idxPx]*ima[idx]+ima[idxPx]*rea[idx], rea[idxPx]*rea[idx]+ima[idxPx]*ima[idx]) 
				- atan2(-rea[idxMx]*ima[idx]+ima[idxMx]*rea[idx], rea[idxMx]*rea[idx]+ima[idxMx]*ima[idx]));
							
    gr_y = 0.5*(  atan2(-rea[idxPy]*ima[idx]+ima[idxPy]*rea[idx], rea[idxPy]*rea[idx]+ima[idxPy]*ima[idx]) 
				- atan2(-rea[idxMy]*ima[idx]+ima[idxMy]*rea[idx], rea[idxMy]*rea[idx]+ima[idxMy]*ima[idx]));
							
    gr_z = 0.5*(  atan2(-rea[idxPz]*ima[idx]+ima[idxPz]*rea[idx], rea[idxPz]*rea[idx]+ima[idxPz]*ima[idx]) 
				- atan2(-rea[idxMz]*ima[idx]+ima[idxMz]*rea[idx], rea[idxMz]*rea[idx]+ima[idxMz]*ima[idx]));
    
    grad3[0] = gr_x;
    grad3[1] = gr_y;
    grad3[2] = gr_z;
}

float mass_idx(Scalar *axion, size_t idx)
{
	float mass_int;
	float  *massArr = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->m2Cpu())));
	mass_int = massArr[idx];
	return mass_int;
}

void grad_interp(Scalar *axion, float * grad3, size_t idx, float x_disp, float y_disp, float z_disp)
{
	const size_t totlX = axion->Length();
    hsize_t S  = axion->Surf();

	size_t xyz, Xyz ,xYz, xyZ;
	size_t XYz, XyZ ,xYZ, XYZ;
	size_t X[3], O[4]; //check grad_idx function

	float gra_xyz[3],gra_Xyz[3],gra_xYz[3],gra_xyZ[3];
	float gra_XYz[3],gra_XyZ[3],gra_xYZ[3],gra_XYZ[3];

	indexXeon::idx2VecNeigh(idx,X,O,totlX);
	xyz = idx;
	Xyz = O[0]; //idxPx
	xYz = O[2]; //idxPy
	xyZ = idx + S; //idxPz

	if (X[1] != totlX - 1)
		XYz = Xyz + totlX;          // (1,1,0)
	else
		XYz = xYz + 1;              // (1,1,0)

	if (X[0] != totlX - 1)
		XyZ = xyZ + 1;              // (1,0,1)
	else 
		XyZ = xyZ + totlX;          // (1,0,1)

	if (X[1] != totlX - 1)
		xYZ = xyZ + totlX;          // (0,1,1)
	else 
		xYZ = xYz + S;              // (0,1,1)

	if (X[1] != totlX - 1)
		XYZ = Xyz + totlX + S;      // (1,1,1)
	else
		XYZ = xYz + 1 + S;          // (1,1,1)
	
	grad_idx(axion,gra_xyz,xyz);
	grad_idx(axion,gra_Xyz,Xyz);
	grad_idx(axion,gra_xYz,xYz);
	grad_idx(axion,gra_xyZ,xyZ);

	grad_idx(axion,gra_XYz,XYz);
	grad_idx(axion,gra_XyZ,XyZ);
	grad_idx(axion,gra_xYZ,xYZ);
	grad_idx(axion,gra_XYZ,XYZ);

	grad3[0] = gra_xyz[0] *((float) ( (1.-x_disp) * (1.-y_disp) * (1.-z_disp) ))
			 + gra_Xyz[0] *((float) ( x_disp      * (1.-y_disp) * (1.-z_disp) )) 
			 + gra_xYz[0] *((float) ( (1.-x_disp) * y_disp      * (1.-z_disp) )) 
			 + gra_XYz[0] *((float) ( x_disp      * y_disp      * (1.-z_disp) )) 
			 + gra_xyZ[0] *((float) ( (1.-x_disp) * (1.-y_disp) * z_disp      )) 
			 + gra_XyZ[0] *((float) ( x_disp      * (1.-y_disp) * z_disp      )) 
			 + gra_xYZ[0] *((float) ((1.-x_disp)  *  y_disp     * z_disp      )) 
			 + gra_XYZ[0] *((float) ( x_disp      *  y_disp     * z_disp      ));

	grad3[1] = gra_xyz[1] *((float) ( (1.-x_disp) * (1.-y_disp) * (1.-z_disp) ))
			 + gra_Xyz[1] *((float) ( x_disp      * (1.-y_disp) * (1.-z_disp) )) 
			 + gra_xYz[1] *((float) ( (1.-x_disp) * y_disp      * (1.-z_disp) )) 
			 + gra_XYz[1] *((float) ( x_disp      * y_disp      * (1.-z_disp) )) 
			 + gra_xyZ[1] *((float) ( (1.-x_disp) * (1.-y_disp) * z_disp      )) 
			 + gra_XyZ[1] *((float) ( x_disp      * (1.-y_disp) * z_disp      )) 
			 + gra_xYZ[1] *((float) ((1.-x_disp)  *  y_disp     * z_disp      )) 
			 + gra_XYZ[1] *((float) ( x_disp      *  y_disp     * z_disp      ));

	grad3[2] = gra_xyz[2] *((float) ( (1.-x_disp) * (1.-y_disp) * (1.-z_disp) ))
			 + gra_Xyz[2] *((float) ( x_disp      * (1.-y_disp) * (1.-z_disp) )) 
			 + gra_xYz[2] *((float) ( (1.-x_disp) * y_disp      * (1.-z_disp) )) 
			 + gra_XYz[2] *((float) ( x_disp      * y_disp      * (1.-z_disp) )) 
			 + gra_xyZ[2] *((float) ( (1.-x_disp) * (1.-y_disp) * z_disp      )) 
			 + gra_XyZ[2] *((float) ( x_disp      * (1.-y_disp) * z_disp      )) 
			 + gra_xYZ[2] *((float) ((1.-x_disp)  *  y_disp     * z_disp      )) 
			 + gra_XYZ[2] *((float) ( x_disp      *  y_disp     * z_disp      ));
}

float mass_interp(Scalar *axion, size_t idx, float x_disp, float y_disp, float z_disp)
{
	const size_t totlX = axion->Length();
    hsize_t S  = axion->Surf();

	float mass;
	float mass_xyz,mass_Xyz,mass_xYz,mass_xyZ;
	float mass_XYz,mass_XyZ,mass_xYZ,mass_XYZ;

	size_t xyz, Xyz ,xYz, xyZ;
	size_t XYz, XyZ ,xYZ, XYZ;
	size_t X[3], O[4]; //check grad_idx function

	indexXeon::idx2VecNeigh(idx,X,O,totlX);
	xyz = idx;
	Xyz = O[0]; //idxPx
	xYz = O[2]; //idxPy
	xyZ = idx + S; //idxPz

	if (X[1] != totlX - 1)
		XYz = Xyz + totlX;          // (1,1,0)
	else
		XYz = xYz + 1;              // (1,1,0)

	if (X[0] != totlX - 1)
		XyZ = xyZ + 1;              // (1,0,1)
	else 
		XyZ = xyZ + totlX;          // (1,0,1)

	if (X[1] != totlX - 1)
		xYZ = xyZ + totlX;          // (0,1,1)
	else 
		xYZ = xYz + S;              // (0,1,1)

	if (X[1] != totlX - 1)
		XYZ = Xyz + totlX + S;      // (1,1,1)
	else
		XYZ = xYz + 1 + S;

	mass_xyz = mass_idx(axion,xyz);
	mass_Xyz = mass_idx(axion,Xyz);
	mass_xYz = mass_idx(axion,xYz);
	mass_xyZ = mass_idx(axion,xyZ);

	mass_XYz = mass_idx(axion,XYz);
	mass_XyZ = mass_idx(axion,XyZ);
	mass_xYZ = mass_idx(axion,xYZ);
	mass_XYZ = mass_idx(axion,XYZ);

	mass = mass_xyz *((float) ( (1.-x_disp) * (1.-y_disp) * (1.-z_disp) ))
	     + mass_Xyz *((float) ( x_disp      * (1.-y_disp) * (1.-z_disp) )) 
		 + mass_xYz *((float) ( (1.-x_disp) * y_disp      * (1.-z_disp) )) 
		 + mass_XYz *((float) ( x_disp      * y_disp      * (1.-z_disp) )) 
		 + mass_xyZ *((float) ( (1.-x_disp) * (1.-y_disp) * z_disp      )) 
		 + mass_XyZ *((float) ( x_disp      * (1.-y_disp) * z_disp      )) 
		 + mass_xYZ *((float) ((1.-x_disp)  *  y_disp     * z_disp      )) 
		 + mass_XYZ *((float) ( x_disp      *  y_disp     * z_disp      ));
	return mass;
}

void	createGadget_Mass (Scalar *axion, size_t realN, size_t nParts, bool map_velocity)
{
	hid_t	file_id, hGrp_id, hGDt_id, attr, plist_id, chunk_id, shunk_id, vhunk_id, mhunk_id;
	hid_t	vSt1_id, vSt2_id, sSts_id, mSts_id, aSpace, status;
	hid_t	vSpc1, vSpc2, sSpce, mSpce, memSpace, semSpace, mesSpace, dataType, totalSpace, scalarSpace, massSpace;
	hsize_t	total, slice, slab, offset, rOff;

	char	prec[16], fStr[16];
	int	length = 8;

	size_t	dataSize;

	int myRank = commRank();

	LogMsg (VERB_NORMAL, "Writing Gadget output file");
	LogMsg (VERB_NORMAL, "");
    LogOut("\n----------------------------------------------------------------------\n");
	LogOut("   GAD_MASS selected!        \n");
	LogOut("----------------------------------------------------------------------\n");
	
	/*      Start profiling         */
	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();
    
        /*      WKB not supported atm   */
        if (axion->Field() == FIELD_WKB) 
        {
            LogError ("Error: WKB field not supported");
            prof.stop();
            exit(1);
        }

	/*      If needed, transfer data to host        */
	if (axion->Device() == DEV_GPU)
		axion->transferCpu(FIELD_M2);

	if (axion->m2Cpu() == nullptr) 
        {
		LogError ("You seem to be using the lowmem option");
		prof.stop();
		return;
	}

	/*	Set up parallel access with Hdf5	*/
	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];
	sprintf(base, "%s/%s.hdf5", outDir, gadName);

	/*	Create the file and release the plist	*/
	if ((file_id = H5Fcreate (base, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) < 0)
	{
		LogError ("Error creating file %s", base);
		return;
	}

	H5Pclose(plist_id);

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	commSync();

	switch (axion->Precision())
	{
		case FIELD_SINGLE:
		{
			dataType = H5T_NATIVE_FLOAT;
			dataSize = sizeof(float);
		}

		break;

		case FIELD_DOUBLE:
		{
			dataType = H5T_NATIVE_DOUBLE;
			dataSize = sizeof(double);
		}

		break;

		default:

		LogError ("Error: Invalid precision. How did you get this far?");
		exit(1);

		break;
	}

	/*	Create header	*/
	hGrp_id = H5Gcreate2(file_id, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
	// Units
	double  L1_in_pc = axion->BckGnd()->ICData().L1_pc; 
	double	bSize  = axion->BckGnd()->PhysSize() * L1_in_pc / 0.7;
	double  Omega0 = 0.3;
	double  met_to_pc = 1/(3.08567758e16);
	double  G_N = 6.67430e-11 * 1.98847e30 * met_to_pc * met_to_pc * met_to_pc; // pc^3/s^2/SolarMass
	double  H0 = 0.1 * met_to_pc; // 100 km/s/Mpc in 1/s 
	double  vel_conv = 299792.458; 

	size_t  nPrt = nParts;
	if (nParts == 0)
	nPrt = axion->TotalSize();

	double  totalMass = Omega0 * (bSize*bSize*bSize) * (3.0 * H0*H0) / (8 * M_PI * G_N);
	double  avMass = totalMass/((double) nPrt);

	LogOut("\n[gadmass] Number of particles (nPrt): %lu\n",nPrt);
	LogOut("[gadmass] Box Length: L = %lf pc/h\n",bSize); 
	LogOut("[gadmass] Total Mass: M = %e Solar Masses\n",totalMass);
	LogOut("[gadmass] Average Particle Mass: m_av = %e Solar Masses\n",avMass);

	size_t  iDummy = 0;
	size_t  oDummy = 1;
	double  fDummy = 0.0;

	hid_t attr_type;
	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size   (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	char gtype[16];
	sprintf(gtype, "gadmass");	

    /* Simple scalar attributes */
	writeAttribute(hGrp_id, &bSize,  "BoxSize",                H5T_NATIVE_DOUBLE);
	writeAttribute(hGrp_id, &L1_in_pc, "L1_pc",                H5T_NATIVE_DOUBLE);
	writeAttribute(hGrp_id, &avMass, "AverageMass",            H5T_NATIVE_DOUBLE);
	writeAttribute(hGrp_id, gtype,   "GadType",                attr_type);
	writeAttribute(hGrp_id, &iDummy, "Flag_Entropy_ICs",       H5T_NATIVE_UINT);
	writeAttribute(hGrp_id, &iDummy, "Flag_Cooling",           H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "Flag_DoublePrecision",   H5T_NATIVE_HSIZE);  
	writeAttribute(hGrp_id, &iDummy, "Flag_Feedback",          H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "Flag_Metals",            H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "Flag_Sfr",               H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "Flag_StellarAge",        H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "HubbleParam",            H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &oDummy, "NumFilesPerSnapshot",    H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &fDummy, "Omega0",                 H5T_NATIVE_DOUBLE); 
	writeAttribute(hGrp_id, &iDummy, "OmegaLambda",            H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &fDummy, "Redshift",               H5T_NATIVE_DOUBLE);
	writeAttribute(hGrp_id, &fDummy, "Time",                   H5T_NATIVE_DOUBLE);

	/* Attribute arrays.
     * These need to be created so gadget4 knows what particles to read.
     * This works with the setup NTYPES=6 in the Config.sh file for the compilation.
     * I guess this could be simplified to NTYPES=2 simulations by using hsize_t dims[1]={2}
     * The mass table mTab[] needs all zero entries so we can use multiple masses. 
     * */ 
    hsize_t	dims[1]  = { 2 };
	double	dAFlt[6] = { 0.0, 0.0  };
	double	mTab[6]  = { 0.0, 0.0  };
	size_t	nPart[6] = {   0, nPrt };

	aSpace = H5Screate_simple (1, dims, nullptr);

	attr   = H5Acreate(hGrp_id, "MassTable",              H5T_NATIVE_DOUBLE, aSpace, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Awrite (attr, H5T_NATIVE_DOUBLE, mTab);
	H5Aclose (attr);

	attr   = H5Acreate(hGrp_id, "NumPart_ThisFile",       H5T_NATIVE_HSIZE,  aSpace, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Awrite (attr, H5T_NATIVE_HSIZE, nPart);
	H5Aclose (attr);

	attr   = H5Acreate(hGrp_id, "NumPart_Total",          H5T_NATIVE_HSIZE,  aSpace, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Awrite (attr, H5T_NATIVE_HSIZE, nPart);
	H5Aclose (attr);

	attr   = H5Acreate(hGrp_id, "NumPart_Total_HighWord", H5T_NATIVE_DOUBLE, aSpace, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Awrite (attr, H5T_NATIVE_DOUBLE, dAFlt);
	H5Aclose (attr);

    /*     Group containing particle information   */
    /*	   Create datagroup	                       */
	hGDt_id = H5Gcreate2(file_id, "/PartType1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	uint totlZ = realN;
	uint totlX = realN;
	uint realDepth = realN/commSize();

	if (totlX == 0) 
    {
		totlZ	  = axion->TotalDepth();
		totlX	  = axion->Length();
		realDepth = axion->Depth();
	}
    
	total = ((hsize_t) totlX)*((hsize_t) totlX)*((hsize_t) totlZ);
	slab  = ((hsize_t) totlX)*((hsize_t) totlX);
	rOff  = ((hsize_t) (totlX))*((hsize_t) (totlX))*(realDepth);
    const hsize_t vSlab[2] = { slab, 3 };
	//const hsize_t mSlab[2] = { slab, 1 };

	LogOut("\n[gadmass] Decomposition: total %lu slab %lu rOff %lu\n",total, slab, rOff);
	LogOut("\n[gadgrid] Computing energy grid ...");

	// We need to insert here the energy 

	if (dataSize == 4)
	{
		float * re    = static_cast<float *>(axion->mStart());
		float * im    = static_cast<float *>(axion->vStart());
		float * newEn = static_cast<float *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static)
		for (size_t idx = 0; idx < rOff; idx++)
		{
			newEn[idx] = re[idx]*re[idx]+im[idx]*im[idx];
		} 
	}
	else
	{
		double * re    = static_cast<double *>(axion->mStart());
		double * im    = static_cast<double *>(axion->vStart());
		double * newEn = static_cast<double *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static)
		for (size_t idx = 0; idx < rOff; idx++)
		{
			newEn[idx] = re[idx]*re[idx]+im[idx]*im[idx];
		} 
	}
	
	LogOut("done!");
    LogOut("\n[gadgrid] Computing eMean using one reduction ...");

	double eMean_local = 0.0;
    double eMean_global;      
	
	double * axArray2 = static_cast<double *> (axion->m2half());

    if (dataSize == 4) 
    {
		float * axArray1 = static_cast<float *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static) reduction(+:eMean_local)
		for (size_t idx =0; idx < rOff/2; idx++)
		{
			axArray2[idx] = (double) (axArray1[2*idx] + axArray1[2*idx+1]) ;
			eMean_local += axArray2[idx];
		}
	} 
    else 
    {
		double * axArray1 = static_cast<double *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static) reduction(+:eMean_local)
		for (size_t idx =0; idx < rOff/2; idx++)
		{
			axArray2[idx] = axArray1[2*idx] + axArray1[2*idx+1] ;
			eMean_local += axArray2[idx];
		}
	}
	
	LogOut("done!");
    //LogOut("\n[gadmass] eMean (local) = %lf \n",eMean_local/rOff);
	eMean_local /= rOff;
	MPI_Allreduce(&eMean_local, &eMean_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	eMean_global /= commSize();
   	LogOut("[gadmass] eMean = %.20f \n",eMean_global);
    

    /* New definition of eMean is related to the number of particles and not the grid points. 
     * If the particle number is equal to the number of grid points neweMean = eMean_global
    */
    double neweMean = eMean_global*(totlX*totlX*totlZ)/nPrt;
	double factor = 1/neweMean;
	scaleField(axion,FIELD_M2,factor);
    //LogOut("\n[gadmass] eMean normalised to particle number: neweMean = %lf\n",neweMean);
	LogOut("\n[gadgrid] Energy field normalised!\n"); // m2 now holds the density contrast

    double gloglo = round(eMean_global*rOff/neweMean);
    size_t nPrt_local = (size_t) gloglo;
    int pp_grid = nPrt/(totlX*totlX*totlZ);
    LogOut("[gadmass] Create %d particle(s) per grid site, so rank %d should take %lu particles ",pp_grid,myRank,nPrt_local);
    fflush(stdout);
	commSync();

	/*	Create space for writing the raw data */
	
	hsize_t nPrt_h = (hsize_t) nPrt;
	const hsize_t dDims[2]  = { nPrt_h , 3 };
	//const hsize_t maDims[2] = { nPrt_h , 1 };

	if ((totalSpace = H5Screate_simple(2, dDims, nullptr)) < 0)	// Whole data
	{
		LogError ("Fatal error H5Screate_simple");
		prof.stop();
		exit (1);
	}

	if ((scalarSpace = H5Screate_simple(1, &nPrt_h, nullptr)) < 0)	// Whole data
	{
		LogError ("Fatal error H5Screate_simple");
		prof.stop();
		exit (1);
	}

	/*	Set chunked access - Coordinates  */
	if ((chunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (chunk_id, 2, vSlab) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

	/*  Set chunked access - Velocities */
	if ((vhunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (vhunk_id, 2, vSlab) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

	/*  Set chunked access - IDs */
	if ((shunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (shunk_id, 1, &slab) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_fill_time");
		prof.stop();
		exit (1);
	}
	
	if (H5Pset_fill_time (shunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_fill_time");
		prof.stop();
		exit (1);
	}
	
    if (H5Pset_fill_time (vhunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_fill_time");
		prof.stop();
		exit (1);
	}

	/*	Create a dataset for the vectors and another for the scalars	*/

	char vDt1[16] = "Coordinates";
	char vDt2[16] = "Velocities";
	char mDts[16] = "Masses"; 
	char sDts[16] = "ParticleIDs";
 
	vSt1_id = H5Dcreate (hGDt_id, vDt1, dataType,         totalSpace,  H5P_DEFAULT, chunk_id, H5P_DEFAULT); // Coordinates
	vSt2_id = H5Dcreate (hGDt_id, vDt2, dataType,         totalSpace,  H5P_DEFAULT, vhunk_id, H5P_DEFAULT); // Velocities
	mSts_id = H5Dcreate (hGDt_id, mDts, dataType,         scalarSpace, H5P_DEFAULT, shunk_id, H5P_DEFAULT); // Masses
	sSts_id = H5Dcreate (hGDt_id, sDts, H5T_NATIVE_HSIZE, scalarSpace, H5P_DEFAULT, shunk_id, H5P_DEFAULT); // ParticleIDs

	vSpc1 = H5Dget_space (vSt1_id);
	vSpc2 = H5Dget_space (vSt2_id);
	mSpce = H5Dget_space (mSts_id);
	sSpce = H5Dget_space (sSts_id);

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/
	memSpace = H5Screate_simple(2, vSlab, nullptr);
	semSpace = H5Screate_simple(1, &slab, nullptr);	
	commSync();

    const hsize_t Lz = realDepth;
    const hsize_t stride[2] = { 1, 1 };

	hsize_t Nslabs = nPrt_h/slab/commSize(); // This only works for one particle per grid
    if (nPrt_h > Nslabs*slab*commSize())
        LogOut("\nError: Nparticles is not a multiple of the slab size!");

	/* Write the values in solar masses in m2*/
	if (dataSize == 4) 
    {   
        float * mData = static_cast<float *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx<rOff; idx++)
		{
			mData[idx] *= avMass;
		}
    }
	else
	{
		LogError ("Double precision not supported yet! Set --prec single");
        prof.stop();
        exit(1);
	}
	
    /*  Fill particle coordinates and velocities  */
	LogOut("\n[gadmass] Creating particle coordinates, velocities and masses ... \n");
	size_t lPos = 0;
	for (hsize_t zDim = 0; zDim < Nslabs; zDim++)
	{
		offset = (((hsize_t) (myRank*Nslabs)) + zDim)*slab;
		hsize_t vOffset[2] = { offset , 0 };

		std::random_device rSd;
		std::mt19937_64 rng(rSd());
		std::uniform_real_distribution<float> uni(0.0, 0.5);

		size_t	idx = lPos;
		size_t	tPrti = 0;
		hssize_t	yC  = lPos/totlX;
		hssize_t	zC  = yC  /totlX;
		hssize_t	xC  = lPos - totlX*yC;
		yC -= zC*totlX;
		zC += myRank*Lz;

		/* main function */
		while ((idx < slab*Lz) && (tPrti < slab))
		{
			int nPrti = pp_grid;
			if (dataSize == 4) 
			{	
				float *axOut = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->mCpu())+dataSize*(slab*(Lz*2+1))));
				float  *vOut = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->vBackGhost())+dataSize*(slab*(Lz*2+1))));
				float  *mOut = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->m2Cpu())+ (slab*(Lz+1))*dataSize));
                for (hssize_t i=0; i<nPrti; i++)
				{	
					float xO,yO,zO,x_disp,y_disp,z_disp;
					
					x_disp = uni(rng);
					y_disp = uni(rng);
					z_disp = uni(rng);
					
					xO = (xC + x_disp)*bSize/((float) totlX); 
					yO = (yC + y_disp)*bSize/((float) totlX);
					zO = (zC + z_disp)*bSize/((float) totlZ);

					if (xO < 0.0f) xO += bSize;
					if (yO < 0.0f) yO += bSize;
					if (zO < 0.0f) zO += bSize;
                    if (xO > bSize) xO -= bSize;
					if (yO > bSize) yO -= bSize;
					if (zO > bSize) zO -= bSize;

					axOut[tPrti*3+0] = xO;
					axOut[tPrti*3+1] = yO;
					axOut[tPrti*3+2] = zO;

					if (map_velocity)
					{
						float grad[3];
						grad_interp(axion,grad,idx,x_disp,y_disp,z_disp);
						vOut[tPrti*3+0] = grad[0]/(*axion->zV() * axion->AxionMass()) * vel_conv;
						vOut[tPrti*3+1] = grad[1]/(*axion->zV() * axion->AxionMass()) * vel_conv;
						vOut[tPrti*3+2] = grad[2]/(*axion->zV() * axion->AxionMass()) * vel_conv;
						// if (sm_vel) // to finish
						// {
						// 	smooth_vel();
						// 	vOut[tPrti*3+0] = smgrad[0]/(*axion->zV() * axion->AxionMass());
						// 	vOut[tPrti*3+1] = smgrad[1]/(*axion->zV() * axion->AxionMass());
						// 	vOut[tPrti*3+2] = smgrad[2]/(*axion->zV() * axion->AxionMass());
						// }
					}
					else
					{
						vOut[tPrti*3+0] = 0.f;
						vOut[tPrti*3+1] = 0.f;
						vOut[tPrti*3+2] = 0.f;
					}
					
					float mass = mass_interp(axion,idx,x_disp,y_disp,z_disp);
					mOut[tPrti] = mass;
					
					tPrti++; 
				} 
			} 
			else 
			{ 
                LogError ("Double precision not supported yet! Set --prec single");
		        prof.stop();
		        exit (1);
			}
			idx++;
			lPos = idx;
			xC++;
			if (xC == totlX) { xC = 0; yC++; }
			if (yC == totlX) { yC = 0; zC++; }
		};

		H5Sselect_hyperslab(vSpc1, H5S_SELECT_SET, vOffset, stride, vSlab, nullptr);
		auto rErr = H5Dwrite (vSt1_id, dataType, memSpace, vSpc1, plist_id, static_cast<char *> (axion->mCpu())+(slab*(Lz*2 + 1)*dataSize));
		
		if ((rErr < 0))
		{
			LogError ("Error writing position dataset");
			prof.stop();
			exit(0);
		}
		
		H5Sselect_hyperslab(vSpc2, H5S_SELECT_SET, vOffset, stride, vSlab, nullptr);
		auto vErr = H5Dwrite (vSt2_id, dataType, memSpace, vSpc2, plist_id, static_cast<char *> (axion->vBackGhost())+(slab*(Lz*2+1)*dataSize));
		if ((vErr < 0))
		{
			LogError ("Error writing velocity dataset");
			prof.stop();
			exit(0);
		}

		H5Sselect_hyperslab(mSpce, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
		auto mErr = H5Dwrite (mSts_id, dataType, semSpace, mSpce, plist_id, (static_cast<char *> (axion->m2Cpu())+(slab*(Lz+1))*dataSize));
		if ((mErr < 0))
		{
			LogError ("Error writing mass dataset");
			prof.stop();
			exit(0);
		}
		
	}

	commSync();
	LogOut("\n[gadgrid] Filled coordinates and velocities!"); 

    /*  Fill particle masses        */

    // for (hsize_t zDim = 0; zDim < Nslabs; zDim++)
    // {	
	// 	float *maOut = static_cast<float*>(axion->m2Cpu());
    //     offset = (((hsize_t) (myRank*Nslabs)) + zDim)*slab;
	// 	hsize_t vOffset[2] = { offset , 0 };
	// 	H5Sselect_hyperslab(mSpce, H5S_SELECT_SET, vOffset, NULL, mSlab, NULL);
	// 	hsize_t *imArray = static_cast<hsize_t*>(mArray);
    //     #pragma omp parallel for shared(mArray) schedule(static)
    //     for (hsize_t idx = 0; idx < slab; idx++)
	// 		imArray[idx] = maOut[idx];
	// 	auto rErr = H5Dwrite (mSts_id, dataType, mesSpace, mSpce, plist_id, (static_cast<char *> (axion->m2Cpu())+(slab*zDim)*dataSize));

    //     if (rErr < 0)
    //     {
    //         LogError ("Error writing particle masses");
    //         prof.stop();
    //         exit(0);
    //     }
                    
    //     commSync();
    // }
    
    // LogOut("\n[gadgrid] Filled particle Masses!"); 
    
    /*  Fill particle ID  */
	void *vArray = static_cast<void*>(static_cast<char*>(axion->m2Cpu())+(slab*Lz)*dataSize); // This will be filled with IDs
    for (hsize_t zDim = 0; zDim < Nslabs; zDim++)
    {
        offset = (((hsize_t) (myRank*Nslabs)) + zDim)*slab;
        H5Sselect_hyperslab(sSpce, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
        hsize_t *iArray = static_cast<hsize_t*>(vArray);
        #pragma omp parallel for shared(vArray) schedule(static)
        for (hsize_t idx = 0; idx < slab; idx++)
            iArray[idx] = offset + idx;
                
        auto rErr = H5Dwrite (sSts_id, H5T_NATIVE_HSIZE, semSpace, sSpce, plist_id, (static_cast<char *> (vArray)));
                
        if (rErr < 0)
        {
            LogError ("Error writing particle tag");
            prof.stop();
            exit(0);
        }
                
        commSync();
    }

    LogOut("\n[gadgrid] Filled particle IDs!");

	/*	Close the datasets,groups,files*/
	H5Dclose (vSt1_id);
	H5Dclose (vSt2_id);
	H5Dclose (sSts_id);
	H5Dclose (mSts_id);
	H5Sclose (vSpc1);
	H5Sclose (sSpce);
	H5Sclose (memSpace);
	H5Sclose (aSpace);
	H5Sclose (scalarSpace);
	H5Sclose (totalSpace);
	H5Pclose (chunk_id);
	H5Pclose (shunk_id);
	H5Pclose (mhunk_id);
	H5Pclose (vhunk_id);

	H5Gclose (hGDt_id);
	H5Gclose (hGrp_id);
	H5Pclose (plist_id);
	H5Fclose (file_id);
    //trackFree(exStat);
    prof.stop();

	
}

void	createGadget_Grid (Scalar *axion, size_t realN, size_t nParts, bool map_velocity)
{
	hid_t	file_id, hGrp_id, hGDt_id, attr, plist_id, chunk_id, shunk_id, vhunk_id, mhunk_id;
	hid_t	vSt1_id, vSt2_id, sSts_id, mSts_id, aSpace, status;
	hid_t	vSpc1, vSpc2, sSpce, mSpce, memSpace, semSpace, mesSpace, dataType, totalSpace, scalarSpace, massSpace;
	hsize_t	total, slice, slab, offset, rOff;

	char	prec[16], fStr[16];
	int	length = 8;

	size_t	dataSize;

	int myRank = commRank();

	LogMsg (VERB_NORMAL, "Writing Gadget output file");
	LogMsg (VERB_NORMAL, "");
	LogOut("\n----------------------------------------------------------------------\n");
	LogOut("   GAD_GRID selected!        \n");
	LogOut("----------------------------------------------------------------------\n");
	
	/*      Start profiling         */
	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();
    
        /*      WKB not supported atm   */
        if (axion->Field() == FIELD_WKB) 
        {
            LogError ("Error: WKB field not supported");
            prof.stop();
            exit(1);
        }

	/*      If needed, transfer data to host        */
	if (axion->Device() == DEV_GPU)
		axion->transferCpu(FIELD_M2);

	if (axion->m2Cpu() == nullptr) 
        {
		LogError ("You seem to be using the lowmem option");
		prof.stop();
		return;
	}

	/*	Set up parallel access with Hdf5	*/
	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];
	sprintf(base, "%s/%s.hdf5", outDir, gadName);

	/*	Create the file and release the plist	*/
	if ((file_id = H5Fcreate (base, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) < 0)
	{
		LogError ("Error creating file %s", base);
		return;
	}

	H5Pclose(plist_id);

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	commSync();

	switch (axion->Precision())
	{
		case FIELD_SINGLE:
		{
			dataType = H5T_NATIVE_FLOAT;
			dataSize = sizeof(float);
		}

		break;

		case FIELD_DOUBLE:
		{
			dataType = H5T_NATIVE_DOUBLE;
			dataSize = sizeof(double);
		}

		break;

		default:

		LogError ("Error: Invalid precision. How did you get this far?");
		exit(1);

		break;
	}

	/*	Create header	*/
	hGrp_id = H5Gcreate2(file_id, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
	// Units
	double  L1_in_pc = axion->BckGnd()->ICData().L1_pc; 
	double	bSize  = axion->BckGnd()->PhysSize() * L1_in_pc / 0.7;
	double  Omega0 = 0.3;
	double  met_to_pc = 1/(3.08567758e16);
	double  G_N = 6.67430e-11 * 1.98847e30 * met_to_pc * met_to_pc * met_to_pc; // pc^3/s^2/SolarMass
	double  H0 = 0.1 * met_to_pc; // 100 km/s/Mpc in 1/s
	double  vel_conv = 299792.458;  

	size_t  nPrt = nParts;
	if (nParts == 0)
	nPrt = axion->TotalSize();

	double  totalMass = Omega0 * (bSize*bSize*bSize) * (3.0 * H0*H0) / (8 * M_PI * G_N);
	double  avMass = totalMass/((double) nPrt);

	LogOut("\n[gadgrid] Number of particles (nPrt): %lu\n",nPrt);
	LogOut("[gadgrid] Box Length: L = %lf pc/h\n",bSize); 
	LogOut("[gadgrid] Total Mass: M = %e Solar Masses\n",totalMass);
	LogOut("[gadgrid] Average Particle Mass: m_av = %e Solar Masses\n",avMass);

	size_t  iDummy = 0;
	size_t  oDummy = 1;
	double  fDummy = 0.0;

	hid_t attr_type;
	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size   (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	char gtype[16];
	sprintf(gtype, "gadgrid");

    /* Simple scalar attributes */
	writeAttribute(hGrp_id, &bSize,  "BoxSize",                H5T_NATIVE_DOUBLE);
	writeAttribute(hGrp_id, &L1_in_pc, "L1_pc",                H5T_NATIVE_DOUBLE);
	writeAttribute(hGrp_id, &avMass, "AverageMass",            H5T_NATIVE_DOUBLE);
	writeAttribute(hGrp_id, gtype,   "GadType",                attr_type);
	writeAttribute(hGrp_id, &iDummy, "Flag_Entropy_ICs",       H5T_NATIVE_UINT);
	writeAttribute(hGrp_id, &iDummy, "Flag_Cooling",           H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "Flag_DoublePrecision",   H5T_NATIVE_HSIZE);  
	writeAttribute(hGrp_id, &iDummy, "Flag_Feedback",          H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "Flag_Metals",            H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "Flag_Sfr",               H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "Flag_StellarAge",        H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "HubbleParam",            H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &oDummy, "NumFilesPerSnapshot",    H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &fDummy, "Omega0",                 H5T_NATIVE_DOUBLE); 
	writeAttribute(hGrp_id, &iDummy, "OmegaLambda",            H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &fDummy, "Redshift",               H5T_NATIVE_DOUBLE);
	writeAttribute(hGrp_id, &fDummy, "Time",                   H5T_NATIVE_DOUBLE);

	/* Attribute arrays.
     * These need to be created so gadget4 knows what particles to read.
     * This works with the setup NTYPES=6 in the Config.sh file for the compilation.
     * I guess this could be simplified to NTYPES=2 simulations by using hsize_t dims[1]={2}
     * The mass table mTab[] needs all zero entries so we can use multiple masses. 
     * */ 
    hsize_t	dims[1]  = { 2 };
	double	dAFlt[6] = { 0.0, 0.0  };
	double	mTab[6]  = { 0.0, 0.0  };
	size_t	nPart[6] = {   0, nPrt };

	aSpace = H5Screate_simple (1, dims, nullptr);

	attr   = H5Acreate(hGrp_id, "MassTable",              H5T_NATIVE_DOUBLE, aSpace, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Awrite (attr, H5T_NATIVE_DOUBLE, mTab);
	H5Aclose (attr);

	attr   = H5Acreate(hGrp_id, "NumPart_ThisFile",       H5T_NATIVE_HSIZE,  aSpace, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Awrite (attr, H5T_NATIVE_HSIZE, nPart);
	H5Aclose (attr);

	attr   = H5Acreate(hGrp_id, "NumPart_Total",          H5T_NATIVE_HSIZE,  aSpace, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Awrite (attr, H5T_NATIVE_HSIZE, nPart);
	H5Aclose (attr);

	attr   = H5Acreate(hGrp_id, "NumPart_Total_HighWord", H5T_NATIVE_DOUBLE, aSpace, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Awrite (attr, H5T_NATIVE_DOUBLE, dAFlt);
	H5Aclose (attr);

    /*     Group containing particle information   */
    /*	   Create datagroup	                       */
	hGDt_id = H5Gcreate2(file_id, "/PartType1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	uint totlZ = realN;
	uint totlX = realN;
	uint realDepth = realN/commSize();

	if (totlX == 0) 
    {
		totlZ	  = axion->TotalDepth();
		totlX	  = axion->Length();
		realDepth = axion->Depth();
	}
    
	total = ((hsize_t) totlX)*((hsize_t) totlX)*((hsize_t) totlZ);
	slab  = ((hsize_t) totlX)*((hsize_t) totlX);
	rOff  = ((hsize_t) (totlX))*((hsize_t) (totlX))*(realDepth);
    const hsize_t vSlab[2] = { slab, 3 };
	//const hsize_t mSlab[2] = { slab, 1 };

	LogOut("\n[gadgrid] Decomposition: total %lu slab %lu rOff %lu\n",total, slab, rOff);
	LogOut("\n[gadgrid] Computing energy grid ...");

	// PropParms ppar;
	
	// ppar.Ng    = axion->getNg();
	// ppar.ood2a = 1.0;
	// ppar.Lx    = axion->Length();
	// ppar.Lz    = axion->Depth();
	// size_t BO  = ppar.Ng*ppar.Lx*ppar.Lx;
	// size_t V   = axion->Size();

	// size_t xBlock, yBlock, zBlock;
	// int tmp   = axion->DataAlign()/axion->DataSize();
	// int shift = 0;
	// 	while (tmp != 1) {
	// 	shift++;
	// 	tmp >>= 1;
	// }

	// xBlock = ppar.Lx << shift;
	// yBlock = ppar.Lx >> shift;
	// zBlock = ppar.Lz;

	// Folder munge(axion);
	
	// if (axion->Folded())
	// 	munge(UNFOLD_ALL);
	
	
	// /*Energy map in m2Start*/
	// void *nada;
	// graviPaxKernelXeon<KIDI_ENE>(axion->mCpu(), axion->vCpu(), nada, axion->m2Cpu(), ppar, BO, V+BO, axion->Precision(), xBlock, yBlock, zBlock);

	if (dataSize == 4)
	{
		float * re    = static_cast<float *>(axion->mStart());
		float * im    = static_cast<float *>(axion->vStart());
		float * newEn = static_cast<float *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static)
		for (size_t idx = 0; idx < rOff; idx++)
		{
			newEn[idx] = re[idx]*re[idx]+im[idx]*im[idx];
		} 
	}
	else
	{
		double * re    = static_cast<double *>(axion->mStart());
		double * im    = static_cast<double *>(axion->vStart());
		double * newEn = static_cast<double *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static)
		for (size_t idx = 0; idx < rOff; idx++)
		{
			newEn[idx] = re[idx]*re[idx]+im[idx]*im[idx];
		} 
	}

	LogOut("done!");
	LogOut("\n[gadgrid] Computing eMean using one reduction ...");
	
	double eMean_local = 0.0;
    double eMean_global;      
   
	double * axArray2 = static_cast<double *> (axion->m2half());

    if (dataSize == 4) 
    {
		float * axArray1 = static_cast<float *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static) reduction(+:eMean_local)
		for (size_t idx =0; idx < rOff/2; idx++)
		{
			axArray2[idx] = (double) (axArray1[2*idx] + axArray1[2*idx+1]) ;
			eMean_local += axArray2[idx];
		}
	} 
    else 
    {
		double * axArray1 = static_cast<double *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static) reduction(+:eMean_local)
		for (size_t idx =0; idx < rOff/2; idx++)
		{
			axArray2[idx] = axArray1[2*idx] + axArray1[2*idx+1] ;
			eMean_local += axArray2[idx];
		}
	}

	LogOut("done!");
	//LogOut("\n[gadgrid] eMean (local rank) = %lf \n",eMean_local/rOff);
	eMean_local /= rOff;
	MPI_Allreduce(&eMean_local, &eMean_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	eMean_global /= commSize();
   	LogOut("[gadgrid] eMean = %.20f \n",eMean_global);
    
	/* New definition of eMean is related to the number of particles and not the grid points. 
     * If the particle number is equal to the number of grid points neweMean = eMean_global
    */
    double neweMean = eMean_global*(totlX*totlX*totlZ)/nPrt;
	double factor = 1/neweMean;
	scaleField(axion,FIELD_M2,factor);
	//LogOut("[gadgrid] Normalise to particle number: neweMean = %lf\n",neweMean);
	LogOut("\n[gadgrid] Energy field normalised!\n"); // m2 now holds the density contrast 

    double gloglo = round(eMean_global*rOff/neweMean);
    size_t nPrt_local = (size_t) gloglo;
    int pp_grid = nPrt/(totlX*totlX*totlZ);
    LogOut("[gadgrid] Create %d particle(s) per grid site, so rank %d should take %lu sparticles ",pp_grid,myRank,nPrt_local);
    fflush(stdout);
	commSync();

	/*	Create space for writing the raw data  */
	
	hsize_t nPrt_h = (hsize_t) nPrt;
	const hsize_t dDims[2]  = { nPrt_h , 3 };
	//const hsize_t maDims[2] = { nPrt_h , 1 };

	if ((totalSpace = H5Screate_simple(2, dDims, nullptr)) < 0)	// Whole data
	{
		LogError ("Fatal error H5Screate_simple");
		prof.stop();
		exit (1);
	}

	if ((scalarSpace = H5Screate_simple(1, &nPrt_h, nullptr)) < 0)	// Whole data
	{
		LogError ("Fatal error H5Screate_simple");
		prof.stop();
		exit (1);
	}

	/*	Set chunked access - Coordinates  */
	if ((chunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (chunk_id, 2, vSlab) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

	/*  Set chunked access - Velocities */
	if ((vhunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (vhunk_id, 2, vSlab) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}
	
	/*  Set chunked access - IDs */
	if ((shunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (shunk_id, 1, &slab) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_fill_time");
		prof.stop();
		exit (1);
	}
	
	if (H5Pset_fill_time (shunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_fill_time");
		prof.stop();
		exit (1);
	}
	
    if (H5Pset_fill_time (vhunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_fill_time");
		prof.stop();
		exit (1);
	}

	/*	Create a dataset for the vectors and another for the scalars	*/

	char vDt1[16] = "Coordinates";
	char vDt2[16] = "Velocities";
	char mDts[16] = "Masses"; 
	char sDts[16] = "ParticleIDs";
 
	vSt1_id = H5Dcreate (hGDt_id, vDt1, dataType,         totalSpace,  H5P_DEFAULT, chunk_id, H5P_DEFAULT); // Coordinates
	vSt2_id = H5Dcreate (hGDt_id, vDt2, dataType,         totalSpace,  H5P_DEFAULT, vhunk_id, H5P_DEFAULT); // Velocities
	mSts_id = H5Dcreate (hGDt_id, mDts, dataType,         scalarSpace, H5P_DEFAULT, shunk_id, H5P_DEFAULT); // Masses
	sSts_id = H5Dcreate (hGDt_id, sDts, H5T_NATIVE_HSIZE, scalarSpace, H5P_DEFAULT, shunk_id, H5P_DEFAULT); // ParticleIDs

	vSpc1 = H5Dget_space (vSt1_id);
	vSpc2 = H5Dget_space (vSt2_id);
	mSpce = H5Dget_space (mSts_id);
	sSpce = H5Dget_space (sSts_id);

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/
	memSpace = H5Screate_simple(2, vSlab, nullptr);
	semSpace = H5Screate_simple(1, &slab, nullptr);	
	commSync();

    const hsize_t Lz = realDepth;
    const hsize_t stride[2] = { 1, 1 };

	hsize_t Nslabs = nPrt_h/slab/commSize(); // This only works for one particle per grid
    if (nPrt_h > Nslabs*slab*commSize())
        LogOut("\nError: Nparticles is not a mltiple of the slab size!");
	
    /*  Fill particle coordinates and velocities  */
	LogOut("\n[gadgrid] Creating particle coordinates and velocities ... \n");
	size_t lPos = 0;
	for (hsize_t zDim = 0; zDim < Nslabs; zDim++)
	{
		offset = (((hsize_t) (myRank*Nslabs)) + zDim)*slab;
		hsize_t vOffset[2] = { offset , 0 };

		std::random_device rSd;
		std::mt19937_64 rng(rSd());
		std::uniform_real_distribution<float> uni(0.0, 1.0);

		size_t	idx = lPos;
		size_t	tPrti = 0;
		hssize_t	yC  = lPos/totlX;
		hssize_t	zC  = yC  /totlX;
		hssize_t	xC  = lPos - totlX*yC;
		yC -= zC*totlX;
		zC += myRank*Lz;

		/* main function */
		while ((idx < slab*Lz) && (tPrti < slab))
		{
			int nPrti = pp_grid;
			if (dataSize == 4) 
			{	
				float *axOut = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->m2Cpu())+dataSize*(slab*(Lz*2+1))));
				float  *vOut = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->vBackGhost())+dataSize*(slab*(Lz*2+1))));
                for (hssize_t i=0; i<nPrti; i++)
				{	
					float xO,yO,zO;
					
					xO = xC*bSize/((float) totlX); 
					yO = yC*bSize/((float) totlX);
					zO = zC*bSize/((float) totlZ);

					if (xO < 0.0f) xO += bSize;
					if (yO < 0.0f) yO += bSize;
					if (zO < 0.0f) zO += bSize;
                    if (xO > bSize) xO -= bSize;
					if (yO > bSize) yO -= bSize;
					if (zO > bSize) zO -= bSize;

					axOut[tPrti*3+0] = xO;
					axOut[tPrti*3+1] = yO;
					axOut[tPrti*3+2] = zO;

					if (map_velocity)
					{
						float grad[3];
						grad_idx(axion,grad,idx);
						vOut[tPrti*3+0] = grad[0]/(*axion->zV() * axion->AxionMass()) * vel_conv;
						vOut[tPrti*3+1] = grad[1]/(*axion->zV() * axion->AxionMass()) * vel_conv;
						vOut[tPrti*3+2] = grad[2]/(*axion->zV() * axion->AxionMass()) * vel_conv;
						// if (sm_vel) // to finish
						// {
						// 	smooth_vel();
						// 	vOut[tPrti*3+0] = smgrad[0]/(*axion->zV() * axion->AxionMass());
						// 	vOut[tPrti*3+1] = smgrad[1]/(*axion->zV() * axion->AxionMass());
						// 	vOut[tPrti*3+2] = smgrad[2]/(*axion->zV() * axion->AxionMass());
						// }
					}
					else
					{
						vOut[tPrti*3+0] = 0.f;
						vOut[tPrti*3+1] = 0.f;
						vOut[tPrti*3+2] = 0.f;
					}
					tPrti++; 
				} 
			} 
			else 
			{ 
                LogError ("Double precision not supported yet! Set --prec single");
		        prof.stop();
		        exit (1);
			}
			idx++;
			lPos = idx;
			xC++;
			if (xC == totlX) { xC = 0; yC++; }
			if (yC == totlX) { yC = 0; zC++; }
		};

		H5Sselect_hyperslab(vSpc1, H5S_SELECT_SET, vOffset, stride, vSlab, nullptr);
		auto rErr = H5Dwrite (vSt1_id, dataType, memSpace, vSpc1, plist_id, static_cast<char *> (axion->m2Cpu())+(slab*(Lz*2 + 1)*dataSize));
		
		if ((rErr < 0))
		{
			LogError ("Error writing position dataset");
			prof.stop();
			exit(0);
		}
		
		H5Sselect_hyperslab(vSpc2, H5S_SELECT_SET, vOffset, stride, vSlab, nullptr);
		auto vErr = H5Dwrite (vSt2_id, dataType, memSpace, vSpc2, plist_id, static_cast<char *> (axion->vBackGhost())+(slab*(Lz*2+1)*dataSize));
		if ((vErr < 0))
		{
			LogError ("Error writing velocity dataset");
			prof.stop();
			exit(0);
		}
		
	}

	commSync();
	LogOut("\n[gadgrid] Filled coordinates and velocities!"); 

    /*  Pointers used to fill data  */
	void *mArray = static_cast<void*>(static_cast<char*>(axion->m2Cpu())+(slab*Lz)*dataSize); // This will be filled with masses
    void *vArray = static_cast<void*>(static_cast<char*>(axion->m2Cpu())+(slab*Lz)*dataSize); // This will be filled with IDs

    /*  Fill particle masses        */
    if (dataSize == 4) 
    {   
        float * mData = static_cast<float *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx<rOff; idx++)
		{
			mData[idx] *= avMass;
		}
    }
	else
	{
		LogError ("Double precision not supported yet! Set --prec single");
        prof.stop();
        exit(1);
	}

    for (hsize_t zDim = 0; zDim < Nslabs; zDim++)
    {	
		float *maOut = static_cast<float*>(axion->m2Cpu());
        offset = (((hsize_t) (myRank*Nslabs)) + zDim)*slab;
		H5Sselect_hyperslab(mSpce, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
		hsize_t *imArray = static_cast<hsize_t*>(mArray);
        #pragma omp parallel for shared(mArray) schedule(static)
        for (hsize_t idx = 0; idx < slab; idx++)		
            imArray[idx] = maOut[idx];
        
		auto rErr = H5Dwrite (mSts_id, dataType, semSpace, mSpce, plist_id, (static_cast<char *> (axion->m2Cpu())+(slab*zDim)*dataSize));

        if (rErr < 0)
        {
            LogError ("Error writing particle masses");
            prof.stop();
            exit(0);
        }
                    
        commSync();
    }
    
    LogOut("\n[gadgrid] Filled particle Masses!"); 
    
    /*  Fill particle ID  */
    for (hsize_t zDim = 0; zDim < Nslabs; zDim++)
    {
        offset = (((hsize_t) (myRank*Nslabs)) + zDim)*slab;
        H5Sselect_hyperslab(sSpce, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
        hsize_t *iArray = static_cast<hsize_t*>(vArray);
        #pragma omp parallel for shared(vArray) schedule(static)
        for (hsize_t idx = 0; idx < slab; idx++)
            iArray[idx] = offset + idx;
                
        auto rErr = H5Dwrite (sSts_id, H5T_NATIVE_HSIZE, semSpace, sSpce, plist_id, (static_cast<char *> (vArray)));
                
        if (rErr < 0)
        {
            LogError ("Error writing particle tag");
            prof.stop();
            exit(0);
        }
                
        commSync();
    }

    LogOut("\n[gadgrid] Filled particle IDs!");
	LogOut("\n[gadgrid] Closing HDF5 file!");

	/*	Close the datasets,groups,files*/
	H5Dclose (vSt1_id);
	H5Dclose (vSt2_id);
	H5Dclose (sSts_id);
	H5Dclose (mSts_id);
	H5Sclose (vSpc1);
	H5Sclose (sSpce);
	H5Sclose (memSpace);
	H5Sclose (aSpace);
	H5Sclose (scalarSpace);
	H5Sclose (totalSpace);
	H5Pclose (chunk_id);
	H5Pclose (shunk_id);
	H5Pclose (vhunk_id);

	H5Gclose (hGDt_id);
	H5Gclose (hGrp_id);
	H5Pclose (plist_id);
	H5Fclose (file_id);
    //trackFree(exStat);
    prof.stop();

	
}

void    createGadget_3(Scalar *axion, size_t realN=0, size_t nParts=0, double sigma = 1.0)
{
	hid_t	file_id, hGrp_id, hGDt_id, attr, plist_id, chunk_id, shunk_id, vhunk_id, mhunk_id;
	hid_t	vSt1_id, vSt2_id, sSts_id, aSpace, status;
	hid_t	vSpc1, vSpc2, sSpce, mSpce, memSpace, semSpace, dataType, totalSpace, scalarSpace;
	hsize_t	total, slice, slab, offset, rOff;

	char	prec[16], fStr[16];
	int	length = 8;

	size_t	dataSize;

	int myRank = commRank();

	LogMsg (VERB_NORMAL, "Writing Gadget output file");
	LogMsg (VERB_NORMAL, "");
    LogOut("\n----------------------------------------------------------------------\n");
	LogOut("   GAD_3 selected!        \n");
	LogOut("----------------------------------------------------------------------\n");
	
	/*      Start profiling         */
	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();
    
	/*      WKB not supported atm   */
	if (axion->Field() == FIELD_WKB) 
	{
		LogError ("Error: WKB field not supported");
		prof.stop();
		exit(1);
	}

	/*      If needed, transfer data to host        */
	if (axion->Device() == DEV_GPU)
		axion->transferCpu(FIELD_M2);

	if (axion->m2Cpu() == nullptr) 
        {
		LogError ("You seem to be using the lowmem option");
		prof.stop();
		return;
	}

	/*	Set up parallel access with Hdf5	*/
	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];
	sprintf(base, "%s/%s.hdf5", outDir, gadName);

	/*	Create the file and release the plist	*/
	if ((file_id = H5Fcreate (base, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) < 0)
	{
		LogError ("Error creating file %s", base);
		return;
	}

	H5Pclose(plist_id);

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	commSync();

	switch (axion->Precision())
	{
		case FIELD_SINGLE:
		{
			dataType = H5T_NATIVE_FLOAT;
			dataSize = sizeof(float);
		}

		break;

		case FIELD_DOUBLE:
		{
			dataType = H5T_NATIVE_DOUBLE;
			dataSize = sizeof(double);
		}

		break;

		default:

		LogError ("Error: Invalid precision. How did you get this far?");
		exit(1);

		break;
	}

	/*	Create header	*/
	hGrp_id = H5Gcreate2(file_id, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
	// Units
	double  L1_in_pc = axion->BckGnd()->ICData().L1_pc; 
	double	bSize  = axion->BckGnd()->PhysSize() * L1_in_pc / 0.7;
	double  Omega0 = 0.3;
	double  met_to_pc = 1/(3.08567758e16);
	double  G_N = 6.67430e-11 * 1.98847e30 * met_to_pc * met_to_pc * met_to_pc; // pc^3/s^2/SolarMass
	double  H0 = 0.1 * met_to_pc; // 100 km/s/Mpc in 1/s 
	double  vel_conv = 299792.458;
	size_t  nPrt = nParts;
	if (nParts == 0)
	nPrt = axion->TotalSize();

	double  totalMass = Omega0 * (bSize*bSize*bSize) * (3.0 * H0*H0) / (8 * M_PI * G_N);
	double  avMass = totalMass/((double) nPrt);

	LogOut("\n[gad3] Number of particles (nPrt): %lu\n",nPrt);
	LogOut("[gad3] Box Length: L = %lf pc/h\n",bSize); 
	LogOut("[gad3] Total Mass: M = %e Solar Masses\n",totalMass);
	LogOut("[gad3] Particle Mass: m_av = %e Solar Masses\n",avMass);

	size_t  iDummy = 0;
	size_t  oDummy = 1;
	double  fDummy = 0.0;

	hid_t attr_type;
	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size   (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	char gtype[16];
	sprintf(gtype, "gad3");

    /* Simple scalar attributes */
	writeAttribute(hGrp_id, &bSize,  "BoxSize",                H5T_NATIVE_DOUBLE);
	writeAttribute(hGrp_id, &L1_in_pc, "L1_pc",                H5T_NATIVE_DOUBLE);
	writeAttribute(hGrp_id, &avMass, "AverageMass",            H5T_NATIVE_DOUBLE);
	writeAttribute(hGrp_id, gtype,   "GadType",                attr_type);
	writeAttribute(hGrp_id, &iDummy, "Flag_Entropy_ICs",       H5T_NATIVE_UINT);
	writeAttribute(hGrp_id, &iDummy, "Flag_Cooling",           H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "Flag_DoublePrecision",   H5T_NATIVE_HSIZE);  
	writeAttribute(hGrp_id, &iDummy, "Flag_Feedback",          H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "Flag_Metals",            H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "Flag_Sfr",               H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "Flag_StellarAge",        H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &iDummy, "HubbleParam",            H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &oDummy, "NumFilesPerSnapshot",    H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &fDummy, "Omega0",                 H5T_NATIVE_DOUBLE); 
	writeAttribute(hGrp_id, &iDummy, "OmegaLambda",            H5T_NATIVE_HSIZE);
	writeAttribute(hGrp_id, &fDummy, "Redshift",               H5T_NATIVE_DOUBLE);
	writeAttribute(hGrp_id, &fDummy, "Time",                   H5T_NATIVE_DOUBLE);

	/* Attribute arrays.
     * These need to be created so gadget4 knows what particles to read.
     * This works with the setup NTYPES=6 in the Config.sh file for the compilation.
     * I guess this could be simplified to NTYPES=2 simulations by using hsize_t dims[1]={2}
     * The mass table mTab[] needs all zero entries so we can use multiple masses. 
     * */ 
    hsize_t	dims[1]  = { 2 };
	double	dAFlt[6] = { 0.0, 0.0  };
	double	mTab[6]  = { 0.0, avMass };
	size_t	nPart[6] = {   0, nPrt };

	aSpace = H5Screate_simple (1, dims, nullptr);

	attr   = H5Acreate(hGrp_id, "MassTable",              H5T_NATIVE_DOUBLE, aSpace, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Awrite (attr, H5T_NATIVE_DOUBLE, mTab);
	H5Aclose (attr);

	attr   = H5Acreate(hGrp_id, "NumPart_ThisFile",       H5T_NATIVE_HSIZE,  aSpace, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Awrite (attr, H5T_NATIVE_HSIZE, nPart);
	H5Aclose (attr);

	attr   = H5Acreate(hGrp_id, "NumPart_Total",          H5T_NATIVE_HSIZE,  aSpace, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Awrite (attr, H5T_NATIVE_HSIZE, nPart);
	H5Aclose (attr);

	attr   = H5Acreate(hGrp_id, "NumPart_Total_HighWord", H5T_NATIVE_DOUBLE, aSpace, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Awrite (attr, H5T_NATIVE_DOUBLE, dAFlt);
	H5Aclose (attr);

    /*     Group containing particle information   */
    /*	   Create datagroup	                       */
	hGDt_id = H5Gcreate2(file_id, "/PartType1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	uint totlZ = realN;
	uint totlX = realN;
	uint realDepth = realN/commSize();

	if (totlX == 0) 
    {
		totlZ	  = axion->TotalDepth();
		totlX	  = axion->Length();
		realDepth = axion->Depth();
	}
    
	total = ((hsize_t) totlX)*((hsize_t) totlX)*((hsize_t) totlZ);
	slab  = ((hsize_t) totlX)*((hsize_t) totlX);
	rOff  = ((hsize_t) (totlX))*((hsize_t) (totlX))*(realDepth);
    const hsize_t vSlab[2] = { slab, 3 };

	LogOut("\n[gad3] Decomposition: total %lu slab %lu rOff %lu\n",total, slab, rOff);
	LogOut("\n[gad3] Computing energy grid ...");

	if (dataSize == 4)
	{
		float * re    = static_cast<float *>(axion->mStart());
		float * im    = static_cast<float *>(axion->vStart());
		float * newEn = static_cast<float *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static)
		for (size_t idx = 0; idx < rOff; idx++)
		{
			newEn[idx] = re[idx]*re[idx]+im[idx]*im[idx];
		} 
	}
	else
	{
		double * re    = static_cast<double *>(axion->mStart());
		double * im    = static_cast<double *>(axion->vStart());
		double * newEn = static_cast<double *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static)
		for (size_t idx = 0; idx < rOff; idx++)
		{
			newEn[idx] = re[idx]*re[idx]+im[idx]*im[idx];
		} 
	}

	LogOut("done!");
	LogOut("\n[gad3] Computing eMean using one reduction ...");
	
	double eMean_local = 0.0;
    double eMean_global;      
   
	double * axArray2 = static_cast<double *> (axion->m2half());

    if (dataSize == 4) 
    {
		float * axArray1 = static_cast<float *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static) reduction(+:eMean_local)
		for (size_t idx =0; idx < rOff/2; idx++)
		{
			axArray2[idx] = (double) (axArray1[2*idx] + axArray1[2*idx+1]) ;
			eMean_local += axArray2[idx];
		}
	} 
    else 
    {
		double * axArray1 = static_cast<double *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static) reduction(+:eMean_local)
		for (size_t idx =0; idx < rOff/2; idx++)
		{
			axArray2[idx] = axArray1[2*idx] + axArray1[2*idx+1] ;
			eMean_local += axArray2[idx];
		}
	}

	LogOut("done!");
	eMean_local /= rOff;
	MPI_Allreduce(&eMean_local, &eMean_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	eMean_global /= commSize();
   	LogOut("[gad3] eMean = %.20f \n",eMean_global);
    
	/* New definition of eMean is related to the number of particles and not the grid points. 
     * If the particle number is equal to the number of grid points neweMean = eMean_global
    */
    double neweMean = eMean_global*(totlX*totlX*totlZ)/nPrt;
	double factor = 1/neweMean;
	scaleField(axion,FIELD_M2,factor);
	LogOut("\n[gad3] Energy field normalised!\n"); // m2 now holds the density contrast 

	double localoca = round(eMean_local*rOff/neweMean);
	size_t nPrt_local = (size_t) localoca;
	printf("[gad] rank %d should take %lu particles\n",commRank(),nPrt_local);
	fflush(stdout);
	commSync();

	size_t nPrt_temp;
	MPI_Allreduce(&nPrt_local, &nPrt_temp, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
	if (nPrt_temp > nPrt)
	{
		//LogOut("[gad3] sum is %lu, which is %lu too many \n",nPrt_temp,nPrt_temp-nPrt);
		LogOut("[gad3] Rank 0 readjusts itself from %lu to nPart = %lu\n",nPrt_local,nPrt_local + (nPrt-nPrt_temp) );
		if (commRank()==0)
			nPrt_local = nPrt_local + (nPrt-nPrt_temp);
	}
	else 
	{
		//LogOut("[gad3] sum is %lu, which is %lu too few \n",nPrt_temp,nPrt-nPrt_temp);
		LogOut("[gad3] Rank 0 readjusts itself from %lu to nPart = %lu ...",nPrt_local,nPrt_local + (nPrt-nPrt_temp) );
		if (commRank()==0){
			nPrt_local = nPrt_local + (nPrt-nPrt_temp);
			LogOut("Done! \n");
			}

	}

	/*	Create space for writing the raw data to disk with chunked access	*/
	
	hsize_t nPrt_h = (hsize_t)  nPrt;
	const hsize_t dDims[2] = { nPrt_h , 3 };
	if ((totalSpace = H5Screate_simple(2, dDims, nullptr)) < 0)	// Whole data
	{
		LogError ("Fatal error H5Screate_simple");
		prof.stop();
		exit (1);
	}

	if ((scalarSpace = H5Screate_simple(1, &nPrt_h, nullptr)) < 0)	// Whole data
	{
		LogError ("Fatal error H5Screate_simple");
		prof.stop();
		exit (1);
	}

	/*	Set chunked access	*/
	if ((chunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (chunk_id, 2, vSlab) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

	if ((vhunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (vhunk_id, 2, vSlab) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

	if ((shunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (shunk_id, 1, &slab) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_fill_time");
		prof.stop();
		exit (1);
	}

	if (H5Pset_fill_time (shunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_fill_time");
		prof.stop();
		exit (1);
	}

	/*	The velocity array is initialized to zero, and it stays that way	*/
	if (H5Pset_alloc_time (vhunk_id, H5D_ALLOC_TIME_EARLY) < 0)
	{
		LogError ("Fatal error H5Pset_alloc_time");
		prof.stop();
		exit (1);
	}

	const double zero = 0.0;

	if (H5Pset_fill_value (vhunk_id, dataType, &zero) < 0)
	{
		LogError ("Fatal error H5Pset_fill_value");
		prof.stop();
		exit (1);
	}

	if (H5Pset_fill_time (vhunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_fill_time");
		prof.stop();
		exit (1);
	}

	/*	Create a dataset for the vectors and another for the scalars	*/

	char vDt1[16] = "Coordinates";
	char vDt2[16] = "Velocities";
	char mDts[16] = "Masses"; // not used
	char sDts[16] = "ParticleIDs";

	vSt1_id = H5Dcreate (hGDt_id, vDt1, dataType,         totalSpace,  H5P_DEFAULT, chunk_id, H5P_DEFAULT);
	vSt2_id = H5Dcreate (hGDt_id, vDt2, dataType,         totalSpace,  H5P_DEFAULT, vhunk_id, H5P_DEFAULT);
	sSts_id = H5Dcreate (hGDt_id, sDts, H5T_NATIVE_HSIZE, scalarSpace, H5P_DEFAULT, shunk_id, H5P_DEFAULT);

	vSpc1 = H5Dget_space (vSt1_id);
	//vSpc2 = H5Dget_space (vSt2_id);
	sSpce = H5Dget_space (sSts_id);


	memSpace = H5Screate_simple(2, vSlab, nullptr);	// Slab
	semSpace = H5Screate_simple(1, &slab, nullptr);	// Slab

	commSync();

	const hsize_t Lz = realDepth;
	const hsize_t stride[2] = { 1, 1 };

	/*	For the something	*/
	void *vArray = static_cast<void*>(static_cast<char*>(axion->m2Cpu())+(slab*Lz)*dataSize);
	/*	pointer to the array of integers[exact number of particles per grid site]	*/
	int	*axTmp  = static_cast<int  *>(static_cast<void*>(static_cast<char *> (axion->m2Cpu())+(slab*Lz)*dataSize));

	size_t	tPart = 0, aPart = 0;

	/* Distribute #total particles */
	if (dataSize == 4) {
		/*	pointer to the array of floats[exact density normalised to particle number/site]	*/
		float	*axData = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->m2Cpu())));

		LogOut("[gad] ... \n");
		LogOut("[gad] Decide 1st bunch #particles from (int) density + binomial\n");
		/*	Set all the 1st deterministic bunch of particles according to density contrast	*/
		#pragma omp parallel shared(axData) reduction(+:tPart)
		{
			std::random_device rSd;
			std::mt19937_64 rng(rSd());


			#pragma omp for schedule(static)
			for (hsize_t idx=0; idx<slab*Lz; idx++) {
				float	cData = axData[idx];
				int	nPrti = cData;
				float	rest  = cData - nPrti;

				std::binomial_distribution<int>	bDt  (1, rest);

				nPrti += bDt(rng);

				axTmp[idx] = nPrti;

				tPart += nPrti;
			}
		}

		MPI_Allreduce(&tPart, &aPart, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

		printf("[gad] Rank %d got %lu (target %lu)\n",commRank(),tPart,nPrt_local);
		fflush(stdout);

		commSync();

		LogOut("[gad] Rank sum %lu (target %lu)\n",aPart,nPrt);

		commSync();

		std::random_device rSd;
		std::mt19937_64 rng(rSd());
		std::uniform_int_distribution<size_t> uni  (0, slab*Lz);

		LogOut("[gad] Balancing locally adding/substracting points...\n",aPart,nPrt);
		// while (aPart != total)   [old global version]
		while (tPart != nPrt_local)
		{
			size_t	nIdx = uni(rng);

			auto	cData = axData[nIdx];
			int	cPart = axTmp[nIdx];
			int	pChange = 0;
			int	tChange = 0;

			// if (cData > cPart) [old version adds and substracts]
			if ( (cData > cPart) && (tPart < nPrt_local) ) // only adds if needed
			{
				std::binomial_distribution<int> bDt  (1, cData - ((float) cPart));
				int dPart = bDt(rng);
				axTmp[nIdx] += dPart;
				pChange	    += dPart;
			}
			else if ( (cData < cPart) && (tPart > nPrt_local) ) // only substracts if needed
			{
				std::binomial_distribution<int> bDt  (1, ((float) cPart) - cData);
				int dPart = bDt(rng);
				axTmp[nIdx] -= dPart;
				pChange	    -= dPart;
			}

			tPart += pChange;
			// MPI_Allreduce(&pChange, &tChange, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			// aPart += tChange;
			// LogOut("Balancing... %010zu / %010zu\r", aPart, total); fflush(stdout);
		}
		MPI_Allreduce(&tPart, &aPart, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
	}
		else
	{
		/*	Horrible code, for double	*/
	}


	LogOut("\nBalanced locally tPart=nPrt_local (globally %lu-%lu=0 too )\n",aPart,nPrt);

	commSync();

	const int cSize = commSize();

	/* hssize_t long long integer? */
	// hssize_t excess = tPart - rOff;
	hssize_t excess = tPart - nPrt/commSize();
	printf("rankd %d (desired particles - average per rank) %d  \n", commRank(),excess); fflush(stdout);

	commSync();

	void *tmpPointer;
	trackAlloc (&tmpPointer, sizeof(hssize_t)*cSize);
	hssize_t *exStat = static_cast<hssize_t*>(tmpPointer);

	hssize_t	   remain  = 0;
	hssize_t	   missing = 0;
	size_t lPos    = 0;

	int	rSend = 0, rRecv = 0;

	commSync();

	if (cSize > 1) {
		LogOut("\nGathering excess (MPI)\n");

		MPI_Allgather (&excess, 1, MPI_LONG_LONG_INT, exStat, 1, MPI_LONG_LONG_INT, MPI_COMM_WORLD);

		for (int i=1; i<cSize; i++) {
			if (exStat[i] > exStat[rSend])
				rSend = i;
		}

		for (int i=1; i<cSize; i++) {
			if (exStat[i] < exStat[rRecv])
				rRecv = i;
		}
		LogOut("Send %d Receive %d \n",rSend,rRecv);
	}

	/*Calculate number of required slabs!*/

hsize_t Nslabs = nPrt_h/slab/commSize();
if (nPrt_h > Nslabs*slab*commSize())
	LogOut("\nError: Nparticles is not a multiple of the slab size! (we do a few less) blame it on Alex!\n",Nslabs);
LogOut("\nRecalculated slabs to print %lu \n",Nslabs);


	/* Main loop, fills slabs with particles and writes to disk */

	LogOut("\nStarting main loop\n");

	for (hsize_t zDim = 0; zDim < Nslabs; zDim++)
	{
		//LogOut("zDim %zu\n", zDim);
		/*	Select the slab in the file	*/
		offset = (((hsize_t) (myRank*Nslabs)) + zDim)*slab;
		hsize_t vOffset[2] = { offset , 0 };

		std::random_device rSd;
		std::mt19937_64 rng(rSd());

		/* initialise random number generator */ //FIXME threads?
		std::normal_distribution<float>	gauss(0.0, sigma);

		/* position of the grid point to convert to particles */
		size_t	idx = lPos;
		hssize_t	yC  = lPos/totlX;
		hssize_t	zC  = yC  /totlX;
		hssize_t	xC  = lPos - totlX*yC;
		yC -= zC*totlX;
		zC += myRank*Lz;

		/* number of particles placed in the coordinate list */
		size_t	tPrti = 0;

		/* the last point of the slab had still some leftover particles to place? */
			if (remain)
			{
				hssize_t	nY  = (lPos-1)/totlX;
				hssize_t	nZ  = nY  /totlX;
				hssize_t	nX  = (lPos-1) - totlX*nY;  //nY was zC
				nY -= nZ*totlX;
				nZ += myRank*Lz;

				if (dataSize == 4) {
					float	*axOut = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->m2Cpu())+dataSize*(slab*(Lz*2+1))));
					for (hssize_t i=0; i<remain; i++)
					{
						float xO = (nX + 0.5f + gauss(rng) )*bSize/((float) totlX);
						float yO = (nY + 0.5f + gauss(rng) )*bSize/((float) totlX);
						float zO = (nZ + 0.5f + gauss(rng) )*bSize/((float) totlZ);

						if (xO < 0.0f) xO += bSize;
						if (yO < 0.0f) yO += bSize;
						if (zO < 0.0f) zO += bSize;

						if (xO > bSize) xO -= bSize;
						if (yO > bSize) yO -= bSize;
						if (zO > bSize) zO -= bSize;

						axOut[tPrti*3+0] = xO;
						axOut[tPrti*3+1] = yO;
						axOut[tPrti*3+2] = zO;

						tPrti++;
					}
				} else {
					// no double precision version
				}

				remain = 0;
			}

		/* main function */
		while ((idx < slab*Lz) && (tPrti < slab))
		{/* number of particles to place from point idx */
			int nPrti = axTmp[idx];
			if (dataSize == 4) {
				float	*axOut = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->m2Cpu())+dataSize*(slab*(Lz*2+1))));
				for (hssize_t i=0; i<nPrti; i++)
				{
					float xO = (xC + 0.5f + gauss(rng) )*bSize/((float) totlX);
					float yO = (yC + 0.5f + gauss(rng) )*bSize/((float) totlX);
					float zO = (zC + 0.5f + gauss(rng) )*bSize/((float) totlZ);

					if (xO < 0.0f) xO += bSize;
					if (yO < 0.0f) yO += bSize;
					if (zO < 0.0f) zO += bSize;

					if (xO > bSize) xO -= bSize;
					if (yO > bSize) yO -= bSize;
					if (zO > bSize) zO -= bSize;

					axOut[tPrti*3+0] = xO;
					axOut[tPrti*3+1] = yO;
					axOut[tPrti*3+2] = zO;

					tPrti++;

					/* when buffer is filled continue */
					if (tPrti == slab) {
						remain = nPrti - i - 1;
						break;
					}
				} } else { /* double precision version does not make sense */ }

			/* move to the next point */
			idx++;
			lPos = idx;
			xC++;
			if (xC == totlX) { xC = 0; yC++; }
			if (yC == totlX) { yC = 0; zC++; }
		};

		if (cSize > 1) {
			missing = slab - tPrti;

			int    aMiss = 0;
			size_t cPos  = lPos;

			MPI_Allreduce(&missing, &aMiss, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);



			/* a rank misses particles > start redistributing loop */
			while (aMiss > 0) {
LogOut("Missing %08d\n", aMiss);


				/* rank rRecv receives and fills the slab */
				if (myRank == rRecv) {
					int	toRecv = 0;
					size_t	iPos   = 0, ePos = 0;
printf("Rank %d requesting %d particles\n", rRecv, missing); fflush(stdout);
					MPI_Send (&missing, 1, MPI_INT, rSend, 27, MPI_COMM_WORLD);
					MPI_Recv (&toRecv,  1, MPI_INT, rSend, 28, MPI_COMM_WORLD, nullptr);
printf("Rank %d will get %d particles\n", rRecv, toRecv); fflush(stdout);
					MPI_Recv (&iPos, 1, MPI_UNSIGNED_LONG_LONG, rSend, 29, MPI_COMM_WORLD, nullptr);
					MPI_Recv (&ePos, 1, MPI_UNSIGNED_LONG_LONG, rSend, 30, MPI_COMM_WORLD, nullptr);
					size_t aLength = ePos - iPos;
					size_t gAddr = slab*Lz;
					MPI_Recv (&(axTmp[gAddr]), aLength, MPI_INT, rSend, 26, MPI_COMM_WORLD, nullptr);
printf("Rank %d data received\n", rRecv); fflush(stdout);

					missing -= toRecv;
					excess  += toRecv;
					cPos    += toRecv;

					yC  = iPos/totlX;
					zC  = yC  /totlX;
					xC  = iPos - totlX*yC;
					yC -= zC*totlX;
					zC += Lz*rSend;

					for (int i=0; i<aLength; i++) {
if ((i%1024) == 0)
printf("Rank %d processing site %06d\n", rRecv, i); fflush(stdout);
						int nPhere = axTmp[gAddr+i];

						if (dataSize == 4) {
							float	*axOut = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->m2Cpu())+dataSize*(slab*(Lz*2+1))));

							for (int j=0; j<nPhere; j++) {
								float xO = (xC + 0.5f + gauss(rng) )*bSize/((float) totlX);
								float yO = (yC + 0.5f + gauss(rng) )*bSize/((float) totlX);
								float zO = (zC + 0.5f + gauss(rng) )*bSize/((float) totlZ);

								if (xO < 0.0f) xO += bSize;
								if (yO < 0.0f) yO += bSize;
								if (zO < 0.0f) zO += bSize;

								if (xO > bSize) xO -= bSize;
								if (yO > bSize) yO -= bSize;
								if (zO > bSize) zO -= bSize;

								axOut[tPrti*3+0] = xO;
								axOut[tPrti*3+1] = yO;
								axOut[tPrti*3+2] = zO;
								tPrti++;
							}
						} else { }

						xC++;

						if (xC == totlX) { xC = 0; yC++; }
						if (yC == totlX) { yC = 0; zC++; }
					}
				}


				/* the rank with the largest excess of particles sends some away*/
				if (myRank == rSend) {
					size_t iPos = lPos;
					int toSend = 0;
					int i = 0;
					MPI_Recv (&toSend, 1, MPI_INT, rRecv, 27, MPI_COMM_WORLD, nullptr);
printf("Rank %d received request for %d particles\n", rSend, toSend); fflush(stdout);

					if (toSend > excess)
						toSend = excess;

					size_t gAddr = slab*Lz;

					int pSent = 0;

					if (remain) {
						iPos = lPos - 1;
						axTmp[gAddr] = 0;
						while ((remain > 0) && (pSent < toSend)) {
							axTmp[gAddr]++;
							pSent++;
							remain--;
						}
						i++;
					}

					if (pSent == toSend)
						break;

printf("Rank %d is preparing the data to be sent\n", rSend); fflush(stdout);
					do {
if ((pSent%1024) == 0)
printf("Rank %d preparing particle %06d\n", rSend, pSent); fflush(stdout);
						int nPhere = axTmp[lPos];
						axTmp[gAddr+i] = 0;

						if (nPhere > 0) {
							while ((nPhere > 0) && (pSent < toSend)) {
								nPhere--;
								axTmp[gAddr+i]++;
								pSent++;
							}
							axTmp[lPos] = nPhere;
						}

						if (pSent < toSend) {
							i++;
							lPos++;
						} else
							break;

						if (i == slab) {
							toSend = pSent;
							break;
						}
					}	while(true);

					MPI_Send (&toSend, 1, MPI_INT, rRecv, 28, MPI_COMM_WORLD);
					size_t ePos = (i==slab) ? lPos : lPos+1;
					MPI_Send(&iPos, 1, MPI_UNSIGNED_LONG_LONG, rRecv, 29, MPI_COMM_WORLD);
					MPI_Send(&ePos, 1, MPI_UNSIGNED_LONG_LONG, rRecv, 30, MPI_COMM_WORLD);
					MPI_Send(&(axTmp[gAddr]), (ePos-iPos), MPI_INT, rRecv, 26, MPI_COMM_WORLD);

					excess -= toSend;
				}

printf("Rank %d syncing\n", myRank); fflush(stdout);
				commSync();

				MPI_Allgather (&excess, 1, MPI_LONG_LONG_INT, exStat, 1, MPI_LONG_LONG_INT, MPI_COMM_WORLD);
LogOut("Excess broadcasted\n");
				rSend = rRecv = 0;

				for (int i=1; i<cSize; i++) {
					if (exStat[i] > exStat[rSend])
						rSend = i;
				}

				for (int i=1; i<cSize; i++) {
					if (exStat[i] < exStat[rRecv])
						rRecv = i;
				}

				LogOut("Excess: ");
				for (int ii = 0; ii<cSize; ii++){
					LogOut("%ld ",exStat[ii]);
				}
				LogOut("\n");

				MPI_Allreduce(&missing, &aMiss, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			} // end aMiss !=0 No rank is missing a particle
		} // END MPI REDISTRIBUTION


// printf("\nSlice completed\n"); fflush(stdout);

		H5Sselect_hyperslab(vSpc1, H5S_SELECT_SET, vOffset, stride, vSlab, nullptr);
//		H5Sselect_hyperslab(vSpc2, H5S_SELECT_SET, vOffset, stride, vSlab, nullptr);

		/*	Write raw data	*/
		auto rErr = H5Dwrite (vSt1_id, dataType, memSpace, vSpc1, plist_id, static_cast<char *> (axion->m2Cpu())+(slab*(Lz*2 + 1)*dataSize));
//		auto vErr = H5Dwrite (vSt2_id, dataType, memSpace, vSpc2, plist_id, vArray);	// Always zero this one

		if ((rErr < 0))// || (vErr < 0))
		{
			LogError ("Error writing position/velocity dataset");
			prof.stop();
			exit(0);
		}

		commSync();
	}

	/*	Now we go with the particle ID	*/
	for (hsize_t zDim = 0; zDim < Nslabs; zDim++)
	{
		/*	Select the slab in the file	*/
		offset = (((hsize_t) (myRank*Nslabs)) + zDim)*slab;

		H5Sselect_hyperslab(sSpce, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

		hsize_t iBase = slab*(myRank*Nslabs + zDim);
		hsize_t *iArray = static_cast<hsize_t*>(vArray);
		#pragma omp parallel for shared(vArray) schedule(static)
		for (hsize_t idx = 0; idx < slab; idx++)
			iArray[idx] = iBase + idx;

		/*	Write raw data	*/
		auto rErr = H5Dwrite (sSts_id, H5T_NATIVE_HSIZE, semSpace, sSpce, plist_id, (static_cast<char *> (vArray)));

		if (rErr < 0)
		{
			LogError ("Error writing particle tag");
			prof.stop();
			exit(0);
		}

		commSync();
	}

	LogMsg (VERB_HIGH, "Write Gadget file successful");

	size_t bytes = 0;

	/*	Close the dataset	*/

	H5Dclose (vSt1_id);
	H5Dclose (vSt2_id);
	H5Dclose (sSts_id);
	H5Sclose (vSpc1);
	H5Sclose (sSpce);

	H5Sclose (memSpace);
	H5Sclose (aSpace);

	/*	Close the file		*/

	H5Sclose (scalarSpace);
	H5Sclose (totalSpace);
	H5Pclose (chunk_id);
	H5Pclose (shunk_id);
	H5Gclose (hGDt_id);
	H5Gclose (hGrp_id);

	H5Pclose (plist_id);
	H5Fclose (file_id);

	trackFree(exStat);
    prof.stop();

}


// OLD VERSION
// void	createGadget_Grid (Scalar *axion, size_t realN, size_t nParts, bool map_velocity)
// {
// 	hid_t	file_id, hGrp_id, hGDt_id, attr, plist_id, chunk_id, shunk_id, vhunk_id, mhunk_id;
// 	hid_t	vSt1_id, vSt2_id, sSts_id, mSts_id, aSpace, status;
// 	hid_t	vSpc1, vSpc2, sSpce, mSpce, memSpace, semSpace, mesSpace, dataType, totalSpace, scalarSpace, massSpace;
// 	hsize_t	total, slice, slab, offset, rOff;

// 	char	prec[16], fStr[16];
// 	int	length = 8;

// 	size_t	dataSize;

// 	int myRank = commRank();

// 	LogMsg (VERB_NORMAL, "Writing Gadget output file");
// 	LogMsg (VERB_NORMAL, "");
// 	LogOut("\n----------------------------------------------------------------------\n");
// 	LogOut("   GAD_GRID selected!        \n");
// 	LogOut("----------------------------------------------------------------------\n");
	
// 	/*      Start profiling         */
// 	Profiler &prof = getProfiler(PROF_HDF5);
// 	prof.start();
    
//         /*      WKB not supported atm   */
//         if (axion->Field() == FIELD_WKB) 
//         {
//             LogError ("Error: WKB field not supported");
//             prof.stop();
//             exit(1);
//         }

// 	/*      If needed, transfer data to host        */
// 	if (axion->Device() == DEV_GPU)
// 		axion->transferCpu(FIELD_M2);

// 	if (axion->m2Cpu() == nullptr) 
//         {
// 		LogError ("You seem to be using the lowmem option");
// 		prof.stop();
// 		return;
// 	}

// 	/*	Set up parallel access with Hdf5	*/
// 	plist_id = H5Pcreate (H5P_FILE_ACCESS);
// 	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

// 	char base[256];
// 	sprintf(base, "%s/%s.hdf5", outDir, gadName);

// 	/*	Create the file and release the plist	*/
// 	if ((file_id = H5Fcreate (base, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) < 0)
// 	{
// 		LogError ("Error creating file %s", base);
// 		return;
// 	}

// 	H5Pclose(plist_id);

// 	plist_id = H5Pcreate(H5P_DATASET_XFER);
// 	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

// 	commSync();

// 	switch (axion->Precision())
// 	{
// 		case FIELD_SINGLE:
// 		{
// 			dataType = H5T_NATIVE_FLOAT;
// 			dataSize = sizeof(float);
// 		}

// 		break;

// 		case FIELD_DOUBLE:
// 		{
// 			dataType = H5T_NATIVE_DOUBLE;
// 			dataSize = sizeof(double);
// 		}

// 		break;

// 		default:

// 		LogError ("Error: Invalid precision. How did you get this far?");
// 		exit(1);

// 		break;
// 	}

// 	/*	Create header	*/
// 	hGrp_id = H5Gcreate2(file_id, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
// 	// Units
// 	double  L1_in_pc = axion->BckGnd()->ICData().L1_pc; 
// 	double	bSize  = axion->BckGnd()->PhysSize() * L1_in_pc / 0.7;
// 	double  Omega0 = 0.3;
// 	double  met_to_pc = 1/(3.08567758e16);
// 	double  G_N = 6.67430e-11 * 1.98847e30 * met_to_pc * met_to_pc * met_to_pc; // pc^3/s^2/SolarMass
// 	double  H0 = 0.1 * met_to_pc; // 100 km/s/Mpc in 1/s
// 	double  vel_conv = 299792.458;  

// 	size_t  nPrt = nParts;
// 	if (nParts == 0)
// 	nPrt = axion->TotalSize();

// 	double  totalMass = Omega0 * (bSize*bSize*bSize) * (3.0 * H0*H0) / (8 * M_PI * G_N);
// 	double  avMass = totalMass/((double) nPrt);

// 	LogOut("\n[gadgrid] Number of particles (nPrt): %lu\n",nPrt);
// 	LogOut("[gadgrid] Box Length: L = %lf pc/h\n",bSize); 
// 	LogOut("[gadgrid] Total Mass: M = %e Solar Masses\n",totalMass);
// 	LogOut("[gadgrid] Average Particle Mass: m_av = %e Solar Masses\n",avMass);

// 	size_t  iDummy = 0;
// 	size_t  oDummy = 1;
// 	double  fDummy = 0.0;

// 	hid_t attr_type;
// 	attr_type = H5Tcopy(H5T_C_S1);
// 	H5Tset_size   (attr_type, length);
// 	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

// 	char gtype[16];
// 	sprintf(gtype, "gadgrid");

//     /* Simple scalar attributes */
// 	writeAttribute(hGrp_id, &bSize,  "BoxSize",                H5T_NATIVE_DOUBLE);
// 	writeAttribute(hGrp_id, &L1_in_pc, "L1_pc",                H5T_NATIVE_DOUBLE);
// 	writeAttribute(hGrp_id, &avMass, "AverageMass",            H5T_NATIVE_DOUBLE);
// 	writeAttribute(hGrp_id, gtype,   "GadType",                attr_type);
// 	writeAttribute(hGrp_id, &iDummy, "Flag_Entropy_ICs",       H5T_NATIVE_UINT);
// 	writeAttribute(hGrp_id, &iDummy, "Flag_Cooling",           H5T_NATIVE_HSIZE);
// 	writeAttribute(hGrp_id, &iDummy, "Flag_DoublePrecision",   H5T_NATIVE_HSIZE);  
// 	writeAttribute(hGrp_id, &iDummy, "Flag_Feedback",          H5T_NATIVE_HSIZE);
// 	writeAttribute(hGrp_id, &iDummy, "Flag_Metals",            H5T_NATIVE_HSIZE);
// 	writeAttribute(hGrp_id, &iDummy, "Flag_Sfr",               H5T_NATIVE_HSIZE);
// 	writeAttribute(hGrp_id, &iDummy, "Flag_StellarAge",        H5T_NATIVE_HSIZE);
// 	writeAttribute(hGrp_id, &iDummy, "HubbleParam",            H5T_NATIVE_HSIZE);
// 	writeAttribute(hGrp_id, &oDummy, "NumFilesPerSnapshot",    H5T_NATIVE_HSIZE);
// 	writeAttribute(hGrp_id, &fDummy, "Omega0",                 H5T_NATIVE_DOUBLE); 
// 	writeAttribute(hGrp_id, &iDummy, "OmegaLambda",            H5T_NATIVE_HSIZE);
// 	writeAttribute(hGrp_id, &fDummy, "Redshift",               H5T_NATIVE_DOUBLE);
// 	writeAttribute(hGrp_id, &fDummy, "Time",                   H5T_NATIVE_DOUBLE);

// 	/* Attribute arrays.
//      * These need to be created so gadget4 knows what particles to read.
//      * This works with the setup NTYPES=6 in the Config.sh file for the compilation.
//      * I guess this could be simplified to NTYPES=2 simulations by using hsize_t dims[1]={2}
//      * The mass table mTab[] needs all zero entries so we can use multiple masses. 
//      * */ 
//     hsize_t	dims[1]  = { 2 };
// 	double	dAFlt[6] = { 0.0, 0.0  };
// 	double	mTab[6]  = { 0.0, 0.0  };
// 	size_t	nPart[6] = {   0, nPrt };

// 	aSpace = H5Screate_simple (1, dims, nullptr);

// 	attr   = H5Acreate(hGrp_id, "MassTable",              H5T_NATIVE_DOUBLE, aSpace, H5P_DEFAULT, H5P_DEFAULT);
// 	status = H5Awrite (attr, H5T_NATIVE_DOUBLE, mTab);
// 	H5Aclose (attr);

// 	attr   = H5Acreate(hGrp_id, "NumPart_ThisFile",       H5T_NATIVE_HSIZE,  aSpace, H5P_DEFAULT, H5P_DEFAULT);
// 	status = H5Awrite (attr, H5T_NATIVE_HSIZE, nPart);
// 	H5Aclose (attr);

// 	attr   = H5Acreate(hGrp_id, "NumPart_Total",          H5T_NATIVE_HSIZE,  aSpace, H5P_DEFAULT, H5P_DEFAULT);
// 	status = H5Awrite (attr, H5T_NATIVE_HSIZE, nPart);
// 	H5Aclose (attr);

// 	attr   = H5Acreate(hGrp_id, "NumPart_Total_HighWord", H5T_NATIVE_DOUBLE, aSpace, H5P_DEFAULT, H5P_DEFAULT);
// 	status = H5Awrite (attr, H5T_NATIVE_DOUBLE, dAFlt);
// 	H5Aclose (attr);

//     /*     Group containing particle information   */
//     /*	   Create datagroup	                       */
// 	hGDt_id = H5Gcreate2(file_id, "/PartType1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

// 	uint totlZ = realN;
// 	uint totlX = realN;
// 	uint realDepth = realN/commSize();

// 	if (totlX == 0) 
//     {
// 		totlZ	  = axion->TotalDepth();
// 		totlX	  = axion->Length();
// 		realDepth = axion->Depth();
// 	}
    
// 	total = ((hsize_t) totlX)*((hsize_t) totlX)*((hsize_t) totlZ);
// 	slab  = ((hsize_t) totlX)*((hsize_t) totlX);
// 	rOff  = ((hsize_t) (totlX))*((hsize_t) (totlX))*(realDepth);
//     const hsize_t vSlab[2] = { slab, 3 };
// 	const hsize_t mSlab[2] = { slab, 1 };

// 	LogOut("\n[gadgrid] Decomposition: total %lu slab %lu rOff %lu\n",total, slab, rOff);

// 	// We need to insert here the energy 

// 	if (dataSize == 4)
// 	{
// 		float * re    = static_cast<float *>(axion->mStart());
// 		float * im    = static_cast<float *>(axion->vStart());
// 		float * newEn = static_cast<float *>(axion->m2Cpu());
// 		#pragma omp parallel for schedule(static)
// 		for (size_t idx = 0; idx < rOff; idx++)
// 		{
// 			newEn[idx] = re[idx]*re[idx]+im[idx]*im[idx];
// 		} 
// 	}
// 	else
// 	{
// 		double * re    = static_cast<double *>(axion->mStart());
// 		double * im    = static_cast<double *>(axion->vStart());
// 		double * newEn = static_cast<double *>(axion->m2Cpu());
// 		#pragma omp parallel for schedule(static)
// 		for (size_t idx = 0; idx < rOff; idx++)
// 		{
// 			newEn[idx] = re[idx]*re[idx]+im[idx]*im[idx];
// 		} 
// 	}
	
//     double eMean_local = 0.0;
//     double eMean_global;      
   
//     LogOut("\n[gadgrid] Recompute eMean using one reduction\n");

// 	double * axArray2 = static_cast<double *> (axion->m2half());

//     if (dataSize == 4) 
//     {
// 		float * axArray1 = static_cast<float *>(axion->m2Cpu());
// 		#pragma omp parallel for schedule(static) reduction(+:eMean_local)
// 		for (size_t idx =0; idx < rOff/2; idx++)
// 		{
// 			axArray2[idx] = (double) (axArray1[2*idx] + axArray1[2*idx+1]) ;
// 			eMean_local += axArray2[idx];
// 		}
// 	} 
//     else 
//     {
// 		double * axArray1 = static_cast<double *>(axion->m2Cpu());
// 		#pragma omp parallel for schedule(static) reduction(+:eMean_local)
// 		for (size_t idx =0; idx < rOff/2; idx++)
// 		{
// 			axArray2[idx] = axArray1[2*idx] + axArray1[2*idx+1] ;
// 			eMean_local += axArray2[idx];
// 		}
// 	}
	
//     LogOut("\n[gadgrid] eMean (local) = %lf \n",eMean_local/rOff);
// 	eMean_local /= rOff;
// 	MPI_Allreduce(&eMean_local, &eMean_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
// 	eMean_global /= commSize();
//    	LogOut("[gadgrid] eMean = %.20f \n",eMean_global);
    
//     /* New definition of eMean is related to the number of particles and not the grid points. 
//      * If the particle number is equal to the number of grid points neweMean = eMean_global
//     */
//     double neweMean = eMean_global*(totlX*totlX*totlZ)/nPrt;
//     LogOut("\n[gadgrid] eMean normalised to particle number: neweMean = %lf\n",neweMean);

// 	/* Normalise m2 to average energy density  */
// 	if (dataSize == 4) 
//     {
// 		float * axArray = static_cast<float *>(axion->m2Cpu());
// 		#pragma omp parallel for schedule(static)
// 		for (size_t idx = 0; idx<rOff; idx++)
// 			axArray[idx] /= neweMean;
// 	} 
//     else 
//     {
// 		double *axArray = static_cast<double*>(axion->m2Cpu());
//     	#pragma omp parallel for schedule(static)
// 		for (size_t idx = 0; idx<rOff; idx++)
// 			axArray[idx] /= neweMean;
// 	}
	
//     LogOut("\n[gadgrid] Normalised m2 to background density\n"); // m2 now holds the density contrast 
    	
//     double gloglo = round(eMean_global*rOff/neweMean);
//     size_t nPrt_local = (size_t) gloglo;
//     int pp_grid = nPrt/(totlX*totlX*totlZ);
//     LogOut("[gadgrid] Create %d particle(s) per grid site, so rank %d should take %lu particles ",pp_grid,myRank,nPrt_local);
//     fflush(stdout);
// 	commSync();

// 	/*	Create space for writing the raw data 
// 	 *   - Coordinates: chunk_id (vSlab)
// 	 *   - Velocities:  vhunk_id (vSlab)
// 	 *   - Masses:      mhunk_id (slab)
// 	 *   - IDs: 		shunk_id (slab)
// 	*/
	
// 	hsize_t nPrt_h = (hsize_t) nPrt;
// 	const hsize_t dDims[2]  = { nPrt_h , 3 };
// 	const hsize_t maDims[2] = { nPrt_h , 1 };

// 	if ((totalSpace = H5Screate_simple(2, dDims, nullptr)) < 0)	// Whole data
// 	{
// 		LogError ("Fatal error H5Screate_simple");
// 		prof.stop();
// 		exit (1);
// 	}

// 	if ((massSpace = H5Screate_simple(2, maDims, nullptr)) < 0)	// Whole data
// 	{
// 		LogError ("Fatal error H5Screate_simple");
// 		prof.stop();
// 		exit (1);
// 	}

// 	if ((scalarSpace = H5Screate_simple(1, &nPrt_h, nullptr)) < 0)	// Whole data
// 	{
// 		LogError ("Fatal error H5Screate_simple");
// 		prof.stop();
// 		exit (1);
// 	}

// 	/*	Set chunked access - Coordinates  */
// 	if ((chunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
// 	{
// 		LogError ("Fatal error H5Pcreate");
// 		prof.stop();
// 		exit (1);
// 	}

// 	if (H5Pset_chunk (chunk_id, 2, vSlab) < 0)
// 	{
// 		LogError ("Fatal error H5Pset_chunk");
// 		prof.stop();
// 		exit (1);
// 	}

// 	/*  Set chunked access - Velocities */
// 	if ((vhunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
// 	{
// 		LogError ("Fatal error H5Pcreate");
// 		prof.stop();
// 		exit (1);
// 	}

// 	if (H5Pset_chunk (vhunk_id, 2, vSlab) < 0)
// 	{
// 		LogError ("Fatal error H5Pset_chunk");
// 		prof.stop();
// 		exit (1);
// 	}

// 	/*  Set chunked access - Masses */
// 	if ((mhunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
// 	{
// 		LogError ("Fatal error H5Pcreate");
// 		prof.stop();
// 		exit (1);
// 	}

// 	if (H5Pset_chunk (mhunk_id, 2, mSlab) < 0)
// 	{
// 		LogError ("Fatal error H5Pset_chunk");
// 		prof.stop();
// 		exit (1);
// 	}
	
// 	/*  Set chunked access - IDs */
// 	if ((shunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
// 	{
// 		LogError ("Fatal error H5Pcreate");
// 		prof.stop();
// 		exit (1);
// 	}

// 	if (H5Pset_chunk (shunk_id, 1, &slab) < 0)
// 	{
// 		LogError ("Fatal error H5Pset_chunk");
// 		prof.stop();
// 		exit (1);
// 	}

// 	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
// 	if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0)
// 	{
// 		LogError ("Fatal error H5Pset_fill_time");
// 		prof.stop();
// 		exit (1);
// 	}
	
// 	if (H5Pset_fill_time (mhunk_id, H5D_FILL_TIME_NEVER) < 0)
// 	{
// 		LogError ("Fatal error H5Pset_fill_time");
// 		prof.stop();
// 		exit (1);
// 	}

// 	if (H5Pset_fill_time (shunk_id, H5D_FILL_TIME_NEVER) < 0)
// 	{
// 		LogError ("Fatal error H5Pset_fill_time");
// 		prof.stop();
// 		exit (1);
// 	}
	
//     if (H5Pset_fill_time (vhunk_id, H5D_FILL_TIME_NEVER) < 0)
// 	{
// 		LogError ("Fatal error H5Pset_fill_time");
// 		prof.stop();
// 		exit (1);
// 	}

// 	/*	Create a dataset for the vectors and another for the scalars	*/

// 	char vDt1[16] = "Coordinates";
// 	char vDt2[16] = "Velocities";
// 	char mDts[16] = "Masses"; 
// 	char sDts[16] = "ParticleIDs";
 
// 	vSt1_id = H5Dcreate (hGDt_id, vDt1, dataType,         totalSpace,  H5P_DEFAULT, chunk_id, H5P_DEFAULT); // Coordinates
// 	vSt2_id = H5Dcreate (hGDt_id, vDt2, dataType,         totalSpace,  H5P_DEFAULT, vhunk_id, H5P_DEFAULT); // Velocities
// 	mSts_id = H5Dcreate (hGDt_id, mDts, dataType,          massSpace,  H5P_DEFAULT, mhunk_id, H5P_DEFAULT); // Masses
// 	sSts_id = H5Dcreate (hGDt_id, sDts, H5T_NATIVE_HSIZE, scalarSpace, H5P_DEFAULT, shunk_id, H5P_DEFAULT); // ParticleIDs

// 	vSpc1 = H5Dget_space (vSt1_id);
// 	vSpc2 = H5Dget_space (vSt2_id);
// 	mSpce = H5Dget_space (mSts_id);
// 	sSpce = H5Dget_space (sSts_id);

// 	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/
// 	memSpace = H5Screate_simple(2, vSlab, nullptr);
// 	mesSpace = H5Screate_simple(2, mSlab, nullptr);
// 	semSpace = H5Screate_simple(1, &slab, nullptr);	
// 	commSync();

//     const hsize_t Lz = realDepth;
//     const hsize_t stride[2] = { 1, 1 };

// 	hsize_t Nslabs = nPrt_h/slab/commSize(); // This only works for one particle per grid
//     if (nPrt_h > Nslabs*slab*commSize())
//         LogOut("\nError: Nparticles is not a multiple of the slab size!");
	
//     /*  Fill particle coordinates and velocities  */
// 	LogOut("\n[gadgrid] Creating particle coordinates and velocities ... \n");
// 	size_t lPos = 0;
// 	for (hsize_t zDim = 0; zDim < Nslabs; zDim++)
// 	{
// 		offset = (((hsize_t) (myRank*Nslabs)) + zDim)*slab;
// 		hsize_t vOffset[2] = { offset , 0 };

// 		std::random_device rSd;
// 		std::mt19937_64 rng(rSd());
// 		std::uniform_real_distribution<float> uni(0.0, 1.0);

// 		size_t	idx = lPos;
// 		size_t	tPrti = 0;
// 		hssize_t	yC  = lPos/totlX;
// 		hssize_t	zC  = yC  /totlX;
// 		hssize_t	xC  = lPos - totlX*yC;
// 		yC -= zC*totlX;
// 		zC += myRank*Lz;

// 		/* main function */
// 		while ((idx < slab*Lz) && (tPrti < slab))
// 		{
// 			int nPrti = pp_grid;
// 			if (dataSize == 4) 
// 			{	
// 				float *axOut = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->m2Cpu())+dataSize*(slab*(Lz*2+1))));
// 				float  *vOut = static_cast<float*>(static_cast<void*>(static_cast<char *> (axion->vBackGhost())+dataSize*(slab*(Lz*2+1))));
//                 for (hssize_t i=0; i<nPrti; i++)
// 				{	
// 					float xO,yO,zO;
					
// 					xO = xC*bSize/((float) totlX); 
// 					yO = yC*bSize/((float) totlX);
// 					zO = zC*bSize/((float) totlZ);

// 					if (xO < 0.0f) xO += bSize;
// 					if (yO < 0.0f) yO += bSize;
// 					if (zO < 0.0f) zO += bSize;
//                     if (xO > bSize) xO -= bSize;
// 					if (yO > bSize) yO -= bSize;
// 					if (zO > bSize) zO -= bSize;

// 					axOut[tPrti*3+0] = xO;
// 					axOut[tPrti*3+1] = yO;
// 					axOut[tPrti*3+2] = zO;

// 					if (map_velocity)
// 					{
// 						float grad[3];
// 						grad_idx(axion,grad,idx);
// 						vOut[tPrti*3+0] = grad[0]/(*axion->zV() * axion->AxionMass()) * vel_conv;
// 						vOut[tPrti*3+1] = grad[1]/(*axion->zV() * axion->AxionMass()) * vel_conv;
// 						vOut[tPrti*3+2] = grad[2]/(*axion->zV() * axion->AxionMass()) * vel_conv;
// 						// if (sm_vel) // to finish
// 						// {
// 						// 	smooth_vel();
// 						// 	vOut[tPrti*3+0] = smgrad[0]/(*axion->zV() * axion->AxionMass());
// 						// 	vOut[tPrti*3+1] = smgrad[1]/(*axion->zV() * axion->AxionMass());
// 						// 	vOut[tPrti*3+2] = smgrad[2]/(*axion->zV() * axion->AxionMass());
// 						// }
// 					}
// 					else
// 					{
// 						vOut[tPrti*3+0] = 0.f;
// 						vOut[tPrti*3+1] = 0.f;
// 						vOut[tPrti*3+2] = 0.f;
// 					}
// 					tPrti++; 
// 				} 
// 			} 
// 			else 
// 			{ 
//                 LogError ("Double precision not supported yet! Set --prec single");
// 		        prof.stop();
// 		        exit (1);
// 			}
// 			idx++;
// 			lPos = idx;
// 			xC++;
// 			if (xC == totlX) { xC = 0; yC++; }
// 			if (yC == totlX) { yC = 0; zC++; }
// 		};

// 		H5Sselect_hyperslab(vSpc1, H5S_SELECT_SET, vOffset, stride, vSlab, nullptr);
// 		auto rErr = H5Dwrite (vSt1_id, dataType, memSpace, vSpc1, plist_id, static_cast<char *> (axion->m2Cpu())+(slab*(Lz*2 + 1)*dataSize));
		
// 		if ((rErr < 0))
// 		{
// 			LogError ("Error writing position dataset");
// 			prof.stop();
// 			exit(0);
// 		}
		
// 		H5Sselect_hyperslab(vSpc2, H5S_SELECT_SET, vOffset, stride, vSlab, nullptr);
// 		auto vErr = H5Dwrite (vSt2_id, dataType, memSpace, vSpc2, plist_id, static_cast<char *> (axion->vBackGhost())+(slab*(Lz*2+1)*dataSize));
// 		if ((vErr < 0))
// 		{
// 			LogError ("Error writing velocity dataset");
// 			prof.stop();
// 			exit(0);
// 		}
		
// 	}

// 	commSync();
// 	LogOut("\n[gadgrid] Filled coordinates and velocities!"); 

//     /*  Pointers used to fill data  */
// 	void *mArray = static_cast<void*>(static_cast<char*>(axion->m2Cpu())+(slab*Lz)*dataSize); // This will be filled with masses
//     void *vArray = static_cast<void*>(static_cast<char*>(axion->m2Cpu())+(slab*Lz)*dataSize); // This will be filled with IDs

//     /*  Fill particle masses        */
//     if (dataSize == 4) 
//     {   
//         float * mData = static_cast<float *>(axion->m2Cpu());
// 		#pragma omp parallel for schedule(static)
//         for (size_t idx = 0; idx<rOff; idx++)
// 		{
// 			mData[idx] *= avMass;
// 		}
//     }
// 	else
// 	{
// 		LogError ("Double precision not supported yet! Set --prec single");
//         prof.stop();
//         exit(1);
// 	}

//     for (hsize_t zDim = 0; zDim < Nslabs; zDim++)
//     {	
// 		float *maOut = static_cast<float*>(axion->m2Cpu());
//         offset = (((hsize_t) (myRank*Nslabs)) + zDim)*slab;
// 		hsize_t vOffset[2] = { offset , 0 };
// 		H5Sselect_hyperslab(mSpce, H5S_SELECT_SET, vOffset, NULL, mSlab, NULL);
// 		hsize_t *imArray = static_cast<hsize_t*>(mArray);
//         #pragma omp parallel for shared(mArray) schedule(static)
//         for (hsize_t idx = 0; idx < slab; idx++)		
//             imArray[idx] = maOut[idx];
        
// 		auto rErr = H5Dwrite (mSts_id, dataType, mesSpace, mSpce, plist_id, (static_cast<char *> (axion->m2Cpu())+(slab*zDim)*dataSize));

//         if (rErr < 0)
//         {
//             LogError ("Error writing particle masses");
//             prof.stop();
//             exit(0);
//         }
                    
//         commSync();
//     }
    
//     LogOut("\n[gadgrid] Filled particle Masses!"); 
    
//     /*  Fill particle ID  */
//     for (hsize_t zDim = 0; zDim < Nslabs; zDim++)
//     {
//         offset = (((hsize_t) (myRank*Nslabs)) + zDim)*slab;
//         H5Sselect_hyperslab(sSpce, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
//         hsize_t *iArray = static_cast<hsize_t*>(vArray);
//         #pragma omp parallel for shared(vArray) schedule(static)
//         for (hsize_t idx = 0; idx < slab; idx++)
//             iArray[idx] = offset + idx;
                
//         auto rErr = H5Dwrite (sSts_id, H5T_NATIVE_HSIZE, semSpace, sSpce, plist_id, (static_cast<char *> (vArray)));
                
//         if (rErr < 0)
//         {
//             LogError ("Error writing particle tag");
//             prof.stop();
//             exit(0);
//         }
                
//         commSync();
//     }

//     LogOut("\n[gadgrid] Filled particle IDs!");

// 	/*	Close the datasets,groups,files*/
// 	H5Dclose (vSt1_id);
// 	H5Dclose (vSt2_id);
// 	H5Dclose (sSts_id);
// 	H5Dclose (mSts_id);
// 	H5Sclose (vSpc1);
// 	H5Sclose (sSpce);
// 	H5Sclose (memSpace);
// 	H5Sclose (aSpace);
// 	H5Sclose (scalarSpace);
// 	H5Sclose (totalSpace);
// 	H5Sclose (massSpace);
// 	H5Pclose (chunk_id);
// 	H5Pclose (shunk_id);
// 	H5Pclose (mhunk_id);
// 	H5Pclose (vhunk_id);

// 	H5Gclose (hGDt_id);
// 	H5Gclose (hGrp_id);
// 	H5Pclose (plist_id);
// 	H5Fclose (file_id);
//     //trackFree(exStat);
//     prof.stop();

	
// }