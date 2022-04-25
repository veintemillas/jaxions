#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <hdf5.h>

#include <random>
#include <fftw3-mpi.h>

#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "utils/parse.h"
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

void grad_idx(Scalar *axion, float * grad3, size_t idx, hssize_t xC, hssize_t yC, hssize_t zC)
{  
    size_t totlX	  = axion->Length();
	size_t Lz = axion->Depth();
    hsize_t slab  = ((hsize_t) totlX)*((hsize_t) totlX);
    size_t idxPx, idxMx, idxPy, idxMy, idxPz, idxMz;
    float gr_x,gr_y,gr_z;				
	
    // x-coord
    if (xC == 0)
        idxMx = idx + totlX - 1;
    else 
        idxMx = idx - 1;
    if (xC == totlX - 1)
        idxPx = idx - xC + 1;
    else
        idxPx = idx + 1;
    // y-coord
    if (yC == 0)
    {
        idxMy = idx + slab;
        idxPy = idx + totlX; 
    }
    else
    {
        idxMy = idx - totlX;
    }
    if (yC == totlX - 1)
        idxPy = idx - slab + totlX;
    else
        idxPy = idx + totlX;
    // z-coord
    if (zC == 0)
    {
        idxPz = idx + slab;
        idxMz = idx + slab*Lz - 1;
    }
    if (zC == slab*Lz -1) // not sure here 
    {
        idxPz = idx - slab*Lz + 1;
        idxMz = idx - slab;
    }
    
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

void	createGadget_Grid (Scalar *axion, size_t realN, size_t nParts, double L1_pc, bool map_velocity)
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
	sprintf(base, "%s/ics.hdf5", outDir, outName);

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
    double  L1_in_pc = L1_pc; 
    double	bSize  = axion->BckGnd()->PhysSize() * L1_in_pc / 0.7;
    double  Omega0 = 0.3;
    double  met_to_pc = 1/(3.08567758e16);
    double  G_N = 6.67430e-11 * 1.98847e30 * met_to_pc * met_to_pc * met_to_pc; // pc^3/s^2/SolarMass
    double  H0 = 0.1 * met_to_pc; // 100 km/s/Mpc in 1/s 
    
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

    /* Simple scalar attributes */
	writeAttribute(hGrp_id, &bSize,  "BoxSize",                H5T_NATIVE_DOUBLE);
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
	const hsize_t mSlab[2] = { slab, 1 };

	LogOut("\n[gadmass] Decomposition: total %lu slab %lu rOff %lu\n",total, slab, rOff);

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
	
    double eMean_local = 0.0;
    double eMean_global;      
   
    LogOut("\n[gadmass] Recompute eMean using one reduction\n");

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
	
    LogOut("\n[gadmass] eMean (local) = %lf \n",eMean_local/rOff);
	eMean_local /= rOff;
	MPI_Allreduce(&eMean_local, &eMean_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	eMean_global /= commSize();
   	LogOut("[gadmass] eMean = %.20f \n",eMean_global);
    
    /* New definition of eMean is related to the number of particles and not the grid points. 
     * If the particle number is equal to the number of grid points neweMean = eMean_global
    */
    double neweMean = eMean_global*(totlX*totlX*totlZ)/nPrt;
    LogOut("\n[gadmass] eMean normalised to particle number: neweMean = %lf\n",neweMean);

	/* Normalise m2 to average energy density  */
	if (dataSize == 4) 
    {
		float * axArray = static_cast<float *>(axion->m2Cpu());
		#pragma omp parallel for schedule(static)
		for (size_t idx = 0; idx<rOff; idx++)
			axArray[idx] /= neweMean;
	} 
    else 
    {
		double *axArray = static_cast<double*>(axion->m2Cpu());
    	#pragma omp parallel for schedule(static)
		for (size_t idx = 0; idx<rOff; idx++)
			axArray[idx] /= neweMean;
	}
	
    LogOut("\n[gadmass] Normalised m2 to background density\n"); // m2 now holds the density contrast 
    	
    double gloglo = round(eMean_global*rOff/neweMean);
    size_t nPrt_local = (size_t) gloglo;
    int pp_grid = nPrt/(totlX*totlX*totlZ);
    LogOut("[gadmass] We create %d particle(s) per grid site, so rank %d should take %lu particles ",pp_grid,myRank,nPrt_local);
    fflush(stdout);
	commSync();

	/*	Create space for writing the raw data 
	 *   - Coordinates: chunk_id (vSlab)
	 *   - Velocities:  vhunk_id (vSlab)
	 *   - Masses:      mhunk_id (slab)
	 *   - IDs: 		shunk_id (slab)
	*/
	
	hsize_t nPrt_h = (hsize_t) nPrt;
	const hsize_t dDims[2]  = { nPrt_h , 3 };
	const hsize_t maDims[2] = { nPrt_h , 1 };

	if ((totalSpace = H5Screate_simple(2, dDims, nullptr)) < 0)	// Whole data
	{
		LogError ("Fatal error H5Screate_simple");
		prof.stop();
		exit (1);
	}

	if ((massSpace = H5Screate_simple(2, maDims, nullptr)) < 0)	// Whole data
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

	/*  Set chunked access - Masses */
	if ((mhunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (mhunk_id, 2, mSlab) < 0)
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
	
	if (H5Pset_fill_time (mhunk_id, H5D_FILL_TIME_NEVER) < 0)
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
	mSts_id = H5Dcreate (hGDt_id, mDts, dataType,          massSpace,  H5P_DEFAULT, mhunk_id, H5P_DEFAULT); // Masses
	sSts_id = H5Dcreate (hGDt_id, sDts, H5T_NATIVE_HSIZE, scalarSpace, H5P_DEFAULT, shunk_id, H5P_DEFAULT); // ParticleIDs

	vSpc1 = H5Dget_space (vSt1_id);
	vSpc2 = H5Dget_space (vSt2_id);
	mSpce = H5Dget_space (mSts_id);
	sSpce = H5Dget_space (sSts_id);

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/
	memSpace = H5Screate_simple(2, vSlab, nullptr);
	mesSpace = H5Screate_simple(2, mSlab, nullptr);
	semSpace = H5Screate_simple(1, &slab, nullptr);	
	commSync();

    const hsize_t Lz = realDepth;
    const hsize_t stride[2] = { 1, 1 };

	hsize_t Nslabs = nPrt_h/slab/commSize(); // This only works for one particle per grid
    if (nPrt_h > Nslabs*slab*commSize())
        LogOut("\nError: Nparticles is not a multiple of the slab size!");
	
    /*  Fill particle coordinates and velocities  */
	LogOut("\n[gadmass] Creating particle coordinates and velocities ... \n");
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
					float xO,yO,zO,x_disp,y_disp,z_disp;
					
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
						grad_idx(axion,grad,idx,xC,yC,zC);

						vOut[tPrti*3+0] = grad[0]/(*axion->zV() * axion->AxionMass());
						vOut[tPrti*3+1] = grad[1]/(*axion->zV() * axion->AxionMass());
						vOut[tPrti*3+2] = grad[2]/(*axion->zV() * axion->AxionMass());
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
	LogOut("\n[gadmass] Filled coordinates and velocities!"); 

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

	//LogOut("\n[gadmass] Normalised m2 by average particle mass"); 

    for (hsize_t zDim = 0; zDim < Nslabs; zDim++)
    {	
		float *maOut = static_cast<float*>(axion->m2Cpu());
        offset = (((hsize_t) (myRank*Nslabs)) + zDim)*slab;
		hsize_t vOffset[2] = { offset , 0 };
		H5Sselect_hyperslab(mSpce, H5S_SELECT_SET, vOffset, NULL, mSlab, NULL);
		hsize_t *imArray = static_cast<hsize_t*>(mArray);
        #pragma omp parallel for shared(mArray) schedule(static)
        for (hsize_t idx = 0; idx < slab; idx++)		
            imArray[idx] = maOut[idx];
        
		auto rErr = H5Dwrite (mSts_id, dataType, mesSpace, mSpce, plist_id, (static_cast<char *> (axion->m2Cpu())+(slab*zDim)*dataSize));

        if (rErr < 0)
        {
            LogError ("Error writing particle masses");
            prof.stop();
            exit(0);
        }
                    
        commSync();
    }
    
    LogOut("\n[gadmass] Filled particle Masses!"); 
    
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

    LogOut("\n[gadmass] Filled particle IDs!");

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
	H5Sclose (massSpace);
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