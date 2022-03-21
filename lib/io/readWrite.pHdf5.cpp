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

#include "utils/memAlloc.h"
#include "utils/profiler.h"
#include "utils/logger.h"

#include "fft/fftCode.h"
#include "scalar/fourier.h"

/* In case one reads from Moore format */
#include "utils/simpleops.h"
#include "scalar/mendTheta.h"

/* In case one reads larger grids */
#include "reducer/reducer.h"

#ifdef USE_NYX_OUTPUT
	#include "io/output_nyx.h"
#endif

#define caspr(lab,var,str)   \
	case lab:                  \
	sprintf(var, str);         \
	break;


//#include "scalar/varNQCD.h"

hid_t	meas_id = -1, mlist_id;
hsize_t	tSize, slabSz, sLz;
bool	opened = false, header = false;

bool	mDisabled = true;
H5E_auto2_t eFunc;
void	   *cData;

using namespace std;
using namespace profiler;

/*	TODO	Añade excepciones para salir limpiamente del programa	*/

herr_t	writeAttribute(hid_t file_id, void *data, const char *name, hid_t h5_type)
{
	hid_t	attr, attr_id;
	herr_t	status;

	attr_id = H5Screate(H5S_SCALAR);
	if ((attr    = H5Acreate2 (file_id, name, h5_type, attr_id, H5P_DEFAULT, H5P_DEFAULT)) < 0)
		LogError ("Error creating attribute %s", name);
	if ((status  = H5Awrite (attr, h5_type, data)) < 0)
		LogError ("Error writing attribute %s to file");
	H5Sclose (attr_id);
	H5Aclose (attr);

		if        (h5_type == H5T_NATIVE_HSIZE){
				LogMsg (VERB_HIGH, "Write attribute %s = %d", name, *(static_cast<size_t*>(data)));
		}	else if (h5_type == H5T_NATIVE_DOUBLE) {
				LogMsg (VERB_HIGH, "Write attribute %s = %e", name, *(static_cast<double*>(data)));
		} else if (h5_type == H5T_NATIVE_INT) {
				LogMsg (VERB_HIGH, "Write attribute %s = %d", name, *(static_cast<int*>(data)));
		} else if (h5_type == H5T_NATIVE_UINT) {
				LogMsg (VERB_HIGH, "Write attribute %s = %u", name, *(static_cast<unsigned int*>(data)));
 		} else {
				// LogMsg (VERB_HIGH, "Write attribute %s ", name);
				LogMsg (VERB_HIGH, "Write attribute %s = %s", name, (char*) data);
		}

	return	status;
}

void    writeAttribute	(double *data, const char *name )
{
	writeAttribute(meas_id, data, name, H5T_NATIVE_DOUBLE);
}

void    writeAttribute	(void *data, const char *name, hid_t h5_Type)
{
	if (h5_Type == H5T_NATIVE_HSIZE)
	{
			writeAttribute(meas_id, (size_t*) data, name, H5T_NATIVE_HSIZE);
	}	else if (h5_Type == H5T_NATIVE_DOUBLE) {
			writeAttribute(meas_id, (double*) data, name, H5T_NATIVE_DOUBLE);
	} else if (h5_Type == H5T_NATIVE_INT) {
			writeAttribute(meas_id, (int*) data, name, H5T_NATIVE_INT);
	}	else if (h5_Type == H5T_NATIVE_UINT) {
			writeAttribute(meas_id, (unsigned int*) data, name, H5T_NATIVE_UINT);
	}	else if (h5_Type == H5T_C_S1) {
				char *arr_ptr = (char *) data;
				int	length = std::max( (int) strlen(arr_ptr), 32);
				hid_t attr_type;
				attr_type = H5Tcopy(H5T_C_S1);
				H5Tset_size   (attr_type, length);
				H5Tset_strpad (attr_type, H5T_STR_NULLTERM);
			writeAttribute(meas_id, arr_ptr, name, attr_type);
			H5Tclose (attr_type);
	} else {
		LogError("Cannot write attribute %s . Type not recognised.",name);
	}

}

void    writeAttributeg	(void *data, const char *group, const char *name, hid_t h5_Type)
{
	hid_t	group_id, base_id;

	if ((group_id = H5Gopen2(meas_id, group, H5P_DEFAULT)) < 0)
	{
		LogError ("Error: couldn't open group %s in measurement file.\n", group);
		return;
	}

	if (h5_Type == H5T_NATIVE_HSIZE)
	{
			writeAttribute(group_id, (size_t*) data, name, H5T_NATIVE_HSIZE);
	}	else if (h5_Type == H5T_NATIVE_DOUBLE) {
			writeAttribute(group_id, (double*) data, name, H5T_NATIVE_DOUBLE);
	} else if (h5_Type == H5T_NATIVE_INT) {
			writeAttribute(group_id, (int*) data, name, H5T_NATIVE_INT);
	}	else if (h5_Type == H5T_NATIVE_UINT) {
			writeAttribute(group_id, (unsigned int*) data, name, H5T_NATIVE_UINT);
	}	else if (h5_Type == H5T_C_S1) {
				char *arr_ptr = (char *) data;
				int	length = std::max( (int) strlen(arr_ptr), 32);
				hid_t attr_type;
				attr_type = H5Tcopy(H5T_C_S1);
				H5Tset_size   (attr_type, length);
				H5Tset_strpad (attr_type, H5T_STR_NULLTERM);
			writeAttribute(group_id, arr_ptr, name, attr_type);
			H5Tclose (attr_type);
	} else {
		LogError("Cannot write attribute %s . Type not recognised.",name);
	}
	H5Gclose (group_id);

}

herr_t	readAttribute(hid_t file_id, void *data, const char *name, hid_t h5_type)
{
	hid_t	attr;
	herr_t	status;

	if ((attr   = H5Aopen_by_name (file_id, ".", name, H5P_DEFAULT, H5P_DEFAULT)) < 0){
		LogError ("Error opening attribute %s", name);
		return attr;
	}
	else
	{
		if ((status = H5Aread (attr, h5_type, data)) < 0)
			LogError ("Error reading attribute %s", name);
		status = H5Aclose(attr);

		if        (h5_type == H5T_NATIVE_HSIZE){
				LogMsg (VERB_HIGH, "h5read attribute %s = %d", name, *(static_cast<size_t*>(data)));
		}	else if (h5_type == H5T_NATIVE_DOUBLE) {
				LogMsg (VERB_HIGH, "h5read attribute %s = %e", name, *(static_cast<double*>(data)));
		} else if (h5_type == H5T_NATIVE_INT) {
				LogMsg (VERB_HIGH, "h5read attribute %s = %d", name, *(static_cast<int*>(data)));
		} else if (h5_type == H5T_NATIVE_UINT) {
				LogMsg (VERB_HIGH, "h5read attribute %s = %u", name, *(static_cast<unsigned int*>(data)));
		} else {
				LogMsg (VERB_HIGH, "h5read attribute %s = %s", name, (char*) data);
		}

		return	status;
	}
}


void	disableErrorStack	()
{
	if (H5Eget_auto2(H5E_DEFAULT, &eFunc, &cData) < 0) {	// Save current error stack
		mDisabled = false;
		LogMsg (VERB_NORMAL, "Warning: couldn't retrieve current hdf5 error stack");
		LogMsg (VERB_NORMAL, "Warning: group existence check might lead to spurious errors");

		return;
	} else {
		if (H5Eset_auto2(H5E_DEFAULT, NULL, NULL) < 0) {	// Turn off error output, we don't want trash if the group doesn't exist
			mDisabled = false;
			LogMsg (VERB_NORMAL, "Warning: couldn't disable current hdf5 error stack");
			LogMsg (VERB_NORMAL, "Warning: group existence check might lead to spurious errors");

			return;
		}
	}

	mDisabled = true;
}

void	enableErrorStack	()
{
	if (mDisabled) {
		if (H5Eset_auto2(H5E_DEFAULT, eFunc, cData) < 0) {	// Save current error stack
			LogMsg (VERB_NORMAL, "Warning: couldn't enable hdf5 error stack");
			LogMsg (VERB_NORMAL, "Warning: hdf5 errors will be silent");

			return;
		}
	}

	mDisabled = false;
}




void	writeConf (Scalar *axion, int index, const bool restart)
{
	hid_t	file_id, mset_id, vset_id, plist_id, chunk_id;
	hid_t	mSpace, vSpace, memSpace, dataType, totalSpace;
	hsize_t	total, slice, slab, offset;

	char	prec[16], fStr[16], lStr[16], rStr[16], dStr[16], vStr[32], vPQStr[32], icStr[16], smStr[16];
	int	length = 32;

	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize;

	int myRank = commRank();

	LogMsg (VERB_NORMAL, "Writing Hdf5 configuration to disk");
	LogMsg (VERB_NORMAL, "");

	/*	Unfold field before writing configuration	*/

	bool	wasFolded = axion->Folded();

	Folder	*munge;

	if (wasFolded)
	{
		LogMsg (VERB_HIGH, "Folded configuration, will unfold and fold back at the end");
		munge	= new Folder(axion);
		(*munge)(UNFOLD_ALL);
	}

	/* Inverse FFT field if in momentum space*/
	if	( axion->MMomSpace() || axion->VMomSpace() )
	{
		FTfield pelota(axion);
		pelota(FIELD_MV, FFT_BCK); // BCK to send to position space
	}
	/*	Start profiling		*/

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	/*      If needed, transfer data to host        */
	if (axion->Device() == DEV_GPU)
		axion->transferCpu(FIELD_MV);

	/*	Set up parallel access with Hdf5	*/
	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];
	/* JAVI if restart do not write number for simplicity */
	if (!restart)
	sprintf(base, "%s/%s.%05d", outDir, outName, index);
	else
		sprintf(base, "%s/%s.restart", outDir, outName);

	/*	Create the file and release the plist	*/
	if ((file_id = H5Fcreate (base, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) < 0)
	{
		LogError ("Error creating file %s", base);
		return;
	}

	H5Pclose(plist_id);

	commSync();

	switch (axion->Precision())
	{
		case FIELD_SINGLE:
		{
			dataType = H5T_NATIVE_FLOAT;
			dataSize = sizeof(float);

			sprintf(prec, "Single");
//			length = strlen(prec)+1;
		}

		break;

		case FIELD_DOUBLE:
		{
			dataType = H5T_NATIVE_DOUBLE;
			dataSize = sizeof(double);

			sprintf(prec, "Double");
//			length = strlen(prec)+1;
		}

		break;

		default:

		LogError ("Error: Invalid precision. How did you get this far?");
		exit(1);

		break;
	}

	int cSteps = dump*index;
	hsize_t totlZ = axion->TotalDepth();
	hsize_t tmpS  = axion->Length();

	switch (axion->Field())
	{
		case 	FIELD_SX_RD:
		case 	FIELD_SAXION:
		{
			total = tmpS*tmpS*totlZ*2;
			slab  = axion->Surf()*2;

			sprintf(fStr, "Saxion");
		}
		break;

		case 	FIELD_AX_MOD_RD:
		case	FIELD_AXION_MOD:
		{
			total = tmpS*tmpS*totlZ;
			slab  = axion->Surf();

			sprintf(fStr, "Axion Mod");
		}
		break;

		case 	FIELD_AX_RD:
		case	FIELD_AXION:
		case	FIELD_WKB:
		{
			total = tmpS*tmpS*totlZ;
			slab  = axion->Surf();

			sprintf(fStr, "Axion");
		}
		break;

		case 	FIELD_NAXION:
		{
			total = tmpS*tmpS*totlZ*2;
			slab  = axion->Surf()*2;
			sprintf(fStr, "Naxion");
		}
		break;

		case 	FIELD_PAXION:
		{
			total = tmpS*tmpS*totlZ;
			slab  = axion->Surf();
			sprintf(fStr, "Paxion");
		}
		break;

		default:
		LogError ("Error: Invalid field type. How did you get this far?");
		exit(1);
		break;
	}

	double	llPhys = axion->BckGnd()->Lambda();

	switch (axion->LambdaT())
	{
		case 	LAMBDA_Z2:
			sprintf(lStr, "z2");
			llPhys /= (*axion->zV())*(*axion->zV());
			break;

		case	LAMBDA_FIXED:
			sprintf(lStr, "Fixed");
			break;

		default:

			LogError ("Error: Invalid lambda type. How did you get this far?");
			exit(1);
			break;
	}

	auto lSize    = axion->BckGnd()->PhysSize();
	auto vqcdType = axion->BckGnd()->QcdPot  ();
	auto nQcd     = axion->BckGnd()->QcdExp  ();
	auto nQcdr    = axion->BckGnd()->QcdExpr ();
	auto gamma    = axion->BckGnd()->Gamma   ();
	auto LL       = axion->BckGnd()->Lambda  ();

	switch (vqcdType & V_QCD)	{
		default:
		caspr(V_QCDC,vStr,"VQcd Cos")
		caspr(V_QCD1,vStr,"VQcd 1")
		caspr(V_QCD0,vStr,"VQcd 0")
		caspr(V_QCDV,vStr,"VQcd Variant")
		caspr(V_QCDL,vStr,"VQcd Linear")
		caspr(V_QCD2,vStr,"VQcd N = 2")
	}

	switch (vqcdType & V_PQ) {
		default:
		caspr(V_PQ1,vPQStr,"VPQ 1")
		caspr(V_PQ2,vPQStr,"VPQ 2")
	}

	switch (vqcdType & V_DAMP) {
		caspr(V_DAMP_RHO,dStr,"Rho")
		caspr(V_DAMP_ALL,dStr,"All")
		default:
		caspr(V_NONE,dStr,"None")
	}

	switch (vqcdType & V_EVOL_RHO)	{
		caspr(V_EVOL_RHO,rStr,"Only Rho")
		default:
		caspr(V_NONE,rStr,"Full")
	}

	/*	Write header	*/
	hid_t attr_type;

	/*	Attributes	*/

	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size   (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	double maa = axion->AxionMass(); //axionmass((*axion->zV()), nQcd, zthres, zrestore);
	double msa = axion->Msa();
	double few = axion->BckGnd()->Frw();

	writeAttribute(file_id, fStr,   "Field type",    attr_type);
	writeAttribute(file_id, prec,   "Precision",     attr_type);
	writeAttribute(file_id, &tmpS,  "Size",          H5T_NATIVE_UINT);
	writeAttribute(file_id, &totlZ, "Depth",         H5T_NATIVE_UINT);
	writeAttribute(file_id, &msa,   "Saxion mass",   H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &maa,   "Axion mass",    H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &lSize, "Physical size", H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, axion->zV(),  "z",       H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, axion->RV(),  "R",       H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &few,  "Frw",            H5T_NATIVE_DOUBLE);
	if (axion->BckGnd()->UeC()){
		double Temp = axion->BckGnd()->T(*axion->zV());
		writeAttribute(file_id, &Temp,  "Temperature", H5T_NATIVE_DOUBLE);
	}
	writeAttribute(file_id, &zInit, "zInitial",      H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &zFinl, "zFinal",        H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &nSteps,"nSteps",        H5T_NATIVE_INT);
	writeAttribute(file_id, &cSteps,"Current step",  H5T_NATIVE_INT);

	/* JAVI index...*/
	if (restart)
		writeAttribute(file_id, &index, "index", H5T_NATIVE_INT);

	/*	Create a group for specific header data	*/
	hid_t vGrp_id = H5Gcreate2(file_id, "/potential", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	double shift = axion->Saskia();//axionshift(maa, llPhys, vqcdType);
	//indi3 =  maa/pow(*axion->zV(), nQcd*0.5);
	double indi3    = axion->BckGnd()->Indi3();
	double zthres   = axion->BckGnd()->ZThRes();
	double zrestore = axion->BckGnd()->ZRestore();

	writeAttribute(vGrp_id, &lStr,    "Lambda type",   attr_type);
	writeAttribute(vGrp_id, &LL,      "Lambda",        H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &vStr,    "VQcd type",     attr_type);
	writeAttribute(vGrp_id, &vPQStr,  "VPQ type",      attr_type);
	writeAttribute(vGrp_id, &dStr,    "Damping type",  attr_type);
	writeAttribute(vGrp_id, &rStr,    "Evolution type",attr_type);
	writeAttribute(vGrp_id, &nQcd,    "nQcd",          H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &gamma,   "Gamma",         H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &shift,   "Shift",         H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &indi3,   "Indi3",         H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &zthres,  "z Threshold",   H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &zrestore,"z Restore",     H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &nQcdr,   "nQcd2",         H5T_NATIVE_DOUBLE);

	H5Gclose(vGrp_id);

	hid_t icGrp_id = H5Gcreate2(file_id, "/ic", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	switch (cType) {
		case	CONF_SMOOTH:
			sprintf(icStr, "Smooth");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &iter,  "Smoothing iterations", H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &alpha, "Smoothing constant",   H5T_NATIVE_DOUBLE);
			break;

		case	CONF_KMAX:
			sprintf(icStr, "kMax");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
			break;

		case	CONF_VILGORK:
		case	CONF_VILGORS:
		case	CONF_VILGOR:
			sprintf(icStr, "VilGor");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",		 H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
			break;

		case	CONF_LOLA:
			sprintf(icStr, "Lola");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",		 H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
			break;

		case	CONF_COLE:
			sprintf(icStr, "Cole");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",		 H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
			/* FIX ME read all attributes from Icdata from myCosmos
			for instance kCrit is ic.kcr, etc. */
			// writeAttribute(icGrp_id, &kCrit, "Correlation length",   H5T_NATIVE_DOUBLE);
			break;

		case	CONF_TKACHEV:
			sprintf(icStr, "Tkachev");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
			break;

		case	CONF_THERMAL:
			sprintf(icStr, "Thermal");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kCrit, "Temperature",       H5T_NATIVE_DOUBLE);
			break;

		case	CONF_SPAX:
			sprintf(icStr, "Axion Spectrum");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
			// writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
			break;

		default:
		case	CONF_NONE:
			sprintf(icStr, "None");
			break;
	}

	switch (smvarType) {
		case	CONF_RAND:
			sprintf(smStr, "Random");
			break;

		case	CONF_STRINGXY:
			sprintf(smStr, "String XY");
			break;

		case	CONF_STRINGYZ:
			sprintf(smStr, "String YZ");
			break;

		case	CONF_MINICLUSTER:
			sprintf(smStr, "Minicluster");
			break;

		case	CONF_MINICLUSTER0:
			sprintf(smStr, "Minicluster 0");
			break;

		case	CONF_AXNOISE:
			sprintf(smStr, "Axion noise");
			break;

		case	CONF_SAXNOISE:
			sprintf(smStr, "Saxion noise");
			break;

		case	CONF_AX1MODE:
			sprintf(smStr, "Axion one mode");
			break;

		case	CONF_AXITON:
			sprintf(smStr, "Minicluster 0");
			break;

		case	CONF_PARRES:
			sprintf(smStr, "Parametric Resonance");
			break;

		case	CONF_STRWAVE:
			sprintf(smStr, "String + wave");
			break;


		default:
			sprintf(smStr, "None");
			break;

	}

	writeAttribute(icGrp_id, &mode0, "Axion zero mode",    H5T_NATIVE_DOUBLE);
	writeAttribute(icGrp_id, &smStr, "Configuration type", attr_type);

	H5Gclose(icGrp_id);

	H5Tclose (attr_type);

	commSync();

	/*	Create plist for collective write	*/
	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	/*	Create space for writing the raw data to disk with chunked access	*/
	if ((totalSpace = H5Screate_simple(1, &total, maxD)) < 0)	// Whole data
	{
		LogError ("Error calling H5Screate_simple");
		exit (1);
	}

	/*	Set chunked access	*/
	if ((chunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Error calling H5Pcreate");
		exit (1);
	}

	if (H5Pset_chunk (chunk_id, 1, &slab) < 0)
	{
		LogError ("Error setting chunked access");
		exit (1);
	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Error calling H5Pset_alloc_time\n");
		exit (1);
	}

	/*	Create a dataset for the whole axion data	*/
	char mCh[8] = "/m";
	char vCh[8] = "/v";

	mset_id = H5Dcreate (file_id, mCh, dataType, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);
	if (axion->Field() != FIELD_NAXION)
		vset_id = H5Dcreate (file_id, vCh, dataType, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

	commSync();

	if ((mset_id < 0) || (vset_id < 0))
	{
		LogError ("Error creating datasets");
		exit (0);
	}

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

	mSpace = H5Dget_space (mset_id);
	if (axion->Field() != FIELD_NAXION)
		vSpace = H5Dget_space (vset_id);
	memSpace = H5Screate_simple(1, &slab, NULL);	// Slab

	commSync();

	LogMsg (VERB_HIGH, "Rank %d ready to write", myRank);

	for (hsize_t zDim=0; zDim<((hsize_t) axion->Depth()); zDim++)
	{
		/*	Select the slab in the file	*/
		offset = (((hsize_t) (myRank*axion->Depth()))+zDim)*slab;
		H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
		if (axion->Field() != FIELD_NAXION)
			H5Sselect_hyperslab(vSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

		/*	Write raw data	*/
		auto mErr = H5Dwrite (mset_id, dataType, memSpace, mSpace, plist_id, (static_cast<char *> (axion->mStart())+slab*zDim*dataSize));
		auto vErr = mErr;
		if (axion->Field() != FIELD_NAXION)
			vErr = H5Dwrite (vset_id, dataType, memSpace, vSpace, plist_id, (static_cast<char *> (axion->vStart())  +slab*zDim*dataSize));

		if (mErr < 0)
		{
			LogError ("Error writing dataset");
			exit(0);
		}
		if (axion->Field() != FIELD_NAXION)
			if (vErr < 0){
				LogError ("Error writing dataset"); exit(0);}

		//commSync();
	}

	/*	Close the dataset	*/
	H5Sclose (mSpace);
	H5Dclose (mset_id);
	if (axion->Field() != FIELD_NAXION){
		H5Sclose (vSpace);
		H5Dclose (vset_id);}

	H5Sclose (memSpace);

	/*	Close the file		*/
	H5Pclose (chunk_id);
	H5Sclose (totalSpace);
	H5Pclose (plist_id);
	H5Fclose (file_id);

	prof.stop();
	prof.add(std::string("Write configuration"), 0., (2.*total*dataSize + 81.)*1e-9);

	LogMsg (VERB_NORMAL, "Written %lu bytes", total*dataSize*2 + 81);

	/*	Fold back the field	*/

	// if (wasFolded)
	// {
	// 	(*munge)(FOLD_ALL);
	// 	delete	munge;
	// }
}





/* Read Configuration and parameters from axion.xxxxx file
 	and prepare Cosmos by merging with commandline parameters

	Includes adjusting size to the desired command line
	*/

	void	readConf (Cosmos *myCosmos, Scalar **axion, int index, const bool restart)
	{

		/* Some definitions */
		hid_t	file_id, mset_id, vset_id, plist_id;
		hid_t	mSpace, vSpace, memSpace, dataType;
		hid_t	attr_type;
		hsize_t	slab, offset;
		FieldPrecision	precision;
		char	prec[16], fStr[16], lStr[16], icStr[16], vStr[32], vPQStr[32], smStr[16];
		int	length = 32;
		const hsize_t maxD[1] = { H5S_UNLIMITED };
		size_t	dataSize;
		int myRank = commRank();
		bool Moore = 0;
		char base[256];

		/* JAVI if restart flag, do not read index number for simplicity */
		/* restart files have to pass the same parameters of the simulation! */
		if (!restart)
			sprintf(base, "%s/%s.%05d", outDir, outName, index);
		else
			sprintf(base, "%s/%s.restart", outDir, outName);

		/* Start */

		LogMsg (VERB_NORMAL, "Reading Hdf5 configuration from disk (%s)", base);
		LogMsg (VERB_NORMAL, "");

		/*	Start profiling		*/

		Profiler &prof = getProfiler(PROF_HDF5);
		prof.start();

		/*	Set up parallel access with Hdf5	*/

		plist_id = H5Pcreate (H5P_FILE_ACCESS);
		H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

		/*	Open the file and release the plist	*/

		if ((file_id = H5Fopen (base, H5F_ACC_RDONLY, plist_id)) < 0)
		{
			*axion == nullptr;
			LogError ("Error opening file %s", base);
			return;
		}
		H5Pclose(plist_id);

		/*	                    Attributes	                         */

		attr_type = H5Tcopy(H5T_C_S1);
		H5Tset_size (attr_type, length);
		H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

		//	GRID PARAMETERS
		//  --------------
		//  (do not change, except precision, which does automatically)


			double	zTmp, RTmp, maaR, fTmp;
			uint	tStep, cStep, totlZ;

			uint ux_read;
			uint uz_read;
			readAttribute (file_id, fStr,   "Field type",   attr_type);
			readAttribute (file_id, &ux_read, "Size",         H5T_NATIVE_UINT);
			readAttribute (file_id, &uz_read, "Depth",        H5T_NATIVE_UINT);
			readAttribute (file_id, prec,   "Precision",    attr_type);
			size_t Nx_read = (size_t) ux_read;
			size_t Nz_read = (size_t) uz_read;

		//	  IRRELEVANT
		//    ----------

			readAttribute (file_id, &tStep, "nSteps",       H5T_NATIVE_INT);
			readAttribute (file_id, &cStep, "Current step", H5T_NATIVE_INT);

		//	  PHYSICAL PARAMETERS
		//    ------------------
		//    (can be modified by commandline, already parsed in Cosmos)

			/* conformal time now (ADM units) */

			readAttribute (file_id, &zTmp,  "z",            H5T_NATIVE_DOUBLE);
			readAttribute (file_id, &RTmp,  "R",            H5T_NATIVE_DOUBLE);

			// note: Field will be created with z=zTmp, R = R(z)
			// if zInit is given in command line -> change it
			if (uZin) {
				zTmp = myCosmos->ICData().zi;
				LogMsg (VERB_NORMAL, "Commandline input for zInit!!!!! Initial time set to: %f",zTmp);
			}

			/* initial time of the read simulation */

			readAttribute (file_id, &zInit, "zInitial", H5T_NATIVE_DOUBLE);

			/* final, stopping time  */

			if (!uZfn) {
				readAttribute (file_id, &zFinl,  "zFinal",  H5T_NATIVE_DOUBLE);
			} else {
				LogMsg (VERB_NORMAL, "zFinal (commandline): %f",zFinl);
			}

			/* expansion parameter; cosmology, R = z^frw  */

			readAttribute (file_id, &fTmp,  "Frw",          H5T_NATIVE_DOUBLE);

			// note: command line is given precedence
			if (myCosmos->Frw() != fTmp) {
				LogMsg (VERB_NORMAL, "Commandline input for frw (%.2f) does not match h5read value (%.2f); frw set to input %.2f",myCosmos->Frw(),fTmp,myCosmos->Frw());
				fTmp = myCosmos->Frw();
				if (fTmp == 0.0 ){
					myCosmos->SetMink(true);
					LogMsg (VERB_NORMAL, "Set Mink (frw = 0, no expansion)");
				}
			}

			/* Box length in ADM units */

			// note: if no value is read in commandline, PhysSize() gives 0
			if (myCosmos->PhysSize() == 0.0) {
				double lSize;
				readAttribute (file_id, &lSize, "Physical size", H5T_NATIVE_DOUBLE);
				myCosmos->SetPhysSize(lSize);
			} else {
				LogMsg (VERB_NORMAL, "PhysSize (commandline): %f", myCosmos->PhysSize());
			}

			// RESTART MODULE !!!
			// ------------------------------------------------------------------------
			//
			// note: if restart, we assume all the parsed parameters are equal to
			// the read values; but we give priority to h5file
			if (restart)
			{

				readAttribute (file_id, &fIndex, "index", H5T_NATIVE_INT);
				LogOut(" Reading index is %d\n",fIndex);
				LogMsg (VERB_NORMAL, "RESTART RUN! index %d\n",fIndex);
				readAttribute (file_id, &zInit, "zInitial", H5T_NATIVE_DOUBLE);
				readAttribute (file_id, &zTmp,  "z",        H5T_NATIVE_DOUBLE);
				LogOut("Reading zTmp = %f, zInit=%f \n",zTmp,zInit);
				LogMsg (VERB_NORMAL, "Reading zTmp = %f, zInit=%f \n",zTmp,zInit);
			}
			//
			// ------------------------------------------------------------------------

			//	  POTENTIAL PARAMETERS
			//    ------------------
			//    (can be modified by commandline, already parsed in Cosmos)

			/*	Open potential group	*/
			auto status = H5Lexists (file_id, "/potential", H5P_DEFAULT);

			if (status <= 0)
				LogMsg(VERB_NORMAL, "Potential data not available, using defaults");
			else
			{
				hid_t vGrp_id = H5Gopen2(file_id, "/potential", H5P_DEFAULT);


				/* Axion mass; model          mA^2 = i3^2 * R^n          */

				readAttribute (file_id, &maaR,  "Axion mass",   H5T_NATIVE_DOUBLE);

				/* n, exponent of topological susceptibility */

				// note: if no value is read in commandline, QcdExp() gives -1.e8
				if (myCosmos->QcdExp() == -1.e8) {
					double nQcd;
					readAttribute (vGrp_id, &nQcd,  "nQcd",	  H5T_NATIVE_DOUBLE);
					myCosmos->SetQcdExp(nQcd);
					double nQcd2;
					herr_t test = readAttribute (vGrp_id, &nQcd2,  "nQcd2",	  H5T_NATIVE_DOUBLE);
					if (!(test < 0)){
						myCosmos->SetQcdExpr(nQcd2);
						LogMsg (VERB_NORMAL, "nQcdr (read and set to)= %f",myCosmos->QcdExpr());
					}
				} else {
					LogMsg (VERB_NORMAL, "nQcd  (commandline) = %.2f", myCosmos->QcdExp());
					LogMsg (VERB_HIGH,   "nQcdr (commandline) = %.2f ",myCosmos->QcdExpr());
				}

				/* i3  prefactor */

				// note: if no value is read in commandline, Indi3() gives -1.e8
				if (myCosmos->Indi3() == -1.e8) {
					double indi3;
					readAttribute (vGrp_id, &indi3, "Indi3", H5T_NATIVE_DOUBLE);
					myCosmos->SetIndi3(indi3);
				} else
					LogMsg (VERB_NORMAL, "Indi3 (commandline) = %.2f",myCosmos->Indi3());


				/* Scale factor R beyond which axion mass growth saturates to n = 0 */

				if (myCosmos->ZThRes() == -1.e8) {
					double zthrs;
					readAttribute (vGrp_id, &zthrs, "z Threshold", H5T_NATIVE_DOUBLE);
					myCosmos->SetZThRes(zthrs);
				} else
					LogMsg (VERB_NORMAL, "z Threshold (commandline) = %.3e",myCosmos->ZThRes());

				/* Scale factor R above which axion mass growth continues with n_r */

				if (myCosmos->ZRestore() == -1.e8) {
					double zrest;
					readAttribute (vGrp_id, &zrest, "z Restore", H5T_NATIVE_DOUBLE);
					myCosmos->SetZRestore(zrest);
				} else
					LogMsg (VERB_NORMAL, "z Restore (commandline) = %.3e",myCosmos->ZRestore());


				// test the axion mass
				// -------------------
				LogMsg (VERB_NORMAL, "Axion mass^2 (h5read) %.2f (calculated) %.2f", maaR*maaR, myCosmos->AxionMass2(zTmp));

				// note: if i3, n, n2, z, frw, zThreshold, zRestore have changed
				// the mass might not coincide
				// To ensure the continuity we need to adapt the parameters,
				// and we will choose those which we have not changed in the commandline

				// The most relevant cases :
				// 1) we have nQcd, zTh, zRe >>> we adapt i3
				// 2) ...
				// anyways these readjustments can always be made from the commandline

				/* Lambda; saxion self-interation coefficient at z = 1 */

				// note: if no value is read in commandline, Lambda() gives -1.e8
				if (myCosmos->Lambda() == -1.e8) {
					double	lda, lz2e;

					readAttribute (vGrp_id, &lda,   "Lambda",      H5T_NATIVE_DOUBLE);
					myCosmos->SetLambda(lda);

					readAttribute (vGrp_id, &lStr,  "Lambda type", attr_type);
					herr_t test = readAttribute (vGrp_id, &lz2e,  "Lambda Z2 exponent", H5T_NATIVE_DOUBLE);

					if (!strcmp(lStr, "z2")){
						lType = LAMBDA_Z2;
						if (!(test < 0))
							myCosmos->SetLambda(lda);
						else
							myCosmos->SetLamZ2Exp(2.0);
					}
					else if (!strcmp(lStr, "Fixed")){
						lType = LAMBDA_FIXED;
						myCosmos->SetLamZ2Exp(0.0);
						}
					else {
						LogError ("Error reading file %s: invalid lambda type %s", base, lStr);
						exit(1);
					}
					LogMsg (VERB_NORMAL, "Lambda (h5read and set)= %.2f/R^%.2f",myCosmos->Lambda(),myCosmos->LamZ2Exp());
				}
				else
					LogMsg (VERB_NORMAL, "Lambda (commandline)   = %.2f/R^%.2f",myCosmos->Lambda(),myCosmos->LamZ2Exp());


				// test LambdaP?
				// -------------------
				// LogMsg (VERB_NORMAL, "Axion mass (h5read) %.2f (calculated) %.2f",maaR, myCosmos->AxionMass());


				/* Gamma; Saxion damping coefficient */


				if (myCosmos->Gamma() == -1.e8) {
					double gm;
					readAttribute (vGrp_id, &gm, "Gamma", H5T_NATIVE_DOUBLE);
					myCosmos->SetGamma(gm);
				}
				else
					LogMsg (VERB_NORMAL, "Gamma (commandline) = %f",myCosmos->Gamma());


				/* V_QCD potential type */


				// note : if no commandline values myCosmos->QcdPot() == V_NONE
				VqcdType vqcdType = V_NONE;
				if (myCosmos->Indi3() == 0.0){
					VqcdType PQEVDA = myCosmos->QcdPot() & (V_PQ|V_TYPE|V_EVOL|V_DAMP);
					myCosmos->SetQcdPot(V_QCD0 | PQEVDA);
					vqcdType = V_QCD0;
				}
				else {
					if ( (myCosmos->QcdPot() & V_QCD) == V_NONE) {

						readAttribute (vGrp_id, &vStr,  "VQcd type",  attr_type);
						LogMsg (VERB_PARANOID, " V_QCD (commandline   ) = %d", myCosmos->QcdPot() & V_QCD);
						LogMsg (VERB_PARANOID, " V_QCD (read from file) = %s", vStr);


						if (!strcmp(vStr, "VQcd 1"))
							vqcdType = V_QCD1;
						else if ( !strcmp(vStr, "VQcd Variant") || !strcmp(vStr, "VQcd 2")) //legacy
							vqcdType = V_QCDV;
						else if (!strcmp(vStr, "VQcd 0"))
							vqcdType = V_QCD0;
						else if (!strcmp(vStr, "VQcd Linear"))
							vqcdType = V_QCDL;
						else if (!strcmp(vStr, "VQcd Cos"))
							vqcdType = V_QCDC;
						else if (!strcmp(vStr, "VQcd N = 2"))
							vqcdType = V_QCD2;
						else {
							LogError ("Error reading file %s: invalid QCD potential type %s. Use QCD1", base, vStr);
							vqcdType = V_QCD1;
						}
					}
					else {
						LogMsg (VERB_NORMAL, "V_QCD (commandline) = %d", myCosmos->QcdPot() & V_QCD);
						vqcdType |= (myCosmos->QcdPot() & V_QCD);
					}

				}

				/* V_PQ potential type */

				if ( (myCosmos->QcdPot() & V_PQ) == V_NONE) {

					readAttribute (vGrp_id, &vStr,  "VPQ type",  attr_type);

					if (!strcmp(vStr, "VPQ 1"))
						vqcdType |= V_PQ1;
					else if ( !strcmp(vStr, "VPQ2"))
						vqcdType |= V_PQ2;
					else {
						LogError ("Error reading file %s: invalid PQ potential type %s. Use PQ1", base, vStr);
						vqcdType |= V_PQ1;
					}
				}
				else {
					LogMsg (VERB_NORMAL, "V_PQ (commandline) = %d", myCosmos->QcdPot() & V_PQ);
					vqcdType |= (myCosmos->QcdPot() & V_PQ);
				}


				if ( (myCosmos->QcdPot() & V_DAMP) == V_NONE) {

					readAttribute (vGrp_id, &vStr,  "Damping type",  attr_type);

					if (!strcmp(vStr, "Rho"))
						vqcdType |= V_DAMP_RHO;
					else if (!strcmp(vStr, "All"))
						vqcdType |= V_DAMP_ALL;
					else if (!strcmp(vStr, "None"))
						vqcdType |= V_NONE;
					else {
						LogError ("Error reading file %s: invalid damping type %s. Ignoring damping", base, vStr);
					}
				}
				else {
					LogMsg (VERB_NORMAL, "V_DAMP (commandline) = %d", myCosmos->QcdPot() & V_DAMP);
					vqcdType |= (myCosmos->QcdPot() & V_DAMP);
				}

				// FIXME
				// If command line theta or rho, respect that, else read
				// If theta and rho, -> All (eq. none)


				if ( (myCosmos->QcdPot() & (V_EVOL_RHO | V_EVOL_THETA)) == V_NONE) {

					readAttribute (vGrp_id, &vStr,  "Evolution type",  attr_type);

					if (!strcmp(vStr, "Only Rho"))
						vqcdType |= V_EVOL_RHO;
					else if (!strcmp(vStr, "Full"))
						vqcdType |= V_NONE;
					else {
						LogError ("Error reading file %s: invalid rho evolution type %s. Ignoring rho evolution", base, vStr);
					}
				}
				else {
					if ((myCosmos->QcdPot() & V_EVOL_THETA) && (myCosmos->QcdPot() & V_EVOL_RHO))
						LogMsg (VERB_NORMAL, "V_EVOL_RHO & V_EVOL_THETA selected (commandline), we use full evolution.");
					else
					{
						if (myCosmos->QcdPot() & V_EVOL_RHO) {
							LogMsg (VERB_NORMAL, "V_EVOL_RHO (commandline)");
							vqcdType |= (myCosmos->QcdPot() & V_EVOL_RHO);
						}
						else if (myCosmos->QcdPot() & V_EVOL_THETA) {
							LogMsg (VERB_NORMAL, "V_EVOL_THETA (commandline)");
							vqcdType |= (myCosmos->QcdPot() & V_EVOL_THETA);
						}
					}

				} // end EVOL RHO/THETA TYPE

				myCosmos->SetQcdPot(vqcdType);
				LogMsg (VERB_NORMAL, "QcdPot set to %d\n",myCosmos->QcdPot());

				H5Gclose(vGrp_id);
			}
			/* end potential group */

			//    INITIAL CONDITIONS DATA
			//    -----------------------
			//

			status = H5Lexists (file_id, "/ic", H5P_DEFAULT);

			LogMsg (VERB_NORMAL, "Ic... \n");
			if (status <= 0)
				LogMsg(VERB_NORMAL, "IC data not available");
			else
			{
				hid_t icGrp_id = H5Gopen2(file_id, "/ic", H5P_DEFAULT);
				readAttribute(icGrp_id, &mode0, "Axion zero mode", H5T_NATIVE_DOUBLE);
				readAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);

				if (!strcmp(icStr, "Smooth")) {
					cType = CONF_SMOOTH;
					readAttribute(icGrp_id, &iter,  "Smoothing iterations", H5T_NATIVE_HSIZE);
					readAttribute(icGrp_id, &alpha, "Smoothing constant",   H5T_NATIVE_DOUBLE);
				} else if (!strcmp(icStr, "kMax")) {
					cType = CONF_KMAX;
					readAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
					readAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
				} else if (!strcmp(icStr, "VilGor")) {
					cType = CONF_VILGOR;
					readAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
					readAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
				} else if (!strcmp(icStr, "Lola")) {
					cType = CONF_LOLA;
					readAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
					readAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
				} else if (!strcmp(icStr, "Cole")) {
					cType = CONF_COLE;
					readAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
					readAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
				} else if (!strcmp(icStr, "Tkachev")) {
					cType = CONF_TKACHEV;
					readAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
					readAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
				} else if (!strcmp(icStr, "Thermal")) {
  				cType = CONF_THERMAL;
  				readAttribute(icGrp_id, &kCrit, "Temperature",       H5T_NATIVE_DOUBLE);
				} else if (!strcmp(icStr, "Axion Spectrum")) {
					cType = CONF_SPAX;
					readAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
					// readAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
				} else if (!strcmp(icStr, "Moore")) {
					cType = CONF_SMOOTH;
					/* The m and v fields are not conformal so we will need to rescale them */
					Moore = true;
				}

				readAttribute(icGrp_id, &icStr, "Configuration type",   attr_type);

				if (!strcmp(icStr, "Random")) {
					smvarType = CONF_RAND;
				} else if (!strcmp(icStr, "String XY")) {
					smvarType = CONF_STRINGXY;
				} else if (!strcmp(icStr, "String YZ")) {
					smvarType = CONF_STRINGYZ;
				} else if (!strcmp(icStr, "Minicluster")) {
					smvarType = CONF_MINICLUSTER;
				} else if (!strcmp(icStr, "Minicluster 0")) {
					smvarType = CONF_MINICLUSTER0;
				} else if (!strcmp(icStr, "Axion noise")) {
					smvarType = CONF_AXNOISE;
				} else if (!strcmp(icStr, "Saxion noise")) {
					smvarType = CONF_SAXNOISE;
				} else if (!strcmp(icStr, "Axion one mode")) {
					smvarType = CONF_AX1MODE;
				} else if (!strcmp(icStr, "Parametric Resonance")) {
					smvarType = CONF_PARRES;
				} else if (!strcmp(icStr, "String + wave")) {
					smvarType = CONF_STRWAVE;
				} else {
					LogError("Error: unrecognized configuration type %s", icStr);
				}
				H5Gclose(icGrp_id);
			}
			/* end IC group */

		/* we do not need this anymore */
		H5Tclose (attr_type);


			//    PRECISION
			//    ---------
			//    will be adapted to the commandline

			if (!uPrec)
			{
				if (!strcmp(prec, "Double"))
				{
					precision = FIELD_DOUBLE;
					sPrec	  = FIELD_DOUBLE;
					dataType  = H5T_NATIVE_DOUBLE;
					dataSize  = sizeof(double);
				} else if (!strcmp(prec, "Single")) {
					precision = FIELD_SINGLE;
					sPrec	  = FIELD_SINGLE;
					dataType  = H5T_NATIVE_FLOAT;
					dataSize  = sizeof(float);
				} else {
					LogError ("Error reading file %s: Invalid precision %s", base, prec);
					exit(1);
				}
			}
			else
			{
				precision = sPrec;

				if (sPrec == FIELD_DOUBLE)
				{
					dataType  = H5T_NATIVE_DOUBLE;
					dataSize  = sizeof(double);

					if (!strcmp(prec, "Single"))
						LogMsg (VERB_NORMAL, "Reading single precision configuration as double precision");
				} else if (sPrec == FIELD_SINGLE) {
					dataType  = H5T_NATIVE_FLOAT;
					dataSize  = sizeof(float);
					if (!strcmp(prec, "Double"))
						LogMsg (VERB_NORMAL, "Reading double precision configuration as single precision");
				} else {
					LogError ("Input error: Invalid precision");
					exit(1);
				}
			}
			/* end precision */



		/*	-------------------------------------------------------------------------
				-------------------------------------------------------------------------
															 Create axion field
				-------------------------------------------------------------------------
				-------------------------------------------------------------------------
		*/

		IcData ictemp   = myCosmos->ICData();
		ictemp.alpha    = alpha;
		ictemp.siter    = iter;
		ictemp.kcr      = kCrit;
		ictemp.kMax     = kMax;
		ictemp.mode0    = mode0;
		ictemp.zi       = zTmp;
		ictemp.cType    = cType;
		ictemp.smvarType= smvarType;
		myCosmos->SetICData(ictemp);

		size_t Nz = Nz_read/zGrid;

		if (Nz_read % zGrid)
		{
			LogError ("Error: Geometry not valid. Try a different partitioning");
			exit (1);
		}

		size_t Nxcreate = Nx_read;
		size_t Nzcreate = Nz;

		if ( (sizeN == Nx_read) && (sizeZ == Nz)){
			LogMsg(VERB_NORMAL,"[rc] Reading exact size %dx%dx%d(x%d), size requested %dx%dx%d(x%d)",Nx_read,Nx_read,Nz,zGrid, sizeN,sizeN,sizeZ,zGrid);
		}
		else if ( (sizeN > Nx_read) && (sizeZ > Nz) )
		{
			LogMsg(VERB_NORMAL,"[rc] We will be expanding from %dx%dx%d(x%d) to %dx%dx%d(x%d)",
				Nx_read,Nx_read,Nz,zGrid, sizeN,sizeN,sizeZ,zGrid);
				Nxcreate = sizeN;
				Nzcreate = sizeZ;
		}
		else if ( (sizeN < Nx_read) && (sizeZ < Nz) )
		{
			LogMsg(VERB_NORMAL,"[rc] We will be reducing from %dx%dx%d(x%d) to %dx%dx%d(x%d)",
			Nx_read,Nx_read,Nz,zGrid, sizeN,sizeN,sizeZ,zGrid);
		}
		// else
		// {
		// 	LogError ("Error: Expanding and reducing in different directions not supported: exit!");
		// 	exit (1);
		// }

		/* We read in an auxiliar Scalar field because we might need to reduce into axion */

		LogMsg(VERB_PARANOID, "[rc] Creating axion field %d %d(x%d)",Nxcreate,Nzcreate,zGrid);

		prof.stop();
		prof.add(std::string("Read configuration"), 0, 0);

		myCosmos->ICData().cType = CONF_NONE;
		slab   = (hsize_t) (Nx_read*Nx_read);
		// We create a larger axion file if we need to expand

		if (!strcmp(fStr, "Saxion"))
		{
			(*axion) = new Scalar(myCosmos, Nxcreate, Nzcreate, precision, cDev, zTmp, lowmem, zGrid, FIELD_SAXION,    lType, myCosmos->ICData().Nghost);
			slab   = (hsize_t) (slab*2);
		}
		else if (!strcmp(fStr, "Axion"))
		{
			(*axion) = new Scalar(myCosmos, Nxcreate, Nzcreate, precision, cDev, zTmp, lowmem, zGrid, FIELD_AXION,    lType, myCosmos->ICData().Nghost);
		}
		else if (!strcmp(fStr, "Axion Mod"))
		{
			(*axion) = new Scalar(myCosmos, Nxcreate, Nzcreate, precision, cDev, zTmp, lowmem, zGrid, FIELD_AXION_MOD, lType, myCosmos->ICData().Nghost);
		}
		else
		{
			LogError ("Input error: Invalid field type");
			exit(1);
		}

		LogMsg(VERB_PARANOID, "[rc] Read start\n");

		prof.start();
		commSync();

		LogMsg(VERB_PARANOID, "[rc] Reading into *axion \n");

		prof.start();
		commSync();

		/*	Create plist for collective read	*/

		plist_id = H5Pcreate(H5P_DATASET_XFER);
		H5Pset_dxpl_mpio(plist_id,H5FD_MPIO_COLLECTIVE);


		/*	Open a dataset for the whole axion data	*/

		if ((mset_id = H5Dopen (file_id, "/m", H5P_DEFAULT)) < 0)
			LogError ("Error opening dataset");

		if ((vset_id = H5Dopen (file_id, "/v", H5P_DEFAULT)) < 0)
			LogError ("Error opening dataset");

		memSpace = H5Screate_simple(1, &slab, NULL);	// Slab
		mSpace   = H5Dget_space (mset_id);
		vSpace   = H5Dget_space (vset_id);

		for (hsize_t zDim = 0; zDim < Nz ; zDim++)
		{

	LogMsg(VERB_PARANOID, "[rc] Reading zDim %d slab %d Nz %d",zDim,slab,Nz);LogFlush();

			/*	Select the slab in the file	*/
			offset = (((hsize_t) (myRank*Nz))+zDim)*slab;
			H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
			H5Sselect_hyperslab(vSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

			/*	Read raw data	*/

			auto mErr = H5Dread (mset_id, dataType, memSpace, mSpace, plist_id, (static_cast<char *> ((*axion)->mStart())+slab*zDim*dataSize));
			auto vErr = H5Dread (vset_id, dataType, memSpace, vSpace, plist_id, (static_cast<char *> ((*axion)->vCpu())  +slab*zDim*dataSize));

			if ((mErr < 0) || (vErr < 0)) {
				LogError ("Error reading dataset from file");
				return;
			}
		}

		(*axion)->setFolded(false);
		// (*axion)->setFolded(false);

		/*	Close the dataset	*/

		H5Sclose (mSpace);
		H5Sclose (vSpace);
		H5Dclose (mset_id);
		H5Dclose (vset_id);
		H5Sclose (memSpace);

		/*	Close the file		*/

		H5Pclose (plist_id);
		H5Fclose (file_id);

		prof.stop();
		prof.add(std::string("Read configuration"), 0, (2.*Nz_read*slab*((*axion))->DataSize() + 77.)*1.e-9);


		/*	If configuration is Moore > convert to jaxions */
		if (Moore)
			{
				LogMsg(VERB_NORMAL, "[RC] Unmooring field");
				prof.start();

				/* Converts Moore format to conformal theta */
				unMoor((*axion), PFIELD_MS);

				/* cVelocity = RVelocity - Theta */
				axby((*axion), PFIELD_MS, PFIELD_V, -1., *(*axion)->RV());

				prof.stop();
				prof.add(std::string("Unmoor configuration"), 0, 10*(totlZ*slab*(*axion)->Precision())*1.e-9);

				/* mendTheta! */
				mendTheta (*axion);
			}

		commSync();

			/* Reduce or expand if required */
		if ((sizeN > Nx_read) && (sizeZ > Nz))
		{
				LogMsg(VERB_NORMAL, "[rC] Expansion from XY %d Z %d to XY %d Z %d",Nx_read,Nz*zGrid, sizeN,sizeZ*zGrid);
				(*axion)->setReduced	(true, Nx_read, Nz);
				expandField(*axion);
				(*axion)->setReduced	(false, 1, 1); // 2,3 entries have no effect
		}
		else if ((sizeN < Nx_read) && (sizeZ < Nz))
		{
			LogMsg(VERB_NORMAL, "[rc] Reduction by a factor %d in x and %d in z",Nx_read/sizeN,Nz/sizeZ);
			LogOut("0\n");
			double eFc_xy  = 2*M_PI*M_PI/((double) sizeN*sizeN);
			double eFc_z   = 2*M_PI*M_PI/((double) sizeZ*sizeZ*zGrid*zGrid);
			double nFc  = 1.;
			LogMsg(VERB_NORMAL, "[rc] 1 - reduce in place in (*axion)");
			if ((*axion)->Precision() == FIELD_DOUBLE) {
			  reduceField((*axion), sizeN, sizeZ, FIELD_MV,
			      [eFc_xy  = eFc_xy, eFc_z = eFc_z, nFc = nFc] (int px, int py, int pz, complex<double> x) -> complex<double> { return x*((double) nFc*exp(-eFc_xy*(px*px + py*py) -eFc_z*pz*pz)); }, true);
			} else {
			  reduceField((*axion), sizeN, sizeZ, FIELD_MV,
			      [eFc_xy = eFc_xy, eFc_z = eFc_z, nFc = nFc] (int px, int py, int pz, complex<float>  x) -> complex<float>  { return x*((float)  (nFc*exp(-eFc_xy*(px*px + py*py) -eFc_z*pz*pz))); }, true);
			}
			// LogMsg(VERB_NORMAL, "[rc] 4 - move reduced data from auxion to axion (%lu/%lu data points)",sizeN*sizeN*sizeZ,auxion->Size());
			// //data when reduced in place is in mCpu ,vCpu, sizeN*sizeN*sizeZ
			// memmove((*axion)->mStart(),auxion->mCpu(), sizeN*sizeN*sizeZ * auxion->DataSize());
			// memmove((*axion)->vCpu(),  auxion->vCpu(), sizeN*sizeN*sizeZ * auxion->DataSize());
			LogMsg(VERB_NORMAL, "[rc] 2 - remove plans from large axion");
			AxionFFT::removePlan("pSpecAx");
			AxionFFT::removePlan("SpSx");
			AxionFFT::removePlan("RdSxV");
			LogMsg(VERB_NORMAL, "[rc] 3 - insert plans for correct size axion");
			AxionFFT::initPlan (*axion, FFT_PSPEC_AX,  FFT_FWDBCK, "pSpecAx");
			AxionFFT::initPlan (*axion, FFT_SPSX,       FFT_FWDBCK,     "SpSx");
			AxionFFT::initPlan (*axion, FFT_RDSX_V,     FFT_FWDBCK,    "RdSxV");
			LogMsg(VERB_NORMAL, "[rc] 4 - Reduction complete!");
			// LogMsg(VERB_NORMAL, "[rc] 8 - Remove auxion");
			// delete auxion; kkils the FFTs do not use!
		}


		commSync();

		// delete auxion;
		// LogMsg(VERB_NORMAL, "AUXION deleted");

		if (cDev == DEV_GPU)
			(*axion)->transferDev(FIELD_MV);

		LogMsg (VERB_NORMAL, "[rC] Read %lu bytes", ((size_t) Nz_read)*slab*2 + 77);
		/* If transformed add information */
		// LogMsg (VERB_NORMAL, "[rC] Read %lu bytes", ((size_t) totlZ)*slab*2 + 77);
		LogFlush();

	}








/*	Creates a hdf5 file to write all the measurements	*/
void	createMeas (Scalar *axion, int index)
{
	hid_t	plist_id, dataType;

	char	prec[16], fStr[16], lStr[16], icStr[16], vStr[32], vPQStr[32], smStr[16], dStr[16], rStr[16];
	int	length = 32;

//	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize;

	int myRank = commRank();

	int cSteps = dump*index;
	hsize_t totlZ = axion->TotalDepth();
	hsize_t tmpS  = axion->Length();

	tSize  = axion->TotalSize();
	slabSz = axion->Surf();
	sLz    = axion->Depth();

	LogMsg (VERB_NORMAL, "Creating measurement file with index %d", index);

	if (opened)
	{
		LogError ("Error: Hdf5 measurement file already opened");
		return;
	}

	/*	Set up parallel access with Hdf5	*/

	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];

	sprintf(base, "%s/%s.m.%05d", outDir, outName, index);

	/*	Create the file and release the plist	*/
	if ((meas_id = H5Fcreate (base, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) < 0)
	{
		LogError ("Error creating file %s", base);
		return;
	}

	opened = true;

	switch (axion->Precision())
	{
		case FIELD_SINGLE:
		{
			dataType = H5T_NATIVE_FLOAT;
			dataSize = sizeof(float);

			sprintf(prec, "Single");
			// length = strlen(prec)+1;
		}

		break;

		case FIELD_DOUBLE:
		{
			dataType = H5T_NATIVE_DOUBLE;
			dataSize = sizeof(double);

			sprintf(prec, "Double");
			// length = strlen(prec)+1;
		}

		break;

		default:

		LogError ("Error: Invalid precision. How did you get this far?");
		exit(1);

		break;
	}

	switch (axion->Field())
	{
		case 	FIELD_SAXION:
			sprintf(fStr, "Saxion");
			break;

		case	FIELD_AXION_MOD:
			sprintf(fStr, "Axion Mod");
			break;

		case	FIELD_AXION:
		case	FIELD_WKB:
			sprintf(fStr, "Axion");
			break;

		case	FIELD_NAXION:
			sprintf(fStr, "Naxion");
			break;

		case	FIELD_PAXION:
			sprintf(fStr, "Paxion");
			break;

		default:
			LogError ("Error: Invalid field type. How did you get this far?");
			exit(1);
			break;
	}

	auto lSize    = axion->BckGnd()->PhysSize();
	auto llPhys   = axion->BckGnd()->Lambda  ();
	auto LL       = axion->BckGnd()->Lambda  ();
	auto nQcd     = axion->BckGnd()->QcdExp  ();
	auto nQcdr    = axion->BckGnd()->QcdExpr ();
	auto gamma    = axion->BckGnd()->Gamma   ();
	auto vqcdType = axion->BckGnd()->QcdPot  ();

	switch (axion->LambdaT())
	{
		case 	LAMBDA_Z2:
			sprintf(lStr, "z2");
			llPhys /= (*axion->zV())*(*axion->zV());
			break;

		case	LAMBDA_FIXED:
			sprintf(lStr, "Fixed");
			break;

		default:
			LogError ("Error: Invalid field type. How did you get this far?");
			exit(1);
			break;
	}

	switch (vqcdType & V_QCD)	{
		default:
		caspr(V_QCDC,vStr,"VQcd Cos")
		caspr(V_QCD1,vStr,"VQcd 1")
		caspr(V_QCD0,vStr,"VQcd 0")
		caspr(V_QCDV,vStr,"VQcd Variant")
		caspr(V_QCDL,vStr,"VQcd Linear")
		caspr(V_QCD2,vStr,"VQcd N = 2")
	}

	switch (vqcdType & V_PQ) {
		default:
		caspr(V_PQ1,vPQStr,"VPQ 1")
		caspr(V_PQ2,vPQStr,"VPQ 2")
	}

	switch (vqcdType & V_DAMP) {
		caspr(V_DAMP_RHO,dStr,"Rho")
		caspr(V_DAMP_ALL,dStr,"All")
		default:
		caspr(V_NONE,dStr,"None")
	}

	switch (vqcdType & V_EVOL_RHO)	{
		caspr(V_EVOL_RHO,rStr,"Only Rho")
		default:
		caspr(V_NONE,rStr,"Full")
	}

	/*	Write header	*/

	hid_t attr_type;

	/*	Attributes	*/

	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size   (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	double maa = axion->AxionMass();
	double ms  = sqrt(axion->SaxionMassSq());
	double msa = axion->Msa();

	writeAttribute(meas_id, fStr,   "Field type",    attr_type);
	writeAttribute(meas_id, prec,   "Precision",     attr_type);
	writeAttribute(meas_id, &tmpS,  "Size",          H5T_NATIVE_HSIZE);
	writeAttribute(meas_id, &totlZ, "Depth",         H5T_NATIVE_HSIZE);
	writeAttribute(meas_id, &ms,    "Saxion mass",   H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &msa,   "msa",           H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &maa,   "Axion mass",    H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &lSize, "Physical size", H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, axion->zV(),  "z",       H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, axion->RV(),  "R",       H5T_NATIVE_DOUBLE);
	if (axion->BckGnd()->UeC()){
		double Temp = axion->BckGnd()->T(*axion->zV());
		writeAttribute(meas_id, &Temp,  "Temperature", H5T_NATIVE_DOUBLE);
	}
	writeAttribute(meas_id, &zInit, "zInitial",      H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &zFinl, "zFinal",        H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &nSteps,"nSteps",        H5T_NATIVE_INT);
	writeAttribute(meas_id, &cSteps,"Current step",  H5T_NATIVE_INT);

	/*	Create a group for specific header data	*/
	hid_t vGrp_id = H5Gcreate2(meas_id, "/potential", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	double shift    = axion->Saskia();//saxionshift(maa, llPhys, vqcdType);

	double indi3    = axion->BckGnd()->Indi3();
	double zthres   = axion->BckGnd()->ZThRes();
	double zrestore = axion->BckGnd()->ZRestore();
	double lz2e     = axion->BckGnd()->LamZ2Exp();
	double laam     = axion->LambdaP();

	writeAttribute(vGrp_id, &lStr,  "Lambda type",        attr_type);
	writeAttribute(vGrp_id, &LL,    "Lambda",             H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &lz2e,  "Lambda Z2 exponent", H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &laam,  "LambdaP",            H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &vStr,  "VQcd type",          attr_type);
	writeAttribute(vGrp_id, &vPQStr,"VPQ type",           attr_type);
	writeAttribute(vGrp_id, &dStr,  "Damping type",       attr_type);
	writeAttribute(vGrp_id, &rStr,  "Evolution type",     attr_type);
	writeAttribute(vGrp_id, &nQcd,  "nQcd",               H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &nQcdr, "nQcd2",              H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &gamma, "Gamma",              H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &shift, "Shift",              H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &indi3, "Indi3",              H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &zthres,"z Threshold",        H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &zrestore,"z Restore",        H5T_NATIVE_DOUBLE);

	H5Gclose(vGrp_id);

	hid_t icGrp_id = H5Gcreate2(meas_id, "/ic", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	switch (cType) {
		case	CONF_SMOOTH:
			sprintf(icStr, "Smooth");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &iter,  "Smoothing iterations", H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &alpha, "Smoothing constant",   H5T_NATIVE_DOUBLE);
			break;

		case	CONF_KMAX:
			sprintf(icStr, "kMax");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
			break;

		case	CONF_VILGOR:
			sprintf(icStr, "VilGor");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
			break;

		case	CONF_LOLA:
			sprintf(icStr, "Lola");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
			break;

		case	CONF_COLE:
			sprintf(icStr, "Cole");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
			break;


		case	CONF_TKACHEV:
			sprintf(icStr, "Tkachev");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
			break;

		case	CONF_THERMAL:
			sprintf(icStr, "Thermal");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kCrit, "Temperature",       H5T_NATIVE_DOUBLE);
			break;

		case	CONF_SPAX:
			sprintf(icStr, "Axion Spectrum");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
			// writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
			break;

		default:
		case	CONF_NONE:
			sprintf(icStr, "None");
	}

	switch (smvarType) {
		case	CONF_RAND:
			sprintf(smStr, "Random");
			break;

		case	CONF_STRINGXY:
			sprintf(smStr, "String XY");
			break;

		case	CONF_STRINGYZ:
			sprintf(smStr, "String YZ");
			break;

		case	CONF_MINICLUSTER:
			sprintf(smStr, "Minicluster");
			break;

		case	CONF_MINICLUSTER0:
			sprintf(smStr, "Minicluster 0");
			break;

		case	CONF_AXNOISE:
			sprintf(smStr, "Axion noise");
			break;

		case	CONF_SAXNOISE:
			sprintf(smStr, "Saxion noise");
			break;

		case	CONF_AX1MODE:
			sprintf(smStr, "Axion one mode");
			break;

		case	CONF_PARRES:
			sprintf(smStr, "Parametric Resonance");
			break;

		case	CONF_STRWAVE:
			sprintf(smStr, "String + wave");
			break;

		default:
			sprintf(smStr, "None");
			break;
	}

	writeAttribute(icGrp_id, &mode0, "Axion zero mode",    H5T_NATIVE_DOUBLE);
	writeAttribute(icGrp_id, &smStr, "Configuration type", attr_type);

	H5Gclose(icGrp_id);

	H5Tclose (attr_type);

	/*	Create plist for collective write	*/

	mlist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(mlist_id, H5FD_MPIO_COLLECTIVE);

	header = true;

	LogMsg (VERB_NORMAL, "Measurement file %s successfuly opened", base);

	return;
}




void	destroyMeas ()
{
	/*	Closes the currently opened file for measurements	*/
	LogMsg (VERB_NORMAL, "Closing measurement file...");LogFlush();


	if (opened) {
		LogMsg (VERB_PARANOID, "opened indeed");LogFlush();
		H5Pclose (mlist_id);
		LogMsg (VERB_PARANOID, "mlist_id closed");LogFlush();
		H5Fclose (meas_id);
		LogMsg (VERB_PARANOID, "meas_id closed");LogFlush();
	}

	opened = false;
	header = false;

	meas_id = -1;

	LogMsg (VERB_NORMAL, "Measurement file successfuly closed\n");LogFlush();
}

void	writeDensity	(Scalar *axion, MapType fMap, double eMax, double eMin)
{
	hid_t	totalSpace, chunk_id, group_id, sSet_id, sSpace, memSpace;
	hid_t	grp_id, datum;

	size_t	sBytes	 = 0;

	int myRank = commRank();

	const hsize_t maxD[1] = { H5S_UNLIMITED };
	unsigned char *eData = static_cast<unsigned char *>(axion->sData());

	Profiler &prof = getProfiler(PROF_HDF5);

	/*	Start profiling		*/
	LogMsg (VERB_NORMAL, "Writing density contrast data");
	prof.start();

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request. %d %d\n", header, opened);
		prof.stop();
		return;
	}

	/*	Create a group for density contrast data		*/
	auto status = H5Lexists (meas_id, "/energy", H5P_DEFAULT);	// Create group if it doesn't exists

	if (!status)
		group_id = H5Gcreate2(meas_id, "/energy", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(meas_id, "/energy", H5P_DEFAULT);	// Group exists
		} else {
			LogError ("Error: can't check whether group /energy/density exists");
			prof.stop();
			return;
		}
	}

	status = H5Lexists (meas_id, "/energy/density", H5P_DEFAULT);	// Create group if it doesn't exists

	if (!status)
		grp_id = H5Gcreate2(meas_id, "/energy/density", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			grp_id = H5Gopen2(meas_id, "/energy/density", H5P_DEFAULT);	// Group exists
		} else {
			LogError ("Error: can't check whether group /energy/density exists");
			prof.stop();
			return;
		}
	}

	uint rLz   = axion->rDepth();
	uint redlZ = axion->rTotalDepth();
	uint redlX = axion->rLength();

	hsize_t total = ((hsize_t) redlX)*((hsize_t) redlX)*((hsize_t) redlZ);
	hsize_t slab  = ((hsize_t) redlX)*((hsize_t) redlX);

	/*	Might be reduced	*/

	status = H5Aexists_by_name(grp_id, ".", "Size",  H5P_DEFAULT);

	if (status < 0)
		LogError ("Error: can't check attribute \"Size\" in file");
	else if (status == 0)
		writeAttribute(grp_id, &redlX, "Size",  H5T_NATIVE_UINT);

	status = H5Aexists_by_name(grp_id, ".", "Depth",  H5P_DEFAULT);

	if (status < 0)
		LogError ("Error: can't check attribute \"Depth\" in file");
	else if (status == 0)
		writeAttribute(grp_id, &redlZ, "Depth", H5T_NATIVE_UINT);

	/*	String metadata		*/
	writeAttribute(grp_id, &eMin, "Minimum energy", H5T_NATIVE_DOUBLE);
	writeAttribute(grp_id, &eMax, "Maximum energy", H5T_NATIVE_DOUBLE);

	/*	Create space for writing the raw data to disk with chunked access	*/
	if((totalSpace = H5Screate_simple(1, &total, maxD)) < 0) {	// Whole data
		LogError ("Fatal error H5Screate_simple");
		prof.stop();
		return;
	}

	/*	Set chunked access and dynamical compression	*/
	if ((chunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0) {
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		return;
	}

	if (H5Pset_chunk (chunk_id, 1, &slab) < 0) {
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		return;
	}
/*	Disabled, until it works properly
	if (H5Pset_deflate (chunk_id, 9) < 0) {	// Maximum compression
		LogError ("Error: couldn't set compression level to 9");
		prof.stop();
		return;
	}
*/
	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0) {
		LogError ("Fatal error H5Pset_alloc_time");
		prof.stop();
		return;
	}

	/*	Create a dataset for string data	*/
	if (fMap == MAP_RHO)
		sSet_id = H5Dcreate (grp_id, "cRho",   H5T_NATIVE_UCHAR, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);
	else
		sSet_id = H5Dcreate (grp_id, "cTheta", H5T_NATIVE_UCHAR, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

	if (sSet_id < 0) {
		LogError ("Fatal error creating dataset");
		prof.stop();
		return;
	}

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

	sSpace = H5Dget_space (sSet_id);
	memSpace = H5Screate_simple(1, &slab, NULL);	// Slab

	commSync();

	for (hsize_t zDim=0; zDim < rLz; zDim++)
	{
		/*	Select the slab in the file	*/
		hsize_t offset = ((hsize_t) (myRank*rLz) + zDim)*slab;
		H5Sselect_hyperslab(sSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

		/*	Write raw data	*/
		auto mErr = H5Dwrite (sSet_id, H5T_NATIVE_UCHAR, memSpace, sSpace, mlist_id, (eData)+slab*zDim);

		if (mErr < 0)
		{
			LogError ("Error writing dataset");
			prof.stop();
			exit(0);
		}
	}


	/*	Close the dataset	*/

	H5Dclose (sSet_id);
	H5Sclose (sSpace);
	H5Sclose (memSpace);

	H5Sclose (totalSpace);
	H5Pclose (chunk_id);

	sBytes = slab*rLz + 24;

	H5Gclose (grp_id);
	H5Gclose (group_id);

	prof.stop();
	prof.add(std::string("Write density"), 0, 1e-9*sBytes);

	LogMsg (VERB_NORMAL, "Written %lu bytes to disk", sBytes);
}




void	writeString	(Scalar *axion, StringData strDat, const bool rData)
{
	hid_t	totalSpace, chunk_id, group_id, sSet_id, sSpace, memSpace;
	hid_t	datum;

	size_t	sBytes	 = 0;

	int myRank = commRank();

	const hsize_t maxD[1] = { H5S_UNLIMITED };
	char *strData = static_cast<char *>(axion->sData());
	char sCh[16] = "/string/data";

	Profiler &prof = getProfiler(PROF_HDF5);

	/*	Start profiling		*/
	LogMsg (VERB_NORMAL, "Writing string data");
	prof.start();

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request. %d %d\n", header, opened);
		prof.stop();
		return;
	}

	/*	Create a group for string data		*/
	auto status = H5Lexists (meas_id, "/string", H5P_DEFAULT);	// Create group if it doesn't exists

	if (!status)
		group_id = H5Gcreate2(meas_id, "/string", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(meas_id, "/string", H5P_DEFAULT);	// Group exists, WTF
			LogMsg(VERB_NORMAL, "Warning: group /string exists!");	// Since this is weird, log it
		} else {
			LogError ("Error: can't check whether group /string exists");
			prof.stop();
			return;
		}
	}

	uint rLz   = axion->rDepth();
	uint redlZ = axion->rTotalDepth();
	uint redlX = axion->rLength();

	hsize_t total = ((hsize_t) redlX)*((hsize_t) redlX)*((hsize_t) redlZ);
	hsize_t slab  = ((hsize_t) redlX)*((hsize_t) redlX);

	/* We only write these if group does not exist, assuming they are there
	a better fix would be to modify the wA function to make a check */
	if (!(status > 0))
	{
		/*	Might be reduced	*/
		writeAttribute(group_id, &redlX, "Size",  H5T_NATIVE_UINT);
		writeAttribute(group_id, &redlZ, "Depth", H5T_NATIVE_UINT);

		/*	String metadata		*/
		writeAttribute(group_id, &(strDat.strDen), "String number",    H5T_NATIVE_HSIZE);
		writeAttribute(group_id, &(strDat.strChr), "String chirality", H5T_NATIVE_HSSIZE);
		writeAttribute(group_id, &(strDat.wallDn), "Wall number",      H5T_NATIVE_HSIZE);
		writeAttribute(group_id, &(strDat.strLen),  "String length",    H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &(strDat.strDeng), "String number with gamma",    H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &(strDat.strVel),  "String velocity",  H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &(strDat.strVel2), "String velocity squared",    H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &(strDat.strGam),  "String gamma",     H5T_NATIVE_DOUBLE);
	}

	if	(rData) {
		/*	Create space for writing the raw data to disk with chunked access	*/
		if((totalSpace = H5Screate_simple(1, &total, maxD)) < 0) {	// Whole data
			LogError ("Fatal error H5Screate_simple");
			prof.stop();
			return;
		}

		/*	Set chunked access and dynamical compression	*/
		if ((chunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0) {
			LogError ("Fatal error H5Pcreate");
			prof.stop();
			return;
		}

		if (H5Pset_chunk (chunk_id, 1, &slab) < 0) {
			LogError ("Fatal error H5Pset_chunk");
			prof.stop();
			return;
		}

		if (H5Pset_deflate (chunk_id, 9) < 0) {	// Maximum compression
			LogError ("Error: couldn't set compression level to 9");
			prof.stop();
			return;
		}

		/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
		if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0) {
			LogError ("Fatal error H5Pset_alloc_time");
			prof.stop();
			return;
		}

		/*	Create a dataset for string data	*/
		sSet_id = H5Dcreate (meas_id, sCh, H5T_NATIVE_CHAR, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

		if (sSet_id < 0) {
			LogError ("Fatal error creating dataset");
			prof.stop();
			return;
		}

		/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

		sSpace = H5Dget_space (sSet_id);
		memSpace = H5Screate_simple(1, &slab, NULL);	// Slab

		commSync();

		for (hsize_t zDim=0; zDim < rLz; zDim++)
		{
			/*	Select the slab in the file	*/
			hsize_t offset = ((hsize_t) (myRank*rLz) + zDim)*slab;
			H5Sselect_hyperslab(sSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

			/*	Write raw data	*/
			auto mErr = H5Dwrite (sSet_id, H5T_NATIVE_CHAR, memSpace, sSpace, mlist_id, (strData)+slab*zDim);

			if (mErr < 0)
			{
				LogError ("Error writing dataset");
				prof.stop();
				exit(0);
			}
		}


		/*	Close the dataset	*/

		H5Dclose (sSet_id);
		H5Sclose (sSpace);
		H5Sclose (memSpace);

		H5Sclose (totalSpace);
		H5Pclose (chunk_id);

		sBytes = slab*rLz + 24;
	} else
		sBytes = 24;

	H5Gclose (group_id);

	prof.stop();
	prof.add(std::string("Write strings"), 0, 1e-9*sBytes);

	LogMsg (VERB_NORMAL, "Written %lu bytes to disk", sBytes);
}

/* New strings */

// void	writeStringCo	(Scalar *axion, StringData strDat, const bool rData)	// TODO Terminar
// {
// 	LogMsg(VERB_NORMAL, "Not implemented (yet) for parallel HDF5. Ignoring request.");
// 	return;
// #if 0
// 	hid_t       dataset_id, dataspace_id;  /* identifiers */
// 	hsize_t     dims[1];
// 	herr_t      status;
//
// 	size_t	sBytes	 = 0;
//
// 	int myRank = commRank();
//
// 	uint nmax = 2*axion->Length();
//
// 	/* String data, different casts */
// 	char *strData;
//
// 	if (axion->LowMem())
// 		{
// 			if (axion->sDStatus() == SD_STRINGCOORD)
// 				strData = static_cast<char *>(axion->sData());
// 			else{
// 				printf("Return!"); // improve
// 				return;
// 			}
// 		}
// 	else
// 	{
// 		if (axion->m2Status() == M2_STRINGCOO)
// 				strData = static_cast<char *>(axion->m2Cpu());
// 			else{
// 				printf("Return!"); // improve
// 				return ;
// 			}
// 	}
//
// 	/* Number of strings per rank, initialise = 0 */
// 	size_t stringN[commSize()];
// 	for (int i=0;i<commSize();i++)
// 		stringN[i]=0;
//
// 	/*send stringN to rank 0*/
// 	MPI_Gather( &strDat.strDen_local , 1, MPI_UNSIGNED_LONG, &stringN, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
// 	commSync();
//
// 	/* Check the total number*/
// 	size_t toti = 0;
// 	for (int i=0;i<commSize();i++){
// 		toti += stringN[i];
// 	}
//
// 	hid_t sSpace;
// 	hid_t memSpace;
// 	hid_t group_id;
// 	hsize_t slab = 0;
//
// 	Profiler &prof = getProfiler(PROF_HDF5);
//
// 	/*	Start profiling		*/
// 	LogMsg (VERB_NORMAL, "Writing string coord.");
// 	prof.start();
//
// 	if (header == false || opened == false)
// 	{
// 		LogError ("Error: measurement file not opened. Ignoring write request. %d %d\n", header, opened);
// 		prof.stop();
// 		return;
// 	}
//
// 	/* Create a group for string data */
// 	auto status = H5Lexists (meas_id, "/string", H5P_DEFAULT);	// Create group if it doesn't exists
// 	if (!status)
// 		group_id = H5Gcreate2(meas_id, "/string", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
// 	else {
// 		if (status > 0) {
// 			group_id = H5Gopen2(meas_id, "/string", H5P_DEFAULT);	// Group exists, WTF
// 			LogMsg(VERB_NORMAL, "Warning: group /string exists!");	// Since this is weird, log it
// 		} else {
// 			LogError ("Error: can't check whether group /string exists");
// 		}
// 	}
//
// 	/*	Maximum coordinate is 2xN	(to avoid 1/2's)*/
// 	writeAttribute(group_id, &nmax, "nmax",  H5T_NATIVE_UINT);
//
// 	/*	String metadata		*/
// 	status = H5Aexists(group_id,"String number");
//
// 	if (!status) {
// 		writeAttribute(group_id, &(strDat.strDen), "String number",    H5T_NATIVE_HSIZE);
// 		writeAttribute(group_id, &(strDat.strChr), "String chirality", H5T_NATIVE_HSSIZE);
// 		writeAttribute(group_id, &(strDat.wallDn), "Wall number",      H5T_NATIVE_HSIZE);
// 	}
//
// 	/* if rData write the coordinates*/
// 	if (rData)
// 	{
// 		/* Total length of coordinates to write */
// 		dims[0] = 3*toti;
// 		/* Create the data space for the dataset. */
// 		dataspace_id = H5Screate_simple(1, dims, NULL);
// 		/* Create the dataset. */
// 		dataset_id = H5Dcreate2(meas_id, "/string/codata", H5T_NATIVE_USHORT, dataspace_id,
// 		            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//
// 		sSpace = H5Dget_space (dataset_id);
// 		memSpace = H5Screate_simple(1, &slab, NULL);
// 	}
//
// 	/* perhaps we need some cross-checks */
// 	/* Write the dataset. */
//
// 	if (rData) {
// 		int tSz = commSize(), test = myRank;
//
// 		commSync();
//
// 		for (int rank=0; rank<tSz; rank++)
// 		{
// 			/* Each rank selects a slab of its own size*/
// 			int tralara =(int) 3*strDat.strDen_local*sizeof(unsigned short);
//
// 			if (myRank != 0)
// 			{
// 				/* Only myRank >0 sends */
// 				if (myRank == rank){
// 				MPI_Send(&(strData[0]), tralara, MPI_CHAR, 0, rank, MPI_COMM_WORLD);
// 				}
// 			}
// 			else
// 			{
// 				/* Only  myRank 0 receives and writes */
// 				slab = 3*stringN[rank];
// 				tralara =(int) slab*sizeof(unsigned short);
//
// 				if (rank != 0)
// 					{
// 						MPI_Recv(&(strData[0]), tralara, MPI_CHAR, rank, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
// 					}
//
// 				/*	Select the slab in the file	*/
// 				toti=0;
// 				for (int i=0;i<rank;i++){
// 					toti += stringN[i];
// 				}
// 				hsize_t offset = ((hsize_t) 3*toti);
//
// 				/* update memSpace with new slab size	*/
// 				/* here one can partition in less than 2Gb if needed in future */
// 				memSpace = H5Screate_simple(1, &slab, NULL);
// 				H5Sselect_hyperslab(sSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
//
// 				/* write file */
// 				H5Dwrite (dataset_id, H5T_NATIVE_USHORT, memSpace, sSpace, H5P_DEFAULT, (void *) &(strData[0]) );
// 			}
//
// 			commSync();
// 		}
//
// 		// H5Dclose(sSpace);
// 		// H5Dclose(memSpace);
// 		H5Dclose(dataset_id);
// 		H5Sclose(dataspace_id);
//
// 		/* the 24 is copied from writeString ... probably a bit less */
// 		sBytes = 3*strDat.strDen*sizeof(unsigned short) + 24;
// 	} else
// 		sBytes = 24;
//
// 	H5Gclose (group_id);
//
// 	prof.stop();
// 	prof.add(std::string("Write string Coord."), 0, 1e-9*sBytes);
//
// 	LogMsg (VERB_NORMAL, "Written %lu bytes to disk", sBytes);
// 	commSync();
// #endif
// }

/* New strings non-parallel version */

void	writeStringCo	(Scalar *axion, StringData strDat, const bool rData)
{
	LogMsg(VERB_NORMAL, "[wsco] Not implemented (yet) for parallel HDF5. Continue with sequential HDF5.");

	hid_t       dataset_id, dataspace_id;  /* identifiers */
	hsize_t     dims[1];
	herr_t      status;

	size_t	sBytes	 = 0;

	int myRank = commRank();

	uint nmax = 2*axion->Length();

	/* String data, different casts */
	char *strData;

	if (axion->LowMem())
	{
		if (axion->sDStatus() == SD_STRINGCOORD)
			strData = static_cast<char *>(axion->sData());
		else {
			printf("Return!"); // improve
			return;
		}
	} else {
		if (axion->m2Status() == M2_STRINGCOO)
			strData = static_cast<char *>(axion->m2Cpu());
		else {
			printf("Return!"); // improve
			return ;
		}
	}

	/* Number of strings per rank, initialise = 0 */
	size_t stringN[commSize()];
	for (int i=0;i<commSize();i++)
		stringN[i]=0;

	/*send stringN to rank 0*/
	MPI_Allgather( &strDat.strDen_local , 1, MPI_UNSIGNED_LONG, &stringN, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
	commSync();

	/* Check the total number*/
	size_t toti = 0;
	for (int i=0;i<commSize();i++){
		toti += stringN[i];
	}

	hid_t sSpace;
	hid_t memSpace;
	hid_t group_id;
	hsize_t slab = 0;
	hsize_t offset = 0;

	Profiler &prof = getProfiler(PROF_HDF5);

	/*	Start profiling		*/
	LogMsg (VERB_NORMAL, "Writing string coord.");
	prof.start();

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request. %d %d", header, opened);
		prof.stop();
		return;
	}

	/* Create a group for string data */
	status = H5Lexists (meas_id, "/string", H5P_DEFAULT);	// Create group if it doesn't exists
	if (!status)
		group_id = H5Gcreate2(meas_id, "/string", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(meas_id, "/string", H5P_DEFAULT);	// Group exists, WTF
			LogMsg(VERB_NORMAL, "Warning: group /string exists!");	// Since this is weird, log it
		} else {
			LogError ("Error: can't check whether group /string exists");
		}
	}

	/*	Maximum coordinate is 2xN	(to avoid 1/2's)*/
	writeAttribute(group_id, &nmax, "nmax",  H5T_NATIVE_UINT);

	/*	String metadata		*/
	status = H5Aexists(group_id, "String number");

	if (status==0){
		writeAttribute(group_id, &(strDat.strDen),  "String number",    H5T_NATIVE_HSIZE);
		writeAttribute(group_id, &(strDat.strChr),  "String chirality", H5T_NATIVE_HSSIZE);
		writeAttribute(group_id, &(strDat.wallDn),  "Wall number",      H5T_NATIVE_HSIZE);
		writeAttribute(group_id, &(strDat.strLen),  "String length",    H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &(strDat.strDeng), "String number with gamma",    H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &(strDat.strVel),  "String velocity",  H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &(strDat.strVel2), "String velocity squared",    H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &(strDat.strGam),  "String gamma",     H5T_NATIVE_DOUBLE);
	}

	/* if rData write the coordinates*/
	if (rData)
	{
		/* Total length of coordinates to write */
		dims[0] = 3*toti;

		/* Create the data space for the dataset. */
		dataspace_id = H5Screate_simple(1, dims, NULL);
		/* Create the dataset. */
		dataset_id = H5Dcreate(group_id, "codata", H5T_NATIVE_USHORT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		sSpace = H5Dget_space (dataset_id);

		int tSz = commSize(), test = myRank;

		commSync();

		for (int rank=0; rank<tSz; rank++)
		{
			/* Each rank selects a slab of its own size*/
			int tralara =(int) 3*strDat.strDen_local*sizeof(unsigned short);

			if (myRank != 0)
			{
				/* Only myRank >0 sends */
				if (myRank == rank){
					MPI_Send(&(strData[0]), tralara, MPI_CHAR, 0, rank, MPI_COMM_WORLD);
				}
			}
			else  /* Only  myRank 0 receives and writes */
			{
				slab = 3*stringN[rank];
				tralara =(int) slab*sizeof(unsigned short);

				if (rank != 0)
					MPI_Recv(&(strData[0]), tralara, MPI_CHAR, rank, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				/*	Select the slab in the file	*/
				toti=0;
				for (int i=0;i<rank;i++)
					toti += stringN[i];

				offset = ((hsize_t) 3*toti);
			}
			/* update memSpace with new slab size	*/
			/* here one can partition in less than 2Gb if needed in future */

			memSpace = H5Screate_simple(1, &slab, NULL);
			if (myRank == 0) {
				H5Sselect_hyperslab(sSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
			} else {
				H5Sselect_none(memSpace);
				H5Sselect_none(sSpace);
			}

			/* write file */
			H5Dwrite (dataset_id, H5T_NATIVE_USHORT, memSpace, sSpace, H5P_DEFAULT, (void *) &(strData[0]) );
			H5Sclose(memSpace);
		}

		commSync();

		H5Sclose(sSpace);
		H5Dclose(dataset_id);
		H5Sclose(dataspace_id);

		sBytes = 3*strDat.strDen*sizeof(unsigned short) + 64;
	} else
		sBytes = 64;

	H5Gclose (group_id);

	prof.stop();
	prof.add(std::string("Write string Coord."), 0, 1e-9*sBytes);

	LogMsg (VERB_NORMAL, "Written %lu bytes to disk", sBytes);
	commSync();
}





void	writeStringEnergy	(Scalar *axion, StringEnergyData strEDat)
{
	hid_t	group_id;

	bool	mpiCheck = true;
	size_t	sBytes	 = 0;

	int myRank = commRank();

	Profiler &prof = getProfiler(PROF_HDF5);

	/*	Start profiling		*/
	LogMsg (VERB_NORMAL, "Writing string energy");
	prof.start();

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request. %d %d\n", header, opened);
		prof.stop();
		return;
	}

	/*	Create a group for string data		*/
	char LABEL[256];
	sprintf(LABEL, "string/rmask_%.2f", strEDat.rmask);

	auto status = H5Lexists (meas_id, LABEL, H5P_DEFAULT);	// Create group if it doesn't exists
	if (!status)
	{
		group_id = H5Gcreate2(meas_id, LABEL, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	}
	else {
		if (status > 0) {
			group_id = H5Gopen2(meas_id, LABEL, H5P_DEFAULT);	// Group exists, perhaps already created in writeString(Co)
		  //LogMsg(VERB_NORMAL, "Warning: group /string exists!");
		} else {
			LogError ("Error: can't check whether group /string exists");
			return;
		}
	}
	/*	write string energy density	*/
	writeAttribute(group_id, &(strEDat.rho_str),    "String energy density",             H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &(strEDat.rho_a),      "Masked axion energy density",       H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &(strEDat.rho_s),      "Masked saxion energy density",      H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &(strEDat.rho_str_Vil),"String energy density (Vil)",       H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &(strEDat.rho_a_Vil),  "Masked axion energy density (Vil)", H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &(strEDat.rho_s_Vil),  "Masked saxion energy density (Vil)",H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &(strEDat.nout),       "nout",                              H5T_NATIVE_HSIZE);
	writeAttribute(group_id, &(strEDat.rmask),      "rmask",                             H5T_NATIVE_DOUBLE);

	sBytes = 56;

	H5Gclose (group_id);

	prof.stop();
	prof.add(std::string("Write string energy density"), 0, 1e-9*sBytes);

	LogMsg (VERB_NORMAL, "Written %lu bytes to disk", sBytes);

	commSync();
}





void	writeEnergy	(Scalar *axion, void *eData_, double rmask)
{

	LogMsg(VERB_NORMAL,"[wEn] Write Energy %f ",rmask);
	bool pe = false ;
	bool pmaskede = false;

	/* At the moment we do not save both masked and unmasked energy at the same time */
	if (rmask < 0.){
			LogMsg(VERB_NORMAL,"[wEn] Saving energy to disk ");
			pe = true;
	}
	if (rmask > 0.){
			LogMsg(VERB_NORMAL,"[wEn] Saving energy - masked to disk (rmask = %.3f)",rmask);
			pmaskede = true;
	}

	char LABEL[256];

	hid_t	group_id, group_id2;

	if (!pe && !pmaskede){
		LogMsg(VERB_NORMAL,"[wEn] Called but nothing to do: Exit! ");
		return;
	}

	double	*eData = static_cast<double *>(eData_);

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request. %d %d", header, opened);
		return;
	}

	/*	Start profiling		*/

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	/*	Create a group for string data if it doesn't exist	*/
	auto status = H5Lexists (meas_id, "/energy", H5P_DEFAULT);	// Create group if it doesn't exists

	if (!status)
		group_id = H5Gcreate2(meas_id, "/energy", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(meas_id, "/energy", H5P_DEFAULT);	// Group exists, so we open it
		} else {
			LogError ("Error: can't check whether group /energy exists");
			prof.stop();
			return;
		}
	}
	// The masked energy goes into another folder with label
	if (pmaskede)
	{
		sprintf(LABEL, "Redmask_%.2f", rmask);
		auto status2 = H5Lexists (group_id, LABEL, H5P_DEFAULT);	// Create group if it doesn't exists

		if (!status2)
			group_id2 = H5Gcreate2(group_id, LABEL, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		else {
			if (status > 0) {
				group_id2 = H5Gopen2(group_id, LABEL, H5P_DEFAULT);		// Group exists, but it shouldn't
				LogMsg(VERB_NORMAL, "Warning: group %s exists!", LABEL);	// Since this is weird, log it
			} else {
				LogError ("Error: can't check whether group %s exists", LABEL);
				return;
			}
		}
	}

	int totalBytes = 0 ;

	if (pe){
		writeAttribute(group_id, &eData[TH_GRX],  "Axion Gr X",      H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &eData[TH_GRY],  "Axion Gr Y",      H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &eData[TH_GRZ],  "Axion Gr Z",      H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &eData[TH_POT],  "Axion Potential", H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &eData[TH_KIN],  "Axion Kinetic",   H5T_NATIVE_DOUBLE);
		totalBytes += 40;
		if (axion->Field() == FIELD_PAXION )//|| axion->Field() == FIELD_NAXION)
		{
			writeAttribute(group_id, &eData[TH_POT],  "Paxion Potential (SI)", H5T_NATIVE_DOUBLE);
			writeAttribute(group_id, &eData[TH_KIN],  "Paxion Number density", H5T_NATIVE_DOUBLE);
		}


		if	(axion->Field() == FIELD_SAXION)
		{
			writeAttribute(group_id, &eData[RH_GRX],  "Saxion Gr X",      H5T_NATIVE_DOUBLE);
			writeAttribute(group_id, &eData[RH_GRY],  "Saxion Gr Y",      H5T_NATIVE_DOUBLE);
			writeAttribute(group_id, &eData[RH_GRZ],  "Saxion Gr Z",      H5T_NATIVE_DOUBLE);
			writeAttribute(group_id, &eData[RH_POT],  "Saxion Potential", H5T_NATIVE_DOUBLE);
			writeAttribute(group_id, &eData[RH_RHO],  "Saxion vev",       H5T_NATIVE_DOUBLE);
			writeAttribute(group_id, &eData[RH_KIN],  "Saxion Kinetic",   H5T_NATIVE_DOUBLE);

			totalBytes += 48;
		}
	}

	if (pmaskede){
		totalBytes += 40;
		writeAttribute(group_id2, &eData[TH_GRXM],  "Axion Gr X nMask",      H5T_NATIVE_DOUBLE);
		writeAttribute(group_id2, &eData[TH_GRYM],  "Axion Gr Y nMask",      H5T_NATIVE_DOUBLE);
		writeAttribute(group_id2, &eData[TH_GRZM],  "Axion Gr Z nMask",      H5T_NATIVE_DOUBLE);
		writeAttribute(group_id2, &eData[TH_POTM],  "Axion Potential nMask", H5T_NATIVE_DOUBLE);
		writeAttribute(group_id2, &eData[TH_KINM],  "Axion Kinetic nMask",   H5T_NATIVE_DOUBLE);

		if	(axion->Field() == FIELD_SAXION)
		{
			writeAttribute(group_id2, &eData[RH_GRXM],  "Saxion Gr X nMask",      H5T_NATIVE_DOUBLE);
			writeAttribute(group_id2, &eData[RH_GRYM],  "Saxion Gr Y nMask",      H5T_NATIVE_DOUBLE);
			writeAttribute(group_id2, &eData[RH_GRZM],  "Saxion Gr Z nMask",      H5T_NATIVE_DOUBLE);
			writeAttribute(group_id2, &eData[RH_POTM],  "Saxion Potential nMask", H5T_NATIVE_DOUBLE);
			writeAttribute(group_id2, &eData[RH_KINM],  "Saxion Kinetic nMask",   H5T_NATIVE_DOUBLE);

			writeAttribute(group_id2, &eData[RH_RHOM],  "Saxion vev nMask",       H5T_NATIVE_DOUBLE);
			writeAttribute(group_id2, &eData[MM_NUMM],  "Number of masked points",H5T_NATIVE_DOUBLE);

			totalBytes += 48;
		}
	}

	/*	TODO	Distinguish the different versions of the potentials	*/

	/*	Close the group		*/
	H5Gclose (group_id);
	if (pmaskede)
		H5Gclose (group_id2);

	prof.stop();
	prof.add(std::string("Write energy"), 0, 1e-9*totalBytes);
}

void	writePoint (Scalar *axion)	// NO PROFILER YET
{
	hid_t	group_id, dataSpace, sSet_id, sSpace, dataSet, dataType;
	hsize_t dims[1];

	size_t	dataSize = axion->DataSize(), S0 = axion->Surf();

	auto myRank = commRank();

	LogMsg (VERB_NORMAL, "Writing single point data to measurement file");

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request. %d %d", header, opened);
		return;
	}

	/*	Create a group for point data if it doesn't exist	*/
	auto status = H5Lexists (meas_id, "/point", H5P_DEFAULT);	// Create group if it doesn't exists

	if (!status)
		group_id = H5Gcreate2(meas_id, "/point", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(meas_id, "/point", H5P_DEFAULT);	// Group exists, but it shouldn't
			LogMsg(VERB_NORMAL, "Warning: group /string exists!");	// Since this is weird, log it
		} else {
			LogError ("Error: can't check whether group /point exists");
			return;
		}
	}

	/*	Create minidataset	*/
	if (axion->Precision() == FIELD_DOUBLE)
	{
			dataType = H5T_NATIVE_DOUBLE;
			dims[0]	 = dataSize/8;
	} else {
			dataType = H5T_NATIVE_FLOAT;
			dims[0]	 = dataSize/4;
	}

	dataSpace = H5Screate_simple(1, dims, NULL);
	dataSet	  = H5Dcreate(group_id, "value", dataType, dataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	sSpace	  = H5Dget_space (dataSet);

	if (myRank == 0) {
		hsize_t offset = 0;
		H5Sselect_hyperslab(sSpace, H5S_SELECT_SET, &offset, NULL, dims, NULL);
	} else {
		H5Sselect_none(dataSpace);
		H5Sselect_none(sSpace);
	}

	/*	Write point data	*/
	if (H5Dwrite(dataSet, dataType, dataSpace, sSpace, H5P_DEFAULT, static_cast<char*>(axion->mStart())) < 0)
		LogError ("Error: couldn't write point data to file");

	/*	Close everything		*/
	H5Sclose (sSpace);
	H5Dclose (dataSet);
	H5Sclose (dataSpace);
	H5Gclose (group_id);

	LogMsg (VERB_NORMAL, "Written %lu bytes", dataSize);
}

void	writeArray (double *aData, size_t aSize, const char *group, const char *dataName, int rango)
{
	hid_t	group_id, base_id, dataSpace, sSpace, dataSet;
	hsize_t dims[1] = { aSize };

	size_t	dataSize;

	LogMsg (VERB_NORMAL, "Writing array to measurement file");

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request.\n");
		return;
	}

	/*	Start profiling		*/

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	auto myRank = commRank();

	/*	Create the group for the data if it doesn't exist	*/
	auto status = H5Lexists (meas_id, group, H5P_DEFAULT);	// Create group if it doesn't exists

	if (!status)
		base_id = H5Gcreate2(meas_id, group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0)
			base_id = H5Gopen2(meas_id, group, H5P_DEFAULT);		// Group exists
		else {
			LogError ("Error: can't check whether group %s exists", group);
			return;
		}
	}

	status = H5Lexists (base_id, dataName, H5P_DEFAULT);	// Create group if it doesn't exists

	if (!status)
		group_id = H5Gcreate2(base_id, dataName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(base_id, dataName, H5P_DEFAULT);		// Group exists, but it shouldn't
			LogMsg(VERB_NORMAL, "Warning: group %s exists!", dataName);	// Since this is weird, log it
		} else {
			LogError ("Error: can't check whether group %s exists", dataName);
			return;
		}
	}
/*
	disableErrorStack();

	if (H5Gget_objinfo (meas_id, group, 0, NULL) < 0)	// Create group if it doesn't exists
		group_id = H5Gcreate2(meas_id, group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else
		group_id = H5Gopen2(meas_id, group, H5P_DEFAULT);

	enableErrorStack();
*/
	/*	Create dataset	*/
	dataSpace = H5Screate_simple(1, dims, NULL);
	dataSet   = H5Dcreate(group_id, "data", H5T_NATIVE_DOUBLE, dataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	sSpace	  = H5Dget_space (dataSet);

	if (myRank == rango) {
		hsize_t offset = 0;
		H5Sselect_hyperslab(sSpace, H5S_SELECT_SET, &offset, NULL, dims, NULL);
	} else {
		H5Sselect_none(dataSpace);
		H5Sselect_none(sSpace);
	}

	/*	Write spectrum data	*/
	if (H5Dwrite(dataSet, H5T_NATIVE_DOUBLE, dataSpace, sSpace, H5P_DEFAULT, aData) < 0) {
		LogError ("Error writing %lu bytes to dataset", aSize*8);
		prof.stop();
		return;
	}

	/*	Close everything		*/
	H5Sclose (sSpace);
	H5Dclose (dataSet);
	H5Sclose (dataSpace);
	H5Gclose (group_id);
	H5Gclose (base_id);

	prof.stop();
	prof.add(std::string("Write array"), 0, 1e-9*(aSize*8));

	LogMsg (VERB_NORMAL, "Written %lu bytes to disk", aSize*8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

void	writeEDens (Scalar *axion, MapType fMap)
{
	hid_t	eGrp_id, group_id, rset_id, tset_id, aset_id, plist_id, chunk_id;
	hid_t	rSpace, tSpace, aSpace, memSpace, dataType, totalSpace;
	hsize_t	total, slice, slab, offset, rOff;

	char	prec[16], fStr[16];
	int	length = 8;

	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize;

	int myRank = commRank();

	LogMsg (VERB_NORMAL, "[wEd] Writing energy density to Hdf5 measurement file");
	LogMsg (VERB_NORMAL, "");

	if ((fMap & MAP_RHO) && (axion->Field() & FIELD_AXION)) {
	        LogMsg (VERB_NORMAL, "[wEd] Requested MAP_RHO with axion field. Request will be ignored");
	        fMap ^= MAP_RHO;
	}

	if ((fMap & MAP_ALL) == MAP_NONE) {
	        LogMsg (VERB_NORMAL, "[wEd] Nothing to map. Skipping writeEDens");
	        return;
	}

	/*      Start profiling         */

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	/*      If needed, transfer data to host        */
	if (axion->Device() == DEV_GPU)
		axion->transferCpu(FIELD_M2);

	if (axion->m2Cpu() == nullptr) {
		LogError ("[wEd] You seem to be using the lowmem option");
		prof.stop();
		return;
	}

	if (header == false || opened == false)
	{
		LogError ("[wEd] Error: measurement file not opened. Ignoring write request.\n");
		return;
	}

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

		LogError ("[wEd] Error: Invalid precision. How did you get this far?");
		exit(1);

		break;
	}

	//uint totlZ = sizeZ*zGrid;
	//uint tmpS  = sizeN;

	//total = ((hsize_t) tmpS)*((hsize_t) tmpS)*((hsize_t) (totlZ));
	//slab  = (hsize_t) axion->Surf();
	uint totlZ = axion->TotalDepth();
	uint totlX = axion->Length();
	uint redlZ = axion->rTotalDepth();
	uint redlX = axion->rLength();

	total = ((hsize_t) redlX)*((hsize_t) redlX)*((hsize_t) redlZ);
	slab  = ((hsize_t) redlX)*((hsize_t) redlX);
	rOff  = ((hsize_t) (totlX))*((hsize_t) (totlX))*(axion->Depth()+2*axion->getNg());

	if (axion->Field() == FIELD_WKB) {
		LogError ("Error: WKB field not supported");
		prof.stop();
		exit(1);
	}

	/*	Create space for writing the raw data to disk with chunked access	*/
	if ((totalSpace = H5Screate_simple(1, &total, maxD)) < 0)	// Whole data
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

	if (H5Pset_chunk (chunk_id, 1, &slab) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_alloc_time");
		prof.stop();
		exit (1);
	}

	/*	Create a group for energy data if it doesn't exist	*/
	auto status = H5Lexists (meas_id, "/energy", H5P_DEFAULT);

	if (!status)
		eGrp_id = H5Gcreate2(meas_id, "/energy", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			eGrp_id = H5Gopen2(meas_id, "/energy", H5P_DEFAULT);		// Group exists
		} else {
			LogError ("Error: can't check whether group /energy exists");
			prof.stop();
			return;
		}
	}

	/*	Create a group for energy density if it doesn't exist
	we call it density for the full resolution and rdensity for the reduced resolution	*/
	char *gr_name;
	if (axion->Reduced())
		gr_name = "rdensity";
	else
		gr_name = "density";

	status = H5Lexists (eGrp_id, gr_name, H5P_DEFAULT);

	if (!status)
		group_id = H5Gcreate2(eGrp_id, gr_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(eGrp_id, gr_name, H5P_DEFAULT);		// Group exists
			LogMsg (VERB_HIGH, "Group /energy/%s exists",gr_name);
		} else {
			LogError ("Error: can't check whether group /energy/%s exists",gr_name);
			prof.stop();
			return;
		}
	}

	/*	Might be reduced	*/
	writeAttribute(group_id, &redlX, "Size",  H5T_NATIVE_UINT);
	writeAttribute(group_id, &redlZ, "Depth", H5T_NATIVE_UINT);

	/*	Create a dataset for the whole axion data	*/

	char rhoCh[24];
	sprintf (rhoCh, "/energy/%s/rho", gr_name);
	char thCh[24];
	sprintf (thCh, "/energy/%s/theta", gr_name);
	char auCh[24];
	sprintf (auCh, "/energy/%s/aux", gr_name);

	if (fMap & MAP_RHO) {
		rset_id = H5Dcreate (meas_id, rhoCh, dataType, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

		if (rset_id < 0)
		{
			LogError("Error creating rho dataset");
			prof.stop();
			exit (0);
		}

		rSpace = H5Dget_space (rset_id);
	}

	if (fMap & MAP_THETA) {
		tset_id = H5Dcreate (meas_id, thCh, dataType, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

		if (tset_id < 0)
		{
			LogError("Error creating theta dataset");
			prof.stop();
			exit (0);
		}

		tSpace = H5Dget_space (tset_id);
	}

	if (fMap & MAP_M2S) {
		aset_id = H5Dcreate (meas_id, auCh, dataType, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

		if (aset_id < 0)
		{
			LogError("Error creating aux dataset");
			prof.stop();
			exit (0);
		}

		aSpace = H5Dget_space (aset_id);
	}
	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

	memSpace = H5Screate_simple(1, &slab, NULL);	// Slab

	commSync();

	LogMsg (VERB_HIGH, "Rank %d ready to write", myRank);

	const hsize_t Lz = axion->rDepth();

	if (fMap & MAP_RHO) {
		for (hsize_t zDim = 0; zDim < Lz; zDim++)
		{
			/*	Select the slab in the file	*/
			offset = (((hsize_t) (myRank*Lz)) + zDim)*slab;
			H5Sselect_hyperslab(rSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

			/*	Write raw data	*/
			auto rErr = H5Dwrite (rset_id, dataType, memSpace, rSpace, mlist_id, (static_cast<char *> (axion->m2Cpu())+(slab*zDim+rOff)*dataSize));

			if (rErr < 0)
			{
				LogError ("Error writing rho dataset");
				prof.stop();
				exit(0);
			}
		}

		commSync();
	}

	if (fMap & MAP_THETA) {
		for (hsize_t zDim = 0; zDim < Lz; zDim++)
		{
			/*	Select the slab in the file	*/
			offset = (((hsize_t) (myRank*Lz)) + zDim)*slab;
			H5Sselect_hyperslab(tSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

			/*	Write raw data	*/
			auto tErr = H5Dwrite (tset_id, dataType, memSpace, tSpace, mlist_id, (static_cast<char *> (axion->m2Cpu())+slab*zDim*dataSize));

			if (tErr < 0)
			{
				LogError ("Error writing theta dataset");
				prof.stop();
				exit(0);
			}
		}

		commSync();
	}

	if (fMap & MAP_M2S) {
		for (hsize_t zDim = 0; zDim < Lz; zDim++)
		{
			/*	Select the slab in the file	*/
			offset = (((hsize_t) (myRank*Lz)) + zDim)*slab;
			H5Sselect_hyperslab(aSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

			/*	Write raw data	*/
			auto tErr = H5Dwrite (aset_id, dataType, memSpace, aSpace, mlist_id, (static_cast<char *> (axion->m2Start())+slab*zDim*dataSize));

			if (tErr < 0)
			{
				LogError ("Error writing theta dataset");
				prof.stop();
				exit(0);
			}
		}

		commSync();
	}

	LogMsg (VERB_HIGH, "Write energy map successful");

	size_t bytes = 0;

	/*	Close the dataset	*/

	if (fMap & MAP_RHO) {
		H5Dclose (rset_id);
		H5Sclose (rSpace);
		bytes += total*dataSize;
	}

	if (fMap & MAP_THETA) {
		H5Dclose (tset_id);
		H5Sclose (tSpace);
		bytes += total*dataSize;
	}

	if (fMap & MAP_M2S) {
		H5Dclose (aset_id);
		H5Sclose (aSpace);
		bytes += total*dataSize;
	}

	H5Sclose (memSpace);

	/*	Close the file		*/

	H5Sclose (totalSpace);
	H5Pclose (chunk_id);
	H5Gclose (group_id);
	H5Gclose (eGrp_id);

        prof.stop();
	prof.add(std::string("Write energy map"), 0., ((double) bytes)*1e-9);

	LogMsg (VERB_NORMAL, "Written %lu bytes", bytes);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

void	writeEDensReduced (Scalar *axion, int index, int newNx, int newNz)
{
	hid_t	group_id, mset_id, plist_id, chunk_id;
	hid_t	mSpace, memSpace, dataType, totalSpace;
	hsize_t	total, slice, slab, newslab, offset;

	char	prec[16], fStr[16];
	int	length = 8;

	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize;

	int myRank = commRank();

	LogMsg (VERB_NORMAL, "[wEDR] Called filter with M2 status %d", axion->m2Status());
	LogMsg (VERB_NORMAL, "Writing energy density REDUCED to Hdf5 measurement file");
	LogMsg (VERB_NORMAL, "");

	/*      Start profiling         */

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	if (axion->m2Cpu() == nullptr) {
		LogError ("You seem to be using the lowmem option");
		prof.stop();
		return;
	}

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request.\n");
		return;
	}

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

	int cSteps = dump*index;
	uint totlZ = newNz*zGrid;
	uint tmpS  = newNx;

	total = ((hsize_t) tmpS)*((hsize_t) tmpS)*((hsize_t) totlZ);
	slab  = (hsize_t) axion->Surf();
	newslab  = (hsize_t) newNx * newNx;

	/*	Create space for writing the raw data to disk with chunked access	*/
	if ((totalSpace = H5Screate_simple(1, &total, maxD)) < 0)	// Whole data
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

	if (H5Pset_chunk (chunk_id, 1, &slab) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_alloc_time");
		prof.stop();
		exit (1);
	}

	/*	Create a group for energy data if it doesn't exist	*/
	auto status = H5Lexists (meas_id, "/energy", H5P_DEFAULT);

	if (!status)
		group_id = H5Gcreate2(meas_id, "/energy", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(meas_id, "/energy", H5P_DEFAULT);		// Group exists
			LogMsg (VERB_HIGH, "Group /energy exists");
		} else {
			LogError ("Error: can't check whether group /energy exists");
			prof.stop();
			return;
		}
	}

	/*	Create a dataset for the whole reduced contrast data	*/

	char mCh[24] = "/energy/redensity";

	mset_id = H5Dcreate (meas_id, mCh, dataType, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

	commSync();

	if (mset_id < 0)
	{
		LogError("Error creating dataset");
		prof.stop();
		exit (0);
	}

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

	mSpace = H5Dget_space (mset_id);
	memSpace = H5Screate_simple(1, &newslab, NULL);	// Slab

	commSync();

	LogMsg (VERB_HIGH, "Rank %d ready to write", myRank);

	//for (hsize_t zDim=0; zDim<((hsize_t) axion->Depth()); zDim++)
	for (hsize_t zDim=0; zDim<((hsize_t) newNz); zDim++)
	{
		/*	Select the slab in the file	*/
		offset = (((hsize_t) (myRank*newNz)) + zDim)*newslab;
		H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &newslab, NULL);

		/*	Write raw data	*/
		auto mErr = H5Dwrite (mset_id, dataType, memSpace, mSpace, mlist_id, (static_cast<char *> (axion->m2Cpu())+slab*dataSize + newslab*(zDim)*dataSize));
			//                    m2Cpu          + ghost bytes  +
		if (mErr < 0)
		{
			LogError ("Error writing dataset");
			prof.stop();
			exit(0);
		}

		//commSync();
	}

	/*	Close the dataset	*/

	H5Dclose (mset_id);
	H5Sclose (mSpace);
	H5Sclose (memSpace);

	/*	Close the file		*/

	H5Sclose (totalSpace);
	H5Pclose (chunk_id);
	H5Pclose (plist_id);
	H5Gclose (group_id);

        prof.stop();
	prof.add(std::string("Write energy map"), 0., (2.*total*dataSize + 78.)*1e-9);

	LogMsg (VERB_HIGH, "Write reduced contrast map successful");
	LogMsg (VERB_NORMAL, "Written %lu bytes", total*dataSize*2 + 78);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

/* This function must be updated to take an object of the SpecBin class */

void	writeSpectrum (Scalar *axion, void *spectrumK, void *spectrumG, void *spectrumV, size_t powMax, bool power)
{
	hid_t	group_id, dataSpace, kSpace, gSpace, vSpace, dataSetK, dataSetG, dataSetV;
	hsize_t dims[1] = { powMax };

	char	dataName[32];
	char	*sK = static_cast<char*>(spectrumK);
	char	*sG = static_cast<char*>(spectrumG);
	char	*sV = static_cast<char*>(spectrumV);

	LogMsg (VERB_NORMAL, "Writing spectrum to Hdf5 measurement file");
	LogMsg (VERB_NORMAL, "");

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request");
		return;
	}

	/*      Start profiling         */

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	if (power == true) {
		LogMsg (VERB_HIGH, "Writing power spectrum");
		sprintf(dataName, "/pSpectrum");
	} else {
		LogMsg (VERB_HIGH, "Writing number spectrum");
		sprintf(dataName, "/nSpectrum");
	}

	/*	Create a group for the spectra if it doesn't exist	*/
	auto status = H5Lexists (meas_id, dataName, H5P_DEFAULT);

	if (!status)
		group_id = H5Gcreate2(meas_id, dataName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(meas_id, dataName, H5P_DEFAULT);		// Group exists
			LogMsg (VERB_HIGH, "Group %s exists", dataName);
		} else {
			LogError ("Error: can't check whether group %s exists", dataName);
			prof.stop();
			return;
		}
	}

	/*	Create datasets	*/
	if ((dataSpace = H5Screate_simple(1, dims, NULL)) < 0) {
		LogError ("Fatal error in H5Screate_simple");
		prof.stop();
		exit(1);
	}

	dataSetK  = H5Dcreate(group_id, "sK", H5T_NATIVE_DOUBLE, dataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataSetG  = H5Dcreate(group_id, "sG", H5T_NATIVE_DOUBLE, dataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataSetV  = H5Dcreate(group_id, "sV", H5T_NATIVE_DOUBLE, dataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	if ((dataSetK < 0) || (dataSetG < 0) || (dataSetV < 0)) {
		LogError ("Fatal error creating datasets");
		prof.stop();
		exit(1);
	}

	kSpace	  = H5Dget_space (dataSetK);
	gSpace	  = H5Dget_space (dataSetG);
	vSpace	  = H5Dget_space (dataSetV);

	if ((kSpace < 0) || (gSpace < 0) || (vSpace < 0)) {
		LogError ("Fatal error in H5Dget_space");
		prof.stop();
		exit(1);
	}

	/*	Write spectrum data	*/
	if (H5Dwrite(dataSetK, H5T_NATIVE_DOUBLE, dataSpace, kSpace, H5P_DEFAULT, sK) < 0) {
		LogError ("Error writing dataset sK");
		prof.stop();
		exit(1);
	}

	if (H5Dwrite(dataSetG, H5T_NATIVE_DOUBLE, dataSpace, gSpace, H5P_DEFAULT, sG) < 0) {
		LogError ("Error writing dataset sG");
		prof.stop();
		exit(1);
	}

	if (H5Dwrite(dataSetV, H5T_NATIVE_DOUBLE, dataSpace, vSpace, H5P_DEFAULT, sV) < 0) {
		LogError ("Error writing dataset sV");
		prof.stop();
		exit(1);
	}

	LogMsg (VERB_HIGH, "Write spectrum successful");

	/*	Close everything		*/
	H5Sclose (kSpace);
	H5Sclose (gSpace);
	H5Sclose (vSpace);
	H5Dclose (dataSetK);
	H5Dclose (dataSetG);
	H5Dclose (dataSetV);
	H5Sclose (dataSpace);
	H5Gclose (group_id);
	prof.stop();

	prof.add(std::string("Write spectrum"), 0, 24e-9*(powMax));
	LogMsg (VERB_NORMAL, "Written %lu bytes", powMax*24);
}





void	writeMapHdf5s	(Scalar *axion, int slicenumbertoprint)
{
	hid_t	mapSpace, chunk_id, group_id, mSet_id, vSet_id, mSpace, vSpace,  dataType;
	hsize_t	dataSize = axion->DataSize();

	int myRank = commRank();

	const hsize_t maxD[1] = { H5S_UNLIMITED };
	hsize_t slb  = axion->Surf();
	hsize_t lSz  = axion->Length();
	char *dataM  = static_cast<char *>(axion->mFrontGhost());
	char *dataV  = static_cast<char *>(axion->mBackGhost());
	char mCh[16] = "/map/m";
	char vCh[16] = "/map/v";

	LogMsg (VERB_NORMAL, "Writing 2D maps to Hdf5 measurement file");LogFlush();
	LogMsg (VERB_NORMAL, "");LogFlush();

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request");
		return;
	}

	/*	Start profiling		*/

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	if ( (axion->Field() == FIELD_SAXION) || (axion->Field() == FIELD_NAXION)) {
		lSz *= 2;
		slb *= 2;
	}

	if (axion->Precision() == FIELD_DOUBLE) {
		dataType = H5T_NATIVE_DOUBLE;
	} else {
		dataType = H5T_NATIVE_FLOAT;
	}

	/*	Unfold field before writing configuration	*/
	//if (axion->Folded())
	//{
		int slicenumber = slicenumbertoprint ;
		if (slicenumbertoprint > axion->Depth())
		{
			LogMsg (VERB_NORMAL, "Sliceprintnumberchanged to 0");
			slicenumber = 0;
		}
		Folder	munge(axion);
		LogMsg (VERB_NORMAL, "If configuration folded, unfold 2D slice");LogFlush();
		munge(UNFOLD_SLICE, slicenumber);
	//}

	/*	Create a group for map data if it doesn't exist	*/
	auto status = H5Lexists (meas_id, "/map", H5P_DEFAULT);

	if (!status)
		group_id = H5Gcreate2(meas_id, "/map", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(meas_id, "/map", H5P_DEFAULT);		// Group exists
			LogMsg (VERB_HIGH, "Group /map exists");
		} else {
			LogError ("Error: can't check whether group /map exists");
			prof.stop();
			return;
		}
	}

	/*	Create space for writing the raw data to disk with chunked access	*/
	if ((mapSpace = H5Screate_simple(1, &slb, maxD)) < 0)	// Whole data
	{
		LogError ("Fatal error H5Screate_simple");
		prof.stop();
		exit (1);
	}

	if (myRank != 0) {
		H5Sselect_none(mapSpace);
	}

	/*	Set chunked access and dynamical compression	*/
	if ((chunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (chunk_id, 1, &lSz) < 0) //slb) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

//	if (H5Pset_deflate (chunk_id, 9) < 0)	// Maximum compression, hoping that the map is a bunch of zeroes
//	{
//		LogError ("Fatal error H5Pset_deflate");
//		prof.stop();
//		exit (1);
//	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_alloc_time");
		prof.stop();
		exit (1);
	}

	/*	Create a dataset for map data	*/
	mSet_id = H5Dcreate (meas_id, mCh, dataType, mapSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);
	if (axion->Field() != FIELD_NAXION)
		vSet_id = H5Dcreate (meas_id, vCh, dataType, mapSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

	if (mSet_id < 0 || vSet_id < 0)
	{
		LogError ("Fatal error creating datasets");
		prof.stop();
		exit (0);
	}

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/
	mSpace = H5Dget_space (mSet_id);
	if (axion->Field() != FIELD_NAXION)
		vSpace = H5Dget_space (vSet_id);

	if (mSpace < 0 || vSpace < 0)
	{
		LogError ("Fatal error");
		prof.stop();
		exit (0);
	}

	hsize_t offset = 0;

	if (myRank == 0) {
		H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slb, NULL);
	} else {
		H5Sselect_none(mSpace);
		//dataM = NULL;
	}

	/*	Write raw data	*/
	if (H5Dwrite (mSet_id, dataType, mapSpace, mSpace, H5P_DEFAULT, dataM) < 0)
	{
		LogError ("Error writing dataset /map/m");
		prof.stop();
		exit(0);
	}

	if (axion->Field() != FIELD_NAXION){
		if (myRank == 0) {
			H5Sselect_hyperslab(vSpace, H5S_SELECT_SET, &offset, NULL, &slb, NULL);
		} else {
			H5Sselect_none(vSpace);
		}

		if (H5Dwrite (vSet_id, dataType, mapSpace, vSpace, H5P_DEFAULT, dataV) < 0)
		{
			LogError ("Error writing dataset /map/v");
			prof.stop();
			exit(0);
		}
	}

	LogMsg (VERB_HIGH, "Write 2D map successful");LogFlush();

	/*	Close the dataset	*/
	H5Dclose (mSet_id);
	H5Sclose (mSpace);

	if (axion->Field() != FIELD_NAXION){
		H5Sclose (vSpace);
		H5Dclose (vSet_id);}

	H5Sclose (mapSpace);
	H5Pclose (chunk_id);
	H5Gclose (group_id);
	prof.stop();

	prof.add(std::string("Write Map"), 0, 2.e-9*slb*dataSize);
	LogMsg (VERB_HIGH, "Written %lu bytes", slb*dataSize*2);
	LogFlush();
}





void	writeMapHdf5	(Scalar *axion)
{
	writeMapHdf5s	(axion, 0);
}





void	writeMapHdf5s2	(Scalar *axion, int slicenumbertoprint)
{
	hid_t	mapSpace, chunk_id, group_id, mSet_id, vSet_id, mSpace, vSpace, memSpace, dataType;
	hsize_t	dataSize = axion->DataSize();

	int myRank = commRank();

	const hsize_t maxD[1] = { H5S_UNLIMITED };
	/* total values to be written is NOT Surf but Ly*Lz*zGrid*/
	hsize_t total  = axion->Length()*axion->TotalDepth();
	/* chunk size */
	hsize_t slab  = axion->Length();
	char mCh[16] = "/mapp/m";
	char vCh[16] = "/mapp/v";

	LogMsg (VERB_NORMAL, "[wm2] Writing 2D maps to Hdf5 measurement file YZ (%dx%d)",axion->Length(), axion->TotalDepth());
	LogMsg (VERB_PARANOID, "[wm2] total %d slab %d myRank %d dataSize %d",total,slab, commRank(),dataSize);	LogFlush();
	LogFlush();

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request");
		return;
	}

	/*	Start profiling		*/

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	if (axion->Field() == FIELD_SAXION) {
		total *= 2;
		slab *= 2;
	}

	if (axion->Precision() == FIELD_DOUBLE) {
		dataType = H5T_NATIVE_DOUBLE;
	} else {
		dataType = H5T_NATIVE_FLOAT;
	}

	/*	Unfold field before writing configuration	*/
	//if (axion->Folded())
	//{
		int slicenumber = slicenumbertoprint ;
		if (slicenumbertoprint > axion->Length())
		{
			LogMsg (VERB_NORMAL, "[wm2] Sliceprintnumberchanged to 0");
			slicenumber = 0;
		}
		Folder	munge(axion);
		LogMsg (VERB_NORMAL, "[wm2] If configuration folded, unfold 2D slice");	LogFlush();
		munge(UNFOLD_SLICEYZ, slicenumber);
	//}


	/*	Create a group for map data if it doesn't exist	*/
	auto status = H5Lexists (meas_id, "/mapp", H5P_DEFAULT);

	if (!status)
		group_id = H5Gcreate2(meas_id, "/mapp", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(meas_id, "/mapp", H5P_DEFAULT);		// Group exists
			LogMsg (VERB_HIGH, "[wm2] Group /map exists");
		} else {
			LogError ("[wm2] Error: can't check whether group /mapp exists");
			prof.stop();
			return;
		}
	}

	/*	Create space for writing the raw data to disk with chunked access	*/
	if ((mapSpace = H5Screate_simple(1, &total, maxD)) < 0)	// Whole data
	{
		LogError ("[wm2] Fatal error H5Screate_simple");
		prof.stop();
		exit (1);
	}

	// if (myRank != 0) {
	// 	H5Sselect_none(mapSpace);
	// }

	/*	Set chunked access and dynamical compression	*/
	if ((chunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("[wm2] Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (chunk_id, 1, &slab) < 0) //slb) < 0)
	{
		LogError ("[wm2] Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("[wm2] Fatal error H5Pset_alloc_time");
		prof.stop();
		exit (1);
	}

	/*	Create a dataset for map data	*/
	mSet_id = H5Dcreate (meas_id, mCh, dataType, mapSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);
	vSet_id = H5Dcreate (meas_id, vCh, dataType, mapSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

	commSync();

	if (mSet_id < 0 || vSet_id < 0)
	{
		LogError ("Fatal error creating datasets");
		prof.stop();
		exit (0);
	}

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/
	mSpace = H5Dget_space (mSet_id);
	vSpace = H5Dget_space (vSet_id);
	memSpace = H5Screate_simple(1, &slab, NULL);	// Slab

	if (mSpace < 0 )
	{
		LogError ("Fatal error");
		prof.stop();
		exit (0);
	}

	hsize_t partial = total/commSize();
	LogMsg (VERB_HIGH, "[wm2] Ready to write");	LogFlush();
	LogMsg (VERB_PARANOID, "[wm2] total %d commSize() %d sizeN %d dataSize %d",total,commSize(),sizeN,dataSize);	LogFlush();
	for (hsize_t yDim=0; yDim < axion->Depth(); yDim++)
	{
		hsize_t offset = (hsize_t) myRank*partial + yDim*slab;

		H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
		H5Sselect_hyperslab(vSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

		LogMsg (VERB_PARANOID, "[wm2] line yDim %d ",yDim);	LogFlush();
		/*	Write raw data	recall slab = sizeN*2*/
		auto mErr = H5Dwrite (mSet_id, dataType, memSpace, mSpace, H5P_DEFAULT, (static_cast<char *> (axion->mFrontGhost())) +axion->Length()*yDim*dataSize);
		auto vErr = H5Dwrite (vSet_id, dataType, memSpace, vSpace, H5P_DEFAULT, (static_cast<char *> (axion->mBackGhost() )) +axion->Length()*yDim*dataSize);

		if ((mErr < 0) || (vErr < 0))
		{
			LogError ("Error writing dataset");
			exit(0);
		}
	}

	LogMsg (VERB_HIGH, "[wm2] Write 2D mapp successful");LogFlush();

	/*	Close the dataset	*/
	H5Dclose (mSet_id);
	H5Dclose (vSet_id);
	H5Sclose (mSpace);
	H5Sclose (vSpace);

	H5Sclose (mapSpace);
	H5Pclose (chunk_id);
	H5Gclose (group_id);
	prof.stop();

	prof.add(std::string("Write Mapp"), 0, 2.e-9*total*dataSize);
	LogMsg (VERB_HIGH, "[wm2] Written %lu bytes", total*dataSize*2);LogFlush();
}




void	writeEMapHdf5s	(Scalar *axion, int slicenumbertoprint, char *eCh)
{
	hid_t	mapSpace, chunk_id, group_id, eSet_id, eSpace, dataType;
	hsize_t	dataSize = axion->Precision();

	int myRank = commRank();

	const hsize_t maxD[1] = { H5S_UNLIMITED };
	hsize_t slb  = axion->Surf();
	hsize_t lSz  = axion->Length();
	char *dataE  = static_cast<char *>(axion->m2Cpu()) + dataSize*axion->Surf()*slicenumbertoprint;
	// char eCh[16] = dataname;

	LogMsg (VERB_NORMAL, "[wem] Writing 2D energy map to Hdf5 measurement file");
	LogMsg (VERB_NORMAL, "");

	switch (axion->m2Status()){
		case M2_ENERGY:
		case M2_MASK_TEST:
		case M2_ANTIMASK:
		case M2_MASK:
			LogMsg (VERB_NORMAL, "[wem] M2 status %d ",axion->m2Status());
		break;
		default:
		LogError ("Error: Energy not available in m2. (status %d) Call energy before calling writeEMapHdf5", axion->m2Status());
		return;
		break;
	}

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request");
		return;
	}

	/*	Start profiling		*/

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	if (axion->Precision() == FIELD_DOUBLE) {
		dataType = H5T_NATIVE_DOUBLE;
	} else {
		dataType = H5T_NATIVE_FLOAT;
	}

	int slicenumber = slicenumbertoprint ;
	if (slicenumbertoprint > axion->Depth())
	{
		LogMsg (VERB_NORMAL, "Sliceprintnumberchanged to 0");
		slicenumber = 0;
	}

	/*	Create a group for map data if it doesn't exist	*/
	auto status = H5Lexists (meas_id, "/map", H5P_DEFAULT);

	if (!status)
		group_id = H5Gcreate2(meas_id, "/map", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(meas_id, "/map", H5P_DEFAULT);		// Group exists
			LogMsg (VERB_HIGH, "Group /map exists");
		} else {
			LogError ("Error: can't check whether group /map exists");
			prof.stop();
			return;
		}
	}

	/*	Create space for writing the raw data to disk with chunked access	*/
	if ((mapSpace = H5Screate_simple(1, &slb, maxD)) < 0)	// Whole data
	{
		LogError ("Fatal error H5Screate_simple");
		prof.stop();
		exit (1);
	}

	if (myRank != 0) {
		H5Sselect_none(mapSpace);
	}

	/*	Set chunked access and dynamical compression	*/

	if ((chunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (chunk_id, 1, &lSz) < 0) //slb) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

//	if (H5Pset_deflate (chunk_id, 9) < 0)	// Maximum compression, hoping that the map is a bunch of zeroes
//	{
//		LogError ("Fatal error H5Pset_deflate");
//		prof.stop();
//		exit (1);
//	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_alloc_time");
		prof.stop();
		exit (1);
	}

	/*	Create a dataset for map data	*/
	eSet_id = H5Dcreate (meas_id, eCh, dataType, mapSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

	if (eSet_id < 0)
	{
		LogError ("Fatal error creating datasets");
		prof.stop();
		exit (0);
	}

	eSpace = H5Dget_space (eSet_id);

	if (eSpace < 0)
	{
		LogError ("Fatal error");
		prof.stop();
		exit (0);
	}

	hsize_t offset = 0;

	if (myRank == 0) {
		H5Sselect_hyperslab(eSpace, H5S_SELECT_SET, &offset, NULL, &slb, NULL);
	} else {
		H5Sselect_none(eSpace);
	}

	/*	Write raw data	*/
	if (H5Dwrite (eSet_id, dataType, mapSpace, eSpace, H5P_DEFAULT, dataE) < 0)
	{
		LogError ("Error writing dataset /map/E");
		prof.stop();
		exit(0);
	}

	LogMsg (VERB_HIGH, "Write energy map successful");

	/*	Close the dataset	*/
	H5Dclose (eSet_id);
	H5Sclose (eSpace);

	H5Sclose (mapSpace);
	H5Pclose (chunk_id);
	H5Gclose (group_id);
	prof.stop();

	prof.add(std::string("Write EMap"), 0, 1.e-9*slb*dataSize);
	LogMsg (VERB_NORMAL, "Written %lu bytes", slb*dataSize);
}





void	writeEMapHdf5	(Scalar *axion)
{
	writeEMapHdf5s	(axion, 0, "/map/E");
}




void	writePMapHdf5s	(Scalar *axion, char *eCh)
{
	hid_t	mapSpace, chunk_id, group_id, eSet_id, eSpace, dataType;
	hsize_t	dataSize = axion->DataSize();

	int myRank = commRank();

	const hsize_t maxD[1] = { H5S_UNLIMITED };
	hsize_t slb  = axion->Surf();
	hsize_t lSz  = axion->Length();
	char *dataE  = static_cast<char *>(axion->mFrontGhost());
	// char eCh[16] = "/map/P";

	LogMsg (VERB_NORMAL, "Writing 2D energy projection to Hdf5 measurement file");
	LogMsg (VERB_NORMAL, "");

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request");
		return;
	}

	/*	Start profiling		*/

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	if (axion->Precision() == FIELD_DOUBLE) {
		dataType = H5T_NATIVE_DOUBLE;
	} else {
		dataType = H5T_NATIVE_FLOAT;
	}

	/*	Create a group for map data if it doesn't exist	*/
	auto status = H5Lexists (meas_id, "/map", H5P_DEFAULT);

	if (!status)
		group_id = H5Gcreate2(meas_id, "/map", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(meas_id, "/map", H5P_DEFAULT);		// Group exists
			LogMsg (VERB_HIGH, "Group /map exists");
		} else {
			LogError ("Error: can't check whether group /map exists");
			prof.stop();
			return;
		}
	}

	/*	Create space for writing the raw data to disk with chunked access	*/
	if ((mapSpace = H5Screate_simple(1, &slb, maxD)) < 0)	// Whole data
	{
		LogError ("Fatal error H5Screate_simple");
		prof.stop();
		exit (1);
	}

	if (myRank != 0) {
		H5Sselect_none(mapSpace);
	}

	/*	Set chunked access and dynamical compression	*/

	if ((chunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (chunk_id, 1, &lSz) < 0) //slb) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

//	if (H5Pset_deflate (chunk_id, 9) < 0)	// Maximum compression, hoping that the map is a bunch of zeroes
//	{
//		LogError ("Fatal error H5Pset_deflate");
//		prof.stop();
//		exit (1);
//	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_alloc_time");
		prof.stop();
		exit (1);
	}

	/*	Create a dataset for map data	*/
	eSet_id = H5Dcreate (meas_id, eCh, dataType, mapSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

	if (eSet_id < 0)
	{
		LogError ("Fatal error creating datasets");
		prof.stop();
		exit (0);
	}

	eSpace = H5Dget_space (eSet_id);

	if (eSpace < 0)
	{
		LogError ("Fatal error");
		prof.stop();
		exit (0);
	}

	hsize_t offset = 0;

	if (myRank == 0) {
		H5Sselect_hyperslab(eSpace, H5S_SELECT_SET, &offset, NULL, &slb, NULL);
	} else {
		H5Sselect_none(eSpace);
	}

	/*	Write raw data	*/
	if (H5Dwrite (eSet_id, dataType, mapSpace, eSpace, H5P_DEFAULT, dataE) < 0)
	{
		LogError ("Error writing dataset %s",eCh);
		prof.stop();
		exit(0);
	}

	LogMsg (VERB_HIGH, "Write energy projection successful");

	/*	Close the dataset	*/
	H5Dclose (eSet_id);
	H5Sclose (eSpace);

	H5Sclose (mapSpace);
	H5Pclose (chunk_id);
	H5Gclose (group_id);
	prof.stop();

	prof.add(std::string("Write PMap"), 0, 1.e-9*slb*dataSize);
	LogMsg (VERB_NORMAL, "Written %lu bytes", slb*dataSize);
}


void	writePMapHdf5	(Scalar *axion)
{
	writePMapHdf5s	(axion, "/map/P");
}


void	writeBinnerMetadata (double max, double min, size_t N, const char *group)
{
	hid_t	group_id;

	LogMsg (VERB_HIGH, "Writing binner metadata to measurement file");

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request.\n");
		return;
	}

	if ((group_id = H5Gopen2(meas_id, group, H5P_DEFAULT)) < 0)
		LogError ("Error: couldn't open group %s in measurement file.\n", group);
	else {
		writeAttribute(group_id, &max, "Maximum", H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &min, "Minimum", H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &N,   "Size",    H5T_NATIVE_HSIZE);
	}

	/*	Close everything		*/
	H5Gclose (group_id);

	LogMsg (VERB_HIGH, "Metadata written");
}

#ifdef USE_NYX_OUTPUT
	void writeConfNyx(Scalar *axion, int index)
	{
		if (axion->Folded())
		{
			Folder	*munge;
			munge	= new Folder(axion);
			(*munge)(UNFOLD_ALL);
			delete munge;
		}

		LogMsg (VERB_NORMAL, "[rw] write Conf NYX called "); LogFlush();
		amrex::nyx_output_plugin *morla;
		morla = new amrex::nyx_output_plugin(axion,index);
		delete morla;
	}
#endif


/* write gadget file */

void	writeGadget (Scalar *axion, double eMean, size_t realN, size_t nParts, double sigma)
{
	hid_t	file_id, hGrp_id, hGDt_id, attr, plist_id, chunk_id, shunk_id, vhunk_id;
	hid_t	vSt1_id, vSt2_id, sSts_id, aSpace, status;
	hid_t	vSpc1, vSpc2, sSpce, memSpace, semSpace, dataType, totalSpace, scalarSpace;
	hsize_t	total, slice, slab, offset, rOff;

	char	prec[16], fStr[16];
	int	length = 8;

	size_t	dataSize;

	int myRank = commRank();

	LogMsg (VERB_NORMAL, "Writing Gadget output file");
	LogMsg (VERB_NORMAL, "");

	/*      Start profiling         */

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	/*      If needed, transfer data to host        */
	if (axion->Device() == DEV_GPU)
		axion->transferCpu(FIELD_M2);

	if (axion->m2Cpu() == nullptr) {
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

	size_t	iDummy = 0;
	size_t	oDummy = 1;
	double	fDummy = 0.0;
	double	bSize  = axion->BckGnd()->PhysSize() * 7430.0 * 0.7;	// FIXME

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

	/*	Deal with the attribute arrays	*/
	/*	Create dataspace		*/

	//changed
	// size_t	nPrt = axion->Size();
	size_t	nPrt = nParts;
	if (nParts == 0)
		nPrt = axion->TotalSize();
	LogOut("[gad] nPart = %lu\n",nPrt);

	//	Total DM density
	double	Omega0 = 0.3;
	//	H0 in units of kyear/h
	double	H0 = 1e15 * M_PI / 3.086e22;
	//	G in units of AU^3/kyear^2 (1/1e-15 Msun)
	double	Grav = (6.67408e09 * 1.989e15 * M_PI * M_PI)/(149.6e9*149.6e9*149.6e9);

	//	Total mass in GADGET units
	double	totalMass = Omega0 * (bSize*bSize*bSize) * (3.0 * H0*H0) / (8 * M_PI * Grav);
	double	mAx  = totalMass/((double) nPrt);

	hsize_t	dims[1]  = { 6 };
	double	dAFlt[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	double	mTab[6]  = { 0.0, mAx, 0.0, 0.0, 0.0, 0.0 };
	size_t	nPart[6] = {   0, nPrt,  0,   0,   0,   0 };

	aSpace = H5Screate_simple (1, dims, nullptr); //dims, nullptr);

	/*	Create the attributes and write them	*/

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

	/*	Create datagroup	*/
	hGDt_id = H5Gcreate2(file_id, "/PartType1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// changed FIXME for asymmetric reduced grids?
	// uint totlZ = axion->TotalDepth();
	// uint totlX = axion->Length();
	uint totlZ = realN;
	uint totlX = realN;
	uint realDepth = realN/commSize();

	if (totlX == 0) {
		totlZ	  = axion->TotalDepth();
		totlX	  = axion->Length();
		realDepth = axion->Depth();
	}

	total = ((hsize_t) totlX)*((hsize_t) totlX)*((hsize_t) totlZ);
	slab  = ((hsize_t) totlX)*((hsize_t) totlX);

	// changed FIXME for asymmetric reduced grids?
	// rOff  = ((hsize_t) (totlX))*((hsize_t) (totlX))*(axion->Depth());
	rOff  = ((hsize_t) (totlX))*((hsize_t) (totlX))*(realDepth);

	const hsize_t vSlab[2] = { slab, 3 };

	LogOut("[gad] total %lu slab %lu r0ff %lu\n",total, slab, rOff);

	/*	1 - Compute eMean_local
			2 - Compute eMean_global (again and compare with input eMean)
			3 - Normalise energy to contrast eMean
			4 - Set the goal in a rank	= llround(Npart*eMean_local/eMean_global)
		  5 - Adjust rank 0  */

	// float version
	// number of down interations
	LogOut("[gad] Recompute eMean with care \n");
	int nred = round( log2( (double) rOff));
	LogOut("[gad] sub-reductions %d (%lu , %lu)\n", nred, rOff, (size_t) round(pow(2,nred)));
	double eMean_local;
	double eMean_global;

	double * axArray2 = static_cast<double *> (axion->m2half());
	// first reduction
	double eMean_local1 = 0.0;
	if (dataSize == 4) {
			float * axArray1 = static_cast<float *>(axion->m2Cpu());
			#pragma omp parallel for schedule(static) reduction(+:eMean_local1)
			for (size_t idx =0; idx < rOff/2; idx++)
			{
				axArray2[idx] = (double) (axArray1[2*idx] + axArray1[2*idx+1]) ;
				eMean_local1 += axArray2[idx];
			}
	} else {
			double * axArray1 = static_cast<double *>(axion->m2Cpu());
			#pragma omp parallel for schedule(static) reduction(+:eMean_local1)
			for (size_t idx =0; idx < rOff/2; idx++)
			{
				axArray2[idx] = axArray1[2*idx] + axArray1[2*idx+1] ;
				eMean_local1 += axArray2[idx];
			}
		}
	LogOut("[gad] emean_local = %lf (eMean/zgrid %lf)\n",eMean_local1/rOff,eMean/commSize());
	eMean_local1 /= rOff;

	MPI_Allreduce(&eMean_local1, &eMean_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	eMean_global /= commSize();
	LogOut("[gad] New eMean (naive) = %.20f (eMean %.20lf)\n",eMean_global,eMean);

	// reduction loop
	size_t ldata = rOff/2;
	{
		size_t lldata;
		/* choose smallest stride */
		hssize_t stride=1;
		hssize_t old_stride=1;
		hssize_t fa ;

		while (ldata > 1)
		{
			for (fa = 2; fa<100; fa++){
				lldata = ldata/fa ;
				if ( lldata*fa == ldata){
					stride *= fa;
					break;}
			}
			LogOut("[gad] lData %lu ... stride %d oldstride %d fa %d",ldata,stride,old_stride, fa);
			/* reduce */
			#pragma omp parallel for schedule(static)
			for (size_t idx =0; idx < rOff/2; idx += stride)
			{
				for (hssize_t ips=1; ips<fa; ips++ ) // 0 is already included
					axArray2[idx] += axArray2[idx + ips*old_stride] ;
			}
			LogOut("[gad] Reduced %lu to %lu stride %d > part buf %f Mean %f\n",ldata,lldata,stride, axArray2[0], axArray2[0]/(2*stride));
			ldata = lldata;
			old_stride = stride;
		} //end while
	}
	LogOut("[gad] emean_local = %lf (eMean/zgrid %lf)\n",axArray2[0]/rOff,eMean/commSize());
	eMean_local = axArray2[0]/rOff;

	MPI_Allreduce(&eMean_local, &eMean_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	eMean_global /= commSize();
	LogOut("[gad] New eMean = %.20lf (input eMean %.20lf)\n",eMean_global,eMean);

	LogOut("[gad] Sum Energy = realN^3 eMean\n");
	LogOut("[gad] To get %lu particles, multiply by nPrt/realN^3\n",nPrt);
	double neweMean = eMean_global*(totlX*totlX*totlZ)/nPrt;

	if (dataSize == 4) {
		float * axArray = static_cast<float *>(axion->m2Cpu());

		#pragma omp parallel for schedule(static)
		for (size_t idx = 0; idx<rOff; idx++)
			axArray[idx] /= neweMean;
	} else { // Assumes double
		double *axArray = static_cast<double*>(axion->m2Cpu());

		#pragma omp parallel for schedule(static)
		for (size_t idx = 0; idx<rOff; idx++)
			axArray[idx] /= neweMean;
	}
	LogOut("[gad] E normalised to contrast\n");




	/* Set deterministic goals for the number of particles in each rank */
	// sum axArray now is ~ nPrt
	// sum local array now is (sum local array before)/neweMean =
	// (eMean_local*rOff)/neweMean
	LogOut("[gad] Each rank takes round(eMean_local*rOff/eMean)\n");
	double localoca = round(eMean_local*rOff/neweMean);
	size_t nPrt_local = (size_t) localoca;
	printf("[gad] rank %d should take (%lf) %lu\n",commRank(),localoca,nPrt_local);
	fflush(stdout);
	commSync();

	size_t nPrt_temp;
	MPI_Allreduce(&nPrt_local, &nPrt_temp, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
	if (nPrt_temp > nPrt)
	{
		LogOut("[gad] sum is %lu, which is %lu too many \n",nPrt_temp,nPrt_temp-nPrt);
		LogOut("[gad] Rank 0 readjusts itself from %lu to nPart = %lu\n",nPrt_local,nPrt_local + (nPrt-nPrt_temp) );
		if (commRank()==0)
			nPrt_local = nPrt_local + (nPrt-nPrt_temp);
		}
	else {
		LogOut("[gad] sum is %lu, which is %lu too few \n",nPrt_temp,nPrt-nPrt_temp);
		LogOut("[gad] Rank 0 readjusts itself from %lu to nPart = %lu ...",nPrt_local,nPrt_local + (nPrt-nPrt_temp) );
		if (commRank()==0){
			nPrt_local = nPrt_local + (nPrt-nPrt_temp);
			LogOut("Done! \n");
			}

	}


	if (axion->Field() == FIELD_WKB) {
		LogError ("Error: WKB field not supported");
		prof.stop();
		exit(1);
	}


	/*	Create space for writing the raw data to disk with chunked access	*/
	// total is the number of points in the grid //changed
	hsize_t nPrt_h = (hsize_t)  nPrt;
	const hsize_t dDims[2] = { nPrt_h , 3 };
	if ((totalSpace = H5Screate_simple(2, dDims, nullptr)) < 0)	// Whole data
	{
		LogError ("Fatal error H5Screate_simple");
		prof.stop();
		exit (1);
	}

	// total is the number of points in the grid //changed
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

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

	memSpace = H5Screate_simple(2, vSlab, nullptr);	// Slab
	semSpace = H5Screate_simple(1, &slab, nullptr);	// Slab

	commSync();

	// changed
	// const hsize_t Lz = axion->Depth();
	const hsize_t Lz = realDepth;

	const hsize_t stride[2] = { 1, 1 };

	/*	For the something	*/
	void *vArray = static_cast<void*>(static_cast<char*>(axion->m2Cpu())+(slab*Lz)*dataSize);
	//memset(vArray, 0, slab*3*dataSize);

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
		LogOut("zDim %zu\n", zDim);
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
//	prof.add(std::string("Write gadget map"), 0., ((double) bytes)*1e-9);

//	LogMsg (VERB_NORMAL, "Written %lu bytes", bytes);
}

/* read edens maps for postprocessing */

double	readEDens (Cosmos *myCosmos, Scalar **axion, int index)
{
	hid_t	file_id, mset_id, vset_id, plist_id, grp_id, grp_id2;
	hid_t	mSpace, vSpace, memSpace, dataType;
	hid_t	attr_type;

	hsize_t	slab, offset;

	FieldPrecision	precision;

	char	prec[16], fStr[16], lStr[16], icStr[16], vStr[32], smStr[16];
	int	length = 32;

	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize;

	int myRank = commRank();

	LogMsg (VERB_NORMAL, "Reading Energy density field from Hdf5 file on disk");
	LogMsg (VERB_NORMAL, "");

	/*	Start profiling		*/

	// Profiler &prof = getProfiler(PROF_HDF5);
	// prof.start();

	/*	Set up parallel access with Hdf5	*/

	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];

	sprintf(base, "%s/%s.m.%05d", outDir, outName, index);

	LogMsg (VERB_NORMAL, "File read: %s",base);

	/*	Open the file and release the plist	*/
	if ((file_id = H5Fopen (base, H5F_ACC_RDONLY, plist_id)) < 0)
	{
		*axion == nullptr;
		LogError ("Error opening file %s", base);
		return 0.0;
	}
	H5Pclose(plist_id);

	/*	Attributes	*/
	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	double	zTmp, RTmp, maaR;
	uint	tStep, cStep, totlZ;
	readAttribute (file_id, fStr,   "Field type",   attr_type);
	readAttribute (file_id, prec,   "Precision",    attr_type);
	readAttribute (file_id, &sizeN, "Size",         H5T_NATIVE_UINT);
	readAttribute (file_id, &totlZ, "Depth",        H5T_NATIVE_UINT);
	readAttribute (file_id, &zTmp,  "z",            H5T_NATIVE_DOUBLE);
	// readAttribute (file_id, &RTmp,  "R",            H5T_NATIVE_DOUBLE);

	LogMsg (VERB_NORMAL, "Field type: %s",fStr);
	LogMsg (VERB_NORMAL, "Precision: %s",prec);
	LogMsg (VERB_NORMAL, "Size: %d",sizeN);
	LogMsg (VERB_NORMAL, "Depth: %d",totlZ);
	LogMsg (VERB_NORMAL, "zTmp: %f",zTmp);
	LogMsg (VERB_NORMAL, "RTmp: %f",RTmp);

LogMsg (VERB_NORMAL, "PhysSize: %f",myCosmos->PhysSize());
	if (myCosmos->PhysSize() == 0.0) {
		double lSize;
		readAttribute (file_id, &lSize, "Physical size", H5T_NATIVE_DOUBLE);
		myCosmos->SetPhysSize(lSize);
		LogMsg (VERB_NORMAL, "read L: %f",lSize);
	}

	if (!uPrec)
	{
		if (!strcmp(prec, "Double"))
		{
			precision = FIELD_DOUBLE;
			sPrec	  = FIELD_DOUBLE;
			dataType  = H5T_NATIVE_DOUBLE;
			dataSize  = sizeof(double);
		} else if (!strcmp(prec, "Single")) {
			precision = FIELD_SINGLE;
			sPrec	  = FIELD_SINGLE;
			dataType  = H5T_NATIVE_FLOAT;
			dataSize  = sizeof(float);
		} else {
			LogError ("Error reading file %s: Invalid precision %s", base, prec);
			exit(1);
		}
	} else {
		precision = sPrec;

		if (sPrec == FIELD_DOUBLE)
		{
			dataType  = H5T_NATIVE_DOUBLE;
			dataSize  = sizeof(double);

			if (!strcmp(prec, "Single"))
				LogMsg (VERB_NORMAL, "Reading single precision configuration as double precision");
		} else if (sPrec == FIELD_SINGLE) {
			dataType  = H5T_NATIVE_FLOAT;
			dataSize  = sizeof(float);
			if (!strcmp(prec, "Double"))
				LogMsg (VERB_NORMAL, "Reading double precision configuration as single precision");
		} else {
			LogError ("Input error: Invalid precision");
			exit(1);
		}
	}

	/* Read mean energy */


	grp_id  = H5Gopen( file_id, "energy", H5P_DEFAULT ) ;

	double eMean = 0.0;
	double eSum = 0.0;
	readAttribute (grp_id, &eMean, "Axion Gr X", H5T_NATIVE_DOUBLE);
	eSum += eMean;
	readAttribute (grp_id, &eMean, "Axion Gr Y", H5T_NATIVE_DOUBLE);
	eSum += eMean;
	readAttribute (grp_id, &eMean, "Axion Gr Z", H5T_NATIVE_DOUBLE);
	eSum += eMean;
	readAttribute (grp_id, &eMean, "Axion Kinetic", H5T_NATIVE_DOUBLE);
	eSum += eMean;
	readAttribute (grp_id, &eMean, "Axion Potential", H5T_NATIVE_DOUBLE);
	eMean += eSum;

	LogMsg (VERB_NORMAL, "read eMean: %f",eMean);


	if (debug) LogOut("[db] Read start\n");
	// prof.stop();

	// prof.add(std::string("Read Energy"), 0, 0);
	// prof.start();
	commSync();

	/*	Create plist for collective read	*/

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id,H5FD_MPIO_COLLECTIVE);

	/*	Open a dataset for the whole axion data	*/

	size_t sizeNe = 0;
	size_t sizeZe = 0;


	if ((mset_id = H5Dopen (file_id, "/energy/density/theta", H5P_DEFAULT)) < 0)
	{
		LogError ("Error opening dataset /energy/density/theta ");

		if ((mset_id = H5Dopen (file_id, "/energy/redensity", H5P_DEFAULT)) < 0)
		{
			LogError ("Error opening dataset /energy/redensity");
		}
		else{
			LogMsg(VERB_NORMAL, "[rEd] from energy/redensity ");
			/* assumes sizeN = sizeZ */
			readAttribute (grp_id, &sizeNe, "Size", H5T_NATIVE_UINT);
			readAttribute (grp_id, &sizeZe, "Size", H5T_NATIVE_UINT);
		}
	}
	else{
		grp_id2 = H5Gopen( file_id, "energy/density", H5P_DEFAULT) ;
		LogMsg(VERB_NORMAL, "[rEd] from energy/density/theta ");
		readAttribute (grp_id2, &sizeNe, "Size", H5T_NATIVE_UINT);
		readAttribute (grp_id2, &sizeZe, "Depth", H5T_NATIVE_UINT);
	}

	LogMsg(VERB_NORMAL, "[rEd] Read N=%lu ",sizeNe);
	LogMsg(VERB_NORMAL, "[rEd] Read Z=%lu ",sizeZe);
	LogMsg(VERB_NORMAL, "[rEd] Read zGrid=%d ",zGrid);

	if (sizeZe % zGrid)
	{
		LogError ("Error: Geometry not valid. Try a different partitioning");
		exit (1);
	}
	else
		sizeZe = (size_t) (sizeZe/zGrid);


	LogMsg(VERB_HIGH, "Creating axion with N=%lu ",sizeNe);
	LogMsg(VERB_HIGH, "Creating axion with Z=%lu ",sizeZe);
	LogMsg(VERB_HIGH, "Creating axion with zGrid=%d",zGrid);


	/*	Create axion field */
	*axion = new Scalar(myCosmos, sizeNe, sizeZe, precision, cDev, zTmp, false, zGrid, FIELD_AXION, lType, myCosmos->ICData().Nghost);

	LogOut("Axion created\n");

	slab   = (hsize_t) ((*axion)->Surf());
	memSpace = H5Screate_simple(1, &slab, NULL);	// Slab
	mSpace   = H5Dget_space (mset_id);

	LogMsg(VERB_HIGH,"check slices %lu slab %lu \n",(*axion)->Depth(),slab);
	for (hsize_t zDim=0; zDim<((hsize_t) (*axion)->Depth()); zDim++)
	{
		/*	Select the slab in the file	*/
		offset = (((hsize_t) (myRank*(*axion)->Depth()))+zDim)*slab;
		H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

		/*	Read raw data	*/
		auto mErr = H5Dread (mset_id, dataType, memSpace, mSpace, plist_id, (static_cast<char *> ((*axion)->m2Cpu())+slab*zDim*dataSize));

		if ((mErr < 0)) {
			LogError ("Error reading dataset from file");
			return 0.0;
		}
	}

	/*	Close the dataset	*/

	H5Sclose (mSpace);
	// H5Sclose (vSpace);
	H5Dclose (mset_id);
	// H5Dclose (vset_id);
	H5Sclose (memSpace);
	H5Gclose (grp_id);
	H5Gclose (grp_id2);

	/*	Close the file		*/

	H5Pclose (plist_id);
	H5Fclose (file_id);

	if (cDev == DEV_GPU)
		(*axion)->transferDev(FIELD_MV);

	// prof.stop();
	// prof.add(std::string("Read configuration"), 0, (totlZ*slab*(*axion)->DataSize() + 77.)*1.e-9);

	LogMsg (VERB_NORMAL, "Read %lu bytes", ((size_t) totlZ)*slab*2 + 77);
	(*axion)->setM2(M2_ENERGY);
	LogMsg (VERB_HIGH, "M2 set to M2_ENERGY");
	LogFlush();

	return eMean;
}
