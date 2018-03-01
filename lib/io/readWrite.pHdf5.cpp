#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <hdf5.h>

#include <fftw3-mpi.h>

#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "utils/parse.h"
#include "comms/comms.h"

#include "utils/memAlloc.h"
#include "utils/profiler.h"
#include "utils/logger.h"

#include "fft/fftCode.h"

#include "scalar/varNQCD.h"

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

	LogMsg (VERB_HIGH, "Write attribute %s", name);	// Usa status para hacer logging de los errores!!!

	return	status;
}

herr_t	readAttribute(hid_t file_id, void *data, const char *name, hid_t h5_type)
{
	hid_t	attr;
	herr_t	status;

	if ((attr   = H5Aopen_by_name (file_id, ".", name, H5P_DEFAULT, H5P_DEFAULT)) < 0)
		LogError ("Error opening attribute %s");
	if ((status = H5Aread (attr, h5_type, data)) < 0)
		LogError ("Error reading attribute %s");
	status = H5Aclose(attr);

	LogMsg (VERB_HIGH, "Read attribute %s", name);

	return	status;
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

void	writeConf (Scalar *axion, int index)
{
	hid_t	file_id, mset_id, vset_id, plist_id, chunk_id;
	hid_t	mSpace, vSpace, memSpace, dataType, totalSpace;
	hsize_t	total, slice, slab, offset;

	char	prec[16], fStr[16], lStr[16], rStr[16], dStr[16], vStr[32], icStr[16], smStr[16];
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
		LogMsg (VERB_HIGH, "Folded configuration, will unfold at the end");
		munge	= new Folder(axion);
		(*munge)(UNFOLD_ALL);
	}

	/*	Start profiling		*/

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	/*	Set up parallel access with Hdf5	*/
	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];

	sprintf(base, "%s/%s.%05d", outDir, outName, index);

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
	uint totlZ = axion->TotalDepth();
	uint tmpS  = axion->Length();

	switch (axion->Field())
	{
		case 	FIELD_SX_RD:
		case 	FIELD_SAXION:
		{
			total = ((hsize_t) tmpS)*((hsize_t) tmpS)*((hsize_t) (totlZ*2));
			slab  = (hsize_t) (axion->Surf()*2);

			sprintf(fStr, "Saxion");
		}
		break;

		case 	FIELD_AX_MOD_RD:
		case	FIELD_AXION_MOD:
		{
			total = ((hsize_t) tmpS)*((hsize_t) tmpS)*((hsize_t) totlZ);
			slab  = (hsize_t) axion->Surf();

			sprintf(fStr, "Axion Mod");
		}
		break;

		case 	FIELD_AX_RD:
		case	FIELD_AXION:
		case	FIELD_WKB:
		{
			total = ((hsize_t) tmpS)*((hsize_t) tmpS)*((hsize_t) totlZ);
			slab  = (hsize_t) axion->Surf();

			sprintf(fStr, "Axion");
		}
		break;

		default:

		LogError ("Error: Invalid field type. How did you get this far?");
		exit(1);

		break;
	}

	double	llPhys = LL;

	switch (axion->Lambda())
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

	switch (vqcdType & VQCD_TYPE)
	{
		case	VQCD_1:
			sprintf(vStr, "VQcd 1");
			break;

		case	VQCD_2:
			sprintf(vStr, "VQcd 2");
			break;

		case	VQCD_1_PQ_2:
			sprintf(vStr, "VQcd 1 Peccei-Quinn 2");
			break;

		default:
			sprintf(vStr, "None");
			break;
	}

	switch (vqcdTypeDamp)
	{
		case	VQCD_DAMP_RHO:
			sprintf(dStr, "Rho");
			break;

		case	VQCD_DAMP_ALL:
			sprintf(dStr, "All");
			break;

		default:
		case	VQCD_NONE:
			sprintf(dStr, "None");
			break;
	}

	switch (vqcdTypeRhoevol)
	{
		case	VQCD_EVOL_RHO:
			sprintf(rStr, "Only Rho");
			break;

		default:
		case	VQCD_NONE:
			sprintf(rStr, "Full");
			break;
	}

	/*	Write header	*/
	hid_t attr_type;

	/*	Attributes	*/

	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size   (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	double maa = axionmass((*axion->zV()), nQcd, zthres, zrestore);

	writeAttribute(file_id, fStr,   "Field type",    attr_type);
	writeAttribute(file_id, prec,   "Precision",     attr_type);
	writeAttribute(file_id, &tmpS,  "Size",          H5T_NATIVE_UINT);
	writeAttribute(file_id, &totlZ, "Depth",         H5T_NATIVE_UINT);
	writeAttribute(file_id, &msa,   "Saxion mass",   H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &maa,   "Axion mass",    H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &sizeL, "Physical size", H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, axion->zV(),  "z",       H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &zInit, "zInitial",      H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &zFinl, "zFinal",        H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &nSteps,"nSteps",        H5T_NATIVE_INT);
	writeAttribute(file_id, &cSteps,"Current step",  H5T_NATIVE_INT);

	/*	Create a group for specific header data	*/
	hid_t vGrp_id = H5Gcreate2(file_id, "/potential", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	double shift = saxionshift(maa, llPhys, vqcdType);
	indi3 =  maa/pow(*axion->zV(), nQcd*0.5);

	writeAttribute(vGrp_id, &lStr,  "Lambda type",   attr_type);
	writeAttribute(vGrp_id, &LL,    "Lambda",        H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &vStr,  "VQcd type",     attr_type);
	writeAttribute(vGrp_id, &dStr,  "Damping type",  attr_type);
	writeAttribute(vGrp_id, &rStr,  "Evolution type",attr_type);
	writeAttribute(vGrp_id, &nQcd,  "nQcd",          H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &gammo, "Gamma",         H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &shift, "Shift",         H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &indi3, "Indi3",         H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &zthres,"z Threshold",   H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &zrestore,"z Restore",   H5T_NATIVE_DOUBLE);

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

		case	CONF_TKACHEV:
			sprintf(icStr, "Tkachev");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
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
	vset_id = H5Dcreate (file_id, vCh, dataType, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

	commSync();

	if ((mset_id < 0) || (vset_id < 0))
	{
		LogError ("Error creating datasets");
		exit (0);
	}

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

	mSpace = H5Dget_space (mset_id);
	vSpace = H5Dget_space (vset_id);
	memSpace = H5Screate_simple(1, &slab, NULL);	// Slab

	commSync();

	LogMsg (VERB_HIGH, "Rank %d ready to write", myRank);

	for (hsize_t zDim=0; zDim<((hsize_t) axion->Depth()); zDim++)
	{
		/*	Select the slab in the file	*/
		offset = (((hsize_t) (myRank*axion->Depth()))+zDim)*slab;
		H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
		H5Sselect_hyperslab(vSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

		/*	Write raw data	*/
		auto mErr = H5Dwrite (mset_id, dataType, memSpace, mSpace, plist_id, (static_cast<char *> (axion->mCpu())+slab*(1+zDim)*dataSize));
		auto vErr = H5Dwrite (vset_id, dataType, memSpace, vSpace, plist_id, (static_cast<char *> (axion->vCpu())+slab*zDim*dataSize));

		if ((mErr < 0) || (vErr < 0))
		{
			LogError ("Error writing dataset");
			exit(0);
		}
		//commSync();
	}

	/*	Close the dataset	*/

	H5Dclose (mset_id);
	H5Dclose (vset_id);
	H5Sclose (mSpace);
	H5Sclose (vSpace);
	H5Sclose (memSpace);

	/*	Close the file		*/

	H5Sclose (totalSpace);
	H5Pclose (chunk_id);
	H5Pclose (plist_id);
	H5Fclose (file_id);

	prof.stop();
	prof.add(std::string("Write configuration"), 0., (2.*total*dataSize + 81.)*1e-9);

	LogMsg (VERB_NORMAL, "Written %lu bytes", total*dataSize*2 + 81);

	/*	Fold back the field	*/

	if (wasFolded)
	{
		(*munge)(FOLD_ALL);
		delete	munge;
	}
}


void	readConf (Scalar **axion, int index)
{
	hid_t	file_id, mset_id, vset_id, plist_id;
	hid_t	mSpace, vSpace, memSpace, dataType;
	hid_t	attr_type;

	hsize_t	slab, offset;

	FieldPrecision	precision;

	char	prec[16], fStr[16], lStr[16], icStr[16], vStr[32], smStr[16];
	int	length = 32;

	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize;

	int myRank = commRank();

	LogMsg (VERB_NORMAL, "Reading Hdf5 configuration from disk");
	LogMsg (VERB_NORMAL, "");

	/*	Start profiling		*/

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	/*	Set up parallel access with Hdf5	*/

	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];

	sprintf(base, "%s/%s.%05d", outDir, outName, index);

	/*	Open the file and release the plist	*/

	if ((file_id = H5Fopen (base, H5F_ACC_RDONLY, plist_id)) < 0)
	{
		*axion == nullptr;
		LogError ("Error opening file %s", base);
		return;
	}

	H5Pclose(plist_id);

	/*	Attributes	*/

	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	double	zTmp;
	uint	tStep, cStep, totlZ;

	readAttribute (file_id, fStr,   "Field type",   attr_type);
	readAttribute (file_id, prec,   "Precision",    attr_type);
	readAttribute (file_id, &sizeN, "Size",         H5T_NATIVE_UINT);
	readAttribute (file_id, &totlZ, "Depth",        H5T_NATIVE_UINT);

	readAttribute (file_id, &zTmp,  "z",        H5T_NATIVE_DOUBLE);

	readAttribute (file_id, &sizeL, "Physical size",H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &tStep, "nSteps",       H5T_NATIVE_INT);
	readAttribute (file_id, &cStep, "Current step", H5T_NATIVE_INT);

	if (endredmap == -1)	// No reduction unless specified
		endredmap = sizeN;

	if (!uZin) {
		readAttribute (file_id, &zInit, "zInitial", H5T_NATIVE_DOUBLE);
		if (zTmp < zInit)
			zInit = zTmp;
	} else {
		zTmp = zInit;
	}

	if (!uZfn) {
		readAttribute (file_id, &zFinl,  "zFinal",       H5T_NATIVE_DOUBLE);
	}

//	if (zInit > zTmp)
//		zTmp = zInit;

	/*	Read potential data		*/
	auto status = H5Lexists (file_id, "/potential", H5P_DEFAULT);

	if (status <= 0)
		LogMsg(VERB_NORMAL, "Potential data not available, using defaults");
	else {
		hid_t vGrp_id = H5Gopen2(file_id, "/potential", H5P_DEFAULT);

		if (uQcd == false)
			readAttribute (vGrp_id, &nQcd,  "nQcd",         H5T_NATIVE_DOUBLE);

		if ((uLambda == false) && (msa == false)) {
			readAttribute (vGrp_id, &LL,    "Lambda",       H5T_NATIVE_DOUBLE);
			readAttribute (file_id, &msa,   "Saxion mass",  H5T_NATIVE_DOUBLE);
			readAttribute (vGrp_id, &lStr,  "Lambda type",  attr_type);

			if (!strcmp(lStr, "z2"))
				lType = LAMBDA_Z2;
			else if (!strcmp(lStr, "Fixed"))
				lType = LAMBDA_FIXED;
			else {
				LogError ("Error reading file %s: invalid lambda type %s", base, lStr);
				exit(1);
			}
		} else {
			if (uMsa) {
				double tmp = (msa*sizeN)/sizeL;
				LL    = 0.5*tmp*tmp;
			} else {
				double tmp = sizeL/sizeN;
				msa = sqrt(2*LL)*tmp;
			}
		}

		double	maa = 0., maaR;
		readAttribute (file_id, &maa,   "Axion mass",   H5T_NATIVE_DOUBLE);
		readAttribute (vGrp_id, &zthres,"z Threshold",  H5T_NATIVE_DOUBLE);
		readAttribute (vGrp_id, &zrestore,"z Restore",  H5T_NATIVE_DOUBLE);
		readAttribute (file_id, &maaR,  "Axion mass",   H5T_NATIVE_DOUBLE);
		readAttribute (vGrp_id, &indi3, "Indi3",        H5T_NATIVE_DOUBLE);

		maa = axionmass(zTmp, nQcd, zthres, zrestore);
		LogMsg(VERB_HIGH, "Chaging axion mass from %e to %e", maaR, maa);
		indi3 =  maa/pow(zTmp, nQcd*0.5);

		if (uGamma == false)
			readAttribute (vGrp_id, &gammo,  "Gamma",       H5T_NATIVE_DOUBLE);

		if (uPot == false) {
			readAttribute (vGrp_id, &vStr,  "VQcd type",  attr_type);

			if (!strcmp(vStr, "VQcd 1"))
				vqcdType = VQCD_1;
			else if (!strcmp(vStr, "VQcd 2"))
				vqcdType = VQCD_2;
			else if (!strcmp(vStr, "VQcd 1 Peccei-Quinn 2"))
				vqcdType = VQCD_1_PQ_2;
			else {
				LogError ("Error reading file %s: invalid potential type %s", base, vStr);
				vqcdType = VQCD_1;
				//exit(1);
			}

			readAttribute (vGrp_id, &vStr,  "Damping type",  attr_type);

			if (!strcmp(vStr, "Rho"))
				vqcdTypeDamp = VQCD_DAMP_RHO;
			else if (!strcmp(vStr, "All"))
				vqcdTypeDamp = VQCD_DAMP_ALL;
			else if (!strcmp(vStr, "None"))
				vqcdTypeDamp = VQCD_NONE;
			else {
				LogError ("Error reading file %s: invalid damping type %s", base, vStr);
				exit(1);
			}

			readAttribute (vGrp_id, &vStr,  "Evolution type",  attr_type);

			if (!strcmp(vStr, "Only Rho"))
				vqcdTypeRhoevol = VQCD_EVOL_RHO;
			else if (!strcmp(vStr, "Full"))
				vqcdTypeRhoevol = VQCD_NONE;
			else {
				LogError ("Error reading file %s: invalid rho evolution type %s", base, vStr);
				exit(1);
			}

			vqcdType |= vqcdTypeDamp | vqcdTypeRhoevol;
		}

		H5Gclose(vGrp_id);
	}

	/*	Read IC data		*/
	status = H5Lexists (file_id, "/ic", H5P_DEFAULT);

	if (status <= 0)
		LogMsg(VERB_NORMAL, "IC data not available");
	else {
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
		} else if (!strcmp(icStr, "Tkachev")) {
			cType = CONF_TKACHEV;
			readAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
			readAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
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
		} else {
			LogError("Error: unrecognized configuration type %s", icStr);
			exit(1);
		}
		H5Gclose(icGrp_id);
	}

	H5Tclose (attr_type);

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
				LogMsg (VERB_HIGH, "Reading single precision configuration as double precision");
		} else if (sPrec == FIELD_SINGLE) {
			dataType  = H5T_NATIVE_FLOAT;
			dataSize  = sizeof(float);
			if (!strcmp(prec, "Double"))
				LogMsg (VERB_HIGH, "Reading double precision configuration as single precision");
		} else {
			LogError ("Input error: Invalid precision");
			exit(1);
		}
	}

	/*	Create axion field	*/

	if (totlZ % zGrid)
	{
		LogError ("Error: Geometry not valid. Try a different partitioning");
		exit (1);
	}
	else
		sizeZ = totlZ/zGrid;

	prof.stop();
	prof.add(std::string("Read configuration"), 0, 0);

	if (!strcmp(fStr, "Saxion"))
	{
		*axion = new Scalar(sizeN, sizeZ, precision, cDev, zTmp, lowmem, zGrid, FIELD_SAXION,    lType, CONF_NONE, 0, 0);
		slab   = (hsize_t) ((*axion)->Surf()*2);
	} else if (!strcmp(fStr, "Axion")) {
		*axion = new Scalar(sizeN, sizeZ, precision, cDev, zTmp, lowmem, zGrid, FIELD_AXION,     lType, CONF_NONE, 0, 0);
		slab   = (hsize_t) ((*axion)->Surf());
	} else if (!strcmp(fStr, "Axion Mod")) {
		*axion = new Scalar(sizeN, sizeZ, precision, cDev, zTmp, lowmem, zGrid, FIELD_AXION_MOD, lType, CONF_NONE, 0, 0);
		slab   = (hsize_t) ((*axion)->Surf());
	} else {
		LogError ("Input error: Invalid field type");
		exit(1);
	}

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

	for (hsize_t zDim=0; zDim<((hsize_t) (*axion)->Depth()); zDim++)
	{
		/*	Select the slab in the file	*/

		offset = (((hsize_t) (myRank*(*axion)->Depth()))+zDim)*slab;
		H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
		H5Sselect_hyperslab(vSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

		/*	Read raw data	*/

		auto mErr = H5Dread (mset_id, dataType, memSpace, mSpace, plist_id, (static_cast<char *> ((*axion)->mCpu())+slab*(1+zDim)*dataSize));
		auto vErr = H5Dread (vset_id, dataType, memSpace, vSpace, plist_id, (static_cast<char *> ((*axion)->vCpu())+slab*zDim*dataSize));

		if ((mErr < 0) || (vErr < 0)) {
			LogError ("Error reading dataset from file");
			return;
		}
	}

	/*	Close the dataset	*/

	H5Sclose (mSpace);
	H5Sclose (vSpace);
	H5Dclose (mset_id);
	H5Dclose (vset_id);
	H5Sclose (memSpace);

	/*	Close the file		*/

	H5Pclose (plist_id);
	H5Fclose (file_id);

	if (cDev == DEV_GPU)
		(*axion)->transferDev(FIELD_MV);

	prof.stop();
	prof.add(std::string("Read configuration"), 0, (2.*totlZ*slab*(*axion)->DataSize() + 77.)*1.e-9);

	LogMsg (VERB_NORMAL, "Read %lu bytes", ((size_t) totlZ)*slab*2 + 77);

	/*	Fold the field		*/

	Folder munge(*axion);
	munge(FOLD_ALL);

}



/*	Creates a hdf5 file to write all the measurements	*/
void	createMeas (Scalar *axion, int index)
{
	hid_t	plist_id, dataType;

	char	prec[16], fStr[16], lStr[16], icStr[16], vStr[32], smStr[16], dStr[16], rStr[16];
	int	length = 32;

//	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize;

	int myRank = commRank();

	int cSteps = dump*index;
	hsize_t totlZ = axion->TotalDepth();
	hsize_t tmpS  = axion->Length();

	tSize  = axion->TotalSize();
	slabSz = tmpS*tmpS;
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
			length = strlen(prec)+1;
		}

		break;

		case FIELD_DOUBLE:
		{
			dataType = H5T_NATIVE_DOUBLE;
			dataSize = sizeof(double);

			sprintf(prec, "Double");
			length = strlen(prec)+1;
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

		default:
			LogError ("Error: Invalid field type. How did you get this far?");
			exit(1);
			break;
	}

	double	llPhys = LL;

	switch (axion->Lambda())
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

	switch (vqcdType)
	{
		case	VQCD_1:
			sprintf(vStr, "VQcd 1");
			break;

		case	VQCD_2:
			sprintf(vStr, "VQcd 2");
			break;

		case	VQCD_1_PQ_2:
			sprintf(vStr, "VQcd 1 Peccei-Quinn 2");
			break;

		default:
			sprintf(vStr, "VQCD_1");
			break;
	}

	switch (vqcdTypeDamp)
	{
		case	VQCD_DAMP_RHO:
			sprintf(dStr, "Rho");
			break;

		case	VQCD_DAMP_ALL:
			sprintf(dStr, "All");
			break;

		default:
		case	VQCD_NONE:
			sprintf(dStr, "None");
			break;
	}

	switch (vqcdTypeRhoevol)
	{
		case	VQCD_EVOL_RHO:
			sprintf(rStr, "Only Rho");
			break;

		default:
		case	VQCD_NONE:
			sprintf(rStr, "Full");
			break;
	}

	/*	Write header	*/

	hid_t attr_type;

	/*	Attributes	*/

	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size   (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	double maa = axionmass((*axion->zV()), nQcd, zthres, zrestore);

	writeAttribute(meas_id, fStr,   "Field type",    attr_type);
	writeAttribute(meas_id, prec,   "Precision",     attr_type);
	writeAttribute(meas_id, &tmpS,  "Size",          H5T_NATIVE_UINT);
	writeAttribute(meas_id, &totlZ, "Depth",         H5T_NATIVE_UINT);
	writeAttribute(meas_id, &msa,   "Saxion mass",   H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &maa,   "Axion mass",    H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &sizeL, "Physical size", H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, axion->zV(),  "z",       H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &zInit, "zInitial",      H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &zFinl, "zFinal",        H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &nSteps,"nSteps",        H5T_NATIVE_INT);
	writeAttribute(meas_id, &cSteps,"Current step",  H5T_NATIVE_INT);

	/*	Create a group for specific header data	*/
	hid_t vGrp_id = H5Gcreate2(meas_id, "/potential", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	double shift = saxionshift(maa, llPhys, vqcdType);
	indi3 =  maa/pow(*axion->zV(), nQcd*0.5);

	writeAttribute(vGrp_id, &lStr,  "Lambda type",   attr_type);
	writeAttribute(vGrp_id, &LL,    "Lambda",        H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &vStr,  "VQcd type",     attr_type);
	writeAttribute(vGrp_id, &dStr,  "Damping type",  attr_type);
	writeAttribute(vGrp_id, &rStr,  "Evolution type",attr_type);
	writeAttribute(vGrp_id, &nQcd,  "nQcd",          H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &gammo, "Gamma",         H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &shift, "Shift",         H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &indi3, "Indi3",         H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &zthres,"z Threshold",   H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &zrestore,"z Restore",   H5T_NATIVE_DOUBLE);

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

		case	CONF_TKACHEV:
			sprintf(icStr, "Tkachev");
			writeAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
			writeAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
			writeAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
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

	if (opened) {
		H5Pclose (mlist_id);
		H5Fclose (meas_id);
	}

	opened = false;
	header = false;

	meas_id = -1;

	LogMsg (VERB_NORMAL, "Measurement file successfuly closed");
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

	/*	Might be reduced	*/
	writeAttribute(group_id, &redlX, "Size",  H5T_NATIVE_UINT);
	writeAttribute(group_id, &redlZ, "Depth", H5T_NATIVE_UINT);

	/*	String metadata		*/
	writeAttribute(group_id, &(strDat.strDen), "String number",    H5T_NATIVE_HSIZE);
	writeAttribute(group_id, &(strDat.strChr), "String chirality", H5T_NATIVE_HSSIZE);
	writeAttribute(group_id, &(strDat.wallDn), "Wall number",      H5T_NATIVE_HSIZE);

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

void	writeEnergy	(Scalar *axion, void *eData_)
{
	hid_t	group_id;

	double	*eData = static_cast<double *>(eData_);

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request. %d %d", header, opened);
		return;
	}

	/*	Start profiling		*/

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	LogMsg (VERB_NORMAL, "Writing energy data");

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

	int totalBytes = 40;

	writeAttribute(group_id, &eData[TH_GRX],  "Axion Gr X", H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &eData[TH_GRY],  "Axion Gr Y", H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &eData[TH_GRZ],  "Axion Gr Z", H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &eData[TH_KIN],  "Axion Kinetic", H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &eData[TH_POT],  "Axion Potential", H5T_NATIVE_DOUBLE);

	if	(axion->Field() == FIELD_SAXION)
	{
		writeAttribute(group_id, &eData[RH_GRX],  "Saxion Gr X", H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &eData[RH_GRY],  "Saxion Gr Y", H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &eData[RH_GRZ],  "Saxion Gr Z", H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &eData[RH_KIN],  "Saxion Kinetic", H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &eData[RH_POT],  "Saxion Potential", H5T_NATIVE_DOUBLE);

		totalBytes += 40;
	}

	/*	TODO	Distinguish the different versions of the potentials	*/

	/*	Close the group		*/
	H5Gclose (group_id);

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
	if (H5Dwrite(dataSet, dataType, dataSpace, sSpace, H5P_DEFAULT, static_cast<char*>(axion->mCpu()) + S0*dataSize) < 0)
		LogError ("Error: couldn't write point data to file");

	/*	Close everything		*/
	H5Sclose (sSpace);
	H5Dclose (dataSet);
	H5Sclose (dataSpace);
	H5Gclose (group_id);

	LogMsg (VERB_NORMAL, "Written %lu bytes", dataSize);
}

void	writeArray (double *aData, size_t aSize, const char *group, const char *dataName)
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

	if (myRank == 0) {
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
	hid_t	eGrp_id, group_id, rset_id, tset_id, plist_id, chunk_id;
	hid_t	rSpace, tSpace, memSpace, dataType, totalSpace;
	hsize_t	total, slice, slab, offset, rOff;

	char	prec[16], fStr[16];
	int	length = 8;

	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize;

	int myRank = commRank();

	LogMsg (VERB_NORMAL, "Writing energy density to Hdf5 measurement file");
	LogMsg (VERB_NORMAL, "");

	if ((fMap & MAP_RHO) && (axion->Field() & FIELD_AXION)) {
	        LogMsg (VERB_NORMAL, "Requested MAP_RHO with axion field. Request will be ignored");
	        fMap ^= MAP_RHO;
	}

	if ((fMap & MAP_ALL) == MAP_NONE) {
	        LogMsg (VERB_NORMAL, "Nothing to map. Skipping writeEDens");
	        return;
	}

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
	rOff  = ((hsize_t) (totlX))*((hsize_t) (totlX))*(axion->Depth()+2);

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

	/*	Create a group for energy density if it doesn't exist	*/
	status = H5Lexists (eGrp_id, "density", H5P_DEFAULT);

	if (!status)
		group_id = H5Gcreate2(eGrp_id, "density", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(eGrp_id, "density", H5P_DEFAULT);		// Group exists
			LogMsg (VERB_HIGH, "Group /energy/density exists");
		} else {
			LogError ("Error: can't check whether group /energy/density exists");
			prof.stop();
			return;
		}
	}

	/*	Might be reduced	*/
	writeAttribute(group_id, &redlX, "Size",  H5T_NATIVE_UINT);
	writeAttribute(group_id, &redlZ, "Depth", H5T_NATIVE_UINT);

	/*	Create a dataset for the whole axion data	*/

	char rhoCh[24] = "/energy/density/rho";
	char thCh[24]  = "/energy/density/theta";

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
	hsize_t slb  = slabSz;
	hsize_t lSz  = sizeN;
	char *dataM  = static_cast<char *>(axion->mCpu());
	char *dataV  = static_cast<char *>(axion->mCpu());
	char mCh[16] = "/map/m";
	char vCh[16] = "/map/v";

	LogMsg (VERB_NORMAL, "Writing 2D maps to Hdf5 measurement file");
	LogMsg (VERB_NORMAL, "");

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request");
		return;
	}

	/*	Start profiling		*/

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	if (axion->Field() == FIELD_SAXION) {
		lSz *= 2;
		slb *= 2;
	}

	if (axion->Precision() == FIELD_DOUBLE) {
		dataV += slb*(axion->Depth()+1)*sizeof(double);
		dataType = H5T_NATIVE_DOUBLE;
	} else {
		dataV += slb*(axion->Depth()+1)*sizeof(float);
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
		LogMsg (VERB_NORMAL, "Folded configuration, unfolding 2D slice");
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
	vSet_id = H5Dcreate (meas_id, vCh, dataType, mapSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

	if (mSet_id < 0 || vSet_id < 0)
	{
		LogError ("Fatal error creating datasets");
		prof.stop();
		exit (0);
	}

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/
	mSpace = H5Dget_space (mSet_id);
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

	if (myRank == 0) {
		H5Sselect_hyperslab(vSpace, H5S_SELECT_SET, &offset, NULL, &slb, NULL);
	} else {
		H5Sselect_none(vSpace);
		//dataV = NULL;
	}

	if (H5Dwrite (vSet_id, dataType, mapSpace, vSpace, H5P_DEFAULT, dataV) < 0)
	{
		LogError ("Error writing dataset /map/v");
		prof.stop();
		exit(0);
	}

	LogMsg (VERB_HIGH, "Write energy map successful");

	/*	Close the dataset	*/
	H5Dclose (mSet_id);
	H5Dclose (vSet_id);
	H5Sclose (mSpace);
	H5Sclose (vSpace);

	H5Sclose (mapSpace);
	H5Pclose (chunk_id);
	H5Gclose (group_id);
	prof.stop();

	prof.add(std::string("Write Map"), 0, 2.e-9*slb*dataSize);
	LogMsg (VERB_NORMAL, "Written %lu bytes", slb*dataSize*2);
}

void	writeMapHdf5	(Scalar *axion)
{
	writeMapHdf5s	(axion, 0);
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
