#include <cstdio>
#include <cstdlib>
#include <cstring>
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

hid_t	meas_id = -1, mlist_id;
hsize_t	tSize, slabSz, sLz;
bool	opened = false, header = false;

bool	mDisabled = true;
H5E_auto2_t eFunc;
void	   *cData;

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

	if ((attr   = H5Aopen_by_name (file_id, "/", name, H5P_DEFAULT, H5P_DEFAULT)) < 0)
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
		LogMsg (VERB_NORMAL, "Warning: retrieve current hdf5 error stack");
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

	char	prec[16], fStr[16];
	int	length = 8;

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

	sprintf(base, "%s.%05d", outName, index);

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
	uint totlZ = sizeZ*zGrid;
	uint tmpS  = sizeN;

	switch (axion->Field())
	{
		case 	FIELD_SAXION:
		{
			total = ((hsize_t) tmpS)*((hsize_t) tmpS)*((hsize_t) (totlZ*2));
			slab  = (hsize_t) (axion->Surf()*2);

			sprintf(fStr, "Saxion");
		}
		break;

		case	FIELD_AXION:
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

	/*	Write header	*/
	hid_t attr_type;

	/*	Attributes	*/

	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size   (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	writeAttribute(file_id, fStr,   "Field type",    attr_type);
	writeAttribute(file_id, prec,   "Precision",     attr_type);
	writeAttribute(file_id, &tmpS,  "Size",          H5T_NATIVE_UINT);
	writeAttribute(file_id, &totlZ, "Depth",         H5T_NATIVE_UINT);
	writeAttribute(file_id, &LL,    "Lambda",        H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &nQcd,  "nQcd",          H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &sizeL, "Physical size", H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, axion->zV(),  "z",       H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &zInit, "zInitial",      H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &zFinl, "zFinal",        H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &nSteps,"nSteps",        H5T_NATIVE_INT);
	writeAttribute(file_id, &cSteps,"Current step",  H5T_NATIVE_INT);

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

	char	prec[16], fStr[16];
	int	length = 8;

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

	sprintf(base, "%s.%05d", outName, index);

	/*	Open the file and release the plist	*/

	if ((file_id = H5Fopen (base, H5F_ACC_RDONLY, plist_id)) < 0)
	{
		*axion == NULL;
		LogError ("Error opening file %s", base);
		return;
	}

	H5Pclose(plist_id);

	/*	Attributes	*/

	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	double	zTmp, zTfl, zTin;
	uint	tStep, cStep, totlZ;

	readAttribute (file_id, fStr,   "Field type",   attr_type);
	readAttribute (file_id, prec,   "Precision",    attr_type);
	readAttribute (file_id, &sizeN, "Size",         H5T_NATIVE_UINT);
	readAttribute (file_id, &totlZ, "Depth",        H5T_NATIVE_UINT);
	readAttribute (file_id, &nQcd,  "nQcd",         H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &LL,    "Lambda",       H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &sizeL, "Physical size",H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &zTmp,  "z",            H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &zTin,  "zInitial",     H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &zTfl,  "zFinal",       H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &tStep, "nSteps",       H5T_NATIVE_INT);
	readAttribute (file_id, &cStep, "Current step", H5T_NATIVE_INT);

	H5Tclose (attr_type);

	if (!uPrec)
	{
		if (!strcmp(prec, "Double"))
		{
			precision = FIELD_DOUBLE;
			dataType  = H5T_NATIVE_DOUBLE;
			dataSize  = sizeof(double);
		} else if (!strcmp(prec, "Single")) {
			precision = FIELD_SINGLE;
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


	if (!strcmp(fStr, "Saxion"))
	{
		*axion = new Scalar(sizeN, sizeZ, precision, cDev, zTmp, lowmem, zGrid, FIELD_SAXION,  CONF_NONE, 0, 0, NULL);
		slab   = (hsize_t) ((*axion)->Surf()*2);
	} else if (!strcmp(fStr, "Axion")) {
		*axion = new Scalar(sizeN, sizeZ, precision, cDev, zTmp, lowmem, zGrid, FIELD_AXION, CONF_NONE, 0, 0, NULL);
		slab   = (hsize_t) ((*axion)->Surf());
	} else {
		LogError ("Input error: Invalid field type");
		exit(1);
	}

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

	prof.stop();
	prof.add(std::string("Read configuration"), 0, (2.*totlZ*slab + 77.)*1.e-9);

	LogMsg (VERB_NORMAL, "Read %lu bytes", ((size_t) totlZ)*slab*2 + 77);

	/*	Fold the field		*/

	Folder munge(*axion);
	munge(FOLD_ALL);
}

/*	Creates a hdf5 file to write all the measurements	*/
void	createMeas (Scalar *axion, int index)
{
	hid_t	plist_id, dataType;

	char	prec[16], fStr[16];
	int	length = 8;

//	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize;

	int myRank = commRank();

	int cSteps = dump*index;
	hsize_t totlZ = sizeZ*zGrid;
	hsize_t tmpS  = sizeN;

	tSize  = axion->TotalSize();
	slabSz = tmpS*tmpS;
	sLz    = sizeZ;

	if (myRank != 0)	// Only rank 0 writes measurement data
		return;

	LogMsg (VERB_NORMAL, "Creating measurement file with index %d", index);

	if (opened)
	{
		LogError ("Error: Hdf5 measurement file already opened");
		return;
	}

	/*	Set up parallel access with Hdf5	*/

//	We give up pHdf5 for the measurements because compression is not supported
//	plist_id = H5Pcreate (H5P_FILE_ACCESS);
//	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];

	sprintf(base, "%s.m.%05d", outName, index);

	/*	Create the file and release the plist	*/
	if ((meas_id = H5Fcreate (base, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)) < 0)	//plist_id)) < 0)
	{
		LogError ("Error creating file %s", base);
		return;
	}

	H5Pclose(plist_id);

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

		case	FIELD_AXION:
			sprintf(fStr, "Axion");
			break;

		default:
			LogError ("Error: Invalid field type. How did you get this far?");
			exit(1);
			break;
	}

	/*	Write header	*/

	hid_t attr_type;

	/*	Attributes	*/

	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size   (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	writeAttribute(meas_id, fStr,   "Field type",    attr_type);
	writeAttribute(meas_id, prec,   "Precision",     attr_type);
	writeAttribute(meas_id, &tmpS,  "Size",          H5T_NATIVE_HSIZE);
	writeAttribute(meas_id, &totlZ, "Depth",         H5T_NATIVE_HSIZE);
	writeAttribute(meas_id, &LL,    "Lambda",        H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &nQcd,  "nQcd",          H5T_NATIVE_INT);
	writeAttribute(meas_id, &sizeL, "Physical size", H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, axion->zV(),  "z",       H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &zInit, "zInitial",      H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &zFinl, "zFinal",        H5T_NATIVE_DOUBLE);
	writeAttribute(meas_id, &nSteps,"nSteps",        H5T_NATIVE_INT);
	writeAttribute(meas_id, &cSteps,"Current step",  H5T_NATIVE_INT);

	H5Tclose (attr_type);

	/*	Create plist for collective write	*/

//	mlist_id = H5Pcreate(H5P_DATASET_XFER);
//	H5Pset_dxpl_mpio(mlist_id,H5FD_MPIO_COLLECTIVE);

	header = true;

	LogMsg (VERB_NORMAL, "Measurement file %s successfuly openend", base);

	return;
}


void	destroyMeas ()
{
	if (commRank() != 0)
		return;

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

void	writeString	(void *str, StringData strDat)
{
	hid_t	totalSpace, chunk_id, group_id, sSet_id, sSpace, memSpace;
	hid_t	datum;

	bool	mpiCheck = true;

	int myRank = commRank();

	const hsize_t maxD[1] = { H5S_UNLIMITED };
	char *strData = static_cast<char *>(str);
	char sCh[16] = "/string/data";

	Profiler &prof = getProfiler(PROF_HDF5);

	if (myRank == 0)
	{
		/*	Start profiling		*/
		prof.start();

		LogMsg (VERB_NORMAL, "Writing string data");

		if (header == false || opened == false)
		{
			LogError ("Error: measurement file not opened. Ignoring write request. %d %d\n", header, opened);
			mpiCheck = false;
			goto bCastAndExit;		// HELL
		}

		/*	Create space for writing the raw data to disk with chunked access	*/
		totalSpace = H5Screate_simple(1, &tSize, maxD);	// Whole data

		if (totalSpace < 0)
		{
			LogError ("Fatal error H5Screate_simple");
			mpiCheck = false;
			goto bCastAndExit;		// Hurts my eyes
		}

		/*	Set chunked access and dynamical compression	*/
		if ((chunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0) {
			LogError ("Fatal error H5Pcreate");
			mpiCheck = false;
			goto bCastAndExit;		// Really?
		}

		if (H5Pset_chunk (chunk_id, 1, &slabSz) < 0) {
			LogError ("Fatal error H5Pset_chunk");
			mpiCheck = false;
			goto bCastAndExit;		// You MUST be kidding
		}

		if (H5Pset_deflate (chunk_id, 9) < 0)	// Maximum compression
		{
			LogError ("Error: couldn't set compression level to 9");
			mpiCheck = false;
			goto bCastAndExit;		// NOOOOOO
		}

		/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
		if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0)
		{
			LogError ("Fatal error H5Pset_alloc_time");
			mpiCheck = false;
			goto bCastAndExit;		// Aaaaaaaaaaarrggggggghhhh
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
				mpiCheck - false;
				goto bCastAndExit;
			}
		}

		writeAttribute(group_id, &(strDat.strDen), "String number",    H5T_NATIVE_HSIZE);
		writeAttribute(group_id, &(strDat.strChr), "String chirality", H5T_NATIVE_HSSIZE);
		writeAttribute(group_id, &(strDat.wallDn), "Wall number",      H5T_NATIVE_HSIZE);

		/*	Create a dataset for string data	*/
		sSet_id = H5Dcreate (meas_id, sCh, H5T_NATIVE_CHAR, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

		if (sSet_id < 0)
		{
			LogError ("Fatal error creating dataset");
			mpiCheck = false;
			goto bCastAndExit;		// adslfkjñasldkñkja
		}

		/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

		sSpace = H5Dget_space (sSet_id);
		memSpace = H5Screate_simple(1, &slabSz, NULL);	// Slab
	}

	bCastAndExit:

	MPI_Bcast(&mpiCheck, sizeof(mpiCheck), MPI_CHAR, 0, MPI_COMM_WORLD);

	if (mpiCheck == false) {	// Prevent non-zero ranks deadlock in case there is an error with rank 0. MPI would force exit anyway..
		if (myRank == 0)
			prof.stop();
		return;
	}

	int tSz = commSize(), test = myRank;

	commSync();

	for (int rank=0; rank<tSz; rank++)
	{
		for (hsize_t zDim=0; zDim<((hsize_t) sLz); zDim++)
		{
			if (myRank != 0)
			{
				if (myRank == rank) {
					LogMsg (VERB_HIGH, "Sending %lu bytes to rank 0", slabSz);
					MPI_Send(&(strData[0]) + slabSz*zDim, slabSz, MPI_CHAR, 0, rank, MPI_COMM_WORLD);
				}
			} else {
				if (rank != 0) {
					LogMsg (VERB_HIGH, "Receiving %lu bytes from rank %d", slabSz, rank);
					MPI_Recv(&(strData[0]) + slabSz*zDim, slabSz, MPI_CHAR, rank, rank, MPI_COMM_WORLD, NULL);
				}

				/*	Select the slab in the file	*/
				hsize_t offset = (((hsize_t) (rank*sLz))+zDim)*slabSz;
				H5Sselect_hyperslab(sSpace, H5S_SELECT_SET, &offset, NULL, &slabSz, NULL);

				/*	Write raw data	*/
				H5Dwrite (sSet_id, H5T_NATIVE_CHAR, memSpace, sSpace, H5P_DEFAULT, (strData)+slabSz*zDim);
			}

			commSync();
		}
	}

	/*	Close the dataset	*/

	if (myRank == 0) {
		H5Dclose (sSet_id);
		H5Sclose (sSpace);
		H5Sclose (memSpace);

		H5Sclose (totalSpace);
		H5Pclose (chunk_id);
		H5Gclose (group_id);

		prof.stop();
		prof.add(std::string("Write strings"), 0, 1e-9*(slabSz*sLz));
	}

	LogMsg (VERB_NORMAL, "Written %lu bytes to disk", slabSz*sLz);

	commSync();
}

void	writeEnergy	(Scalar *axion, void *eData_)
{
	hid_t	group_id;

	double	*eData = static_cast<double *>(eData_);

	if (commRank() != 0)
		return;

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

	if (commRank() != 0)
		return;

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

void	writeArray (Scalar *axion, void *aData, size_t aSize, const char *group, const char *dataName)
{
	hid_t	group_id, dataSpace, sSpace, dataSet;
	hsize_t dims[1] = { aSize };

	size_t	dataSize;

	if (commRank() != 0)
		return;

	LogMsg (VERB_NORMAL, "Writing array to measurement file");

	if (header == false || opened == false)
	{
		LogError ("Error: measurement file not opened. Ignoring write request.\n");
		return;
	}

	/*	Start profiling		*/

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	/*	Create the group for the data if it doesn't exist	*/
	auto status = H5Lexists (meas_id, group, H5P_DEFAULT);	// Create group if it doesn't exists

	if (!status)
		group_id = H5Gcreate2(meas_id, group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(meas_id, group, H5P_DEFAULT);		// Group exists, but it shouldn't
			LogMsg(VERB_NORMAL, "Warning: group %s exists!", group);	// Since this is weird, log it
		} else {
			LogError ("Error: can't check whether group %s exists", group);
			return;
		}
	}
/*	disableErrorStack();

	if (H5Gget_objinfo (meas_id, group, 0, NULL) < 0)	// Create group if it doesn't exists
		group_id = H5Gcreate2(meas_id, group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else
		group_id = H5Gopen2(meas_id, group, H5P_DEFAULT);

	enableErrorStack();
*/
	/*	Create dataset	*/
	dataSpace = H5Screate_simple(1, dims, NULL);
	dataSet   = H5Dcreate(group_id, dataName, H5T_NATIVE_DOUBLE, dataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	sSpace	  = H5Dget_space (dataSet);

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

	prof.stop();
	prof.add(std::string("Write array"), 0, 1e-9*(aSize*8));

	LogMsg (VERB_NORMAL, "Written %lu bytes to disk", aSize*8);
}

void	writeEDens (Scalar *axion, int index)
{
	hid_t	file_id, group_id, mset_id, plist_id, chunk_id;
	hid_t	mSpace, memSpace, dataType, totalSpace;
	hsize_t	total, slice, slab, offset;

	char	prec[16], fStr[16];
	int	length = 8;

	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize;

	int myRank = commRank();

	LogMsg (VERB_NORMAL, "Writing energy density to Hdf5 measurement file");
	LogMsg (VERB_NORMAL, "");

	/*      Start profiling         */

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	if (axion->m2Cpu() == nullptr) {
		LogError ("You seem to be using the lowmem option");
		prof.stop();
		return;
	}

	/*	Set up parallel access with Hdf5	*/
	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];

	sprintf(base, "%s.m.%05d", outName, index);

	/*	Broadcast the values of opened/header	*/
	MPI_Bcast(&opened, sizeof(opened), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

	/*	If the measurement file is opened, we reopen it with parallel access	*/
	if (opened == true)
	{
		destroyMeas();

		if ((file_id = H5Fopen (base, H5F_ACC_RDWR, plist_id)) < 0)
		{
			LogError ("Error opening file %s", base);
			prof.stop();
			return;
		}
	} else {
		/*	Else we create the file		*/
		if ((file_id = H5Fcreate (base, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) < 0)
		{
			LogError ("Error creating file %s", base);
			prof.stop();
			return;
		}
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
	uint totlZ = sizeZ*zGrid;
	uint tmpS  = sizeN;

	switch (axion->Field())
	{
		case 	FIELD_SAXION:
		{
			total = ((hsize_t) tmpS)*((hsize_t) tmpS)*((hsize_t) (totlZ*2));
			slab  = (hsize_t) (axion->Surf()*2);

			sprintf(fStr, "Saxion");
		}
		break;

		case	FIELD_AXION:
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

	if (header == false)
	{
		/*	Write header	*/
		hid_t attr_type;

		/*	Attributes	*/
		attr_type = H5Tcopy(H5T_C_S1);
		H5Tset_size   (attr_type, length);
		H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

		writeAttribute(file_id, fStr,   "Field type",    attr_type);
		writeAttribute(file_id, prec,   "Precision",     attr_type);
		writeAttribute(file_id, &tmpS,  "Size",          H5T_NATIVE_UINT);
		writeAttribute(file_id, &totlZ, "Depth",         H5T_NATIVE_UINT);
		writeAttribute(file_id, &LL,    "Lambda",        H5T_NATIVE_DOUBLE);
		writeAttribute(file_id, &nQcd,  "nQcd",          H5T_NATIVE_DOUBLE);
		writeAttribute(file_id, &sizeL, "Physical size", H5T_NATIVE_DOUBLE);
		writeAttribute(file_id, axion->zV(),  "z",       H5T_NATIVE_DOUBLE);
		writeAttribute(file_id, &zInit, "zInitial",      H5T_NATIVE_DOUBLE);
		writeAttribute(file_id, &zFinl, "zFinal",        H5T_NATIVE_DOUBLE);
		writeAttribute(file_id, &nSteps,"nSteps",        H5T_NATIVE_INT);
		writeAttribute(file_id, &cSteps,"Current step",  H5T_NATIVE_INT);

		H5Tclose (attr_type);

		header = true;
	}

	commSync();

	/*	Create plist for collective write	*/
	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

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

	/*	Create a group for string data if it doesn't exist	*/
	auto status = H5Lexists (file_id, "/energy", H5P_DEFAULT);	// Create group if it doesn't exists

	if (!status)
		group_id = H5Gcreate2(file_id, "/energy", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0)
			group_id = H5Gopen2(file_id, "/energy", H5P_DEFAULT);		// Group exists
		else {
			LogError ("Error: can't check whether group /energy exists");
			prof.stop();
			return;
		}
	}

	/*	Create a dataset for the whole axion data	*/

	char mCh[24] = "/energy/density";

	mset_id = H5Dcreate (file_id, mCh, dataType, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

	commSync();

	if (mset_id < 0)
	{
		LogError("Error creating dataset");
		prof.stop();
		exit (0);
	}

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

	mSpace = H5Dget_space (mset_id);
	memSpace = H5Screate_simple(1, &slab, NULL);	// Slab

	commSync();

	LogMsg (VERB_HIGH, "Rank %d ready to write", myRank);

	for (hsize_t zDim=0; zDim<((hsize_t) axion->Depth()); zDim++)
	{
		/*	Select the slab in the file	*/
		offset = (((hsize_t) (myRank*axion->Depth())) + zDim)*slab;
		H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

		/*	Write raw data	*/
		// CAMBIAR A mCpu!!!
		auto mErr = H5Dwrite (mset_id, dataType, memSpace, mSpace, plist_id, (static_cast<char *> (axion->m2Cpu())+slab*zDim*dataSize));

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
	H5Fclose (file_id);

        prof.stop();
	prof.add(std::string("Write energy map"), 0., (2.*total*dataSize + 78.)*1e-9);

	LogMsg (VERB_NORMAL, "Written %lu bytes", total*dataSize*2 + 78);

	/*	If there was a file opened for measurements, open it again	*/

	if (opened == true)
	{
		hid_t	plist_id;

		if (myRank != 0)	// Only rank 0 writes measurement data
			return;

		LogMsg (VERB_NORMAL, "Opening measurement file");

		/*	This would be weird indeed	*/

		if (meas_id >= 0)
		{
			LogError ("Error, a hdf5 file is already opened");
			return;
		}

		/*	Open the file and release the plist	*/

		if ((meas_id = H5Fopen (base, H5F_ACC_RDWR, H5P_DEFAULT)) < 0)
		{
			LogError ("Error opening file %s", base);
			return;
		}

		H5Pclose(plist_id);
	}
}

void	writeSpectrum (Scalar *axion, void *spectrumK, void *spectrumG, void *spectrumV, size_t powMax, bool power)
{
	hid_t	group_id, dataSpace, kSpace, gSpace, vSpace, dataSetK, dataSetG, dataSetV;
	herr_t	status;
	hsize_t dims[1] = { powMax };

	char	dataName[32];
	char	*sK = static_cast<char*>(spectrumK);
	char	*sG = static_cast<char*>(spectrumG);
	char	*sV = static_cast<char*>(spectrumV);

	if (commRank() != 0)
		return;

	if (header == false || opened == false)
	{
		printf("Error: measurement file not opened. Ignoring write request. %d %d\n",header,opened);
		return;
	}

	if (power == true)
		sprintf(dataName, "/pSpectrum");
	else
		sprintf(dataName, "/nSpectrum");

	/*	Create a group for the spectra if it doesn't exist	*/
	status = H5Eset_auto(H5E_DEFAULT, NULL, NULL);	// Turn off error output, we don't want trash if the group doesn't exist

	if (H5Gget_objinfo (meas_id, dataName, 0, NULL) < 0)	// Create group if it doesn't exist
		group_id = H5Gcreate2(meas_id, dataName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else
		group_id = H5Gopen2(meas_id, dataName, H5P_DEFAULT);

//	status = H5Eset_auto(H5E_DEFAULT, H5Eprint2, stderr);	// Restore error output

	/*	Create datasets	*/
	dataSpace = H5Screate_simple(1, dims, NULL);
	dataSetK  = H5Dcreate(group_id, "sK", H5T_NATIVE_DOUBLE, dataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataSetG  = H5Dcreate(group_id, "sG", H5T_NATIVE_DOUBLE, dataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataSetV  = H5Dcreate(group_id, "sV", H5T_NATIVE_DOUBLE, dataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	kSpace	  = H5Dget_space (dataSetK);
	gSpace	  = H5Dget_space (dataSetG);
	vSpace	  = H5Dget_space (dataSetV);

	/*	Write spectrum data	*/
	status = H5Dwrite(dataSetK, H5T_NATIVE_DOUBLE, dataSpace, kSpace, H5P_DEFAULT, sK);
	status = H5Dwrite(dataSetG, H5T_NATIVE_DOUBLE, dataSpace, gSpace, H5P_DEFAULT, sG);
	status = H5Dwrite(dataSetV, H5T_NATIVE_DOUBLE, dataSpace, vSpace, H5P_DEFAULT, sV);

	/*	Close everything		*/
	H5Sclose (kSpace);
	H5Sclose (gSpace);
	H5Sclose (vSpace);
	H5Dclose (dataSetK);
	H5Dclose (dataSetG);
	H5Dclose (dataSetV);
	H5Sclose (dataSpace);
	H5Gclose (group_id);
}

void	writeMapHdf5	(Scalar *axion)
{
	hid_t	mapSpace, chunk_id, group_id, mSet_id, vSet_id, mSpace, vSpace,  dataType;
	hsize_t	dataSize = axion->DataSize();

	int myRank = commRank();

	if (myRank != 0)
		return;

	const hsize_t maxD[1] = { H5S_UNLIMITED };
	char *dataM  = static_cast<char *>(axion->mCpu());
	char *dataV  = static_cast<char *>(axion->vCpu());
	char mCh[16] = "/map/m";
	char vCh[16] = "/map/v";

	if (header == false || opened == false)
	{
		printf("Error: measurement file not opened. Ignoring write request. %d %d\n", header, opened);
		return;
	}

	if (axion->Precision() == FIELD_DOUBLE)
		dataType = H5T_NATIVE_DOUBLE;
	else
		dataType = H5T_NATIVE_FLOAT;

	/*	Create space for writing the raw data to disk with chunked access	*/
	mapSpace = H5Screate_simple(1, &slabSz, NULL);	// Whole data

	if (mapSpace < 0)
	{
		printf ("Fatal error H5Screate_simple\n");
		exit (1);
	}

	/*	Set chunked access and dynamical compression	*/

	herr_t status;

	chunk_id = H5Pcreate (H5P_DATASET_CREATE);

	if (chunk_id < 0)
	{
		printf ("Fatal error H5Pcreate\n");
		exit (1);
	}

	status = H5Pset_chunk (chunk_id, 1, &slabSz);

	if (status < 0)
	{
		printf ("Fatal error H5Pset_chunk\n");
		exit (1);
	}

	status = H5Pset_deflate (chunk_id, 9);	// Maximum compression, hoping that the map is a bunch of zeroes

	if (status < 0)
	{
		printf ("Fatal error H5Pset_deflate\n");
		exit (1);
	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	status = H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER);

	if (status < 0)
	{
		printf ("Fatal error H5Pset_alloc_time\n");
		exit (1);
	}

	/*	Create a group for map data	*/
	group_id = H5Gcreate2(meas_id, "/map", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	/*	Create a dataset for map data	*/

	//sSet_id = H5Dcreate (meas_id, sCh, datum, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);
	mSet_id = H5Dcreate (meas_id, mCh, dataType, mapSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);
	vSet_id = H5Dcreate (meas_id, vCh, dataType, mapSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

	if (mSet_id < 0 || vSet_id < 0)
	{
		printf	("Fatal error.\n");
		exit (0);
	}

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

	mSpace = H5Dget_space (mSet_id);
	vSpace = H5Dget_space (vSet_id);

	/*	Select the slab in the file	*/
//	hsize_t offset = 0;
//	H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slabSz, NULL);
//	H5Sselect_hyperslab(vSpace, H5S_SELECT_SET, &offset, NULL, &slabSz, NULL);

	/*	Write raw data	*/
	H5Dwrite (mSet_id, dataType, mapSpace, mSpace, H5P_DEFAULT, dataM);
	H5Dwrite (vSet_id, dataType, mapSpace, vSpace, H5P_DEFAULT, dataV);
	

	/*	Close the dataset	*/

	H5Dclose (mSet_id);
	H5Dclose (vSet_id);
	H5Sclose (mSpace);
	H5Sclose (vSpace);

	H5Sclose (mapSpace);
	H5Pclose (chunk_id);
	H5Gclose (group_id);
}





void	reduceEDens (int index, uint newLx, uint newLz)
{
	hid_t	file_id, eset_id, plist_id, chunk_id, group_id;
	hid_t	eSpace, memSpace, totalSpace, dataType;
	hid_t	attr_type;

	hsize_t	slab, offset, nSlb, total;

	FieldPrecision	precision;

	char	prec[16], fStr[16];
	int	length = 8;

	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize, newSz;

	int myRank = commRank();

	void *axionIn  = nullptr;
	void *axionOut = nullptr;

	LogMsg (VERB_NORMAL, "Reading Hdf5 measurement file");
	LogMsg (VERB_NORMAL, "");

	/*	Start profiling		*/

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	/*	Set up parallel access with Hdf5	*/

	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char baseIn[256], baseOut[256];

	sprintf(baseIn, "%s.m.%05d", outName, index);

	/*	Open the file and release the plist	*/

	if ((file_id = H5Fopen (baseIn, H5F_ACC_RDONLY, plist_id)) < 0)
	{
		LogError ("Error opening file %s", baseIn);
		return;
	}

	H5Pclose(plist_id);

	/*	Attributes	*/

	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	double	zTmp, zTfl, zTin;
	uint	tStep, cStep, totlZ;

	readAttribute (file_id, fStr,   "Field type",   attr_type);
	readAttribute (file_id, prec,   "Precision",    attr_type);
	readAttribute (file_id, &sizeN, "Size",         H5T_NATIVE_UINT);
	readAttribute (file_id, &totlZ, "Depth",        H5T_NATIVE_UINT);
	readAttribute (file_id, &nQcd,  "nQcd",         H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &LL,    "Lambda",       H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &sizeL, "Physical size",H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &zTmp,  "z",            H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &zTin,  "zInitial",     H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &zTfl,  "zFinal",       H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &tStep, "nSteps",       H5T_NATIVE_INT);
	readAttribute (file_id, &cStep, "Current step", H5T_NATIVE_INT);

	H5Tclose (attr_type);

	if ((newLx > sizeN) || (newLz > totlZ)) {
		LogError ("Error: new size must be smaller");
		prof.stop();
		exit(1);
	}

	if (!uPrec)
	{
		if (!strcmp(prec, "Double"))
		{
			precision = FIELD_DOUBLE;
			dataType  = H5T_NATIVE_DOUBLE;
			dataSize  = sizeof(double);
		} else if (!strcmp(prec, "Single")) {
			precision = FIELD_SINGLE;
			dataType  = H5T_NATIVE_FLOAT;
			dataSize  = sizeof(float);
		} else {
			LogError ("Error reading file %s: Invalid precision %s", baseIn, prec);
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
			prof.stop();
			exit(1);
		}
	}

	if ((totlZ % zGrid) || (newLz % zGrid))
	{
		LogError ("Error: Geometry not valid. Try a different partitioning");
		prof.stop();
		exit (1);
	}

	sizeZ = totlZ/zGrid;
	newSz = newLz/zGrid;
	slab  = ((hsize_t) (sizeN)) * ((hsize_t) (sizeN));
	nSlb  = ((hsize_t) (newLx)) * ((hsize_t) (newLx));

	total = nSlb*newSz;

	trackAlloc(&axionIn,  (slab+1)*sizeZ*dataSize);	// The extra-slab is for FFTW with MPI, just in case
	trackAlloc(&axionOut, (slab+1)*sizeZ*dataSize*2);

	/*	Create plist for collective read	*/

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id,H5FD_MPIO_COLLECTIVE);

	/*	Open a dataset for the whole axion data	*/

	if ((eset_id = H5Dopen (file_id, "/energy/density", H5P_DEFAULT)) < 0) {
		prof.stop();
		LogError ("Error opening /energy/density dataset");
		return;
	}

	memSpace = H5Screate_simple(1, &slab, NULL);	// Slab
	eSpace   = H5Dget_space (eset_id);

	for (hsize_t zDim=0; zDim < ((hsize_t) sizeZ); zDim++)
	{
		/*	Select the slab in the file	*/

		offset = ((hsize_t) (myRank*sizeZ)+zDim)*slab;
		H5Sselect_hyperslab(eSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

		/*	Read raw data	*/

		auto eErr = H5Dread (eset_id, dataType, memSpace, eSpace, plist_id, static_cast<char *> (axionIn+slab*zDim*dataSize));

		if (eErr < 0) {
			LogError ("Error reading dataset from file");
			prof.stop();
			return;
		}
	}

	/*	Close the dataset	*/

	H5Sclose (eSpace);
	H5Dclose (eset_id);
	H5Sclose (memSpace);

	/*	Close the file		*/

	H5Pclose (plist_id);
	H5Fclose (file_id);

//	prof.add(std::string("Read data"), 0, (dataSize*totlZ*slab + 77.)*1.e-9);

	LogMsg (VERB_NORMAL, "Read %lu bytes", ((size_t) totlZ)*slab + 77);

	/*	FFT		*/
	LogMsg (VERB_HIGH, "Creating FFT plans");

	initFFT(precision);

	fftw_plan	planDoubleForward, planDoubleBackward;
	fftwf_plan	planSingleForward, planSingleBackward;

	switch (precision)
	{
		case FIELD_DOUBLE:
			if (myRank == 0) {
				if (fftw_import_wisdom_from_filename("../fftWisdom.double") == 0)
					LogMsg (VERB_HIGH, "  Warning: could not import wisdom from fftWisdom.double");
                	}

			fftwf_mpi_broadcast_wisdom(MPI_COMM_WORLD);

			LogMsg (VERB_HIGH, "  Plan 3d (%lld x %lld x %lld)", sizeN, sizeN, totlZ);
			planDoubleForward  = fftw_mpi_plan_dft_r2c_3d(totlZ, sizeN, sizeN, static_cast<double*>(axionIn),  static_cast<fftw_complex*>(axionOut), MPI_COMM_WORLD, FFTW_MEASURE);

			LogMsg (VERB_HIGH, "  Execute");
			fftw_execute (planDoubleForward);

			LogMsg (VERB_HIGH, "  Remove high modes");
			for (hsize_t zDim = 0; zDim < sizeZ; zDim++) {
				hsize_t offIn  = zDim*slab*dataSize;
				hsize_t offOut = zDim*nSlb*dataSize;
				memcpy (static_cast<char*> (axionIn) + offOut, static_cast<char*> (axionOut) + offIn, nSlb*dataSize);
			} 

			LogMsg (VERB_HIGH, "  Plan 3d (%lld x %lld x %lld)", newLx, newLx, newLz);
			planDoubleBackward = fftw_mpi_plan_dft_c2r_3d(newLz, newLx, newLx, static_cast<fftw_complex*>(axionOut),  static_cast<double*>(axionIn),  MPI_COMM_WORLD, FFTW_MEASURE);

			LogMsg (VERB_HIGH, "  Execute");
			fftw_execute (planDoubleBackward);

			break;

		case FIELD_SINGLE:
			if (myRank == 0) {
				if (fftw_import_wisdom_from_filename("../fftWisdom.single") == 0)
					LogMsg (VERB_HIGH, "  Warning: could not import wisdom from fftWisdom.single");
                	}

			fftwf_mpi_broadcast_wisdom(MPI_COMM_WORLD);

			LogMsg (VERB_HIGH, "  Plan 3d (%lld x %lld x %lld)", sizeN, sizeN, totlZ);
			planSingleForward  = fftwf_mpi_plan_dft_r2c_3d(totlZ, sizeN, sizeN, static_cast<float*>(axionIn),  static_cast<fftwf_complex*>(axionOut), MPI_COMM_WORLD, FFTW_MEASURE);

			LogMsg (VERB_HIGH, "  Execute");
			fftwf_execute (planSingleForward);

			LogMsg (VERB_HIGH, "  Remove high modes");
			for (hsize_t zDim = 0; zDim < sizeZ; zDim++) {
				hsize_t offIn  = zDim*slab*dataSize;
				hsize_t offOut = zDim*nSlb*dataSize;
				memcpy (static_cast<char*> (axionIn) + offOut, static_cast<char*> (axionOut) + offIn, nSlb*dataSize);
			} 

			LogMsg (VERB_HIGH, "  Plan 3d (%lld x %lld x %lld)", newLx, newLx, newLz);
			planSingleBackward = fftwf_mpi_plan_dft_c2r_3d(newLz, newLx, newLx, static_cast<fftwf_complex*>(axionOut), static_cast<float*>(axionIn),  MPI_COMM_WORLD, FFTW_MEASURE);

			LogMsg (VERB_HIGH, "  Execute");
			fftwf_execute (planSingleBackward);

			break;
	}

	/*	Open the file again and release the plist	*/
	sprintf(baseOut, "%s.r.%05d", outName, index);

	if ((file_id = H5Fopen (baseOut, H5F_ACC_RDONLY, plist_id)) < 0)
	{
		LogError ("Error opening file %s", baseOut);
		return;
	}

	H5Pclose(plist_id);

	/*	Write header	*/
	/*	Attributes	*/

	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size   (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

	writeAttribute(file_id, fStr,   "Field type",    attr_type);
	writeAttribute(file_id, prec,   "Precision",     attr_type);
	writeAttribute(file_id, &newLx, "Size",          H5T_NATIVE_UINT);
	writeAttribute(file_id, &newLz, "Depth",         H5T_NATIVE_UINT);
	writeAttribute(file_id, &LL,    "Lambda",        H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &nQcd,  "nQcd",          H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &sizeL, "Physical size", H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &zTmp,  "z",             H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &zTin,  "zInitial",      H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &zTfl,  "zFinal",        H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &tStep, "nSteps",        H5T_NATIVE_INT);
	writeAttribute(file_id, &cStep, "Current step",  H5T_NATIVE_INT);

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

	if (H5Pset_chunk (chunk_id, 1, &nSlb) < 0)
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

	/*	Create a group for string data if it doesn't exist	*/
	auto status = H5Lexists (file_id, "/energy", H5P_DEFAULT);	// Create group if it doesn't exists

	if (!status)
		group_id = H5Gcreate2(file_id, "/energy", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0)
			group_id = H5Gopen2(file_id, "/energy", H5P_DEFAULT);		// Group exists
		else {
			LogError ("Error: can't check whether group /energy exists");
			prof.stop();
			return;
		}
	}

	/*	Create a dataset for the whole axion data	*/

	char mCh[24] = "/energy/density";

	eset_id = H5Dcreate (file_id, mCh, dataType, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

	commSync();

	if (eset_id < 0)
	{
		LogError("Error creating dataset");
		prof.stop();
		exit (0);
	}

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

	eSpace = H5Dget_space (eset_id);
	memSpace = H5Screate_simple(1, &nSlb, NULL);	// Slab

	commSync();

	LogMsg (VERB_HIGH, "Rank %d ready to write", myRank);

	for (hsize_t zDim=0; zDim < newSz; zDim++)
	{
		/*	Select the slab in the file	*/
		offset = ((hsize_t) (myRank*newSz) + zDim)*nSlb;
		H5Sselect_hyperslab(eSpace, H5S_SELECT_SET, &offset, NULL, &nSlb, NULL);

		/*	Write raw data	*/
		auto eErr = H5Dwrite (eset_id, dataType, memSpace, eSpace, plist_id, (static_cast<char *> (axionIn) + nSlb*zDim*dataSize));

		if (eErr < 0)
		{
			LogError ("Error writing dataset");
			prof.stop();
			exit(0);
		}

		//commSync();
	}

	/*	Close the dataset	*/

	H5Dclose (eset_id);
	H5Sclose (eSpace);
	H5Sclose (memSpace);

	/*	Close the file		*/

	H5Sclose (totalSpace);
	H5Pclose (chunk_id);
	H5Pclose (plist_id);
	H5Fclose (file_id);

	trackFree (&axionOut, ALLOC_TRACK);
	trackFree (&axionIn,  ALLOC_TRACK);

        prof.stop();
//	prof.add(std::string("Reduced energy map"), 0., (2.*total*dataSize + 78.)*1e-9);

	LogMsg (VERB_NORMAL, "Written %lu bytes", nSlb*newLz*dataSize + 78);
}

