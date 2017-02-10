#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hdf5.h>

#include "scalar/scalarField.h"
#include "utils/parse.h"
#include "comms/comms.h"

hid_t	meas_id = -1, mlist_id;
hsize_t	tSize, slabSz, sLz;
bool	opened = false, header = false;

herr_t	writeAttribute(hid_t file_id, void *data, const char *name, hid_t h5_type)
{
	hid_t	attr, attr_id;
	herr_t	status;

	attr_id = H5Screate(H5S_SCALAR);
	attr    = H5Acreate2 (file_id, name, h5_type, attr_id, H5P_DEFAULT, H5P_DEFAULT);
	status  = H5Awrite (attr, h5_type, data);
	H5Sclose (attr_id);
	H5Aclose (attr);

	return	status;
}

herr_t	readAttribute(hid_t file_id, void *data, const char *name, hid_t h5_type)
{
	hid_t	attr;
	herr_t	status;

	attr   = H5Aopen_by_name (file_id, "/", name, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Aread (attr, h5_type, data);
	status = H5Aclose(attr);

	return	status;
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

	commSync();

	/*	Set up parallel access with Hdf5	*/
	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];

	sprintf(base, "%s.%05d", outName, index);

	/*	Create the file and release the plist	*/
	if ((file_id = H5Fcreate (base, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) < 0)
	{
		printf ("Error creating file %s\n", base);
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

		printf("Error: Invalid precision. How did you get this far?\n");
		exit(1);

		break;
	}

	int cSteps = dump*index;
	hsize_t totlZ = sizeZ*zGrid;
	hsize_t tmpS  = sizeN;

	switch (axion->Field())
	{
		case 	FIELD_SAXION:
		{
			total = tmpS*tmpS*totlZ*2;
			slab  = (hsize_t) (axion->Surf()*2);

			sprintf(fStr, "Saxion");
		}
		break;

		case	FIELD_AXION:
		{
			total = tmpS*tmpS*totlZ;
			slab  = (hsize_t) axion->Surf();

			sprintf(fStr, "Axion");
		}
		break;

		default:

		printf("Error: Invalid field type. How did you get this far?\n");
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
	writeAttribute(file_id, &tmpS,  "Size",          H5T_NATIVE_HSIZE);
	writeAttribute(file_id, &totlZ, "Depth",         H5T_NATIVE_HSIZE);
	writeAttribute(file_id, &LL,    "Lambda",        H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &nQcd,  "nQcd",          H5T_NATIVE_INT);
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
	totalSpace = H5Screate_simple(1, &total, maxD);	// Whole data

	if (totalSpace < 0)
	{
		printf ("Fatal error H5Screate_simple\n");
		exit (1);
	}

	/*	Set chunked access	*/
	herr_t status;
	chunk_id = H5Pcreate (H5P_DATASET_CREATE);

	if (chunk_id < 0)
	{
		printf ("Fatal error H5Pcreate\n");
		exit (1);
	}

	status = H5Pset_chunk (chunk_id, 1, &slab);

	if (status < 0)
	{
		printf ("Fatal error H5Pset_chunk\n");
		exit (1);
	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	status = H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER);

	if (status < 0)
	{
		printf ("Fatal error H5Pset_alloc_time\n");
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
		printf	("Fatal error.\n");
		exit (0);
	}

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

	mSpace = H5Dget_space (mset_id);
	vSpace = H5Dget_space (vset_id);
	memSpace = H5Screate_simple(1, &slab, NULL);	// Slab

	commSync();

	printf ("Rank %d ready to write\n", myRank);
	fflush (stdout);

	for (hsize_t zDim=0; zDim<((hsize_t) axion->Depth()); zDim++)
	{
		/*	Select the slab in the file	*/
		offset = (((hsize_t) (myRank*axion->Depth()))+zDim)*slab;
		H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
		H5Sselect_hyperslab(vSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

		/*	Write raw data	*/

		// JAVIER CHANGED 2*axion->Surf to axion->Surf
		H5Dwrite (mset_id, dataType, memSpace, mSpace, plist_id, (static_cast<char *> (axion->mCpu())+slab*(1+zDim)*dataSize));
		H5Dwrite (vset_id, dataType, memSpace, vSpace, plist_id, (static_cast<char *> (axion->vCpu())+slab*zDim*dataSize));

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

	/*	Set up parallel access with Hdf5	*/

	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];

	sprintf(base, "%s.%05d", outName, index);

	/*	Open the file and release the plist	*/

	if ((file_id = H5Fopen (base, H5F_ACC_RDONLY, plist_id)) < 0)
	{
		*axion == NULL;
		printf ("Error opening file %s\n", base);
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
	readAttribute (file_id, &nQcd,  "nQcd",         H5T_NATIVE_INT);
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
			printf("Error reading file %s: Invalid precision %s\n", base, prec);
			exit(1);
		}
	} else {
		precision = sPrec;

		if (sPrec == FIELD_DOUBLE)
		{
			dataType  = H5T_NATIVE_DOUBLE;
			dataSize  = sizeof(double);

			if (!strcmp(prec, "Single"))
				printf("Reading double precision configuration as single precision\n");
		} else if (sPrec == FIELD_SINGLE) {
			dataType  = H5T_NATIVE_FLOAT;
			dataSize  = sizeof(float);
			if (!strcmp(prec, "Double"))
				printf("Reading single precision configuration as double precision\n");
		} else {
			printf("Input error: Invalid precision\n");
			exit(1);
		}
	}

	// OJO que LO he CAMBIADOOOOOO
	//printf("RW/ %d %d",sizeN,sizeZ);
	hsize_t sizeZ = totlZ/zGrid;

	if (!strcmp(fStr, "Saxion"))
	{
		*axion = new Scalar(sizeN, sizeZ, precision, cDev, zTmp, lowmem, zGrid, FIELD_SAXION,  CONF_NONE, 0, 0, NULL);
		slab   = (hsize_t) ((*axion)->Surf()*2);
	} else if (!strcmp(fStr, "Axion")) {
		*axion = new Scalar(sizeN, sizeZ, precision, cDev, zTmp, lowmem, zGrid, FIELD_AXION, CONF_NONE, 0, 0, NULL);
		slab   = (hsize_t) ((*axion)->Surf());
	} else {
		printf("Input error: Invalid field type\n");
		exit(1);
	}

	/*	Create axion field	*/

	if (totlZ % zGrid)
	{
		printf("Error: Geometry not valid. Try a different partitioning.\n");
		exit (1);
	}
	else
		sizeZ = totlZ/zGrid;

	/*	Create plist for collective read	*/

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id,H5FD_MPIO_COLLECTIVE);

	/*	Open a dataset for the whole axion data	*/

	mset_id = H5Dopen (file_id, "/m", H5P_DEFAULT);
	vset_id = H5Dopen (file_id, "/v", H5P_DEFAULT);


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

		H5Dread (mset_id, dataType, memSpace, mSpace, plist_id, (static_cast<char *> ((*axion)->mCpu())+slab*(1+zDim)*dataSize));
		H5Dread (vset_id, dataType, memSpace, vSpace, plist_id, (static_cast<char *> ((*axion)->vCpu())+slab*zDim*dataSize));
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

	if (myRank != 0)	// Only rank 0 writes measurement data
		return;

	if (opened)
	{
		printf ("Error, a hdf5 file is already opened\n");
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
		printf ("Error creating file %s\n", base);
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

		printf("Error: Invalid precision. How did you get this far?\n");
		exit(1);

		break;
	}

	int cSteps = dump*index;
	hsize_t totlZ = sizeZ*zGrid;
	hsize_t tmpS  = sizeN;

	switch (axion->Field())
	{
		case 	FIELD_SAXION:
			sprintf(fStr, "Saxion");
			break;

		case	FIELD_AXION:
			sprintf(fStr, "Axion");
			break;

		default:
			printf("Error: Invalid field type. How did you get this far?\n");
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

	tSize  = tmpS*tmpS*totlZ;
	slabSz = tmpS*tmpS;
	sLz    = sizeZ;

	header = true;

	printf("\n\n\nHeader %d Opened %d\n\n\n", header, opened); fflush(stdout);

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
}

void	writeString	(void *str, size_t strDen)
{
	hid_t	totalSpace, chunk_id, group_id, sSet_id, sSpace, memSpace;
	hid_t	datum;

	int myRank = commRank();

	const hsize_t maxD[1] = { H5S_UNLIMITED };
	char *strData = static_cast<char *>(str);
	char sCh[16] = "/string/data";

	if (myRank == 0)
	{
		if (header == false || opened == false)
		{
			printf("Error: measurement file not opened. Ignoring write request. %d %d\n", header, opened);
			return;
		}

		/*	Create space for writing the raw data to disk with chunked access	*/
		totalSpace = H5Screate_simple(1, &tSize, maxD);	// Whole data

		if (totalSpace < 0)
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

		status = H5Pset_deflate (chunk_id, 9);	// Maximum compression

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

		/*	Create a group for string data	*/
		group_id = H5Gcreate2(meas_id, "/string", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		writeAttribute(group_id, &strDen, "String number", H5T_NATIVE_HSIZE);

		/*	Create a dataset for string data	*/
		sSet_id = H5Dcreate (meas_id, sCh, H5T_NATIVE_CHAR, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

		if (sSet_id < 0)
		{
			printf	("Fatal error.\n");
			exit (0);
		}

		/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

		sSpace = H5Dget_space (sSet_id);
		memSpace = H5Screate_simple(1, &slabSz, NULL);	// Slab
	}

	int tSz = commSize();

	for (int rank=0; rank<tSz; rank++)
	{
		if (myRank != 0)
		{
			if (myRank == rank)
				MPI_Send(strData, slabSz*sLz, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
		} else {
			if (rank != 0)
				MPI_Recv(strData, slabSz*sLz, MPI_CHAR, rank, 0, MPI_COMM_WORLD, NULL);

			for (hsize_t zDim=0; zDim<((hsize_t) sLz); zDim++)
			{
				/*	Select the slab in the file	*/
				hsize_t offset = (((hsize_t) (rank*sLz))+zDim)*slabSz;
				H5Sselect_hyperslab(sSpace, H5S_SELECT_SET, &offset, NULL, &slabSz, NULL);

				/*	Write raw data	*/

				H5Dwrite (sSet_id, H5T_NATIVE_CHAR, memSpace, sSpace, H5P_DEFAULT, (strData)+slabSz*zDim);
			}
		}
		commSync();
		printf ("Rank %d in %d iteration\n", myRank, rank); fflush(stdout);
	}

	/*	Close the dataset	*/

	if (myRank == 0) {
		H5Dclose (sSet_id);
		H5Sclose (sSpace);
		H5Sclose (memSpace);

		H5Sclose (totalSpace);
		H5Pclose (chunk_id);
		H5Gclose (group_id);
	}

	commSync();
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

void	writeEnergy	(double *eData)
{
	hid_t	group_id;
	herr_t	status;

	if (commRank() != 0)
		return;

	if (header == false || opened == false)
	{
		printf("Error: measurement file not opened. Ignoring write request. %d %d\n", header, opened);
		return;
	}

	/*	Create a group for string data if it doesn't exist	*/
	status = H5Eset_auto(H5E_DEFAULT, NULL, NULL);	// Turn off error output, we don't want trash if the group doesn't exist

	if (status = H5Gget_objinfo (meas_id, "/energy", 0, NULL))	// Create group if it doesn't exists
		group_id = H5Gcreate2(meas_id, "/energy", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else
		group_id = H5Gopen2(meas_id, "/energy", H5P_DEFAULT);

//	status = H5Eset_auto(H5E_DEFAULT, H5Eprint2, stderr);	// Restore error output

	/* TODO distinguir axion de saxion */
	/* escribir energia total */

	writeAttribute(group_id, &eData[0],  "Grx", H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &eData[1],  "Gax", H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &eData[2],  "Gry", H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &eData[3],  "Gay", H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &eData[4],  "Grz", H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &eData[5],  "Gaz", H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &eData[6],  "Vr",  H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &eData[7],  "Va",  H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &eData[8],  "Kr",  H5T_NATIVE_DOUBLE);
	writeAttribute(group_id, &eData[9],  "Ka",  H5T_NATIVE_DOUBLE);

	/*	Close the group		*/
	H5Gclose (group_id);
}

void	writePoint (Scalar *axion)
{
	hid_t	group_id, dataSpace, sSet_id, sSpace, dataSet, dataType;
	herr_t	status;
	hsize_t dims[1];

	size_t	dataSize = axion->DataSize(), S0 = axion->Surf();

	if (commRank() != 0)
		return;

	if (header == false || opened == false)
	{
		printf("Error: measurement file not opened. Ignoring write request. %d %d\n", header, opened);
		return;
	}

	/*	Create a group for point data if it doesn't exist	*/
	status = H5Eset_auto(H5E_DEFAULT, NULL, NULL);	// Turn off error output, we don't want trash if the group doesn't exist

	if (status = H5Gget_objinfo (meas_id, "/point", 0, NULL))	// Create group if it doesn't exist
		group_id = H5Gcreate2(meas_id, "/point", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else
		group_id = H5Gopen2(meas_id, "/point", H5P_DEFAULT);

//	status = H5Eset_auto(H5E_DEFAULT, H5Eprint2, stderr);	// Restore error output

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
	status = H5Dwrite(dataSet, dataType, dataSpace, sSpace, H5P_DEFAULT, static_cast<char*>(axion->mCpu()) + S0*dataSize);

	/*	Close everything		*/
	H5Sclose (sSpace);
	H5Dclose (dataSet);
	H5Sclose (dataSpace);
	H5Gclose (group_id);
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

	commSync();

	/*	Set up parallel access with Hdf5	*/

	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];

	sprintf(base, "%s.m.%05d", outName, index);

	/*	Broadcast the values of opened/header	*/
	MPI_Bcast(&opened, sizeof(opened), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&opened, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

	/*	If the measurement file is opened, we reopen it with parallel access	*/
	if (opened == true)
	{
		printf ("All ranks here\n");
		destroyMeas();

		if ((file_id = H5Fopen (base, H5F_ACC_RDWR, plist_id)) < 0)
		{
			printf ("Error creating file %s\n", base);
			return;
		}
	} else {
		/*	Else we create the file		*/
		printf ("All ranks there\n");
		if ((file_id = H5Fcreate (base, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) < 0)
		{
			printf ("Error creating file %s\n", base);
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

		printf("Error: Invalid precision. How did you get this far?\n");
		exit(1);

		break;
	}

	int cSteps = dump*index;
	hsize_t totlZ = sizeZ*zGrid;
	hsize_t tmpS  = sizeN;

	switch (axion->Field())
	{
		case 	FIELD_SAXION:
		{
			total = tmpS*tmpS*totlZ*2;
			slab  = (hsize_t) (axion->Surf()*2);

			sprintf(fStr, "Saxion");
		}
		break;

		case	FIELD_AXION:
		{
			total = tmpS*tmpS*totlZ;
			slab  = (hsize_t) axion->Surf();

			sprintf(fStr, "Axion");
		}
		break;

		default:

		printf("Error: Invalid field type. How did you get this far?\n");
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
		writeAttribute(file_id, &tmpS,  "Size",          H5T_NATIVE_HSIZE);
		writeAttribute(file_id, &totlZ, "Depth",         H5T_NATIVE_HSIZE);
		writeAttribute(file_id, &LL,    "Lambda",        H5T_NATIVE_DOUBLE);
		writeAttribute(file_id, &nQcd,  "nQcd",          H5T_NATIVE_INT);
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
	H5Pset_dxpl_mpio(plist_id,H5FD_MPIO_COLLECTIVE);

	/*	Create space for writing the raw data to disk with chunked access	*/

	totalSpace = H5Screate_simple(1, &total, maxD);	// Whole data

	if (totalSpace < 0)
	{
		printf ("Fatal error H5Screate_simple\n");
		exit (1);
	}

	/*	Set chunked access	*/

	herr_t status;

	chunk_id = H5Pcreate (H5P_DATASET_CREATE);

	if (chunk_id < 0)
	{
		printf ("Fatal error H5Pcreate\n");
		exit (1);
	}

	status = H5Pset_chunk (chunk_id, 1, &slab);

	if (status < 0)
	{
		printf ("Fatal error H5Pset_chunk\n");
		exit (1);
	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	status = H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER);

	if (status < 0)
	{
		printf ("Fatal error H5Pset_alloc_time\n");
		exit (1);
	}

	/*	Create a group for string data if it doesn't exist	*/
	status = H5Eset_auto(H5E_DEFAULT, NULL, NULL);	// Turn off error output, we don't want trash if the group doesn't exist

	if (status = H5Gget_objinfo (file_id, "/energy", 0, NULL))	// Create group if it doesn't exist
		group_id = H5Gcreate2(file_id, "/energy", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else
		group_id = H5Gopen2(file_id, "/energy", H5P_DEFAULT);

//	status = H5Eset_auto(H5E_DEFAULT, H5Eprint2, stderr);	// Restore error output

	/*	Create a dataset for the whole axion data	*/

	char mCh[24] = "/energy/density";

	mset_id = H5Dcreate (file_id, mCh, dataType, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

	commSync();

	if (mset_id < 0)
	{
		printf	("Fatal error.\n");
		exit (0);
	}

	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

	mSpace = H5Dget_space (mset_id);
	memSpace = H5Screate_simple(1, &slab, NULL);	// Slab

	commSync();

	printf ("Rank %d ready to write\n", myRank);
	fflush (stdout);

	for (hsize_t zDim=0; zDim<((hsize_t) axion->Depth()); zDim++)
	{
		/*	Select the slab in the file	*/
		offset = (((hsize_t) (myRank*axion->Depth()))+zDim)*slab;
		H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

		/*	Write raw data	*/

		H5Dwrite (mset_id, dataType, memSpace, mSpace, plist_id, (static_cast<char *> (axion->mCpu())+slab*(1+zDim)*dataSize));

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
	H5Fclose (file_id);

	/*	If there was a file opened for measurements, open it again	*/

	if (opened == true)
	{
		hid_t	plist_id;

		if (myRank != 0)	// Only rank 0 writes measurement data
			return;

		/*	This would be weird indeed	*/

		if (meas_id >= 0)
		{
			printf ("Error, a hdf5 file is already opened\n");
			return;
		}

		/*	Open the file and release the plist	*/

		if ((meas_id = H5Fopen (base, H5F_ACC_RDWR, H5P_DEFAULT)) < 0)
		{
			printf ("Error opening file %s\n", base);
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

	if (status = H5Gget_objinfo (meas_id, dataName, 0, NULL))	// Create group if it doesn't exist
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

void	writeArray (Scalar *axion, void *aData, size_t aSize, const char *group, const char *dataName)
{
	hid_t	group_id, dataSpace, sSpace, dataSet;
	herr_t	status;
	hsize_t dims[1] = { aSize };

	size_t	dataSize;

	if (commRank() != 0)
		return;

	if (header == false || opened == false)
	{
		printf("Error: measurement file not opened. Ignoring write request.\n");
		return;
	}

	/*	Create the group for the data if it doesn't exist	*/
	status = H5Eset_auto(H5E_DEFAULT, NULL, NULL);	// Turn off error output, we don't want trash if the group doesn't exist

	if (status = H5Gget_objinfo (meas_id, group, 0, NULL))	// Create group if it doesn't exists
		group_id = H5Gcreate2(meas_id, group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else
		group_id = H5Gopen2(meas_id, group, H5P_DEFAULT);

//	status = H5Eset_auto(H5E_DEFAULT, H5Eprint2, stderr);	// Restore error output

	/*	Create dataset	*/
	dataSpace = H5Screate_simple(1, dims, NULL);
	dataSet   = H5Dcreate(group_id, dataName, H5T_NATIVE_DOUBLE, dataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	sSpace	  = H5Dget_space (dataSet);

	/*	Write spectrum data	*/
	status = H5Dwrite(dataSet, H5T_NATIVE_DOUBLE, dataSpace, sSpace, H5P_DEFAULT, aData);

	/*	Close everything		*/
	H5Sclose (sSpace);
	H5Dclose (dataSet);
	H5Sclose (dataSpace);
	H5Gclose (group_id);
}

