#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hdf5.h>

#include "scalar/scalarField.h"
#include "utils/parse.h"
#include "comms/comms.h"

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

	char	prec[16];
	int	length;

	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize;

	int myRank = commRank();

	commSync();

	/*	Set up parallel access with Hdf5	*/

	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];

	sprintf(base, "out/dump/%s.%05d", outName, index);

	/*	Create the file and release the plist	*/

	if ((file_id = H5Fcreate (base, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) < 0)
	{
		printf ("Error creating file %s\n", base);
		return;
	}

	H5Pclose(plist_id);

	commSync();

	/* Puedes juntar casi todo el código y que el switch elija el datatype y el sizeof() */

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


	/*	Write header	*/

	hid_t attr_type;

	/*	Attributes	*/

	int cSteps = dump*index;
	hsize_t totlZ = sizeZ*zGrid;
	hsize_t tmpS  = sizeN;

	attr_type = H5Tcopy(H5T_C_S1);
	H5Tset_size   (attr_type, length);
	H5Tset_strpad (attr_type, H5T_STR_NULLTERM);

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
	H5Pset_dxpl_mpio(plist_id,H5FD_MPIO_COLLECTIVE);

	/*	Create space for writing the raw data to disk with chunked access	*/

	total = sizeN*sizeN*totlZ*2;
	//slab  = axion->Surf()*2;
	slab  = axion->Surf();

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

	status = H5Pset_layout (chunk_id, H5D_CHUNKED);

	if (status < 0)
	{
		printf ("Fatal error H5Pset_layout\n");
		exit (1);
	}

	status = H5Pset_chunk (chunk_id, 1, &slab);

	if (status < 0)
	{
		printf ("Fatal error H5Pset_chunk\n");
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
		//JAVIER commented next
		//printf ("Rank %d writting slab %ld\n", myRank, zDim + myRank*axion->Depth());
		fflush (stdout);

		/*	Select the slab in the file	*/
		// JAVIER CHANGED 2*axion->Surf to axion->Surf
		//offset = (((hsize_t) (myRank*axion->Depth()))+zDim)*((hsize_t) (2*axion->Surf()));
		offset = (((hsize_t) (myRank*axion->Depth()))+zDim)*((hsize_t) (axion->Surf()));
		H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
		H5Sselect_hyperslab(vSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
		//JAVIER commented next
		//printf ("Rank %d select hyperslab at offset %ld\n", myRank, offset);
		//fflush (stdout);

		/*	Write raw data	*/

		// JAVIER CHANGED 2*axion->Surf to axion->Surf
		//H5Dwrite (mset_id, dataType, memSpace, mSpace, plist_id, (static_cast<char *> (axion->mCpu())+((hsize_t) (axion->Surf()*2))*(1+zDim)*dataSize));
		H5Dwrite (mset_id, dataType, memSpace, mSpace, plist_id, (static_cast<char *> (axion->mCpu())+((hsize_t) (axion->Surf()))*(1+zDim)*dataSize));
		//JAVIER commented next
		//printf ("Rank %d write m\n", myRank);
		//fflush (stdout);
		//H5Dwrite (vset_id, dataType, memSpace, vSpace, plist_id, (static_cast<char *> (axion->vCpu())+((hsize_t) (axion->Surf()*2))*zDim*dataSize));
		H5Dwrite (vset_id, dataType, memSpace, vSpace, plist_id, (static_cast<char *> (axion->vCpu())+((hsize_t) (axion->Surf()))*zDim*dataSize));
		//JAVIER commented next
		//printf ("Rank %d write v\n", myRank);
		fflush (stdout);
		//JAVIER commented next
		//printf ("Rank %d done\n", myRank);
		//fflush (stdout);

		commSync();
	}

	//JAVIER commented next
	//printf ("Rank %d closing\n", myRank);
	//fflush (stdout);

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

	char	prec[16];
	int	length = 8;

	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize;

	int myRank = commRank();

//	MPI_Info mpiInfo;

	/*	Set up parallel access with Hdf5	*/

	plist_id = H5Pcreate (H5P_FILE_ACCESS);
	H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	char base[256];

	sprintf(base, "out/dump/%s.%05d", outName, index);

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

	/*	Create axion field	*/

	if (totlZ % zGrid)
	{
		printf("Error: Geometry not valid. Try a different partitioning.\n");
		exit (1);
	}
	else
		sizeZ = totlZ/zGrid;

	*axion = new Scalar(sizeN, sizeZ, precision, cDev, zTmp, NULL, lowmem, zGrid, CONF_NONE, 0, 0, NULL);

	/*	Create plist for collective read	*/

	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id,H5FD_MPIO_COLLECTIVE);

	/*	Open a dataset for the whole axion data	*/

	mset_id = H5Dopen (file_id, "/m", H5P_DEFAULT);
	vset_id = H5Dopen (file_id, "/v", H5P_DEFAULT);

	slab   = (hsize_t) ((*axion)->Surf()*2);

	memSpace = H5Screate_simple(1, &slab, NULL);	// Slab
	mSpace   = H5Dget_space (mset_id);
	vSpace   = H5Dget_space (vset_id);

	for (hsize_t zDim=0; zDim<((hsize_t) (*axion)->Depth()); zDim++)
	{
		/*	Select the slab in the file	*/

		offset = (((hsize_t) (myRank*(*axion)->Depth()))+zDim)*((hsize_t) (2*(*axion)->Surf()));
		H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
		H5Sselect_hyperslab(vSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

		/*	Read raw data	*/

		H5Dread (mset_id, dataType, memSpace, mSpace, plist_id, (static_cast<char *> ((*axion)->mCpu())+((hsize_t) ((*axion)->Surf()*2))*(1+zDim)*dataSize));
		H5Dread (vset_id, dataType, memSpace, vSpace, plist_id, (static_cast<char *> ((*axion)->vCpu())+((hsize_t) ((*axion)->Surf()*2))*zDim*dataSize));
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
