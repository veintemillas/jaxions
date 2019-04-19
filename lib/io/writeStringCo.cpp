#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <hdf5.h>

#include <fftw3-mpi.h>

#include "scalar/scalarField.h"
#include "utils/parse.h"
#include "comms/comms.h"

#include "utils/memAlloc.h"
#include "utils/logger.h"

void	writeStringCoA	(Scalar *axion, StringData strDat, const bool rData)
{

	hid_t       file_id, dataset_id, dataspace_id;  /* identifiers */
	hsize_t     dims[1];
	herr_t      status;
	/*Chose the file for simplicity*/
	char FILE[256];
	sprintf(FILE, "./out/m/axion.m.01010");
	int myRank = commRank();

	/* String data, different casts */
	char *strData = static_cast<char *>(axion->sData());
	unsigned short *strDatau = static_cast<unsigned short *>(axion->sData());

	/* Number of strings per rank, initialise = 0 */
	size_t stringN[commSize()];
	for (int i=0;i<commSize();i++)
		stringN[i]=0;

	/*send stringN to rank 0*/
	MPI_Gather( &strDat.strDen_local , 1, MPI_UNSIGNED_LONG, &stringN, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
	commSync();

	/* Check the total number*/
	size_t toti = 0;
	for (int i=0;i<commSize();i++){
	LogOut("r %d st %lu \n", i, stringN[i]);
	toti += stringN[i];
	}
	LogOut("needs %ld which we knew %ld\n", toti, strDat.strDen );

	hid_t sSpace;
	hid_t memSpace;
	hid_t group_id;
	hsize_t slab;

	if (myRank == 0)
	{
		/* Create a new file using default properties. */
		file_id = H5Fcreate(FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

		/* Create a group for string data */
		auto status = H5Lexists (file_id, "/string", H5P_DEFAULT);	// Create group if it doesn't exists
		if (!status)
			group_id = H5Gcreate2(file_id, "/string", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		else {
			if (status > 0) {
				group_id = H5Gopen2(file_id, "/string", H5P_DEFAULT);	// Group exists, WTF
				LogMsg(VERB_NORMAL, "Warning: group /string exists!");	// Since this is weird, log it
			} else {
				LogError ("Error: can't check whether group /string exists");
			}
		}

		/* if rData write the coordinates*/
		if (rData)
		{
			/* Total length of coordinates to write */
			dims[0] = 3*toti;
			/* Create the data space for the dataset. */
			dataspace_id = H5Screate_simple(1, dims, NULL);
			LogOut("strings plaq. to be written in rank 0 = %ld \n", dims[0]/3);

			/* Create the dataset. */
			dataset_id = H5Dcreate2(file_id, "/string/codata", H5T_NATIVE_USHORT, dataspace_id,
			            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

			LogOut("created! \n");

			sSpace = H5Dget_space (dataset_id);
			memSpace = H5Screate_simple(1, &slab, NULL);

			LogOut("spaced! \n\n");
		}
	}

	/* Write the dataset. */

	if (rData) {
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
				printf("...SEND %d CHAR from rank %d to 0 US1=(%hu, %hu, %hu) in rank %d\n", tralara, rank, strDatau[0], strDatau[1], strDatau[2], myRank);
				fflush(stdout);
				MPI_Send(&(strData[0]), tralara, MPI_CHAR, 0, rank, MPI_COMM_WORLD);
				}
			}
			else
			{
				/* Only  myRank 0 receives and writes */
				slab = 3*stringN[rank];
				tralara =(int) slab*sizeof(unsigned short);

				if (rank != 0)
					{
						MPI_Recv(&(strData[0]), tralara, MPI_CHAR, rank, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						LogOut("RECEIVE %d CHAR from rank %d to 0 US1=(%hu, %hu, %hu) in rank %d\n", tralara, rank, strDatau[0], strDatau[1], strDatau[2], myRank);
					}

				/*	Select the slab in the file	*/
				toti=0;
				for (int i=0;i<rank;i++){
					toti += stringN[i];
				}
				hsize_t offset = ((hsize_t) 3*toti);

				/* update memSpace with new slab size	*/
				/* here one can partition in less than 2Gb if needed in future */
				memSpace = H5Screate_simple(1, &slab, NULL);
				LogOut(" offset is %d and slab is %d\n",offset/3, slab/3);
				H5Sselect_hyperslab(sSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

				/* write file */
				H5Dwrite (dataset_id, H5T_NATIVE_USHORT, memSpace, sSpace, H5P_DEFAULT, (void *) &(strDatau[0]) );
				LogOut("written rank %d myrank %d (%p)\n", rank, myRank, (void *) &(strData[0]));
				LogOut("pos~~~ (%hu, %hu, %hu) in rank %d (rank %d in loop)\n\n", strDatau[0], strDatau[1], strDatau[2], myRank, rank);
			}

			commSync();
		}

		if (myRank == 0)
		{
		// H5Dclose(sSpace);
		// H5Dclose(memSpace);
		H5Dclose(dataset_id);
		H5Sclose(dataspace_id);
		H5Fclose(file_id);
		}
	}
}
