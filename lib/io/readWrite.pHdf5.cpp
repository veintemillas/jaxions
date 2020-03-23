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
#include "scalar/fourier.h"

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
	//
	// switch (h5_type)
	// {
	// 	case H5T_NATIVE_UINT:
	// 		LogMsg (VERB_HIGH, "Write attribute %s = %u", name, *(static_cast<size_t*>(data)));
	// 	break;
	// 	case H5T_NATIVE_DOUBLE:
	// 		LogMsg (VERB_HIGH, "Write attribute %s = %e", name, *(static_cast<double*>(data)));
	// 	break;
	// 	case H5T_NATIVE_INT:
	// 		LogMsg (VERB_HIGH, "Write attribute %s = %d", name, *(static_cast<int*>(data)));	// Usa status para hacer logging de los errores!!!
	// 	break;
	// 	default:
	// 	LogMsg (VERB_HIGH, "Write attribute %s", name);	// Usa status para hacer logging de los errores!!!
	// 	break;
	// }

		if (h5_type == H5T_NATIVE_UINT)
		{
				LogMsg (VERB_HIGH, "Write attribute %s = %u", name, *(static_cast<size_t*>(data)));
		}	else if (h5_type == H5T_NATIVE_DOUBLE) {
				LogMsg (VERB_HIGH, "Write attribute %s = %e", name, *(static_cast<double*>(data)));
		} else if (h5_type == H5T_NATIVE_INT) {
				LogMsg (VERB_HIGH, "Write attribute %s = %d", name, *(static_cast<int*>(data)));
		}	else {
				LogMsg (VERB_HIGH, "Write attribute %s", name);	// Usa status para hacer logging de los errores!!!
		}
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

// Cosmos readCosmoAttributes(hid_t file_id)
// {
// 	hid_t	attr;
// 	herr_t	status;
//
//	readAttribute (vGrp_id, &nQcd,  "nQcd",	  H5T_NATIVE_DOUBLE);
// myCosmos->SetQcdExp(nQcd);
//	readAttribute (vGrp_id, &lda,   "Lambda",      H5T_NATIVE_DOUBLE);
// myCosmos->SetLambda(lda);
// readAttribute (vGrp_id, &lStr,  "Lambda type", attr_type);
// readAttribute (vGrp_id, &indi3, "Indi3", H5T_NATIVE_DOUBLE);
// myCosmos->SetIndi3(indi3);
// readAttribute (vGrp_id, &zthrs, "z Threshold", H5T_NATIVE_DOUBLE);
// myCosmos->SetZThRes(zthrs);

// readAttribute (vGrp_id, &zrest, "z Restore", H5T_NATIVE_DOUBLE);
// myCosmos->SetZRestore(zrest);
//
// readAttribute (vGrp_id, &gm, "Gamma", H5T_NATIVE_DOUBLE);
// myCosmos->SetGamma(gm);
//
// readAttribute (vGrp_id, &vStr,  "VQcd type",  attr_type);
//
// if (!strcmp(vStr, "VQcd 1"))
// 	vqcdType = VQCD_1;
// else if (!strcmp(vStr, "VQcd 2"))
// 	vqcdType = VQCD_2;
// else if (!strcmp(vStr, "VQcd 0"))
// 	vqcdType = VQCD_0;
// else if (!strcmp(vStr, "VQcd 1 Peccei-Quinn 2"))
// 	vqcdType = VQCD_1_PQ_2;
// else if (!strcmp(vStr, "VQcd 1 N=2"))
// 	vqcdType = VQCD_1N2;
// else {
// 	LogError ("Error reading file %s: invalid potential type %s", base, vStr);
// 	vqcdType = VQCD_1;
//
// 	readAttribute (vGrp_id, &vStr,  "Damping type",  attr_type);
//
// 	if (!strcmp(vStr, "Rho"))
// 		vqcdType |= VQCD_DAMP_RHO;
// 	else if (!strcmp(vStr, "All"))
// 		vqcdType |= VQCD_DAMP_ALL;
// 	else if (!strcmp(vStr, "None"))
// 		vqcdType |= VQCD_NONE;
// 	else {
// 		LogError ("Error reading file %s: invalid damping type %s. Ignoring damping", base, vStr);
// 	}
//
// 	readAttribute (vGrp_id, &vStr,  "Evolution type",  attr_type);
//
// 	if (!strcmp(vStr, "Only Rho"))
// 		vqcdType |= VQCD_EVOL_RHO;
// 	else if (!strcmp(vStr, "Full"))
// 		vqcdType |= VQCD_NONE;
// 	else {
// 		LogError ("Error reading file %s: invalid rho evolution type %s. Ignoring rho evolution", base, vStr);
// 		exit(1);
// 	}
//
// 	myCosmos->SetQcdPot(vqcdType);
//
// 	LogMsg (VERB_NORMAL, "Ic... \n");
// 		if (status <= 0)
// 			LogMsg(VERB_NORMAL, "IC data not available");
// 		else {
// 			hid_t icGrp_id = H5Gopen2(file_id, "/ic", H5P_DEFAULT);
// 			readAttribute(icGrp_id, &mode0, "Axion zero mode", H5T_NATIVE_DOUBLE);
// 			readAttribute(icGrp_id, &icStr, "Initial conditions",   attr_type);
//
// 			if (!strcmp(icStr, "Smooth")) {
// 				cType = CONF_SMOOTH;
// 				readAttribute(icGrp_id, &iter,  "Smoothing iterations", H5T_NATIVE_HSIZE);
// 				readAttribute(icGrp_id, &alpha, "Smoothing constant",   H5T_NATIVE_DOUBLE);
// 			} else if (!strcmp(icStr, "kMax")) {
// 				cType = CONF_KMAX;
// 				readAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
// 				readAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
// 			} else if (!strcmp(icStr, "VilGor")) {
// 				cType = CONF_VILGOR;
// 				readAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
// 				readAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
// 			} else if (!strcmp(icStr, "Tkachev")) {
// 				cType = CONF_TKACHEV;
// 				readAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
// 				readAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
// 			}
//
// 			readAttribute(icGrp_id, &icStr, "Configuration type",   attr_type);
//
// 			if (!strcmp(icStr, "Random")) {
// 				smvarType = CONF_RAND;
// 			} else if (!strcmp(icStr, "String XY")) {
// 				smvarType = CONF_STRINGXY;
// 			} else if (!strcmp(icStr, "String YZ")) {
// 				smvarType = CONF_STRINGYZ;
// 			} else if (!strcmp(icStr, "Minicluster")) {
// 				smvarType = CONF_MINICLUSTER;
// 			} else if (!strcmp(icStr, "Minicluster 0")) {
// 				smvarType = CONF_MINICLUSTER0;
// 			} else if (!strcmp(icStr, "Axion noise")) {
// 				smvarType = CONF_AXNOISE;
// 			} else if (!strcmp(icStr, "Saxion noise")) {
// 				smvarType = CONF_SAXNOISE;
// 			} else if (!strcmp(icStr, "Axion one mode")) {
// 				smvarType = CONF_AX1MODE;
// 			} else {
// 				LogError("Error: unrecognized configuration type %s", icStr);
// 				exit(1);
// }

//	readAttribute (file_id, &zInit, "zInitial", H5T_NATIVE_DOUBLE);
// 	return	status;
// }



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
	auto gamma    = axion->BckGnd()->Gamma   ();
	auto LL       = axion->BckGnd()->Lambda  ();

	switch (vqcdType & VQCD_TYPE)
	{
		case	VQCD_1:
			sprintf(vStr, "VQcd 1");
			break;

		case	VQCD_2:
			sprintf(vStr, "VQcd 2");
			break;

		case	VQCD_0:
			sprintf(vStr, "VQcd 0");
			break;

		case	VQCD_QUAD:
			sprintf(vStr, "VQcd Linear");
			break;

		case	VQCD_PQ_ONLY:
			sprintf(vStr, "VQcd PQ only");
			break;

		case	VQCD_1_PQ_2:
			sprintf(vStr, "VQcd 1 Peccei-Quinn 2");
			break;

		case	VQCD_1N2:
			sprintf(vStr, "VQcd 1 N=2");
			break;

		default:
			sprintf(vStr, "VQcd 1");
			break;
	}

	switch (vqcdType & VQCD_DAMP)
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

	switch (vqcdType & VQCD_EVOL_RHO)
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

	double maa = axion->AxionMass(); //axionmass((*axion->zV()), nQcd, zthres, zrestore);
	double msa = axion->Msa();

	writeAttribute(file_id, fStr,   "Field type",    attr_type);
	writeAttribute(file_id, prec,   "Precision",     attr_type);
	writeAttribute(file_id, &tmpS,  "Size",          H5T_NATIVE_UINT);
	writeAttribute(file_id, &totlZ, "Depth",         H5T_NATIVE_UINT);
	writeAttribute(file_id, &msa,   "Saxion mass",   H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &maa,   "Axion mass",    H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, &lSize, "Physical size", H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, axion->zV(),  "z",       H5T_NATIVE_DOUBLE);
	writeAttribute(file_id, axion->RV(),  "R",       H5T_NATIVE_DOUBLE);
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

	writeAttribute(vGrp_id, &lStr,  "Lambda type",   attr_type);
	writeAttribute(vGrp_id, &LL,    "Lambda",        H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &vStr,  "VQcd type",     attr_type);
	writeAttribute(vGrp_id, &dStr,  "Damping type",  attr_type);
	writeAttribute(vGrp_id, &rStr,  "Evolution type",attr_type);
	writeAttribute(vGrp_id, &nQcd,  "nQcd",          H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &gamma, "Gamma",         H5T_NATIVE_DOUBLE);
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
			vErr = H5Dwrite (vset_id, dataType, memSpace, vSpace, plist_id, (static_cast<char *> (axion->vCpu())  +slab*zDim*dataSize));

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

	if (wasFolded)
	{
		(*munge)(FOLD_ALL);
		delete	munge;
	}
}





/* JAVI ADDED THAT */
void	readConf (Cosmos *myCosmos, Scalar **axion, int index, const bool restart)
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

	/* JAVI if restart flag, do not read index number for simplicity */
	if (!restart)
	sprintf(base, "%s/%s.%05d", outDir, outName, index);
	else
		sprintf(base, "%s/%s.restart", outDir, outName);

	if (debug) LogOut("[db] File read: %s\n",base);
	LogMsg (VERB_NORMAL, "File read: %s",base);
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

	double	zTmp, RTmp, maaR;
	uint	tStep, cStep, totlZ;
	readAttribute (file_id, fStr,   "Field type",   attr_type);
	readAttribute (file_id, prec,   "Precision",    attr_type);
	readAttribute (file_id, &sizeN, "Size",         H5T_NATIVE_UINT);
	readAttribute (file_id, &totlZ, "Depth",        H5T_NATIVE_UINT);
	readAttribute (file_id, &zTmp,  "z",            H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &RTmp,  "R",            H5T_NATIVE_DOUBLE);
	readAttribute (file_id, &tStep, "nSteps",       H5T_NATIVE_INT);
	readAttribute (file_id, &cStep, "Current step", H5T_NATIVE_INT);
LogMsg (VERB_NORMAL, "Field type: %s",fStr);
LogMsg (VERB_NORMAL, "Precision: %s",prec);
LogMsg (VERB_NORMAL, "Size: %d",sizeN);
LogMsg (VERB_NORMAL, "Depth: %d",totlZ);
LogMsg (VERB_NORMAL, "zTmp: %f",zTmp);
LogMsg (VERB_NORMAL, "RTmp: %f",RTmp);
LogMsg (VERB_NORMAL, "tStep: %d",tStep);
LogMsg (VERB_NORMAL, "cStep: %d",cStep);

LogMsg (VERB_NORMAL, "PhysSize: %f",myCosmos->PhysSize());
	if (myCosmos->PhysSize() == 0.0) {
		double lSize;
		readAttribute (file_id, &lSize, "Physical size", H5T_NATIVE_DOUBLE);
		myCosmos->SetPhysSize(lSize);
	}

	//initial time; axion will be created with z=zTmp
	readAttribute (file_id, &zTmp,  "z", H5T_NATIVE_DOUBLE);
	//if zInit is given in command line change it
	if (uZin) {
		zTmp = zInit;
LogMsg (VERB_NORMAL, "Commandline input for zInit!! \nzTmp (set to): %f",zTmp);
		//unless zTmp is before [Im not sure in which case this is relevant]
	} else {
		//but a record of the true z of the read confifuration is kept in zInit
		readAttribute (file_id, &zInit, "zInitial", H5T_NATIVE_DOUBLE);
		LogMsg (VERB_NORMAL, "zInit (read): %f",zInit);

	}


	if (!uZfn) {
		readAttribute (file_id, &zFinl,  "zFinal",  H5T_NATIVE_DOUBLE);
	}
LogMsg (VERB_NORMAL, "zFinal (read): %f",zFinl);
//	if (zInit > zTmp)
//		zTmp = zInit;
	if (restart)
	{
		readAttribute (file_id, &fIndex, "index", H5T_NATIVE_INT);
		LogOut("Reading index is %d\n",fIndex);
		LogMsg (VERB_NORMAL, "Reading index is %d\n",fIndex);
		/* It is very easy, we keep zInit and take z=zTmp we trust everything was properly specified in the file */
		readAttribute (file_id, &zInit, "zInitial", H5T_NATIVE_DOUBLE);
		readAttribute (file_id, &zTmp,  "z",            H5T_NATIVE_DOUBLE);
		LogOut("Reading zTmp = %f, zInit=%f \n",zTmp,zInit);
		LogMsg (VERB_NORMAL, "Reading zTmp = %f, zInit=%f \n",zTmp,zInit);

	}

	/*	Read potential data	*/
	auto status = H5Lexists (file_id, "/potential", H5P_DEFAULT);

	if (status <= 0)
		LogMsg(VERB_NORMAL, "Potential data not available, using defaults");
	else {
		hid_t vGrp_id = H5Gopen2(file_id, "/potential", H5P_DEFAULT);

LogMsg (VERB_NORMAL, "nQcd = %f\n",myCosmos->QcdExp());
		if (myCosmos->QcdExp() == -1.e8) {
			double nQcd;
			readAttribute (vGrp_id, &nQcd,  "nQcd",	  H5T_NATIVE_DOUBLE);
			myCosmos->SetQcdExp(nQcd);
			LogMsg (VERB_NORMAL, "nQcd (read and set to)= %f\n",myCosmos->QcdExp());
		}

		if (myCosmos->Lambda() == -1.e8) {
			double	lda;
			readAttribute (vGrp_id, &lda,   "Lambda",      H5T_NATIVE_DOUBLE);
			//readAttribute (file_id, &msa,   "Saxion mass", H5T_NATIVE_DOUBLE);	// Useless, I guess
			readAttribute (vGrp_id, &lStr,  "Lambda type", attr_type);

			if (!strcmp(lStr, "z2"))
				lType = LAMBDA_Z2;
			else if (!strcmp(lStr, "Fixed"))
				lType = LAMBDA_FIXED;
			else {
				LogError ("Error reading file %s: invalid lambda type %s", base, lStr);
				exit(1);
			}

			myCosmos->SetLambda(lda);
			LogMsg (VERB_NORMAL, "Lambda (read and set)= %f\n",myCosmos->Lambda());

		} /*else {	// Ya se ha hecho en Cosmos
			if (uMsa) {
				double tmp = (msa*sizeN)/sizeL;
				LL    = 0.5*tmp*tmp;
			} else {
				double tmp = sizeL/sizeN;
				msa = sqrt(2*LL)*tmp;
			}
		}*/
LogMsg (VERB_NORMAL, "Indi3 = %f\n",myCosmos->Indi3());
		readAttribute (file_id, &maaR,  "Axion mass",   H5T_NATIVE_DOUBLE);
		if (myCosmos->Indi3() == -1.e8) {
			double indi3;
			readAttribute (vGrp_id, &indi3, "Indi3", H5T_NATIVE_DOUBLE);
			myCosmos->SetIndi3(indi3);
			LogMsg (VERB_NORMAL, "Indi3 (read and set to)= %f\n",myCosmos->Indi3());
		}

LogMsg (VERB_NORMAL, "z Threshold = %f\n",myCosmos->ZThRes());
		if (myCosmos->ZThRes() == -1.e8) {
			double zthrs;
			readAttribute (vGrp_id, &zthrs, "z Threshold", H5T_NATIVE_DOUBLE);
			myCosmos->SetZThRes(zthrs);
			LogMsg (VERB_NORMAL, "z Threshold (read and set) = %f\n",myCosmos->ZThRes());
		}

LogMsg (VERB_NORMAL, "z Restore = %f\n",myCosmos->ZRestore());
		if (myCosmos->ZRestore() == -1.e8) {
			double zrest;
			readAttribute (vGrp_id, &zrest, "z Restore", H5T_NATIVE_DOUBLE);
			myCosmos->SetZRestore(zrest);
			LogMsg (VERB_NORMAL, "z Restore (read and set) = %f\n",myCosmos->ZRestore());
		}

		//indi3 =  maa/pow(zTmp, nQcd*0.5);

LogMsg (VERB_NORMAL, "Gamma = %f\n",myCosmos->Gamma());
		if (myCosmos->Gamma() == -1.e8) {
			double gm;
			readAttribute (vGrp_id, &gm, "Gamma", H5T_NATIVE_DOUBLE);
			myCosmos->SetGamma(gm);
			LogMsg (VERB_NORMAL, "Gamma (read and set) = %f\n",myCosmos->Gamma());
		}

LogMsg (VERB_NORMAL, "QcdPot = %d\n",myCosmos->QcdPot());
		if (myCosmos->QcdPot() == VQCD_NONE) {
			VqcdType vqcdType = VQCD_NONE;

			readAttribute (vGrp_id, &vStr,  "VQcd type",  attr_type);

			if (!strcmp(vStr, "VQcd 1"))
				vqcdType = VQCD_1;
			else if (!strcmp(vStr, "VQcd 2"))
				vqcdType = VQCD_2;
			else if (!strcmp(vStr, "VQcd 0"))
				vqcdType = VQCD_0;
			else if (!strcmp(vStr, "VQcd Linear"))
				vqcdType = VQCD_QUAD;
			else if (!strcmp(vStr, "VQcd PQ only"))
				vqcdType = VQCD_PQ_ONLY;
			else if (!strcmp(vStr, "VQcd 1 Peccei-Quinn 2"))
				vqcdType = VQCD_1_PQ_2;
			else if (!strcmp(vStr, "VQcd 1 N=2"))
				vqcdType = VQCD_1N2;
			else {
				LogError ("Error reading file %s: invalid potential type %s", base, vStr);
				vqcdType = VQCD_1;
				//exit(1);
			}

			readAttribute (vGrp_id, &vStr,  "Damping type",  attr_type);

			if (!strcmp(vStr, "Rho"))
				vqcdType |= VQCD_DAMP_RHO;
			else if (!strcmp(vStr, "All"))
				vqcdType |= VQCD_DAMP_ALL;
			else if (!strcmp(vStr, "None"))
				vqcdType |= VQCD_NONE;
			else {
				LogError ("Error reading file %s: invalid damping type %s. Ignoring damping", base, vStr);
			}

			readAttribute (vGrp_id, &vStr,  "Evolution type",  attr_type);

			if (!strcmp(vStr, "Only Rho"))
				vqcdType |= VQCD_EVOL_RHO;
			else if (!strcmp(vStr, "Full"))
				vqcdType |= VQCD_NONE;
			else {
				LogError ("Error reading file %s: invalid rho evolution type %s. Ignoring rho evolution", base, vStr);
				exit(1);
			}

			myCosmos->SetQcdPot(vqcdType);
			LogMsg (VERB_NORMAL, "QcdPot (read and set)= %d\n",myCosmos->QcdPot());
		}

		H5Gclose(vGrp_id);
	}

	/*	Read IC data		*/
	status = H5Lexists (file_id, "/ic", H5P_DEFAULT);

LogMsg (VERB_NORMAL, "Ic... \n");
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
		} else if (!strcmp(icStr, "Axion Spectrum")) {
			cType = CONF_SPAX;
			readAttribute(icGrp_id, &kMax,  "Max k",                H5T_NATIVE_HSIZE);
			// readAttribute(icGrp_id, &kCrit, "Critical kappa",       H5T_NATIVE_DOUBLE);
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

	/*	Create axion field	*/

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

	if (totlZ % zGrid)
	{
		LogError ("Error: Geometry not valid. Try a different partitioning");
		exit (1);
	}
	else
		sizeZ = totlZ/zGrid;

	if (debug) LogOut("[db] Read start\n");
	prof.stop();
	prof.add(std::string("Read configuration"), 0, 0);

	myCosmos->ICData().cType = CONF_NONE;
	if (!strcmp(fStr, "Saxion"))
	{
		*axion = new Scalar(myCosmos, sizeN, sizeZ, precision, cDev, zTmp, lowmem, zGrid, FIELD_SAXION,    lType, myCosmos->ICData().Nghost);
		slab   = (hsize_t) ((*axion)->Surf()*2);
	} else if (!strcmp(fStr, "Axion")) {
		*axion = new Scalar(myCosmos, sizeN, sizeZ, precision, cDev, zTmp, lowmem, zGrid, FIELD_AXION,     lType, myCosmos->ICData().Nghost);
		slab   = (hsize_t) ((*axion)->Surf());
	} else if (!strcmp(fStr, "Axion Mod")) {
		*axion = new Scalar(myCosmos, sizeN, sizeZ, precision, cDev, zTmp, lowmem, zGrid, FIELD_AXION_MOD, lType, myCosmos->ICData().Nghost);
		slab   = (hsize_t) ((*axion)->Surf());
	} else {
		LogError ("Input error: Invalid field type");
		exit(1);
	}

	double maa = (*axion)->AxionMass();

	if (fabs((maa - maaR)/std::max(maaR,maa)) > 1e-5)
		LogMsg(VERB_NORMAL, "Chaging axion mass from %e to %e (difference %.3f %%)", maaR, maa, 100.*fabs((maaR-maa)/std::max(maaR,maa)));

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

		auto mErr = H5Dread (mset_id, dataType, memSpace, mSpace, plist_id, (static_cast<char *> ((*axion)->mStart())+slab*zDim*dataSize));
		auto vErr = H5Dread (vset_id, dataType, memSpace, vSpace, plist_id, (static_cast<char *> ((*axion)->vCpu())  +slab*zDim*dataSize));

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

	// Folder munge(*axion);
	// munge(FOLD_ALL);
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

	switch (vqcdType & VQCD_TYPE)
	{
		case	VQCD_1:
			sprintf(vStr, "VQcd 1");
			break;
		case	VQCD_2:
			sprintf(vStr, "VQcd 2");
			break;
		case	VQCD_0:
			sprintf(vStr, "VQcd 0");
			break;
		case	VQCD_QUAD:
			sprintf(vStr, "VQcd Linear");
			break;
		case	VQCD_PQ_ONLY:
			sprintf(vStr, "VQcd PQ only");
			break;
		case	VQCD_1_PQ_2:
			sprintf(vStr, "VQcd 1 Peccei-Quinn 2");
			break;

		case	VQCD_1N2:
			sprintf(vStr, "VQcd 1 N=2");
			break;

		default:
			sprintf(vStr, "None");
			break;
	}

	switch (vqcdType & VQCD_DAMP)
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

	switch (vqcdType & VQCD_EVOL_RHO)
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

	double maa = axion->AxionMass();//axionmass((*axion->zV()), nQcd, zthres, zrestore);
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
	//indi3 =  maa/pow(*axion->zV(), nQcd*0.5);
	double indi3    = axion->BckGnd()->Indi3();
	double zthres   = axion->BckGnd()->ZThRes();
	double zrestore = axion->BckGnd()->ZRestore();
	double lz2e     = axion->BckGnd()->LamZ2Exp();
	double laam     = axion->LambdaP();

	writeAttribute(vGrp_id, &lStr,  "Lambda type",   attr_type);
	writeAttribute(vGrp_id, &LL,    "Lambda",        H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &lz2e,  "Lambda Z2 exponent",H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &laam,  "LambdaP",       H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &vStr,  "VQcd type",     attr_type);
	writeAttribute(vGrp_id, &dStr,  "Damping type",  attr_type);
	writeAttribute(vGrp_id, &rStr,  "Evolution type",attr_type);
	writeAttribute(vGrp_id, &nQcd,  "nQcd",          H5T_NATIVE_DOUBLE);
	writeAttribute(vGrp_id, &gamma, "Gamma",         H5T_NATIVE_DOUBLE);
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
	LogMsg (VERB_NORMAL, "Closing measurement file...");

	if (opened) {
		H5Pclose (mlist_id);
		H5Fclose (meas_id);
	}

	opened = false;
	header = false;

	meas_id = -1;

	LogMsg (VERB_NORMAL, "Measurement file successfuly closed\n");
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
	hsize_t slb  = slabSz;
	hsize_t lSz  = sizeN;
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
	/* total values to be written */
	hsize_t total  = slabSz;
	/* chunk size */
	hsize_t slab  = sizeN;
	char mCh[16] = "/mapp/m";
	char vCh[16] = "/mapp/v";

	LogMsg (VERB_NORMAL, "[wm2] Writing 2D maps to Hdf5 measurement file YZ");
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

	LogMsg (VERB_HIGH, "[wm2] Ready to write");	LogFlush();
	hsize_t partial = total/commSize();
	for (hsize_t yDim=0; yDim < axion->Depth(); yDim++)
	{
		hsize_t offset = (hsize_t) myRank*partial + yDim*slab;

		H5Sselect_hyperslab(mSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);
		H5Sselect_hyperslab(vSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

		/*	Write raw data	recall slab = sizeN*2*/
		auto mErr = H5Dwrite (mSet_id, dataType, memSpace, mSpace, H5P_DEFAULT, (static_cast<char *> (axion->mFrontGhost())) +sizeN*yDim*dataSize);
		auto vErr = H5Dwrite (vSet_id, dataType, memSpace, vSpace, H5P_DEFAULT, (static_cast<char *> (axion->mBackGhost() )) +sizeN*yDim*dataSize);

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
	hsize_t	dataSize = axion->DataSize();

	int myRank = commRank();

	const hsize_t maxD[1] = { H5S_UNLIMITED };
	hsize_t slb  = slabSz;
	hsize_t lSz  = sizeN;
	char *dataE  = static_cast<char *>(axion->m2Cpu());
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

	/*	Unfold field before writing configuration	*/
	int slicenumber = slicenumbertoprint ;
	if (slicenumbertoprint > axion->Depth())
	{
		LogMsg (VERB_NORMAL, "Sliceprintnumberchanged to 0");
		slicenumber = 0;
	}
	// Folder	munge(axion);
	// munge(UNFOLD_SLICE, slicenumber);

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
	hsize_t slb  = slabSz;
	hsize_t lSz  = sizeN;
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
		LogError ("Error writing dataset /map/P");
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
