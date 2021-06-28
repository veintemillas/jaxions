#include <memory>
#include <cstring>
#include <complex>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#include "strings/strings.h"

#include "utils/utils.h"

#include <vector>
#include "utils/index.h"

#include <mpi.h>


void setCross (std::complex<double> m, std::complex<double> mu, std::complex<double> mv, std::complex<double> muv, double * dua)
{
	std::complex<double> phic = m;
	std::complex<double> phicu = mu;
	std::complex<double> phicv = mv;
	std::complex<double> phicuv = muv;

	std::vector<double> nodereal;
	std::vector<double> nodeimag;
	nodereal.clear();
	nodeimag.clear();

	// identify node positions where Re(m)=0 or Im(m)=0
	// on each node we save its (x,y) coordinates (interval of [0,1]) at 2 elements of nodereal or nodeimag vector.
	if(std::real(phic)*std::real(phicu)<=0) {
		nodereal.push_back(std::real(phic)/(std::real(phic)-std::real(phicu)));
		nodereal.push_back(0.);
	}
	if(std::real(phicu)*std::real(phicuv)<=0) {
		nodereal.push_back(1.0);
		nodereal.push_back(std::real(phicu)/(std::real(phicu)-std::real(phicuv)));
	}
	if(std::real(phicuv)*std::real(phicv)<=0) {
		nodereal.push_back(std::real(phicv)/(std::real(phicv)-std::real(phicuv)));
		nodereal.push_back(1.0);
	}
	if(std::real(phicv)*std::real(phic)<=0) {
		nodereal.push_back(0.);
		nodereal.push_back(std::real(phic)/(std::real(phic)-std::real(phicv)));
	}
	if(std::imag(phic)*std::imag(phicu)<=0) {
		nodeimag.push_back(std::imag(phic)/(std::imag(phic)-std::imag(phicu)));
		nodeimag.push_back(0.);
	}
	if(std::imag(phicu)*std::imag(phicuv)<=0) {
		nodeimag.push_back(1.0);
		nodeimag.push_back(std::imag(phicu)/(std::imag(phicu)-std::imag(phicuv)));
	}
	if(std::imag(phicuv)*std::imag(phicv)<=0) {
		nodeimag.push_back(std::imag(phicv)/(std::imag(phicv)-std::imag(phicuv)));
		nodeimag.push_back(1.0);
	}
	if(std::imag(phicv)*std::imag(phic)<=0) {
		nodeimag.push_back(0.);
		nodeimag.push_back(std::imag(phic)/(std::imag(phic)-std::imag(phicv)));
	}

	// if there are two nodes, calculate crosspoint and save it
	double du, dv;
	if(nodereal.size()==4 && nodeimag.size()==4) {
		double Ar, Br, Cr, Ai, Bi, Ci;
		Ar = nodereal.at(0) - nodereal.at(2);
		Br = nodereal.at(3) - nodereal.at(1);
		Cr = nodereal.at(1)*nodereal.at(2) - nodereal.at(3)*nodereal.at(0);
		Ai = nodeimag.at(0) - nodeimag.at(2);
		Bi = nodeimag.at(3) - nodeimag.at(1);
		Ci = nodeimag.at(1)*nodeimag.at(2) - nodeimag.at(3)*nodeimag.at(0);
		du = -(Cr*Ai-Ci*Ar)/(Br*Ai-Bi*Ar);
		dv = -(Cr*Bi-Ci*Br)/(Ar*Bi-Ai*Br);
	} else {
		// LogMsg(VERB_PARANOID,"[stringlength] string position was not properly identified");
		du = 0.5;
		dv = 0.5;
	}
	dua[0] = du;
	dua[1] = dv;
}


// -----------------------------------------------------
// Function that calculates the length of strings
// -----------------------------------------------------

	/* As it is written, it requires string information from
	the slice forward in z, but stringData is not ghosted,
	and so the last slice cannot be computed reliably,
	the plaquette XY at Z+1 does not exist in the desired rank
	THIS NEEDS TO BE FIXED
	The current PATCH simply ignores the problematic plaquettes */

template<typename Float>
StringData	stringlength	(Scalar *field, StringData strDen_in, StringMeasureType strmeas)
{
	LogMsg	(VERB_NORMAL, "[stringlength] Called stringlength (StringMeasType %d 0124 string/length/gamma/energy)", strmeas);
	LogMsg	(VERB_HIGH, "[stringlength] Points to check %d", strDen_in.strDen);

	StringData	strDen;

	strDen.strDen = strDen_in.strDen;
	strDen.strChr = strDen_in.strChr;
	strDen.wallDn = strDen_in.wallDn;
	strDen.strDen_local = strDen_in.strDen_local;
	strDen.strChr_local = strDen_in.strChr_local;
	strDen.wallDn_local = strDen_in.wallDn_local;

	strDen.strLen = 0.;
	strDen.strDeng = 0.;
	strDen.strVel = 0.;
	strDen.strVel2 = 0.;
	strDen.strGam = 0.;
	strDen.strLen_local = 0.;
	strDen.strDeng_local = 0.;

	if ((field->Field() & FIELD_AXION) || (field->Field() == FIELD_WKB) || !(strmeas & (STRMEAS_LENGTH | STRMEAS_GAMMA))) {
		LogMsg	(VERB_HIGH, "[stringlength] Exit without doing anything");
		return strDen;
	}

	int rank = commRank();

	size_t carde = strDen.strDen_local;
	size_t Lx = field->Length();
	size_t Lz = field->Depth();

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	field->sendGhosts(FIELD_M,COMM_SDRV);
	field->sendGhosts(FIELD_M,COMM_WAIT);

	char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));
	std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mStart());

	if(strmeas & STRMEAS_LENGTH) {
		LogMsg	(VERB_HIGH, "[stringlength] Measure length");

		double length = 0.;

		#pragma omp parallel for reduction(+:length)
		for (size_t iz=0; iz < Lz; iz++) {
			size_t zi = Lx*Lx*iz ;
			size_t zp = Lx*Lx*(iz+1) ;
			for (size_t iy=0; iy < Lx; iy++) {
				size_t yi = Lx*iy ;
				size_t yp = Lx*((iy+1)%Lx) ;
				for (size_t ix=0; ix < Lx; ix++) {
					size_t idx = ix + yi + zi;
					size_t ixM = ((ix + 1) % Lx) + yi + zi;
					size_t iyM = ix + yp + zi;
					size_t izM = ix + yi + zp;
					size_t ixyM = ((ix + 1) % Lx) + yp + zi;
					size_t iyzM = ix + yp + zp;
					size_t izxM = ((ix + 1) % Lx) + yi + zp;
					size_t ixyzM = ((ix + 1) % Lx) + yp + zp;

					std::vector<double> pos_x;
					std::vector<double> pos_y;
					std::vector<double> pos_z;
					pos_x.clear();
					pos_y.clear();
					pos_z.clear();

					if (strdaa[idx] & STRING_XY) {
						double du[2];
						setCross(ma[idx],ma[ixM],ma[iyM],ma[ixyM],du);
						pos_x.push_back(ix + du[0]);
						pos_y.push_back(iy + du[1]);
						pos_z.push_back(rank*Lz + iz);
					}
					if (strdaa[idx] & STRING_YZ) {
						double du[2];
						setCross(ma[idx],ma[iyM],ma[izM],ma[iyzM],du);
						pos_x.push_back(ix);
						pos_y.push_back(iy + du[0]);
						pos_z.push_back(rank*Lz + iz + du[1]);
					}
					if (strdaa[idx] & STRING_ZX) {
						double du[2];
						setCross(ma[idx],ma[izM],ma[ixM],ma[izxM],du);
						pos_x.push_back(ix + du[1]);
						pos_y.push_back(iy);
						pos_z.push_back(rank*Lz + iz + du[0]);
					}
					if (strdaa[ixM] & STRING_YZ) {
						double du[2];
						setCross(ma[ixM],ma[ixyM],ma[izxM],ma[ixyzM],du);
						pos_x.push_back(ix + 1.);
						pos_y.push_back(iy + du[0]);
						pos_z.push_back(rank*Lz + iz + du[1]);
					}
					if (strdaa[iyM] & STRING_ZX) {
						double du[2];
						setCross(ma[iyM],ma[iyzM],ma[ixyM],ma[ixyzM],du);
						pos_x.push_back(ix + du[1]);
						pos_y.push_back(iy + 1.);
						pos_z.push_back(rank*Lz + iz + du[0]);
					}
					/* In the case iz = Lz-1 this would segfault */
					if ( iz < Lz - 1 )
					if (strdaa[izM] & STRING_XY) {
						double du[2];
						setCross(ma[izM],ma[izxM],ma[iyzM],ma[ixyzM],du);
						pos_x.push_back(ix + du[0]);
						pos_y.push_back(iy + du[1]);
						pos_z.push_back(rank*Lz + iz + 1.);
					}

					if(pos_x.size() == 2) {
						// one string is piercing the cube
						double dl = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)));
						length += dl;
					} else if (pos_x.size() == 4) {
						// two strings are piercing the cube
						// we consider three possible connection patterns and average over them
						double dl1 = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)))
                       + sqrt((pos_x.at(3)-pos_x.at(2))*(pos_x.at(3)-pos_x.at(2))+(pos_y.at(3)-pos_y.at(2))*(pos_y.at(3)-pos_y.at(2))+(pos_z.at(3)-pos_z.at(2))*(pos_z.at(3)-pos_z.at(2)));
						double dl2 = sqrt((pos_x.at(2)-pos_x.at(0))*(pos_x.at(2)-pos_x.at(0))+(pos_y.at(2)-pos_y.at(0))*(pos_y.at(2)-pos_y.at(0))+(pos_z.at(2)-pos_z.at(0))*(pos_z.at(2)-pos_z.at(0)))
					 					   + sqrt((pos_x.at(3)-pos_x.at(1))*(pos_x.at(3)-pos_x.at(1))+(pos_y.at(3)-pos_y.at(1))*(pos_y.at(3)-pos_y.at(1))+(pos_z.at(3)-pos_z.at(1))*(pos_z.at(3)-pos_z.at(1)));
					  double dl3 = sqrt((pos_x.at(3)-pos_x.at(0))*(pos_x.at(3)-pos_x.at(0))+(pos_y.at(3)-pos_y.at(0))*(pos_y.at(3)-pos_y.at(0))+(pos_z.at(3)-pos_z.at(0))*(pos_z.at(3)-pos_z.at(0)))
					 	  				 + sqrt((pos_x.at(2)-pos_x.at(1))*(pos_x.at(2)-pos_x.at(1))+(pos_y.at(2)-pos_y.at(1))*(pos_y.at(2)-pos_y.at(1))+(pos_z.at(2)-pos_z.at(1))*(pos_z.at(2)-pos_z.at(1)));
					  length += (dl1 + dl2 + dl3)/3.;
					} else if (pos_x.size() == 6) {
						// three strings are piercing the cube
						// we consider 15 possible connection patterns and average over them
						double dl1 = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)))
					             + sqrt((pos_x.at(3)-pos_x.at(2))*(pos_x.at(3)-pos_x.at(2))+(pos_y.at(3)-pos_y.at(2))*(pos_y.at(3)-pos_y.at(2))+(pos_z.at(3)-pos_z.at(2))*(pos_z.at(3)-pos_z.at(2)));
						  				 + sqrt((pos_x.at(5)-pos_x.at(4))*(pos_x.at(5)-pos_x.at(4))+(pos_y.at(5)-pos_y.at(4))*(pos_y.at(5)-pos_y.at(4))+(pos_z.at(5)-pos_z.at(4))*(pos_z.at(5)-pos_z.at(4)));
					  double dl2 = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)))
					 	  				 + sqrt((pos_x.at(4)-pos_x.at(2))*(pos_x.at(4)-pos_x.at(2))+(pos_y.at(4)-pos_y.at(2))*(pos_y.at(4)-pos_y.at(2))+(pos_z.at(4)-pos_z.at(2))*(pos_z.at(4)-pos_z.at(2)));
					 	  				 + sqrt((pos_x.at(5)-pos_x.at(3))*(pos_x.at(5)-pos_x.at(3))+(pos_y.at(5)-pos_y.at(3))*(pos_y.at(5)-pos_y.at(3))+(pos_z.at(5)-pos_z.at(3))*(pos_z.at(5)-pos_z.at(3)));
					  double dl3 = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)))
					 	  				 + sqrt((pos_x.at(5)-pos_x.at(2))*(pos_x.at(5)-pos_x.at(2))+(pos_y.at(5)-pos_y.at(2))*(pos_y.at(5)-pos_y.at(2))+(pos_z.at(5)-pos_z.at(2))*(pos_z.at(5)-pos_z.at(2)));
					 		  			 + sqrt((pos_x.at(3)-pos_x.at(4))*(pos_x.at(3)-pos_x.at(4))+(pos_y.at(3)-pos_y.at(4))*(pos_y.at(3)-pos_y.at(4))+(pos_z.at(3)-pos_z.at(4))*(pos_z.at(3)-pos_z.at(4)));
				    double dl4 = sqrt((pos_x.at(2)-pos_x.at(0))*(pos_x.at(2)-pos_x.at(0))+(pos_y.at(2)-pos_y.at(0))*(pos_y.at(2)-pos_y.at(0))+(pos_z.at(2)-pos_z.at(0))*(pos_z.at(2)-pos_z.at(0)))
						  				 + sqrt((pos_x.at(3)-pos_x.at(1))*(pos_x.at(3)-pos_x.at(1))+(pos_y.at(3)-pos_y.at(1))*(pos_y.at(3)-pos_y.at(1))+(pos_z.at(3)-pos_z.at(1))*(pos_z.at(3)-pos_z.at(1)));
							  			 + sqrt((pos_x.at(5)-pos_x.at(4))*(pos_x.at(5)-pos_x.at(4))+(pos_y.at(5)-pos_y.at(4))*(pos_y.at(5)-pos_y.at(4))+(pos_z.at(5)-pos_z.at(4))*(pos_z.at(5)-pos_z.at(4)));
					  double dl5 = sqrt((pos_x.at(2)-pos_x.at(0))*(pos_x.at(2)-pos_x.at(0))+(pos_y.at(2)-pos_y.at(0))*(pos_y.at(2)-pos_y.at(0))+(pos_z.at(2)-pos_z.at(0))*(pos_z.at(2)-pos_z.at(0)))
					 	  				 + sqrt((pos_x.at(4)-pos_x.at(1))*(pos_x.at(4)-pos_x.at(1))+(pos_y.at(4)-pos_y.at(1))*(pos_y.at(4)-pos_y.at(1))+(pos_z.at(4)-pos_z.at(1))*(pos_z.at(4)-pos_z.at(1)));
					 		  			 + sqrt((pos_x.at(5)-pos_x.at(3))*(pos_x.at(5)-pos_x.at(3))+(pos_y.at(5)-pos_y.at(3))*(pos_y.at(5)-pos_y.at(3))+(pos_z.at(5)-pos_z.at(3))*(pos_z.at(5)-pos_z.at(3)));
					  double dl6 = sqrt((pos_x.at(2)-pos_x.at(0))*(pos_x.at(2)-pos_x.at(0))+(pos_y.at(2)-pos_y.at(0))*(pos_y.at(2)-pos_y.at(0))+(pos_z.at(2)-pos_z.at(0))*(pos_z.at(2)-pos_z.at(0)))
					 	  				 + sqrt((pos_x.at(5)-pos_x.at(1))*(pos_x.at(5)-pos_x.at(1))+(pos_y.at(5)-pos_y.at(1))*(pos_y.at(5)-pos_y.at(1))+(pos_z.at(5)-pos_z.at(1))*(pos_z.at(5)-pos_z.at(1)));
					 	  				 + sqrt((pos_x.at(3)-pos_x.at(4))*(pos_x.at(3)-pos_x.at(4))+(pos_y.at(3)-pos_y.at(4))*(pos_y.at(3)-pos_y.at(4))+(pos_z.at(3)-pos_z.at(4))*(pos_z.at(3)-pos_z.at(4)));
					  double dl7 = sqrt((pos_x.at(3)-pos_x.at(0))*(pos_x.at(3)-pos_x.at(0))+(pos_y.at(3)-pos_y.at(0))*(pos_y.at(3)-pos_y.at(0))+(pos_z.at(3)-pos_z.at(0))*(pos_z.at(3)-pos_z.at(0)))
					 	  				 + sqrt((pos_x.at(2)-pos_x.at(1))*(pos_x.at(2)-pos_x.at(1))+(pos_y.at(2)-pos_y.at(1))*(pos_y.at(2)-pos_y.at(1))+(pos_z.at(2)-pos_z.at(1))*(pos_z.at(2)-pos_z.at(1)));
					 		  			 + sqrt((pos_x.at(5)-pos_x.at(4))*(pos_x.at(5)-pos_x.at(4))+(pos_y.at(5)-pos_y.at(4))*(pos_y.at(5)-pos_y.at(4))+(pos_z.at(5)-pos_z.at(4))*(pos_z.at(5)-pos_z.at(4)));
					  double dl8 = sqrt((pos_x.at(3)-pos_x.at(0))*(pos_x.at(3)-pos_x.at(0))+(pos_y.at(3)-pos_y.at(0))*(pos_y.at(3)-pos_y.at(0))+(pos_z.at(3)-pos_z.at(0))*(pos_z.at(3)-pos_z.at(0)))
					 	  				 + sqrt((pos_x.at(4)-pos_x.at(1))*(pos_x.at(4)-pos_x.at(1))+(pos_y.at(4)-pos_y.at(1))*(pos_y.at(4)-pos_y.at(1))+(pos_z.at(4)-pos_z.at(1))*(pos_z.at(4)-pos_z.at(1)));
					 		  			 + sqrt((pos_x.at(5)-pos_x.at(2))*(pos_x.at(5)-pos_x.at(2))+(pos_y.at(5)-pos_y.at(2))*(pos_y.at(5)-pos_y.at(2))+(pos_z.at(5)-pos_z.at(2))*(pos_z.at(5)-pos_z.at(2)));
					  double dl9 = sqrt((pos_x.at(3)-pos_x.at(0))*(pos_x.at(3)-pos_x.at(0))+(pos_y.at(3)-pos_y.at(0))*(pos_y.at(3)-pos_y.at(0))+(pos_z.at(3)-pos_z.at(0))*(pos_z.at(3)-pos_z.at(0)))
					 	  				 + sqrt((pos_x.at(5)-pos_x.at(1))*(pos_x.at(5)-pos_x.at(1))+(pos_y.at(5)-pos_y.at(1))*(pos_y.at(5)-pos_y.at(1))+(pos_z.at(5)-pos_z.at(1))*(pos_z.at(5)-pos_z.at(1)));
					 	  				 + sqrt((pos_x.at(2)-pos_x.at(4))*(pos_x.at(2)-pos_x.at(4))+(pos_y.at(2)-pos_y.at(4))*(pos_y.at(2)-pos_y.at(4))+(pos_z.at(2)-pos_z.at(4))*(pos_z.at(2)-pos_z.at(4)));
					  double dl10 = sqrt((pos_x.at(4)-pos_x.at(0))*(pos_x.at(4)-pos_x.at(0))+(pos_y.at(4)-pos_y.at(0))*(pos_y.at(4)-pos_y.at(0))+(pos_z.at(4)-pos_z.at(0))*(pos_z.at(4)-pos_z.at(0)))
					 	  				  + sqrt((pos_x.at(2)-pos_x.at(1))*(pos_x.at(2)-pos_x.at(1))+(pos_y.at(2)-pos_y.at(1))*(pos_y.at(2)-pos_y.at(1))+(pos_z.at(2)-pos_z.at(1))*(pos_z.at(2)-pos_z.at(1)));
					 		  			  + sqrt((pos_x.at(5)-pos_x.at(3))*(pos_x.at(5)-pos_x.at(3))+(pos_y.at(5)-pos_y.at(3))*(pos_y.at(5)-pos_y.at(3))+(pos_z.at(5)-pos_z.at(3))*(pos_z.at(5)-pos_z.at(3)));
					  double dl11 = sqrt((pos_x.at(4)-pos_x.at(0))*(pos_x.at(4)-pos_x.at(0))+(pos_y.at(4)-pos_y.at(0))*(pos_y.at(4)-pos_y.at(0))+(pos_z.at(4)-pos_z.at(0))*(pos_z.at(4)-pos_z.at(0)))
						  					+ sqrt((pos_x.at(3)-pos_x.at(1))*(pos_x.at(3)-pos_x.at(1))+(pos_y.at(3)-pos_y.at(1))*(pos_y.at(3)-pos_y.at(1))+(pos_z.at(3)-pos_z.at(1))*(pos_z.at(3)-pos_z.at(1)));
							  				+ sqrt((pos_x.at(5)-pos_x.at(2))*(pos_x.at(5)-pos_x.at(2))+(pos_y.at(5)-pos_y.at(2))*(pos_y.at(5)-pos_y.at(2))+(pos_z.at(5)-pos_z.at(2))*(pos_z.at(5)-pos_z.at(2)));
					  double dl12 = sqrt((pos_x.at(4)-pos_x.at(0))*(pos_x.at(4)-pos_x.at(0))+(pos_y.at(4)-pos_y.at(0))*(pos_y.at(4)-pos_y.at(0))+(pos_z.at(4)-pos_z.at(0))*(pos_z.at(4)-pos_z.at(0)))
						  					+ sqrt((pos_x.at(5)-pos_x.at(1))*(pos_x.at(5)-pos_x.at(1))+(pos_y.at(5)-pos_y.at(1))*(pos_y.at(5)-pos_y.at(1))+(pos_z.at(5)-pos_z.at(1))*(pos_z.at(5)-pos_z.at(1)));
							  				+ sqrt((pos_x.at(2)-pos_x.at(3))*(pos_x.at(2)-pos_x.at(3))+(pos_y.at(2)-pos_y.at(3))*(pos_y.at(2)-pos_y.at(3))+(pos_z.at(2)-pos_z.at(3))*(pos_z.at(2)-pos_z.at(3)));
					  double dl13 = sqrt((pos_x.at(5)-pos_x.at(0))*(pos_x.at(5)-pos_x.at(0))+(pos_y.at(5)-pos_y.at(0))*(pos_y.at(5)-pos_y.at(0))+(pos_z.at(5)-pos_z.at(0))*(pos_z.at(5)-pos_z.at(0)))
						  					+ sqrt((pos_x.at(2)-pos_x.at(1))*(pos_x.at(2)-pos_x.at(1))+(pos_y.at(2)-pos_y.at(1))*(pos_y.at(2)-pos_y.at(1))+(pos_z.at(2)-pos_z.at(1))*(pos_z.at(2)-pos_z.at(1)));
						  					+ sqrt((pos_x.at(4)-pos_x.at(3))*(pos_x.at(4)-pos_x.at(3))+(pos_y.at(4)-pos_y.at(3))*(pos_y.at(4)-pos_y.at(3))+(pos_z.at(4)-pos_z.at(3))*(pos_z.at(4)-pos_z.at(3)));
					  double dl14 = sqrt((pos_x.at(5)-pos_x.at(0))*(pos_x.at(5)-pos_x.at(0))+(pos_y.at(5)-pos_y.at(0))*(pos_y.at(5)-pos_y.at(0))+(pos_z.at(5)-pos_z.at(0))*(pos_z.at(5)-pos_z.at(0)))
						  					+ sqrt((pos_x.at(3)-pos_x.at(1))*(pos_x.at(3)-pos_x.at(1))+(pos_y.at(3)-pos_y.at(1))*(pos_y.at(3)-pos_y.at(1))+(pos_z.at(3)-pos_z.at(1))*(pos_z.at(3)-pos_z.at(1)));
							  			  + sqrt((pos_x.at(4)-pos_x.at(2))*(pos_x.at(4)-pos_x.at(2))+(pos_y.at(4)-pos_y.at(2))*(pos_y.at(4)-pos_y.at(2))+(pos_z.at(4)-pos_z.at(2))*(pos_z.at(4)-pos_z.at(2)));
					  double dl15 = sqrt((pos_x.at(5)-pos_x.at(0))*(pos_x.at(5)-pos_x.at(0))+(pos_y.at(5)-pos_y.at(0))*(pos_y.at(5)-pos_y.at(0))+(pos_z.at(5)-pos_z.at(0))*(pos_z.at(5)-pos_z.at(0)))
						  					+ sqrt((pos_x.at(4)-pos_x.at(1))*(pos_x.at(4)-pos_x.at(1))+(pos_y.at(4)-pos_y.at(1))*(pos_y.at(4)-pos_y.at(1))+(pos_z.at(4)-pos_z.at(1))*(pos_z.at(4)-pos_z.at(1)));
						  				  + sqrt((pos_x.at(2)-pos_x.at(3))*(pos_x.at(2)-pos_x.at(3))+(pos_y.at(2)-pos_y.at(3))*(pos_y.at(2)-pos_y.at(3))+(pos_z.at(2)-pos_z.at(3))*(pos_z.at(2)-pos_z.at(3)));
					  length += (dl1 + dl2 + dl3 + dl4 + dl5 + dl6 + dl7 + dl8 + dl9 + dl10 + dl11 + dl12 + dl13 + dl14 + dl15)/15.;
				  }
				  //else if ((pos_x.size() == 1)||(pos_x.size() == 3)||(pos_x.size() == 5)) {
					//	LogMsg(VERB_HIGH,"[stringlength] length is not calculable: idx = (%d,%d,%d), # of ends = %d",ix,iy,iz,pos_x.size());
					//}

				}
			}
		} // end of iz loop

		strDen.strLen_local = length;

		MPI_Allreduce(&(strDen.strLen_local), &(strDen.strLen), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		LogMsg(VERB_HIGH, "[stringlength] length = %f",strDen.strLen);LogFlush();
  	commSync();

	}

	// next calculate string velocity and gamma factor

	if(strmeas & STRMEAS_GAMMA) {
		LogMsg	(VERB_HIGH, "[stringlength] Measure Gamma");

		double gamma  = 0.;
  	double gamma2 = 0.;
  	double vel    = 0.;
  	double vel2   = 0.;

		double mssq = field->SaxionMassSq();
		double c = 0.41238;

		std::complex<Float> Rscale((Float)*field->RV(),0.);
		std::complex<Float> Hc((Float)field->HubbleConformal(),0.);

		std::complex<Float> *va     = static_cast<std::complex<Float>*>(field->vCpu());

		#pragma omp parallel for reduction(+:gamma,gamma2,vel,vel2)
		for (size_t iz=0; iz < Lz; iz++) {
			size_t zi = Lx*Lx*iz ;
			size_t zp = Lx*Lx*(iz+1) ;
			for (size_t iy=0; iy < Lx; iy++) {
				size_t yi = Lx*iy ;
				size_t yp = Lx*((iy+1)%Lx) ;
				for (size_t ix=0; ix < Lx; ix++) {
					size_t idx = ix + yi + zi;
					size_t ixM = ((ix + 1) % Lx) + yi + zi;
					size_t iyM = ix + yp + zi;
					size_t izM = ix + yi + zp;
					size_t ixyM = ((ix + 1) % Lx) + yp + zi;
					size_t iyzM = ix + yp + zp;
					size_t izxM = ((ix + 1) % Lx) + yi + zp;

					if((strdaa[idx] & STRING_ONLY) != 0)
					{
						double g2v2a, g2v2b, g2v2c, g2v2d;
						std::complex<double> phia = ma[idx]/Rscale;
						std::complex<double> dphia = (va[idx]-Hc*ma[idx])/(Rscale*Rscale);
						// Eq. (A.10) of 1509.00026
						g2v2a = std::abs(dphia)*std::abs(dphia)*(1.+std::abs(phia)*std::abs(phia)/(8.*c*c))/(mssq*c*c)
				          + std::abs(std::conj(phia)*dphia+phia*std::conj(dphia))*std::abs(std::conj(phia)*dphia+phia*std::conj(dphia))/(16.*mssq*c*c*c*c);

						if(strdaa[idx] & STRING_XY) {
							std::complex<double> phib = ma[ixM]/Rscale;
							std::complex<double> phic = ma[iyM]/Rscale;
							std::complex<double> phid = ma[ixyM]/Rscale;
							std::complex<double> dphib = (va[ixM]-Hc*ma[ixM])/(Rscale*Rscale);
							std::complex<double> dphic = (va[iyM]-Hc*ma[iyM])/(Rscale*Rscale);
							std::complex<double> dphid = (va[ixyM]-Hc*ma[ixyM])/(Rscale*Rscale);
							g2v2b = std::abs(dphib)*std::abs(dphib)*(1.+std::abs(phib)*std::abs(phib)/(8.*c*c))/(mssq*c*c)
					          + std::abs(std::conj(phib)*dphib+phib*std::conj(dphib))*std::abs(std::conj(phib)*dphib+phib*std::conj(dphib))/(16.*mssq*c*c*c*c);
					    g2v2c = std::abs(dphic)*std::abs(dphic)*(1.+std::abs(phic)*std::abs(phic)/(8.*c*c))/(mssq*c*c)
							  	  + std::abs(std::conj(phic)*dphic+phic*std::conj(dphic))*std::abs(std::conj(phic)*dphic+phic*std::conj(dphic))/(16.*mssq*c*c*c*c);
						  g2v2d = std::abs(dphid)*std::abs(dphid)*(1.+std::abs(phid)*std::abs(phid)/(8.*c*c))/(mssq*c*c)
							  	  + std::abs(std::conj(phid)*dphid+phid*std::conj(dphid))*std::abs(std::conj(phid)*dphid+phid*std::conj(dphid))/(16.*mssq*c*c*c*c);
							double g2v2 = (g2v2a + g2v2b + g2v2c + g2v2d)/4.;
							double g = sqrt(1.+g2v2); // local gamma
							gamma += g;
							gamma2 += 1.+g2v2;
							vel += g*sqrt(g2v2/(1.+g2v2));
							vel2 += g*g2v2/(1.+g2v2);
						}
						if(strdaa[idx] & STRING_YZ) {
							std::complex<double> phib = ma[iyM]/Rscale;
							std::complex<double> phic = ma[izM]/Rscale;
							std::complex<double> phid = ma[iyzM]/Rscale;
							std::complex<double> dphib = (va[iyM]-Hc*ma[iyM])/(Rscale*Rscale);
							std::complex<double> dphic = (va[izM]-Hc*ma[izM])/(Rscale*Rscale);
							std::complex<double> dphid = (va[iyzM]-Hc*ma[iyzM])/(Rscale*Rscale);
							g2v2b = std::abs(dphib)*std::abs(dphib)*(1.+std::abs(phib)*std::abs(phib)/(8.*c*c))/(mssq*c*c)
					          + std::abs(std::conj(phib)*dphib+phib*std::conj(dphib))*std::abs(std::conj(phib)*dphib+phib*std::conj(dphib))/(16.*mssq*c*c*c*c);
					    g2v2c = std::abs(dphic)*std::abs(dphic)*(1.+std::abs(phic)*std::abs(phic)/(8.*c*c))/(mssq*c*c)
							  	  + std::abs(std::conj(phic)*dphic+phic*std::conj(dphic))*std::abs(std::conj(phic)*dphic+phic*std::conj(dphic))/(16.*mssq*c*c*c*c);
						  g2v2d = std::abs(dphid)*std::abs(dphid)*(1.+std::abs(phid)*std::abs(phid)/(8.*c*c))/(mssq*c*c)
							  	  + std::abs(std::conj(phid)*dphid+phid*std::conj(dphid))*std::abs(std::conj(phid)*dphid+phid*std::conj(dphid))/(16.*mssq*c*c*c*c);
							double g2v2 = (g2v2a + g2v2b + g2v2c + g2v2d)/4.;
							double g = sqrt(1.+g2v2); // local gamma
							gamma += g;
							gamma2 += 1.+g2v2;
							vel += g*sqrt(g2v2/(1.+g2v2));
							vel2 += g*g2v2/(1.+g2v2);
						}
						if(strdaa[idx] & STRING_ZX) {
							std::complex<double> phib = ma[izM]/Rscale;
							std::complex<double> phic = ma[ixM]/Rscale;
							std::complex<double> phid = ma[izxM]/Rscale;
							std::complex<double> dphib = (va[izM]-Hc*ma[izM])/(Rscale*Rscale);
							std::complex<double> dphic = (va[ixM]-Hc*ma[ixM])/(Rscale*Rscale);
							std::complex<double> dphid = (va[izxM]-Hc*ma[izxM])/(Rscale*Rscale);
							g2v2b = std::abs(dphib)*std::abs(dphib)*(1.+std::abs(phib)*std::abs(phib)/(8.*c*c))/(mssq*c*c)
					          + std::abs(std::conj(phib)*dphib+phib*std::conj(dphib))*std::abs(std::conj(phib)*dphib+phib*std::conj(dphib))/(16.*mssq*c*c*c*c);
					    g2v2c = std::abs(dphic)*std::abs(dphic)*(1.+std::abs(phic)*std::abs(phic)/(8.*c*c))/(mssq*c*c)
							  	  + std::abs(std::conj(phic)*dphic+phic*std::conj(dphic))*std::abs(std::conj(phic)*dphic+phic*std::conj(dphic))/(16.*mssq*c*c*c*c);
						  g2v2d = std::abs(dphid)*std::abs(dphid)*(1.+std::abs(phid)*std::abs(phid)/(8.*c*c))/(mssq*c*c)
							  	  + std::abs(std::conj(phid)*dphid+phid*std::conj(dphid))*std::abs(std::conj(phid)*dphid+phid*std::conj(dphid))/(16.*mssq*c*c*c*c);
					    double g2v2 = (g2v2a + g2v2b + g2v2c + g2v2d)/4.;
					    double g = sqrt(1.+g2v2); // local gamma
					  	gamma += g;
						  gamma2 += 1.+g2v2;
						  vel += g*sqrt(g2v2/(1.+g2v2));
						  vel2 += g*g2v2/(1.+g2v2);
						}
					}

				}
			}
		} // end of iz loop

	  double gamma_tot  = 0.;
    double gamma2_tot = 0.;
    double vel_tot    = 0.;
    double vel2_tot   = 0.;

	  MPI_Allreduce(&(gamma), &(gamma_tot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	  MPI_Allreduce(&(gamma2), &(gamma2_tot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	  MPI_Allreduce(&(vel), &(vel_tot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	  MPI_Allreduce(&(vel2), &(vel2_tot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	  strDen.strDeng = gamma_tot;
	  strDen.strVel = vel_tot/gamma_tot;
	  strDen.strVel2 = vel2_tot/gamma_tot;
	  strDen.strGam = gamma2_tot/gamma_tot;
	  strDen.strDeng_local = gamma;

		LogMsg(VERB_HIGH, "[stringlength] gamma_tot = %f",strDen.strDeng);

	  commSync();
  }

	return	strDen;
}

StringData stringlength (Scalar *field, StringData strDen_in, StringMeasureType strmeas)
{
	if (field->Precision() == FIELD_SINGLE)
	{
		return stringlength<float> (field, strDen_in, strmeas);
	}
	else
	{
		return stringlength<double>(field, strDen_in, strmeas);
	}
}

// -----------------------------------------------------
// Function that saves positions of strings
// and calculate their length
// -----------------------------------------------------

template<typename Float>
StringData	stringlength2	(Scalar *field, StringData strDen_in, StringMeasureType strmeas)
{
	LogMsg	(VERB_NORMAL, "[stringlength2] Called stringlength2");

	StringData	strDen;

	strDen.strDen = strDen_in.strDen;
	strDen.strChr = strDen_in.strChr;
	strDen.wallDn = strDen_in.wallDn;
	strDen.strDen_local = strDen_in.strDen_local;
	strDen.strChr_local = strDen_in.strChr_local;
	strDen.wallDn_local = strDen_in.wallDn_local;

	strDen.strLen = 0.;
	strDen.strDeng = 0.;
	strDen.strVel = 0.;
	strDen.strVel2 = 0.;
	strDen.strGam = 0.;
	strDen.strLen_local = 0.;
	strDen.strDeng_local = 0.;

	if ((field->Field() & FIELD_AXION) || (field->Field() == FIELD_WKB) || !(strmeas & (STRMEAS_LENGTH | STRMEAS_GAMMA))) {
		return strDen;
	}

	auto	eStr = std::make_unique<Strings> (field);

	int rank = commRank();

	eStr->resizePos ();

	size_t carde = strDen.strDen_local;
	size_t Lx = field->Length();
	size_t Lz = field->Depth();

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	field->sendGhosts(FIELD_M,COMM_SDRV);
	field->sendGhosts(FIELD_M,COMM_WAIT);

	char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));
	std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mStart());

  if(strmeas & STRMEAS_LENGTH) {

		double length = 0.;
		size_t unilab = 0;

		#pragma omp parallel for shared(unilab) reduction(+:length)
		for (size_t iz=0; iz < Lz; iz++) {
			size_t zi = Lx*Lx*iz ;
			size_t zp = Lx*Lx*(iz+1) ;
			for (size_t iy=0; iy < Lx; iy++) {
				size_t yi = Lx*iy ;
				size_t yp = Lx*((iy+1)%Lx) ;
				for (size_t ix=0; ix < Lx; ix++) {
					size_t idx = ix + yi + zi;
					size_t ixM = ((ix + 1) % Lx) + yi + zi;
					size_t iyM = ix + yp + zi;
					size_t izM = ix + yi + zp;
					size_t ixyM = ((ix + 1) % Lx) + yp + zi;
					size_t iyzM = ix + yp + zp;
					size_t izxM = ((ix + 1) % Lx) + yi + zp;
					size_t ixyzM = ((ix + 1) % Lx) + yp + zp;

					std::vector<double> pos_x;
					std::vector<double> pos_y;
					std::vector<double> pos_z;
					pos_x.clear();
					pos_y.clear();
					pos_z.clear();

					size_t besidefresi;

					if (strdaa[idx] & STRING_XY) {
						double du[2];
						setCross(ma[idx],ma[ixM],ma[iyM],ma[ixyM],du);
						#pragma omp atomic capture
						{ besidefresi = unilab ; unilab += 1 ; }
						eStr->Pos()[besidefresi*3+2] = rank*Lz + iz;
						eStr->Pos()[besidefresi*3+1] = iy + du[1];
						eStr->Pos()[besidefresi*3]   = ix + du[0];
						pos_x.push_back(ix + du[0]);
						pos_y.push_back(iy + du[1]);
						pos_z.push_back(rank*Lz + iz);
					}
					if (strdaa[idx] & STRING_YZ) {
						double du[2];
						setCross(ma[idx],ma[iyM],ma[izM],ma[iyzM],du);
						#pragma omp atomic capture
						{ besidefresi = unilab ; unilab += 1 ; }
						eStr->Pos()[besidefresi*3+2] = rank*Lz + iz + du[1];
						eStr->Pos()[besidefresi*3+1] = iy + du[0];
						eStr->Pos()[besidefresi*3]   = ix;
						pos_x.push_back(ix);
						pos_y.push_back(iy + du[0]);
						pos_z.push_back(rank*Lz + iz + du[1]);
					}
					if (strdaa[idx] & STRING_ZX) {
						double du[2];
						setCross(ma[idx],ma[izM],ma[ixM],ma[izxM],du);
						#pragma omp atomic capture
						{ besidefresi = unilab ; unilab += 1 ; }
						eStr->Pos()[besidefresi*3+2] = rank*Lz + iz + du[0];
						eStr->Pos()[besidefresi*3+1] = iy;
						eStr->Pos()[besidefresi*3]   = ix + du[1];
						pos_x.push_back(ix + du[1]);
						pos_y.push_back(iy);
						pos_z.push_back(rank*Lz + iz + du[0]);
					}
					if (strdaa[ixM] & STRING_YZ) {
						double du[2];
						setCross(ma[ixM],ma[ixyM],ma[izxM],ma[ixyzM],du);
						pos_x.push_back(ix + 1.);
						pos_y.push_back(iy + du[0]);
						pos_z.push_back(rank*Lz + iz + du[1]);
					}
					if (strdaa[iyM] & STRING_ZX) {
						double du[2];
						setCross(ma[iyM],ma[iyzM],ma[ixyM],ma[ixyzM],du);
						pos_x.push_back(ix + du[1]);
						pos_y.push_back(iy + 1.);
						pos_z.push_back(rank*Lz + iz + du[0]);
					}
					if (strdaa[izM] & STRING_XY) {
						double du[2];
						setCross(ma[izM],ma[izxM],ma[iyzM],ma[ixyzM],du);
						pos_x.push_back(ix + du[0]);
						pos_y.push_back(iy + du[1]);
						pos_z.push_back(rank*Lz + iz + 1.);
					}

					if(pos_x.size() == 2) {
						// one string is piercing the cube
						double dl = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)));
						length += dl;
					} else if (pos_x.size() == 4) {
						// two strings are piercing the cube
						// we consider three possible connection patterns and average over them
						double dl1 = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)))
						  				 + sqrt((pos_x.at(3)-pos_x.at(2))*(pos_x.at(3)-pos_x.at(2))+(pos_y.at(3)-pos_y.at(2))*(pos_y.at(3)-pos_y.at(2))+(pos_z.at(3)-pos_z.at(2))*(pos_z.at(3)-pos_z.at(2)));
					  double dl2 = sqrt((pos_x.at(2)-pos_x.at(0))*(pos_x.at(2)-pos_x.at(0))+(pos_y.at(2)-pos_y.at(0))*(pos_y.at(2)-pos_y.at(0))+(pos_z.at(2)-pos_z.at(0))*(pos_z.at(2)-pos_z.at(0)))
						  				 + sqrt((pos_x.at(3)-pos_x.at(1))*(pos_x.at(3)-pos_x.at(1))+(pos_y.at(3)-pos_y.at(1))*(pos_y.at(3)-pos_y.at(1))+(pos_z.at(3)-pos_z.at(1))*(pos_z.at(3)-pos_z.at(1)));
					  double dl3 = sqrt((pos_x.at(3)-pos_x.at(0))*(pos_x.at(3)-pos_x.at(0))+(pos_y.at(3)-pos_y.at(0))*(pos_y.at(3)-pos_y.at(0))+(pos_z.at(3)-pos_z.at(0))*(pos_z.at(3)-pos_z.at(0)))
						  				 + sqrt((pos_x.at(2)-pos_x.at(1))*(pos_x.at(2)-pos_x.at(1))+(pos_y.at(2)-pos_y.at(1))*(pos_y.at(2)-pos_y.at(1))+(pos_z.at(2)-pos_z.at(1))*(pos_z.at(2)-pos_z.at(1)));
					  length += (dl1 + dl2 + dl3)/3.;
					} else if (pos_x.size() == 6) {
						// three strings are piercing the cube
						// we consider 15 possible connection patterns and average over them
						double dl1 = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)))
						  				 + sqrt((pos_x.at(3)-pos_x.at(2))*(pos_x.at(3)-pos_x.at(2))+(pos_y.at(3)-pos_y.at(2))*(pos_y.at(3)-pos_y.at(2))+(pos_z.at(3)-pos_z.at(2))*(pos_z.at(3)-pos_z.at(2)));
							  			 + sqrt((pos_x.at(5)-pos_x.at(4))*(pos_x.at(5)-pos_x.at(4))+(pos_y.at(5)-pos_y.at(4))*(pos_y.at(5)-pos_y.at(4))+(pos_z.at(5)-pos_z.at(4))*(pos_z.at(5)-pos_z.at(4)));
					  double dl2 = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)))
						  				 + sqrt((pos_x.at(4)-pos_x.at(2))*(pos_x.at(4)-pos_x.at(2))+(pos_y.at(4)-pos_y.at(2))*(pos_y.at(4)-pos_y.at(2))+(pos_z.at(4)-pos_z.at(2))*(pos_z.at(4)-pos_z.at(2)));
							  			 + sqrt((pos_x.at(5)-pos_x.at(3))*(pos_x.at(5)-pos_x.at(3))+(pos_y.at(5)-pos_y.at(3))*(pos_y.at(5)-pos_y.at(3))+(pos_z.at(5)-pos_z.at(3))*(pos_z.at(5)-pos_z.at(3)));
					  double dl3 = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)))
						  				 + sqrt((pos_x.at(5)-pos_x.at(2))*(pos_x.at(5)-pos_x.at(2))+(pos_y.at(5)-pos_y.at(2))*(pos_y.at(5)-pos_y.at(2))+(pos_z.at(5)-pos_z.at(2))*(pos_z.at(5)-pos_z.at(2)));
						  				 + sqrt((pos_x.at(3)-pos_x.at(4))*(pos_x.at(3)-pos_x.at(4))+(pos_y.at(3)-pos_y.at(4))*(pos_y.at(3)-pos_y.at(4))+(pos_z.at(3)-pos_z.at(4))*(pos_z.at(3)-pos_z.at(4)));
					  double dl4 = sqrt((pos_x.at(2)-pos_x.at(0))*(pos_x.at(2)-pos_x.at(0))+(pos_y.at(2)-pos_y.at(0))*(pos_y.at(2)-pos_y.at(0))+(pos_z.at(2)-pos_z.at(0))*(pos_z.at(2)-pos_z.at(0)))
						  				 + sqrt((pos_x.at(3)-pos_x.at(1))*(pos_x.at(3)-pos_x.at(1))+(pos_y.at(3)-pos_y.at(1))*(pos_y.at(3)-pos_y.at(1))+(pos_z.at(3)-pos_z.at(1))*(pos_z.at(3)-pos_z.at(1)));
						  				 + sqrt((pos_x.at(5)-pos_x.at(4))*(pos_x.at(5)-pos_x.at(4))+(pos_y.at(5)-pos_y.at(4))*(pos_y.at(5)-pos_y.at(4))+(pos_z.at(5)-pos_z.at(4))*(pos_z.at(5)-pos_z.at(4)));
					 double dl5 = sqrt((pos_x.at(2)-pos_x.at(0))*(pos_x.at(2)-pos_x.at(0))+(pos_y.at(2)-pos_y.at(0))*(pos_y.at(2)-pos_y.at(0))+(pos_z.at(2)-pos_z.at(0))*(pos_z.at(2)-pos_z.at(0)))
					  					 + sqrt((pos_x.at(4)-pos_x.at(1))*(pos_x.at(4)-pos_x.at(1))+(pos_y.at(4)-pos_y.at(1))*(pos_y.at(4)-pos_y.at(1))+(pos_z.at(4)-pos_z.at(1))*(pos_z.at(4)-pos_z.at(1)));
						  				 + sqrt((pos_x.at(5)-pos_x.at(3))*(pos_x.at(5)-pos_x.at(3))+(pos_y.at(5)-pos_y.at(3))*(pos_y.at(5)-pos_y.at(3))+(pos_z.at(5)-pos_z.at(3))*(pos_z.at(5)-pos_z.at(3)));
					  double dl6 = sqrt((pos_x.at(2)-pos_x.at(0))*(pos_x.at(2)-pos_x.at(0))+(pos_y.at(2)-pos_y.at(0))*(pos_y.at(2)-pos_y.at(0))+(pos_z.at(2)-pos_z.at(0))*(pos_z.at(2)-pos_z.at(0)))
						  				 + sqrt((pos_x.at(5)-pos_x.at(1))*(pos_x.at(5)-pos_x.at(1))+(pos_y.at(5)-pos_y.at(1))*(pos_y.at(5)-pos_y.at(1))+(pos_z.at(5)-pos_z.at(1))*(pos_z.at(5)-pos_z.at(1)));
							  			 + sqrt((pos_x.at(3)-pos_x.at(4))*(pos_x.at(3)-pos_x.at(4))+(pos_y.at(3)-pos_y.at(4))*(pos_y.at(3)-pos_y.at(4))+(pos_z.at(3)-pos_z.at(4))*(pos_z.at(3)-pos_z.at(4)));
					  double dl7 = sqrt((pos_x.at(3)-pos_x.at(0))*(pos_x.at(3)-pos_x.at(0))+(pos_y.at(3)-pos_y.at(0))*(pos_y.at(3)-pos_y.at(0))+(pos_z.at(3)-pos_z.at(0))*(pos_z.at(3)-pos_z.at(0)))
						  				 + sqrt((pos_x.at(2)-pos_x.at(1))*(pos_x.at(2)-pos_x.at(1))+(pos_y.at(2)-pos_y.at(1))*(pos_y.at(2)-pos_y.at(1))+(pos_z.at(2)-pos_z.at(1))*(pos_z.at(2)-pos_z.at(1)));
							  			 + sqrt((pos_x.at(5)-pos_x.at(4))*(pos_x.at(5)-pos_x.at(4))+(pos_y.at(5)-pos_y.at(4))*(pos_y.at(5)-pos_y.at(4))+(pos_z.at(5)-pos_z.at(4))*(pos_z.at(5)-pos_z.at(4)));
					  double dl8 = sqrt((pos_x.at(3)-pos_x.at(0))*(pos_x.at(3)-pos_x.at(0))+(pos_y.at(3)-pos_y.at(0))*(pos_y.at(3)-pos_y.at(0))+(pos_z.at(3)-pos_z.at(0))*(pos_z.at(3)-pos_z.at(0)))
						  				 + sqrt((pos_x.at(4)-pos_x.at(1))*(pos_x.at(4)-pos_x.at(1))+(pos_y.at(4)-pos_y.at(1))*(pos_y.at(4)-pos_y.at(1))+(pos_z.at(4)-pos_z.at(1))*(pos_z.at(4)-pos_z.at(1)));
							  			 + sqrt((pos_x.at(5)-pos_x.at(2))*(pos_x.at(5)-pos_x.at(2))+(pos_y.at(5)-pos_y.at(2))*(pos_y.at(5)-pos_y.at(2))+(pos_z.at(5)-pos_z.at(2))*(pos_z.at(5)-pos_z.at(2)));
					  double dl9 = sqrt((pos_x.at(3)-pos_x.at(0))*(pos_x.at(3)-pos_x.at(0))+(pos_y.at(3)-pos_y.at(0))*(pos_y.at(3)-pos_y.at(0))+(pos_z.at(3)-pos_z.at(0))*(pos_z.at(3)-pos_z.at(0)))
						  				 + sqrt((pos_x.at(5)-pos_x.at(1))*(pos_x.at(5)-pos_x.at(1))+(pos_y.at(5)-pos_y.at(1))*(pos_y.at(5)-pos_y.at(1))+(pos_z.at(5)-pos_z.at(1))*(pos_z.at(5)-pos_z.at(1)));
							  			 + sqrt((pos_x.at(2)-pos_x.at(4))*(pos_x.at(2)-pos_x.at(4))+(pos_y.at(2)-pos_y.at(4))*(pos_y.at(2)-pos_y.at(4))+(pos_z.at(2)-pos_z.at(4))*(pos_z.at(2)-pos_z.at(4)));
					  double dl10 = sqrt((pos_x.at(4)-pos_x.at(0))*(pos_x.at(4)-pos_x.at(0))+(pos_y.at(4)-pos_y.at(0))*(pos_y.at(4)-pos_y.at(0))+(pos_z.at(4)-pos_z.at(0))*(pos_z.at(4)-pos_z.at(0)))
						  					+ sqrt((pos_x.at(2)-pos_x.at(1))*(pos_x.at(2)-pos_x.at(1))+(pos_y.at(2)-pos_y.at(1))*(pos_y.at(2)-pos_y.at(1))+(pos_z.at(2)-pos_z.at(1))*(pos_z.at(2)-pos_z.at(1)));
							  				+ sqrt((pos_x.at(5)-pos_x.at(3))*(pos_x.at(5)-pos_x.at(3))+(pos_y.at(5)-pos_y.at(3))*(pos_y.at(5)-pos_y.at(3))+(pos_z.at(5)-pos_z.at(3))*(pos_z.at(5)-pos_z.at(3)));
					  double dl11 = sqrt((pos_x.at(4)-pos_x.at(0))*(pos_x.at(4)-pos_x.at(0))+(pos_y.at(4)-pos_y.at(0))*(pos_y.at(4)-pos_y.at(0))+(pos_z.at(4)-pos_z.at(0))*(pos_z.at(4)-pos_z.at(0)))
						  					+ sqrt((pos_x.at(3)-pos_x.at(1))*(pos_x.at(3)-pos_x.at(1))+(pos_y.at(3)-pos_y.at(1))*(pos_y.at(3)-pos_y.at(1))+(pos_z.at(3)-pos_z.at(1))*(pos_z.at(3)-pos_z.at(1)));
							  				+ sqrt((pos_x.at(5)-pos_x.at(2))*(pos_x.at(5)-pos_x.at(2))+(pos_y.at(5)-pos_y.at(2))*(pos_y.at(5)-pos_y.at(2))+(pos_z.at(5)-pos_z.at(2))*(pos_z.at(5)-pos_z.at(2)));
				  	double dl12 = sqrt((pos_x.at(4)-pos_x.at(0))*(pos_x.at(4)-pos_x.at(0))+(pos_y.at(4)-pos_y.at(0))*(pos_y.at(4)-pos_y.at(0))+(pos_z.at(4)-pos_z.at(0))*(pos_z.at(4)-pos_z.at(0)))
					   						+ sqrt((pos_x.at(5)-pos_x.at(1))*(pos_x.at(5)-pos_x.at(1))+(pos_y.at(5)-pos_y.at(1))*(pos_y.at(5)-pos_y.at(1))+(pos_z.at(5)-pos_z.at(1))*(pos_z.at(5)-pos_z.at(1)));
						  					+ sqrt((pos_x.at(2)-pos_x.at(3))*(pos_x.at(2)-pos_x.at(3))+(pos_y.at(2)-pos_y.at(3))*(pos_y.at(2)-pos_y.at(3))+(pos_z.at(2)-pos_z.at(3))*(pos_z.at(2)-pos_z.at(3)));
					  double dl13 = sqrt((pos_x.at(5)-pos_x.at(0))*(pos_x.at(5)-pos_x.at(0))+(pos_y.at(5)-pos_y.at(0))*(pos_y.at(5)-pos_y.at(0))+(pos_z.at(5)-pos_z.at(0))*(pos_z.at(5)-pos_z.at(0)))
						  					+ sqrt((pos_x.at(2)-pos_x.at(1))*(pos_x.at(2)-pos_x.at(1))+(pos_y.at(2)-pos_y.at(1))*(pos_y.at(2)-pos_y.at(1))+(pos_z.at(2)-pos_z.at(1))*(pos_z.at(2)-pos_z.at(1)));
							  				+ sqrt((pos_x.at(4)-pos_x.at(3))*(pos_x.at(4)-pos_x.at(3))+(pos_y.at(4)-pos_y.at(3))*(pos_y.at(4)-pos_y.at(3))+(pos_z.at(4)-pos_z.at(3))*(pos_z.at(4)-pos_z.at(3)));
					  double dl14 = sqrt((pos_x.at(5)-pos_x.at(0))*(pos_x.at(5)-pos_x.at(0))+(pos_y.at(5)-pos_y.at(0))*(pos_y.at(5)-pos_y.at(0))+(pos_z.at(5)-pos_z.at(0))*(pos_z.at(5)-pos_z.at(0)))
						  					+ sqrt((pos_x.at(3)-pos_x.at(1))*(pos_x.at(3)-pos_x.at(1))+(pos_y.at(3)-pos_y.at(1))*(pos_y.at(3)-pos_y.at(1))+(pos_z.at(3)-pos_z.at(1))*(pos_z.at(3)-pos_z.at(1)));
							  				+ sqrt((pos_x.at(4)-pos_x.at(2))*(pos_x.at(4)-pos_x.at(2))+(pos_y.at(4)-pos_y.at(2))*(pos_y.at(4)-pos_y.at(2))+(pos_z.at(4)-pos_z.at(2))*(pos_z.at(4)-pos_z.at(2)));
					  double dl15 = sqrt((pos_x.at(5)-pos_x.at(0))*(pos_x.at(5)-pos_x.at(0))+(pos_y.at(5)-pos_y.at(0))*(pos_y.at(5)-pos_y.at(0))+(pos_z.at(5)-pos_z.at(0))*(pos_z.at(5)-pos_z.at(0)))
						  					+ sqrt((pos_x.at(4)-pos_x.at(1))*(pos_x.at(4)-pos_x.at(1))+(pos_y.at(4)-pos_y.at(1))*(pos_y.at(4)-pos_y.at(1))+(pos_z.at(4)-pos_z.at(1))*(pos_z.at(4)-pos_z.at(1)));
							  				+ sqrt((pos_x.at(2)-pos_x.at(3))*(pos_x.at(2)-pos_x.at(3))+(pos_y.at(2)-pos_y.at(3))*(pos_y.at(2)-pos_y.at(3))+(pos_z.at(2)-pos_z.at(3))*(pos_z.at(2)-pos_z.at(3)));
					  length += (dl1 + dl2 + dl3 + dl4 + dl5 + dl6 + dl7 + dl8 + dl9 + dl10 + dl11 + dl12 + dl13 + dl14 + dl15)/15.;
				  }
			  	//else if (pos_x.size() > 0) {
				  //	LogMsg(VERB_HIGH,"[stringlength2] length is not calculable: idx = (%d,%d,%d), # of ends = %d",ix,iy,iz,pos_x.size());
				  //}

			  }
		  }
	  } // end of iz loop

	  strDen.strLen_local = length;

	  MPI_Allreduce(&(strDen.strLen_local), &(strDen.strLen), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	  double red = (double) sizeof(eStr->Pos()[0])*carde/ (double)(sizeof(strdaa[0])*field->Size());
	  LogMsg	(VERB_NORMAL, "[stringlength2] red = %f ", red);

	  /* make sure that the sData buffer never explotes in lowmem */
	  /* if no lowmem use m2 ! */
	  char *dest ;
	  size_t charmax ;
  	if (field->LowMem()){
	  	dest = strdaa;
		  LogOut("can fail!");
		  charmax = field->Size();
	  }
	  else
		  {
			  dest = static_cast<char *>(field->m2Cpu());
			  charmax = field->Size()*field->DataSize ();
		  }

	  size_t carde3 =carde*3*sizeof(eStr->Pos()[0]);
  	size_t trinitrize = (charmax/3)*3;
  	LogMsg	(VERB_NORMAL, "[stringlength2] charmax = %lu (%lu), needed %lu ", charmax,trinitrize,carde3);

	  char *orig = static_cast<char*>( static_cast<void*>( &eStr->Pos()[0] ));

  	carde3 = std::min(carde3, trinitrize);

	  if (field->LowMem()){
		  LogMsg(VERB_HIGH,"[stringlength2] copyng %d bytes to sData() [%lu bytes]",carde3,field->Size());
		  memcpy (dest, orig, carde3);
		  field->setSD(SD_STRINGCOORD);
	  }
	  else
	  {
		  LogMsg(VERB_HIGH,"[stringlength2] copyng %d bytes to m2Cpu() [%lu bytes]",carde3,field->Size());
		  memcpy (dest, orig, carde3);
		  field->setM2(M2_STRINGCOO);
		  // unsigned short *cerda = static_cast<unsigned short *>( static_cast<void*>( &eStr->Pos()[0] ));
		  // printf("strings.cpp rank %d   %hu, %hu, %hu\n", rank, cerda[0],cerda[1],cerda[2]);
	  }

    commSync();

  }

	// next calculate string velocity and gamma factor

	if(strmeas & STRMEAS_GAMMA) {

	  double gamma  = 0.;
	  double gamma2 = 0.;
		double vel    = 0.;
		double vel2   = 0.;

		double mssq = field->SaxionMassSq();
		double c = 0.41238;

		std::complex<Float> Rscale((Float)*field->RV(),0.);
		std::complex<Float> Hc((Float)field->HubbleConformal(),0.);

		std::complex<Float> *va     = static_cast<std::complex<Float>*>(field->vCpu());

		#pragma omp parallel for reduction(+:gamma,gamma2,vel,vel2)
		for (size_t iz=0; iz < Lz; iz++) {
			size_t zi = Lx*Lx*iz ;
			size_t zp = Lx*Lx*(iz+1) ;
			for (size_t iy=0; iy < Lx; iy++) {
				size_t yi = Lx*iy ;
				size_t yp = Lx*((iy+1)%Lx) ;
				for (size_t ix=0; ix < Lx; ix++) {
					size_t idx = ix + yi + zi;
					size_t ixM = ((ix + 1) % Lx) + yi + zi;
				  size_t iyM = ix + yp + zi;
				  size_t izM = ix + yi + zp;
				  size_t ixyM = ((ix + 1) % Lx) + yp + zi;
				  size_t iyzM = ix + yp + zp;
				  size_t izxM = ((ix + 1) % Lx) + yi + zp;

				  if((strdaa[idx] & STRING_ONLY) != 0)
			  	{
				  	double g2v2a, g2v2b, g2v2c, g2v2d;
					  std::complex<double> phia = ma[idx]/Rscale;
						std::complex<double> dphia = (va[idx]-Hc*ma[idx])/(Rscale*Rscale);
						// Eq. (A.10) of 1509.00026
						g2v2a = std::abs(dphia)*std::abs(dphia)*(1.+std::abs(phia)*std::abs(phia)/(8.*c*c))/(mssq*c*c)
						  		+ std::abs(std::conj(phia)*dphia+phia*std::conj(dphia))*std::abs(std::conj(phia)*dphia+phia*std::conj(dphia))/(16.*mssq*c*c*c*c);

						if(strdaa[idx] & STRING_XY) {
							std::complex<double> phib = ma[ixM]/Rscale;
							std::complex<double> phic = ma[iyM]/Rscale;
							std::complex<double> phid = ma[ixyM]/Rscale;
							std::complex<double> dphib = (va[ixM]-Hc*ma[ixM])/(Rscale*Rscale);
							std::complex<double> dphic = (va[iyM]-Hc*ma[iyM])/(Rscale*Rscale);
							std::complex<double> dphid = (va[ixyM]-Hc*ma[ixyM])/(Rscale*Rscale);
							g2v2b = std::abs(dphib)*std::abs(dphib)*(1.+std::abs(phib)*std::abs(phib)/(8.*c*c))/(mssq*c*c)
										+ std::abs(std::conj(phib)*dphib+phib*std::conj(dphib))*std::abs(std::conj(phib)*dphib+phib*std::conj(dphib))/(16.*mssq*c*c*c*c);
							g2v2c = std::abs(dphic)*std::abs(dphic)*(1.+std::abs(phic)*std::abs(phic)/(8.*c*c))/(mssq*c*c)
							  		+ std::abs(std::conj(phic)*dphic+phic*std::conj(dphic))*std::abs(std::conj(phic)*dphic+phic*std::conj(dphic))/(16.*mssq*c*c*c*c);
						  g2v2d = std::abs(dphid)*std::abs(dphid)*(1.+std::abs(phid)*std::abs(phid)/(8.*c*c))/(mssq*c*c)
							  		+ std::abs(std::conj(phid)*dphid+phid*std::conj(dphid))*std::abs(std::conj(phid)*dphid+phid*std::conj(dphid))/(16.*mssq*c*c*c*c);
							double g2v2 = (g2v2a + g2v2b + g2v2c + g2v2d)/4.;
							double g = sqrt(1.+g2v2); // local gamma
							gamma += g;
							gamma2 += 1.+g2v2;
					  	vel += g*sqrt(g2v2/(1.+g2v2));
							vel2 += g*g2v2/(1.+g2v2);
						}
						if(strdaa[idx] & STRING_YZ) {
							std::complex<double> phib = ma[iyM]/Rscale;
							std::complex<double> phic = ma[izM]/Rscale;
							std::complex<double> phid = ma[iyzM]/Rscale;
							std::complex<double> dphib = (va[iyM]-Hc*ma[iyM])/(Rscale*Rscale);
							std::complex<double> dphic = (va[izM]-Hc*ma[izM])/(Rscale*Rscale);
							std::complex<double> dphid = (va[iyzM]-Hc*ma[iyzM])/(Rscale*Rscale);
						  g2v2b = std::abs(dphib)*std::abs(dphib)*(1.+std::abs(phib)*std::abs(phib)/(8.*c*c))/(mssq*c*c)
							  		+ std::abs(std::conj(phib)*dphib+phib*std::conj(dphib))*std::abs(std::conj(phib)*dphib+phib*std::conj(dphib))/(16.*mssq*c*c*c*c);
						  g2v2c = std::abs(dphic)*std::abs(dphic)*(1.+std::abs(phic)*std::abs(phic)/(8.*c*c))/(mssq*c*c)
							  		+ std::abs(std::conj(phic)*dphic+phic*std::conj(dphic))*std::abs(std::conj(phic)*dphic+phic*std::conj(dphic))/(16.*mssq*c*c*c*c);
						  g2v2d = std::abs(dphid)*std::abs(dphid)*(1.+std::abs(phid)*std::abs(phid)/(8.*c*c))/(mssq*c*c)
							  		+ std::abs(std::conj(phid)*dphid+phid*std::conj(dphid))*std::abs(std::conj(phid)*dphid+phid*std::conj(dphid))/(16.*mssq*c*c*c*c);
						  double g2v2 = (g2v2a + g2v2b + g2v2c + g2v2d)/4.;
							double g = sqrt(1.+g2v2); // local gamma
							gamma += g;
							gamma2 += 1.+g2v2;
							vel += g*sqrt(g2v2/(1.+g2v2));
							vel2 += g*g2v2/(1.+g2v2);
						}
						if(strdaa[idx] & STRING_ZX) {
							std::complex<double> phib = ma[izM]/Rscale;
							std::complex<double> phic = ma[ixM]/Rscale;
							std::complex<double> phid = ma[izxM]/Rscale;
							std::complex<double> dphib = (va[izM]-Hc*ma[izM])/(Rscale*Rscale);
							std::complex<double> dphic = (va[ixM]-Hc*ma[ixM])/(Rscale*Rscale);
							std::complex<double> dphid = (va[izxM]-Hc*ma[izxM])/(Rscale*Rscale);
						  g2v2b = std::abs(dphib)*std::abs(dphib)*(1.+std::abs(phib)*std::abs(phib)/(8.*c*c))/(mssq*c*c)
							  		+ std::abs(std::conj(phib)*dphib+phib*std::conj(dphib))*std::abs(std::conj(phib)*dphib+phib*std::conj(dphib))/(16.*mssq*c*c*c*c);
						  g2v2c = std::abs(dphic)*std::abs(dphic)*(1.+std::abs(phic)*std::abs(phic)/(8.*c*c))/(mssq*c*c)
							  		+ std::abs(std::conj(phic)*dphic+phic*std::conj(dphic))*std::abs(std::conj(phic)*dphic+phic*std::conj(dphic))/(16.*mssq*c*c*c*c);
						  g2v2d = std::abs(dphid)*std::abs(dphid)*(1.+std::abs(phid)*std::abs(phid)/(8.*c*c))/(mssq*c*c)
							  		+ std::abs(std::conj(phid)*dphid+phid*std::conj(dphid))*std::abs(std::conj(phid)*dphid+phid*std::conj(dphid))/(16.*mssq*c*c*c*c);
						  double g2v2 = (g2v2a + g2v2b + g2v2c + g2v2d)/4.;
							double g = sqrt(1.+g2v2); // local gamma
							gamma += g;
							gamma2 += 1.+g2v2;
							vel += g*sqrt(g2v2/(1.+g2v2));
							vel2 += g*g2v2/(1.+g2v2);
						}
					}

				}
			}
		} // end of iz loop

		double gamma_tot  = 0.;
		double gamma2_tot = 0.;
		double vel_tot    = 0.;
		double vel2_tot   = 0.;

		MPI_Allreduce(&(gamma), &(gamma_tot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&(gamma2), &(gamma2_tot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&(vel), &(vel_tot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&(vel2), &(vel2_tot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		strDen.strDeng = gamma_tot;
		strDen.strVel = vel_tot/gamma_tot;
		strDen.strVel2 = vel2_tot/gamma_tot;
		strDen.strGam = gamma2_tot/gamma_tot;
		strDen.strDeng_local = gamma;

		commSync();
	}

	return	strDen;
}

StringData stringlength2 (Scalar *field, StringData strDen_in, StringMeasureType strmeas)
{
	if (field->Precision() == FIELD_SINGLE)
	{
		return stringlength2<float> (field, strDen_in, strmeas);
	}
	else
	{
		return stringlength2<double>(field, strDen_in, strmeas);
	}
}

// ------------------------------------------------------------------
// Function that calculates the energy density of strings
// NOTE: It assumes that string core is identified by masker function
//       and correponding grid points are saved in sData.
// ------------------------------------------------------------------

template<typename Float>
StringEnergyData stringenergy (Scalar *field)
{
	LogMsg	(VERB_NORMAL, "[stringenergy] Called stringenergy");

	size_t Lx = field->Length();
	size_t Lz = field->Depth();

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	field->sendGhosts(FIELD_M,COMM_SDRV);
	field->sendGhosts(FIELD_M,COMM_WAIT);

	double Rscale = *field->RV();
	std::complex<Float> Hc((Float)field->HubbleConformal(),0.);
	std::complex<Float> zaskaF((Float)field->Saskia()*Rscale, 0.);
	double depta = field->BckGnd()->PhysSize()/Lx;
	double lambda = field->LambdaP();

	char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));
	std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mStart());
	std::complex<Float> *va     = static_cast<std::complex<Float>*>(field->vCpu());

	double rhotot = 0.;
	double rhoa = 0.;
	double rhos = 0.;
	size_t nout = 0; // # of points outside the core of strings
	double rhoaV = 0.;
	double rhosV = 0.;

	#pragma omp parallel for reduction(+:rhotot,rhoa,rhos,rhoaV,rhosV,nout)
	for (size_t iz=0; iz < Lz; iz++) {
		size_t zi = Lx*Lx*iz ;
		size_t zp = Lx*Lx*(iz+1) ;
		for (size_t iy=0; iy < Lx; iy++) {
			size_t yi = Lx*iy ;
			size_t yp = Lx*((iy+1)%Lx) ;
			for (size_t ix=0; ix < Lx; ix++) {
				size_t idx = ix + yi + zi;
				size_t ixM = ((ix + 1) % Lx) + yi + zi;
				size_t iyM = ix + yp + zi;
				size_t izM = ix + yi + zp;

				// total energy of complex scalar field
				double rhokin = .5*std::abs(va[idx]-Hc*ma[idx])*std::abs(va[idx]-Hc*ma[idx])/(Rscale*Rscale*Rscale*Rscale);
				double rhograd = .5*(std::abs(ma[ixM]-ma[idx])*std::abs(ma[ixM]-ma[idx])+std::abs(ma[iyM]-ma[idx])*std::abs(ma[iyM]-ma[idx])+std::abs(ma[izM]-ma[idx])*std::abs(ma[izM]-ma[idx]))/(Rscale*Rscale*Rscale*Rscale*depta*depta);
				// NOTE: In the following the PQ1 potential is assumed. It must be modified for PQ2 potential.
				double rhopot = 0.25*lambda*(std::abs(ma[idx])*std::abs(ma[idx])/(Rscale*Rscale)-1.)*(std::abs(ma[idx])*std::abs(ma[idx])/(Rscale*Rscale)-1.);
				rhotot += rhokin + rhograd + rhopot;
				// axion and saxion energies are evaluated only outside the core of strings
				// axion energy evaluated from kinetic energy
				double rhoakin = .5*std::imag((va[idx]-Hc*ma[idx])/(ma[idx]-zaskaF))*std::imag((va[idx]-Hc*ma[idx])/(ma[idx]-zaskaF))/(Rscale*Rscale);
				// saxion kinetic and gradient energy
				Float modu = std::abs(ma[idx]-zaskaF);
				double rhoskin = .5*std::real((va[idx]-Hc*ma[idx])*modu/(ma[idx]-zaskaF))*std::real((va[idx]-Hc*ma[idx])*modu/(ma[idx]-zaskaF))/(Rscale*Rscale*Rscale*Rscale);
				double rhosgrad = .5*((std::abs(ma[ixM]-zaskaF)-modu)*(std::abs(ma[ixM]-zaskaF)-modu)+(std::abs(ma[iyM]-zaskaF)-modu)*(std::abs(ma[iyM]-zaskaF)-modu)+(std::abs(ma[izM]-zaskaF)-modu)*(std::abs(ma[izM]-zaskaF)-modu))/(Rscale*Rscale*Rscale*Rscale*depta*depta);
				// Villadoro's masking squared coincides with the real definition of energy
				rhoaV += pow(std::abs(ma[idx]-zaskaF)/Rscale,2)*2.*rhoakin;
				rhosV += pow(std::abs(ma[idx]-zaskaF)/Rscale,2)*(rhoskin + rhosgrad + rhopot);
				if (!(strdaa[idx] & STRING_MASK)) {
					// Redondo's masking
					rhoa += 2.*rhoakin;
				  rhos += rhoskin + rhosgrad + rhopot;
					nout += 1;
				}
			}
		}
	} // end of iz loop

	double rhotot_sum = 0.;
	double rhoa_sum = 0.;
	double rhos_sum = 0.;
	size_t nout_sum = 0;
	double rhoaV_sum = 0.;
	double rhosV_sum = 0.;

	MPI_Allreduce(&(rhotot), &(rhotot_sum), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&(rhoa), &(rhoa_sum), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&(rhos), &(rhos_sum), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&(nout), &(nout_sum), 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&(rhoaV), &(rhoaV_sum), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&(rhosV), &(rhosV_sum), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	StringEnergyData strE;

	strE.rho_a   = rhoa_sum/nout_sum;
	strE.rho_s   = rhos_sum/nout_sum;
	strE.rho_str = rhotot_sum/(Lx*Lx*Lx) - strE.rho_a - strE.rho_s;
	strE.rho_a_Vil = rhoaV_sum/(Lx*Lx*Lx);
	strE.rho_s_Vil = rhosV_sum/(Lx*Lx*Lx);
	strE.rho_str_Vil = rhotot_sum/(Lx*Lx*Lx) - strE.rho_a_Vil - strE.rho_s_Vil;
	strE.nout = nout_sum;

	commSync();
	return strE;
}

StringEnergyData stringenergy (Scalar *field)
{
	if (field->Precision() == FIELD_SINGLE)
	{
		return stringenergy<float> (field);
	}
	else
	{
		return stringenergy<double>(field);
	}
}
