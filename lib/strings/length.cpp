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
		nodereal.push_back(0);
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
		nodereal.push_back(0);
		nodereal.push_back(std::real(phic)/(std::real(phic)-std::real(phicv)));
	}
	if(std::imag(phic)*std::imag(phicu)<=0) {
		nodeimag.push_back(std::imag(phic)/(std::imag(phic)-std::imag(phicu)));
		nodeimag.push_back(0);
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
		nodeimag.push_back(0);
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
		//LogMsg(VERB_HIGH,"[Strings3] string position was not properly identified");
		du = 0.5;
		dv = 0.5;
	}
	dua[0] = du;
	dua[1] = dv;
}


// -----------------------------------------------------
// Function that calculates the length of strings
// -----------------------------------------------------

template<typename Float>
StringData	stringlength	(Scalar *field, StringData strDen_in)
{
	LogMsg	(VERB_NORMAL, "[stringlength] Called stringlength");

	StringData	strDen;

	strDen.strDen = strDen_in.strDen;
	strDen.strChr = strDen_in.strChr;
	strDen.wallDn = strDen_in.wallDn;
	strDen.strDen_local = strDen_in.strDen_local;
	strDen.strChr_local = strDen_in.strChr_local;
	strDen.wallDn_local = strDen_in.wallDn_local;

	if ((field->Field() & FIELD_AXION) || (field->Field() == FIELD_WKB)) {
		strDen.strLen = 0;
		strDen.strLen_local = 0;
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

	double length = 0;

	#pragma omp parallel for shared(strdaa, ma, length)
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

				if (strdaa[idx] & (STRING_XY_POSITIVE | STRING_XY_NEGATIVE)) {
					double du[2];
					setCross(ma[idx],ma[ixM],ma[iyM],ma[ixyM],du);
					pos_x.push_back(ix + du[0]);
					pos_y.push_back(iy + du[1]);
					pos_z.push_back(rank*Lz + iz);
				}
				if (strdaa[idx] & (STRING_YZ_POSITIVE | STRING_YZ_NEGATIVE)) {
					double du[2];
					setCross(ma[idx],ma[iyM],ma[izM],ma[iyzM],du);
					pos_x.push_back(ix);
					pos_y.push_back(iy + du[0]);
					pos_z.push_back(rank*Lz + iz + du[1]);
				}
				if (strdaa[idx] & (STRING_ZX_POSITIVE | STRING_ZX_NEGATIVE)) {
					double du[2];
					setCross(ma[idx],ma[izM],ma[ixM],ma[izxM],du);
					pos_x.push_back(ix + du[1]);
					pos_y.push_back(iy);
					pos_z.push_back(rank*Lz + iz + du[0]);
				}
				if (strdaa[ixM] & (STRING_YZ_POSITIVE | STRING_YZ_NEGATIVE)) {
					double du[2];
					setCross(ma[ixM],ma[ixyM],ma[izxM],ma[ixyzM],du);
					pos_x.push_back(ix + 1.);
					pos_y.push_back(iy + du[0]);
					pos_z.push_back(rank*Lz + iz + du[1]);
				}
				if (strdaa[iyM] & (STRING_ZX_POSITIVE | STRING_ZX_NEGATIVE)) {
					double du[2];
					setCross(ma[iyM],ma[iyzM],ma[ixyM],ma[ixyzM],du);
					pos_x.push_back(ix + du[1]);
					pos_y.push_back(iy + 1.);
					pos_z.push_back(rank*Lz + iz + du[0]);
				}
				if (strdaa[izM] & (STRING_XY_POSITIVE | STRING_XY_NEGATIVE)) {
					double du[2];
					setCross(ma[izM],ma[izxM],ma[iyzM],ma[ixyzM],du);
					pos_x.push_back(ix + du[0]);
					pos_y.push_back(iy + du[1]);
					pos_z.push_back(rank*Lz + iz + 1.);
				}

				if(pos_x.size() == 2) {
					// one string is piercing the cube
					double dl = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)));
					#pragma omp atomic update
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
					#pragma omp atomic update
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
					#pragma omp atomic update
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

  commSync();
	return	strDen;
}

StringData stringlength (Scalar *field, StringData strDen_in)
{
	if (field->Precision() == FIELD_SINGLE)
	{
		return stringlength<float> (field, strDen_in);
	}
	else
	{
		return stringlength<double>(field, strDen_in);
	}
}

// -----------------------------------------------------
// Function that saves positions of strings
// and calculate their length
// -----------------------------------------------------

template<typename Float>
StringData	stringlength2	(Scalar *field, StringData strDen_in)
{
	LogMsg	(VERB_NORMAL, "[stringlength2] Called stringlength2");

	StringData	strDen;

	strDen.strDen = strDen_in.strDen;
	strDen.strChr = strDen_in.strChr;
	strDen.wallDn = strDen_in.wallDn;
	strDen.strDen_local = strDen_in.strDen_local;
	strDen.strChr_local = strDen_in.strChr_local;
	strDen.wallDn_local = strDen_in.wallDn_local;

	if ((field->Field() & FIELD_AXION) || (field->Field() == FIELD_WKB)) {
		strDen.strLen = 0;
		strDen.strLen_local = 0;
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

	size_t unilab = 0;
	char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));
	std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mStart());

	double length = 0;

	#pragma omp parallel for shared(unilab, strdaa, ma, length)
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

				if (strdaa[idx] & (STRING_XY_POSITIVE | STRING_XY_NEGATIVE)) {
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
				if (strdaa[idx] & (STRING_YZ_POSITIVE | STRING_YZ_NEGATIVE)) {
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
				if (strdaa[idx] & (STRING_ZX_POSITIVE | STRING_ZX_NEGATIVE)) {
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
				if (strdaa[ixM] & (STRING_YZ_POSITIVE | STRING_YZ_NEGATIVE)) {
					double du[2];
					setCross(ma[ixM],ma[ixyM],ma[izxM],ma[ixyzM],du);
					pos_x.push_back(ix + 1.);
					pos_y.push_back(iy + du[0]);
					pos_z.push_back(rank*Lz + iz + du[1]);
				}
				if (strdaa[iyM] & (STRING_ZX_POSITIVE | STRING_ZX_NEGATIVE)) {
					double du[2];
					setCross(ma[iyM],ma[iyzM],ma[ixyM],ma[ixyzM],du);
					pos_x.push_back(ix + du[1]);
					pos_y.push_back(iy + 1.);
					pos_z.push_back(rank*Lz + iz + du[0]);
				}
				if (strdaa[izM] & (STRING_XY_POSITIVE | STRING_XY_NEGATIVE)) {
					double du[2];
					setCross(ma[izM],ma[izxM],ma[iyzM],ma[ixyzM],du);
					pos_x.push_back(ix + du[0]);
					pos_y.push_back(iy + du[1]);
					pos_z.push_back(rank*Lz + iz + 1.);
				}

				if(pos_x.size() == 2) {
					// one string is piercing the cube
					double dl = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)));
					#pragma omp atomic update
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
					#pragma omp atomic update
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
					#pragma omp atomic update
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
	return	strDen;
}

StringData stringlength2 (Scalar *field, StringData strDen_in)
{
	if (field->Precision() == FIELD_SINGLE)
	{
		return stringlength2<float> (field, strDen_in);
	}
	else
	{
		return stringlength2<double>(field, strDen_in);
	}
}
