#ifndef	_LENGTH_
	#define	_LENGTH_

#include "utils/utils.h"

#include <cstdint>
#include <array>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>

/* Calculates the next cube chosing the next STRCUB data exit */
int how_many_plaquettes(char sc)
{
  int n_pla=0;
	// size_t next_idx =0;
	if (sc & STRCUB_XY)
		n_pla++;
	if (sc & STRCUB_YZ)
		n_pla++;
	if (sc & STRCUB_ZX)
		n_pla++;
	if (sc & STRCUB_XY2)
		n_pla++;
	if (sc & STRCUB_YZ2)
		n_pla++;
	if (sc & STRCUB_ZX2)
		n_pla++;
  return n_pla;
}

/* Calculates the next cube chosing the next STRCUB data exit */
/* THE ORDER in next_cube_idx and close_exit_cube MUST BE THE SAME!*/
size_t next_cube_idx(char sc, size_t X[3],size_t Lx, size_t Lz, size_t Sf, bool *out, size_t ng = 0){
	*out = false;
	if (sc & STRCUB_YZ)
		return (X[0]+Lx-1)%Lx+X[1]*Lx+X[2]*Sf;
	if (sc & STRCUB_ZX)
		return X[0]+((X[1]+Lx-1)%Lx)*Lx+X[2]*Sf;
	if (sc & STRCUB_YZ2)
		return (X[0]+1)%Lx+X[1]*Lx+X[2]*Sf;
	if (sc & STRCUB_ZX2)
		return X[0]+((X[1]+1)%Lx)*Lx+X[2]*Sf;
  if (sc & STRCUB_XY)  // this can be negative
  	{ if (X[2] == ng) *out = true;
      return X[0]+X[1]*Lx+(X[2]-1)*Sf;}
  if (sc & STRCUB_XY2) // this can be larger than current volume
		{ if (X[2] == Lz+ng-1) *out = true;
      return X[0]+X[1]*Lx+(X[2]+1)*Sf;}
}

size_t prev_cube_idx(char sc_in, size_t X[3], size_t Lx, size_t Lz, size_t Sf, bool* out, size_t ng =0) {
    *out = false;

    if (sc_in & STRCUB_YZ)
  		return (X[0]+Lx-1)%Lx+X[1]*Lx+X[2]*Sf;
  	if (sc_in & STRCUB_ZX)
  		return X[0]+((X[1]+Lx-1)%Lx)*Lx+X[2]*Sf;
  	if (sc_in & STRCUB_YZ2)
  		return (X[0]+1)%Lx+X[1]*Lx+X[2]*Sf;
  	if (sc_in & STRCUB_ZX2)
  		return X[0]+((X[1]+1)%Lx)*Lx+X[2]*Sf;
    if (sc_in & STRCUB_XY)  // this can be negative
    	{ if (X[2] == ng) *out = true;
        return X[0]+X[1]*Lx+(X[2]-1)*Sf;}
    if (sc_in & STRCUB_XY2) // this can be larger than current volume
  		{ if (X[2] == Lz+ng-1) *out = true;
        return X[0]+X[1]*Lx+(X[2]+1)*Sf;}

    return 0;
}


/* Returns a STRCHAR with the 1st exit CLOSED*/
char close_exit_cube(char sc, size_t X[3],size_t Lx, size_t Sf)
{
	// size_t next_idx =0;
  char out = sc;
	if (sc & STRCUB_YZ)
		return sc ^ STRCUB_YZ;
	if (sc & STRCUB_ZX)
		return sc ^ STRCUB_ZX;
	if (sc & STRCUB_YZ2)
		return sc ^ STRCUB_YZ2;
	if (sc & STRCUB_ZX2)
		return sc ^ STRCUB_ZX2;
  if (sc & STRCUB_XY)
		return sc ^ STRCUB_XY ;
  if (sc & STRCUB_XY2)
		return sc ^ STRCUB_XY2;
}

size_t label_map_index(size_t *cumlabels, unsigned short int tid, unsigned short int local_label)
{
  return cumlabels[tid]+local_label-1;
}

std::pair<unsigned short int,unsigned short int> inverse_label_map_index(size_t *cumlabels, size_t uid)
{
  int thread = 0;
  while (uid > cumlabels[thread])
  {
    thread++;
  }
  return {(unsigned short int) thread, (unsigned short int) (uid-cumlabels[thread])};
}

template <typename Float>
Float dl_cal(const Float* px, const Float* py, const Float* pz, int np) {
    Float length = 0;

    auto norm = [&](int i1, int i2) -> Float {
        Float dx = px[i1] - px[i2];
        Float dy = py[i1] - py[i2];
        Float dz = pz[i1] - pz[i2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    if (np == 2) {
        length += norm(0, 1);
    } else if (np == 4) {
        length += (norm(0, 1) + norm(2, 3) +
                   norm(1, 2) + norm(0, 3) +
                   norm(2, 0) + norm(1, 3)) / Float(3);
    } else if (np == 6) {
        length += (norm(0, 1) + norm(0, 2) + norm(0, 3) + norm(0, 4) + norm(0, 5) +
                   norm(1, 2) + norm(1, 3) + norm(1, 4) + norm(1, 5) +
                   norm(2, 3) + norm(2, 4) + norm(2, 5) +
                   norm(3, 4) + norm(3, 5) +
                   norm(4, 5)) / Float(5);
    }

    return length;
}

double dl_cal(double *px, double *py, double *pz, int np)
  {
    double length = 0;

    auto norm = [&](int i1, int i2) {
        return sqrt(pow(px[i1]-px[i2],2)+pow(py[i1]-py[i2],2)+pow(pz[i1]-pz[i2],2));
    };

    if(np == 2) {
      // one string is piercing the cube
      double dl = norm(0,1);
      length += dl;
    } else if (np == 4) {
      // two strings are piercing the cube
      // we consider three possible connection patterns and average over them
      length += (norm(0,1)+norm(2,3)+norm(1,2)+norm(0,3)+norm(2,0)+norm(1,3))/3.;
    } else if (np == 6) {
      // three strings are piercing the cube
      // we consider 15 possible connection patterns and average over them
      length += (norm(0,1)+norm(0,2)+norm(0,3)+norm(0,4)+norm(0,5)+
      norm(1,2)+norm(1,3)+norm(1,4)+norm(1,5)+
      norm(2,3)+norm(2,4)+norm(2,5)+
      norm(3,4)+norm(3,5)+
      norm(4,5))/5.;
    }
    return length;
  }

double dl_cal(std::vector<double> pos_x,std::vector<double> pos_y,std::vector<double> pos_z)
{
  double length = 0;
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
               + sqrt((pos_x.at(3)-pos_x.at(2))*(pos_x.at(3)-pos_x.at(2))+(pos_y.at(3)-pos_y.at(2))*(pos_y.at(3)-pos_y.at(2))+(pos_z.at(3)-pos_z.at(2))*(pos_z.at(3)-pos_z.at(2)))
               + sqrt((pos_x.at(5)-pos_x.at(4))*(pos_x.at(5)-pos_x.at(4))+(pos_y.at(5)-pos_y.at(4))*(pos_y.at(5)-pos_y.at(4))+(pos_z.at(5)-pos_z.at(4))*(pos_z.at(5)-pos_z.at(4)));
    double dl2 = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)))
               + sqrt((pos_x.at(4)-pos_x.at(2))*(pos_x.at(4)-pos_x.at(2))+(pos_y.at(4)-pos_y.at(2))*(pos_y.at(4)-pos_y.at(2))+(pos_z.at(4)-pos_z.at(2))*(pos_z.at(4)-pos_z.at(2)))
               + sqrt((pos_x.at(5)-pos_x.at(3))*(pos_x.at(5)-pos_x.at(3))+(pos_y.at(5)-pos_y.at(3))*(pos_y.at(5)-pos_y.at(3))+(pos_z.at(5)-pos_z.at(3))*(pos_z.at(5)-pos_z.at(3)));
    double dl3 = sqrt((pos_x.at(1)-pos_x.at(0))*(pos_x.at(1)-pos_x.at(0))+(pos_y.at(1)-pos_y.at(0))*(pos_y.at(1)-pos_y.at(0))+(pos_z.at(1)-pos_z.at(0))*(pos_z.at(1)-pos_z.at(0)))
               + sqrt((pos_x.at(5)-pos_x.at(2))*(pos_x.at(5)-pos_x.at(2))+(pos_y.at(5)-pos_y.at(2))*(pos_y.at(5)-pos_y.at(2))+(pos_z.at(5)-pos_z.at(2))*(pos_z.at(5)-pos_z.at(2)))
               + sqrt((pos_x.at(3)-pos_x.at(4))*(pos_x.at(3)-pos_x.at(4))+(pos_y.at(3)-pos_y.at(4))*(pos_y.at(3)-pos_y.at(4))+(pos_z.at(3)-pos_z.at(4))*(pos_z.at(3)-pos_z.at(4)));
    double dl4 = sqrt((pos_x.at(2)-pos_x.at(0))*(pos_x.at(2)-pos_x.at(0))+(pos_y.at(2)-pos_y.at(0))*(pos_y.at(2)-pos_y.at(0))+(pos_z.at(2)-pos_z.at(0))*(pos_z.at(2)-pos_z.at(0)))
               + sqrt((pos_x.at(3)-pos_x.at(1))*(pos_x.at(3)-pos_x.at(1))+(pos_y.at(3)-pos_y.at(1))*(pos_y.at(3)-pos_y.at(1))+(pos_z.at(3)-pos_z.at(1))*(pos_z.at(3)-pos_z.at(1)))
               + sqrt((pos_x.at(5)-pos_x.at(4))*(pos_x.at(5)-pos_x.at(4))+(pos_y.at(5)-pos_y.at(4))*(pos_y.at(5)-pos_y.at(4))+(pos_z.at(5)-pos_z.at(4))*(pos_z.at(5)-pos_z.at(4)));
    double dl5 = sqrt((pos_x.at(2)-pos_x.at(0))*(pos_x.at(2)-pos_x.at(0))+(pos_y.at(2)-pos_y.at(0))*(pos_y.at(2)-pos_y.at(0))+(pos_z.at(2)-pos_z.at(0))*(pos_z.at(2)-pos_z.at(0)))
               + sqrt((pos_x.at(4)-pos_x.at(1))*(pos_x.at(4)-pos_x.at(1))+(pos_y.at(4)-pos_y.at(1))*(pos_y.at(4)-pos_y.at(1))+(pos_z.at(4)-pos_z.at(1))*(pos_z.at(4)-pos_z.at(1)))
               + sqrt((pos_x.at(5)-pos_x.at(3))*(pos_x.at(5)-pos_x.at(3))+(pos_y.at(5)-pos_y.at(3))*(pos_y.at(5)-pos_y.at(3))+(pos_z.at(5)-pos_z.at(3))*(pos_z.at(5)-pos_z.at(3)));
    double dl6 = sqrt((pos_x.at(2)-pos_x.at(0))*(pos_x.at(2)-pos_x.at(0))+(pos_y.at(2)-pos_y.at(0))*(pos_y.at(2)-pos_y.at(0))+(pos_z.at(2)-pos_z.at(0))*(pos_z.at(2)-pos_z.at(0)))
               + sqrt((pos_x.at(5)-pos_x.at(1))*(pos_x.at(5)-pos_x.at(1))+(pos_y.at(5)-pos_y.at(1))*(pos_y.at(5)-pos_y.at(1))+(pos_z.at(5)-pos_z.at(1))*(pos_z.at(5)-pos_z.at(1)))
               + sqrt((pos_x.at(3)-pos_x.at(4))*(pos_x.at(3)-pos_x.at(4))+(pos_y.at(3)-pos_y.at(4))*(pos_y.at(3)-pos_y.at(4))+(pos_z.at(3)-pos_z.at(4))*(pos_z.at(3)-pos_z.at(4)));
    double dl7 = sqrt((pos_x.at(3)-pos_x.at(0))*(pos_x.at(3)-pos_x.at(0))+(pos_y.at(3)-pos_y.at(0))*(pos_y.at(3)-pos_y.at(0))+(pos_z.at(3)-pos_z.at(0))*(pos_z.at(3)-pos_z.at(0)))
               + sqrt((pos_x.at(2)-pos_x.at(1))*(pos_x.at(2)-pos_x.at(1))+(pos_y.at(2)-pos_y.at(1))*(pos_y.at(2)-pos_y.at(1))+(pos_z.at(2)-pos_z.at(1))*(pos_z.at(2)-pos_z.at(1)))
               + sqrt((pos_x.at(5)-pos_x.at(4))*(pos_x.at(5)-pos_x.at(4))+(pos_y.at(5)-pos_y.at(4))*(pos_y.at(5)-pos_y.at(4))+(pos_z.at(5)-pos_z.at(4))*(pos_z.at(5)-pos_z.at(4)));
    double dl8 = sqrt((pos_x.at(3)-pos_x.at(0))*(pos_x.at(3)-pos_x.at(0))+(pos_y.at(3)-pos_y.at(0))*(pos_y.at(3)-pos_y.at(0))+(pos_z.at(3)-pos_z.at(0))*(pos_z.at(3)-pos_z.at(0)))
               + sqrt((pos_x.at(4)-pos_x.at(1))*(pos_x.at(4)-pos_x.at(1))+(pos_y.at(4)-pos_y.at(1))*(pos_y.at(4)-pos_y.at(1))+(pos_z.at(4)-pos_z.at(1))*(pos_z.at(4)-pos_z.at(1)))
               + sqrt((pos_x.at(5)-pos_x.at(2))*(pos_x.at(5)-pos_x.at(2))+(pos_y.at(5)-pos_y.at(2))*(pos_y.at(5)-pos_y.at(2))+(pos_z.at(5)-pos_z.at(2))*(pos_z.at(5)-pos_z.at(2)));
    double dl9 = sqrt((pos_x.at(3)-pos_x.at(0))*(pos_x.at(3)-pos_x.at(0))+(pos_y.at(3)-pos_y.at(0))*(pos_y.at(3)-pos_y.at(0))+(pos_z.at(3)-pos_z.at(0))*(pos_z.at(3)-pos_z.at(0)))
               + sqrt((pos_x.at(5)-pos_x.at(1))*(pos_x.at(5)-pos_x.at(1))+(pos_y.at(5)-pos_y.at(1))*(pos_y.at(5)-pos_y.at(1))+(pos_z.at(5)-pos_z.at(1))*(pos_z.at(5)-pos_z.at(1)))
               + sqrt((pos_x.at(2)-pos_x.at(4))*(pos_x.at(2)-pos_x.at(4))+(pos_y.at(2)-pos_y.at(4))*(pos_y.at(2)-pos_y.at(4))+(pos_z.at(2)-pos_z.at(4))*(pos_z.at(2)-pos_z.at(4)));
    double dl10 = sqrt((pos_x.at(4)-pos_x.at(0))*(pos_x.at(4)-pos_x.at(0))+(pos_y.at(4)-pos_y.at(0))*(pos_y.at(4)-pos_y.at(0))+(pos_z.at(4)-pos_z.at(0))*(pos_z.at(4)-pos_z.at(0)))
                + sqrt((pos_x.at(2)-pos_x.at(1))*(pos_x.at(2)-pos_x.at(1))+(pos_y.at(2)-pos_y.at(1))*(pos_y.at(2)-pos_y.at(1))+(pos_z.at(2)-pos_z.at(1))*(pos_z.at(2)-pos_z.at(1)))
                + sqrt((pos_x.at(5)-pos_x.at(3))*(pos_x.at(5)-pos_x.at(3))+(pos_y.at(5)-pos_y.at(3))*(pos_y.at(5)-pos_y.at(3))+(pos_z.at(5)-pos_z.at(3))*(pos_z.at(5)-pos_z.at(3)));
    double dl11 = sqrt((pos_x.at(4)-pos_x.at(0))*(pos_x.at(4)-pos_x.at(0))+(pos_y.at(4)-pos_y.at(0))*(pos_y.at(4)-pos_y.at(0))+(pos_z.at(4)-pos_z.at(0))*(pos_z.at(4)-pos_z.at(0)))
                + sqrt((pos_x.at(3)-pos_x.at(1))*(pos_x.at(3)-pos_x.at(1))+(pos_y.at(3)-pos_y.at(1))*(pos_y.at(3)-pos_y.at(1))+(pos_z.at(3)-pos_z.at(1))*(pos_z.at(3)-pos_z.at(1)))
                + sqrt((pos_x.at(5)-pos_x.at(2))*(pos_x.at(5)-pos_x.at(2))+(pos_y.at(5)-pos_y.at(2))*(pos_y.at(5)-pos_y.at(2))+(pos_z.at(5)-pos_z.at(2))*(pos_z.at(5)-pos_z.at(2)));
    double dl12 = sqrt((pos_x.at(4)-pos_x.at(0))*(pos_x.at(4)-pos_x.at(0))+(pos_y.at(4)-pos_y.at(0))*(pos_y.at(4)-pos_y.at(0))+(pos_z.at(4)-pos_z.at(0))*(pos_z.at(4)-pos_z.at(0)))
                + sqrt((pos_x.at(5)-pos_x.at(1))*(pos_x.at(5)-pos_x.at(1))+(pos_y.at(5)-pos_y.at(1))*(pos_y.at(5)-pos_y.at(1))+(pos_z.at(5)-pos_z.at(1))*(pos_z.at(5)-pos_z.at(1)))
                + sqrt((pos_x.at(2)-pos_x.at(3))*(pos_x.at(2)-pos_x.at(3))+(pos_y.at(2)-pos_y.at(3))*(pos_y.at(2)-pos_y.at(3))+(pos_z.at(2)-pos_z.at(3))*(pos_z.at(2)-pos_z.at(3)));
    double dl13 = sqrt((pos_x.at(5)-pos_x.at(0))*(pos_x.at(5)-pos_x.at(0))+(pos_y.at(5)-pos_y.at(0))*(pos_y.at(5)-pos_y.at(0))+(pos_z.at(5)-pos_z.at(0))*(pos_z.at(5)-pos_z.at(0)))
                + sqrt((pos_x.at(2)-pos_x.at(1))*(pos_x.at(2)-pos_x.at(1))+(pos_y.at(2)-pos_y.at(1))*(pos_y.at(2)-pos_y.at(1))+(pos_z.at(2)-pos_z.at(1))*(pos_z.at(2)-pos_z.at(1)))
                + sqrt((pos_x.at(4)-pos_x.at(3))*(pos_x.at(4)-pos_x.at(3))+(pos_y.at(4)-pos_y.at(3))*(pos_y.at(4)-pos_y.at(3))+(pos_z.at(4)-pos_z.at(3))*(pos_z.at(4)-pos_z.at(3)));
    double dl14 = sqrt((pos_x.at(5)-pos_x.at(0))*(pos_x.at(5)-pos_x.at(0))+(pos_y.at(5)-pos_y.at(0))*(pos_y.at(5)-pos_y.at(0))+(pos_z.at(5)-pos_z.at(0))*(pos_z.at(5)-pos_z.at(0)))
                + sqrt((pos_x.at(3)-pos_x.at(1))*(pos_x.at(3)-pos_x.at(1))+(pos_y.at(3)-pos_y.at(1))*(pos_y.at(3)-pos_y.at(1))+(pos_z.at(3)-pos_z.at(1))*(pos_z.at(3)-pos_z.at(1)))
                + sqrt((pos_x.at(4)-pos_x.at(2))*(pos_x.at(4)-pos_x.at(2))+(pos_y.at(4)-pos_y.at(2))*(pos_y.at(4)-pos_y.at(2))+(pos_z.at(4)-pos_z.at(2))*(pos_z.at(4)-pos_z.at(2)));
    double dl15 = sqrt((pos_x.at(5)-pos_x.at(0))*(pos_x.at(5)-pos_x.at(0))+(pos_y.at(5)-pos_y.at(0))*(pos_y.at(5)-pos_y.at(0))+(pos_z.at(5)-pos_z.at(0))*(pos_z.at(5)-pos_z.at(0)))
                + sqrt((pos_x.at(4)-pos_x.at(1))*(pos_x.at(4)-pos_x.at(1))+(pos_y.at(4)-pos_y.at(1))*(pos_y.at(4)-pos_y.at(1))+(pos_z.at(4)-pos_z.at(1))*(pos_z.at(4)-pos_z.at(1)))
                + sqrt((pos_x.at(2)-pos_x.at(3))*(pos_x.at(2)-pos_x.at(3))+(pos_y.at(2)-pos_y.at(3))*(pos_y.at(2)-pos_y.at(3))+(pos_z.at(2)-pos_z.at(3))*(pos_z.at(2)-pos_z.at(3)));
    length += (dl1 + dl2 + dl3 + dl4 + dl5 + dl6 + dl7 + dl8 + dl9 + dl10 + dl11 + dl12 + dl13 + dl14 + dl15)/15.;
  }
  return length;
}

template <typename T>
T clamp(const T& val, const T& low, const T& high) {
    return std::min(std::max(val, low), high);
}


template <typename Float>
inline void set_cross_and_velocity(
    std::complex<Float> m00, std::complex<Float> m10,
    std::complex<Float> m11, std::complex<Float> m01,
    std::complex<Float> v00, std::complex<Float> v10,
    std::complex<Float> v11, std::complex<Float> v01,
    Float* du_out, Float& vel_out,
    Float ms2, Float c)
{
    using std::real;
    using std::imag;
    using std::norm;
    using std::conj;
    using std::sqrt;
    using std::abs;

    // Bilinear interpolation coefficients for φ
    auto a = m00;
    auto b = m10 - m00;
    auto c1 = m01 - m00;
    auto d = m11 + m00 - m10 - m01;

    Float a_r = real(a), a_i = imag(a);
    Float b_r = real(b), b_i = imag(b);
    Float c_r = real(c1), c_i = imag(c1);
    Float d_r = real(d), d_i = imag(d);

    Float A = a_i * c_r - a_r * c_i;
    Float B = a_i * d_r + b_i * c_r - a_r * d_i - b_r * c_i;
    Float C = b_i * d_r - b_r * d_i;

    Float u = Float(0.5), v = Float(0.5);
    Float discr = B * B - Float(4) * A * C;

    if (abs(C) > Float(1e-6) && discr >= Float(0)) {
        Float sqrtD = sqrt(discr);
        Float u1 = (-B + sqrtD) / (Float(2) * C);
        Float u2 = (-B - sqrtD) / (Float(2) * C);
        if (u1 >= Float(0) && u1 <= Float(1)) u = u1;
        else if (u2 >= Float(0) && u2 <= Float(1)) u = u2;
    }

    Float denom = c_r + d_r * u;
    if (abs(denom) > Float(1e-6))
        v = -(a_r + b_r * u) / denom;

    u = clamp(u, Float(0), Float(1));
    v = clamp(v, Float(0), Float(1));

    du_out[0] = u;
    du_out[1] = v;

    // Bilinear interpolation
    auto interp = [u, v](std::complex<Float> a, std::complex<Float> b,
                         std::complex<Float> c, std::complex<Float> d) {
        return a * (Float(1) - u) * (Float(1) - v)
             + b * u * (Float(1) - v)
             + d * u * v
             + c * (Float(1) - u) * v;
    };

    std::complex<Float> psi_interp = interp(v00, v10, v01, v11);
    std::complex<Float> phi_interp = interp(m00, m10, m01, m11);

    Float phi2 = norm(phi_interp);
    Float psi2 = norm(psi_interp);
    Float re_mix = real(conj(phi_interp) * psi_interp);

    Float term1 = psi2 / (ms2 * c * c);
    term1 *= (Float(1) + phi2 / (Float(8) * c * c));

    Float term2 = Float(4) * (re_mix * re_mix) / (Float(16) * ms2 * c * c * c * c);

    vel_out = term1 + term2;

}


void mpi_sum_vector_inplace_D(std::vector<double>& local_vec, MPI_Comm comm = MPI_COMM_WORLD) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // In-place reduction: rank 0 sends and receives from the same buffer
    MPI_Allreduce(rank == 0 ? MPI_IN_PLACE : local_vec.data(),
                  local_vec.data(),
                  static_cast<int>(local_vec.size()),
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);
}


template<typename T>
MPI_Datatype mpi_type();

template<> inline MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }
template<> inline MPI_Datatype mpi_type<float>()  { return MPI_FLOAT; }
template<> inline MPI_Datatype mpi_type<int>()    { return MPI_INT; }
template<> inline MPI_Datatype mpi_type<size_t>() {
    static_assert(sizeof(size_t) == sizeof(unsigned long), "Adjust MPI type for size_t");
    return MPI_UNSIGNED_LONG;
}

// template <typename T>
// void mpi_sum_vector_inplace(std::vector<T>& vec, MPI_Comm comm = MPI_COMM_WORLD) {
// 		if (vec.empty()) return;
//
//     int rank;
//     MPI_Comm_rank(comm, &rank);
//
//     MPI_Datatype dtype = mpi_type<T>();
//
//     MPI_Allreduce(rank == 0 ? MPI_IN_PLACE : vec.data(),
//                   vec.data(),
//                   static_cast<int>(vec.size()),
//                   dtype,
//                   MPI_SUM,
//                   comm);
// }

template <typename T>
void mpi_sum_vector_inplace(std::vector<T>& vec, MPI_Comm comm = MPI_COMM_WORLD) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    MPI_Datatype dtype = mpi_type<T>();

    std::vector<T> result(vec.size(), T(0));

    MPI_Allreduce(vec.data(), result.data(), static_cast<int>(vec.size()), dtype, MPI_SUM, comm);

    vec = std::move(result); // overwrite local data
}

// -----------------------------------------------------------------------------
// Class(es) to densify labels reducing equivalences, migrated to labeling tools.
// -----------------------------------------------------------------------------


// class UnionFind {
//     std::unordered_map<unsigned, unsigned> parent;
//
// public:
//     unsigned find(unsigned x) {
//         auto it = parent.find(x);
//         if (it == parent.end())
//             return parent[x] = x;
//         if (it->second != x)
//             it->second = find(it->second);
//         return it->second;
//     }
//
//     void unite(unsigned x, unsigned y) {
//         parent[find(x)] = find(y);
//     }
// };
//
// std::unordered_map<unsigned, unsigned>
// assign_dense_labels(const std::vector<std::pair<unsigned, unsigned>>& equivalences,
//                     const std::vector<unsigned>& object_labels) {
//     std::unordered_set<unsigned> label_set(object_labels.begin(), object_labels.end());
//
//     // Validate equivalence labels
//     for (const auto& [a, b] : equivalences) {
//         if (label_set.find(a) == label_set.end() || label_set.find(b) == label_set.end()) {
//             throw std::invalid_argument(
//                 "Equivalence refers to a label not in object_labels: (" +
//                 std::to_string(a) + ", " + std::to_string(b) + ")"
//             );
//         }
//     }
//
//     UnionFind uf;
//
//     // Unite equivalences
//     for (const auto& [a, b] : equivalences)
//         uf.unite(a, b);
//
//     // Assign dense labels
//     std::unordered_map<unsigned, unsigned> root_to_dense;
//     std::unordered_map<unsigned, unsigned> dense_labels;
//     root_to_dense.reserve(object_labels.size());
//     dense_labels.reserve(object_labels.size());
//
//     unsigned current_label = 1;
//
//     for (unsigned label : object_labels) {
//         unsigned root = uf.find(label);
//         auto [it, inserted] = root_to_dense.emplace(root, current_label);
//         if (inserted) ++current_label;
//         dense_labels[label] = it->second;
//     }
//
//     return dense_labels;
// }






class LabelEquivalence {
private:
    std::unordered_map<size_t, size_t> parent;

    size_t find(size_t x) {
        if (parent.find(x) == parent.end())
            parent[x] = x;
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(size_t a, size_t b) {
        size_t rootA = find(a);
        size_t rootB = find(b);
        size_t newRoot = std::min(rootA, rootB);
        parent[rootA] = parent[rootB] = newRoot;
    }

public:
    void build(const std::vector<std::pair<size_t, size_t>>& pairs) {
        for (const auto& [a, b] : pairs)
            unite(a, b);
    }

    size_t get_min_label(size_t x) {
        return find(x);
    }

    // Map canonical labels to dense 0...N-1
    std::unordered_map<size_t, size_t> get_dense_map(const std::vector<size_t>& labels) {
        std::unordered_set<size_t> unique;
        for (size_t label : labels) {
            unique.insert(get_min_label(label));
        }

        std::vector<size_t> sorted(unique.begin(), unique.end());
        std::sort(sorted.begin(), sorted.end());

        std::unordered_map<size_t, size_t> dense;
        for (size_t i = 0; i < sorted.size(); ++i) {
            dense[sorted[i]] = i;
        }

        return dense;
    }

    // Relabel given labels using dense indices
    std::vector<size_t> relabel_to_dense(const std::vector<size_t>& labels) {
        auto dense_map = get_dense_map(labels);
        std::vector<size_t> result;
        result.reserve(labels.size());

        for (size_t label : labels) {
            size_t canonical = get_min_label(label);
            result.push_back(dense_map[canonical]);
        }

        return result;
    }
};


	/* tools for the segment identifiers */

struct SegRec {
    uint32_t label;      // global dense label
    uint64_t a_key;      // canonical plaquette ID (endpoint A)
    uint64_t b_key;      // canonical plaquette ID (endpoint B)
    float ax, ay, az;    // wrapped endpoint coords
		// TODO REMOVE THOSE!
    float bx, by, bz;    // wrapped endpoint coords //
};


inline uint64_t canonical_plaq_key(uint32_t ix, uint32_t iy, uint32_t iz,
                                   unsigned short flag,
                                   uint32_t Lx, uint32_t Tz)
{
    uint32_t jx = ix, jy = iy, jz = iz;

    switch(flag){
				//wrapping issue
        // case STRCUB_XY2: jz = (iz + 1) % Lz; break;
				case STRCUB_XY2: jz = (iz + 1) % Tz; break;
        case STRCUB_YZ2: jx = (ix + 1) % Lx; break;
        case STRCUB_ZX2: jy = (iy + 1) % Lx; break;
        default: break;
    }

    // pack into 64 bits:     Z  |   Y   |   X   | which-face
    uint32_t face = (flag & (STRCUB_XY|STRCUB_XY2)) ? 0 :
                    (flag & (STRCUB_YZ|STRCUB_YZ2)) ? 1 : 2;

    return (uint64_t(jz) << 42) | (uint64_t(jy) << 21) | (uint64_t(jx) << 2) | face;
}

inline float wrapf(double a, double L){
    a = fmod(a, L);
    return (a < 0 ? a + L : a);
}

// Minimal torus delta in (-L/2, L/2]
static inline double mindelta(double d, double L) {
    d = std::fmod(d + 0.5*L, L);
    if (d < 0.0) d += L;
    return d - 0.5*L;
}

struct V3 {
    double x, y, z;
};

struct Acc {
    double L  = 0.0;   // total "mass" (length)
    double Mx = 0.0;   // first moments (∫ r ds)
    double My = 0.0;
    double Mz = 0.0;

    // Inertia about the origin (∫ (r^2 I - r r^T) ds)
    double Ixx = 0.0, Iyy = 0.0, Izz = 0.0;
    double Ixy = 0.0, Ixz = 0.0, Iyz = 0.0;
};

// Add the contribution of a single polyline segment [a,b].
// We approximate the line integral by lumping its mass (ds) at the midpoint.
// (Good for short segments; your segments are small by construction.)
static inline void accum_seg(Acc& A, const V3& a, const V3& b)
{
    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    const double dz = b.z - a.z;
    const double ds = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (ds == 0.0) return;

    const double mx = 0.5*(a.x + b.x);
    const double my = 0.5*(a.y + b.y);
    const double mz = 0.5*(a.z + b.z);

    A.L  += ds;
    A.Mx += ds * mx;
    A.My += ds * my;
    A.Mz += ds * mz;

    const double r2 = mx*mx + my*my + mz*mz;
    // inertia tensor contribution at the origin: ds * (r^2 I - r r^T)
    A.Ixx += ds * (r2 - mx*mx);
    A.Iyy += ds * (r2 - my*my);
    A.Izz += ds * (r2 - mz*mz);
    A.Ixy -= ds * (mx*my);
    A.Ixz -= ds * (mx*mz);
    A.Iyz -= ds * (my*mz);
}

// Convert accumulators to COM and inertia about COM.
// We have I_origin = I_com + M (r_c^2 I - c c^T)  ⇒
// I_com = I_origin - M (r_c^2 I - c c^T).
static inline void finalize(const Acc& A, V3& com, double I6[6])
{
    if (A.L <= 0.0) {
        com = {0.0,0.0,0.0};
        I6[0]=I6[1]=I6[2]=I6[3]=I6[4]=I6[5]=0.0;
        return;
    }

    const double invM = 1.0 / A.L;
    com.x = A.Mx * invM;
    com.y = A.My * invM;
    com.z = A.Mz * invM;

    const double cx = com.x, cy = com.y, cz = com.z;
    const double r2 = cx*cx + cy*cy + cz*cz;
    const double M  = A.L;

    // Diagonals:
    const double Ixx_com = A.Ixx - M * (r2 - cx*cx);
    const double Iyy_com = A.Iyy - M * (r2 - cy*cy);
    const double Izz_com = A.Izz - M * (r2 - cz*cz);
    // Off-diagonals:
    const double Ixy_com = A.Ixy + M * (cx*cy);
    const double Ixz_com = A.Ixz + M * (cx*cz);
    const double Iyz_com = A.Iyz + M * (cy*cz);

    // Pack as (Ixx,Iyy,Izz,Ixy,IXz,Iyz)
    I6[0] = Ixx_com;
    I6[1] = Iyy_com;
    I6[2] = Izz_com;
    I6[3] = Ixy_com;
    I6[4] = Ixz_com;
    I6[5] = Iyz_com;
}

static inline void inertia_principal_eigs(
    double Ixx, double Iyy, double Izz,
    double Ixy, double Ixz, double Iyz,
    double evals[3])
{
    // Build symmetric matrix A
    const double a11 = Ixx, a22 = Iyy, a33 = Izz;
    const double a12 = Ixy, a13 = Ixz, a23 = Iyz;

    // Invariants
    const double trace = (a11 + a22 + a33);
    const double q = trace / 3.0;

    // A - qI
    const double b11 = a11 - q, b22 = a22 - q, b33 = a33 - q;

    // p^2 = (1/6) * (sum diag^2 + 2*sum off^2)
    const double off2 = a12*a12 + a13*a13 + a23*a23;
    const double diag2 = b11*b11 + b22*b22 + b33*b33;
    const double p2 = (diag2 + 2.0*off2) / 6.0;
    const double p  = std::sqrt(std::max(0.0, p2));

    // If p ~ 0, matrix ≈ scalar * I
    if (p < 1e-30) {
        evals[0] = evals[1] = evals[2] = q;
        return;
    }

    // B = (1/p) * (A - qI); compute r = det(B)/2
    const double c11 = b11 / p, c22 = b22 / p, c33 = b33 / p;
    const double c12 = a12 / p, c13 = a13 / p, c23 = a23 / p;

    // det of symmetric 3x3
    const double detB =
          c11*(c22*c33 - c23*c23)
        - c12*(c12*c33 - c13*c23)
        + c13*(c12*c23 - c13*c22);

    double r = 0.5 * detB;
    // Clamp for safety
    if (r >  1.0) r =  1.0;
    if (r < -1.0) r = -1.0;

    const double phi = std::acos(r) / 3.0;

    // Three eigenvalues
    const double two_p = 2.0 * p;
    evals[0] = q + two_p * std::cos(        phi);
    evals[1] = q + two_p * std::cos( 2.0*M_PI/3.0 + phi);
    evals[2] = q + two_p * std::cos(-2.0*M_PI/3.0 + phi);

    // Sort descending (largest principal moment first)
    if (evals[0] < evals[1]) std::swap(evals[0], evals[1]);
    if (evals[1] < evals[2]) std::swap(evals[1], evals[2]);
    if (evals[0] < evals[1]) std::swap(evals[0], evals[1]);
}


struct GatherBlock {
    void*     ptr;    // rank 0: start of gathered block inside m2Cpu
    uint64_t  count;  // total elements across ranks
};

// Gathers 'local.size()' items into rank 0 at 'dst_bytes' and advances dst_bytes.
// Requires: on rank 0, 'dst_bytes' points into a large-enough buffer (m2Cpu).
template <typename T>
static inline GatherBlock gather_append_to_rank0(const std::vector<T>& local,
                                                 uint8_t*& dst_bytes,
                                                 MPI_Datatype mpi_type)
{
    int rank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    uint64_t localCount = (uint64_t)local.size();
    std::vector<uint64_t> counts(nRanks), displs(nRanks);

    MPI_Allgather(&localCount, 1, MPI_UINT64_T,
                  counts.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);

    displs[0] = 0;
    for (int r = 1; r < nRanks; ++r) displs[r] = displs[r-1] + counts[r-1];
    uint64_t totalCount = displs[nRanks-1] + counts[nRanks-1];

    void* recv_ptr = (rank == 0) ? (void*)dst_bytes : nullptr;

    // Safe cast: MPI_Gatherv counts/disp are int — use a temp int vec
    std::vector<int> counts_i(nRanks), displs_i(nRanks);
    for (int r=0; r<nRanks; ++r) {
        counts_i[r] = (int)counts[r];
        displs_i[r] = (int)displs[r];
    }

    MPI_Gatherv(local.data(), (int)localCount, mpi_type,
                recv_ptr, counts_i.data(), displs_i.data(), mpi_type,
                0, MPI_COMM_WORLD);

    GatherBlock blk{nullptr, totalCount};
    if (rank == 0) {
        blk.ptr = recv_ptr;
        dst_bytes += totalCount * sizeof(T);  // advance byte pointer
    }
    return blk;
}

#endif
