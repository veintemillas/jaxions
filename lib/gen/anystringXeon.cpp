#include <complex>
#include <random>
#include <omp.h>

#include "scalar/scalarField.h"
#include "enum-field.h"
#include "utils/memAlloc.h"
#include "utils/parse.h"

#include "enum-field.h"
#include "comms/comms.h"


/* computes
  (vs -rc) / (rc^2v sqrt[ (vs-rc)^2+(rs)^2]) [s=1-s=0]
  from xs0 xs1, x
  from the Boit-Savart integral of a straight-current section
  r(s)=x(s)-x= xs0-x + (xs1-xs0)ds = r0 + vds , ds in (0,1)
  v = xs1-xs0
  rc = (xs0-x).v/|v|
  rs^2 = r^2-rc^2

  by construction we will avoid |v| = 0 by avoiding equal points.


  */
double mora(double* xs,double* ys,double* zs, int lo, double x, double y, double z, double *out)
{
	return 0;
}

template<typename Float>
void	anystringXeon (std::complex<Float> * __restrict__ m, Scalar *field, IcData ic, double *xs, double *ys, double *zs, size_t len)
{
	LogMsg(VERB_NORMAL,"[rX] Any string configuration");

	const size_t Lx = field->Length();
	const size_t Sf = field->Surf();
	const size_t V  = field->Size();
	const size_t nz = field->Depth();
	const double L  = field->BckGnd()->PhysSize();


	Float* th = static_cast<Float*> (field->vCpu());
	Float* Bx = static_cast<Float*> (field->m2Cpu());
	Float* By = static_cast<Float*> (field->m2half()) ;
	Float* Bz = static_cast<Float*> (field->m2half()) + 2*Sf;

	/*	For MPI		*/
	const int nSplit  = commSize();
	const int rank    = commRank();
	//const int fwdNeig = (rank + 1) % nSplit;
	const int bckNeig = (rank - 1 + nSplit) % nSplit;

	size_t Lz = Lx/commSize();
	size_t Tz = field->TotalDepth();
	size_t local_z_start = rank*Lz;

	LogMsg(VERB_NORMAL,"[asX] Loop");

	#pragma omp parallel for default(shared)
	for (size_t idx=0; idx<V; idx++)
	{
		size_t ix, iy, iz, iz_l;
		Float   x,  y,  z, dx, dy, dz;
		Float   rx,  ry,  rz, r3;
		Float   bx = 0., by = 0., bz = 0.;
		iz_l = idx/Sf;
	        iz   = iz_l + local_z_start;
		iy = (idx%Sf)/Lx ;
		ix = (idx%Sf)%Lx ;
		z = iz;
		y = iy;
		x = ix;
		// Biot-Savart
		for (int il = 0; il < len-1; il++)
		{
			/* linear approx is very good far away from strings */
			dz = zs[il+1]-zs[il];
			dy = ys[il+1]-ys[il];
			rx = (xs[il+1]+xs[il])*0.5 - (x+0.5) ; // links live between lattice sites
			ry = (ys[il+1]+ys[il])*0.5 - y ;
			rz = (zs[il+1]+zs[il])*0.5 - z ;
			r3 = pow(rx*rx+ry*ry+rz*rz,1.5);
			bx += (ry*dz-rz*dy)/r3;
			if (ix==0)
			{
				dx = xs[il+1]-xs[il];
				rx += 0.5;
				ry -= 0.5;
				r3 = pow(rx*rx+ry*ry+rz*rz,1.5);
				by += (-rx*dz+rz*dx)/r3;
				if (iy==0)
				{
					ry += 0.5;
					rz -= 0.5;
					bz += (rx*dy-ry*dx)/r3;
				}
			}

			/* integral version is much better close to strings, it can be regularised too */

		} // end x string loop
		Bx[idx] = 0.5*bx;
		if(ix==0){
			By[iy + Lx*iz_l] = 0.5*by;
			if (iy==0){
				Bz[iz_l] = 0.5*bz;
				// printf("bz[%d] %f \n",iz_l,bz);
			}}
		//printf("x y z %d %d %d bx %f\n",ix,iy,iz,bx);


/*
		// If x = 0 we need to calculate also By and Bz
		if (ix == 0)
		{
			for (int il = 0; il < len; il++)
			{
				dz = zs[il+1]-zs[il];
				dx = xs[il+1]-xs[il];
				rx = (xs[il+1]+xs[il])*0.5 - x ;
				ry = (ys[il+1]+ys[il])*0.5 - (y+0.5) ; // links live between lattice sites
				rz = (zs[il+1]+zs[il])*0.5 - z ;
				by += (-rx*dz+rz*dx)/pow(rx*rx+ry*ry+rz*rz,1.5);
			} // end y string loop
			By[iy + Lx*iz_l] = 0.5*by;
			printf("x y z %d %d %d by %f\n",ix,iy,iz,by);

			// If x = 0 we need to calculate also Bz
			if (iy == 0)
			{
				for (int il = 0; il < len; il++)
				{
					dy = ys[il+1]-ys[il];
					dx = xs[il+1]-xs[il];
					rx = (xs[il+1]+xs[il])*0.5 - x ;
					ry = (ys[il+1]+ys[il])*0.5 - y ; // links live between lattice sites
					rz = (zs[il+1]+zs[il])*0.5 - (z+0.5) ;
					bz += (rx*dy-ry*dx)/pow(rx*rx+ry*ry+rz*rz,1.5);
				} // end y string loop
				Bz[iz_l] = 0.5*bz;
				printf("x y z %d %d %d bz %f\n",ix,iy,iz,bz);
			}

		}
*/
	} // end Volume parallel for


	LogMsg(VERB_NORMAL,"[asX] Building theta(0,0,z) ...");
	// Now we are going to fill the whole volume by summing up gradients
	// (in units of lattice spacing for simplicity)
	// first we need to do the x,y=0 line across ranks
 	// only 1 rank at work

	/* initial condition can be changed later */
	if (rank == 0)
		th[0] == 0.;

	// Build theta(0,0,z)
	for (int cRank = 0; cRank < commSize(); cRank++) {

		const int cBckNeig = (cRank - 1 + nSplit) % nSplit;

		// calculate the line
		if (rank == cRank)
			for (size_t idx=Sf,iz=0; iz<nz; iz++,idx+=Sf)
			{
				th[idx] = th[idx-Sf] + Bz[iz];
printf("[asX] bz[%d] %f thz[%d] %f\n",idx,Bz[iz],iz,th[idx]);
			}

		// send the last value of theta
		if (commSize() == 1) {
			th[0] = 0.; //th[Sf*nz];
		} else {
			if (rank == cBckNeig) {
				// We can overflow V in vCpu cause it has two extra ghost slices (and is complex)
				MPI_Send(&th[Sf*(nz)], sizeof(Float), MPI_CHAR, cRank, cRank, MPI_COMM_WORLD);
				LogMsg(VERB_NORMAL,"[asX] rank %d send %f for local_z=-1",cBckNeig,th[Sf*(nz)]);
				// printf("[asX] rank %d send %f for local_z=-1",cBckNeig,th[Sf*(nz)]);
			}

			if (rank == cRank) {
				MPI_Recv(&th[0]  , sizeof(Float), MPI_CHAR, bckNeig, cRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				LogMsg(VERB_NORMAL,"[asX] rank %d received %f for local_z=-1",cRank,th[0]);
				// printf("[asX] rank %d received %f for local_z=-1",cRank,th[0]);
			}
		}
	}

	LogMsg(VERB_NORMAL,"[asX] theta(0,0,z) built");

	// Build theta(0,y,z)

	LogMsg(VERB_NORMAL,"[asX] Building theta(0,y,z) ...");


	#pragma omp parallel for default(shared) schedule(static)
	for (size_t iz=0; iz<nz; iz++)
	{
		for (size_t iy=1; iy<Lx; iy++)
			th[iz*Sf+iy*Lx] = th[iz*Sf+(iy-1)*Lx] + By[iy-1 + Lx*iz];
	}

	LogMsg(VERB_NORMAL,"[asX] theta(0,y,z) built");

	// Build theta(0,y,z)

	LogMsg(VERB_NORMAL,"[asX] Building theta(x,y,z) ...");

	#pragma omp parallel for default(shared) schedule(static)
	for (size_t iz=0; iz<nz; iz++)
		for (size_t iy=0; iy<Lx; iy++)
		{
			size_t os = iz*Sf+iy*Lx;
			for (size_t ix=1; ix<Lx; ix++){
				th[os+ix] = th[os+ix-1] + Bx[os+ix-1];
				// printf(">>> x y z %d %d %d the %f\n",ix, iy, iz, th[os+ix]);
				}
		}

	LogMsg(VERB_NORMAL,"[asX] theta(x,y,z) built!");

	LogMsg(VERB_NORMAL,"[asX] Create complex scalar !");

	std::complex<Float>* mc = static_cast<std::complex<Float>*> (field->mStart());
	std::complex<Float>* vc = static_cast<std::complex<Float>*> (field->vStart());

	#pragma omp parallel for default(shared)
	for (size_t idx=0; idx<V; idx++)
	{
		mc[idx] = exp(std::complex<Float>(0,th[idx]));
	}

	#pragma omp parallel for default(shared)
	for (size_t idx=0; idx<V; idx++)
	{
		vc[idx] = std::complex<Float>(0,0);
	}

}

void	anystringConf (Scalar *field, IcData ic, double *x, double *y, double *z, size_t len)
{
	switch (field->Precision())
	{
		case FIELD_DOUBLE:
		{
		std::complex<double>* ma;
		if (ic.fieldindex == FIELD_M){
		 	ma = static_cast<std::complex<double>*> (field->mStart());
			LogMsg(VERB_NORMAL,"[RC] Generating double conf in mS! ");
		}
		else if (ic.fieldindex == FIELD_V){
			ma = static_cast<std::complex<double>*> (field->vCpu());
			LogMsg(VERB_NORMAL,"[RC] Generating double conf in v! ");
		}
		else if (ic.fieldindex == FIELD_M2){
			ma = static_cast<std::complex<double>*> (field->m2Cpu());
			LogMsg(VERB_NORMAL,"[RC] Generating double conf in m2! ");
		}

		anystringXeon<double>(ma, field, ic, x,y,z,len);
		}
		break;

		case FIELD_SINGLE:
		{
		std::complex<float>* ma;
		if (ic.fieldindex == FIELD_M){
			ma = static_cast<std::complex<float>*> (field->mStart());
			LogMsg(VERB_NORMAL,"[RC] Generating single conf in mS! ");
		}
		else if (ic.fieldindex == FIELD_V){
			ma = static_cast<std::complex<float>*> (field->vCpu());
			LogMsg(VERB_NORMAL,"[RC] Generating single conf in v! type %d",ic.smvarType);
		}
		else if (ic.fieldindex == FIELD_M2){
			ma = static_cast<std::complex<float>*> (field->m2Cpu());
			LogMsg(VERB_NORMAL,"[RC] Generating single conf in m2! ");
		}

		anystringXeon<float> (ma, field, ic, x,y,z,len);
		}
		break;

		default:
		break;
	}
}
