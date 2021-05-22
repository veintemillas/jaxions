#ifndef	__SIMPLOPS__
	#define	__SIMPLOPS__

	#include "enum-field.h"
	#include "scalar/scalarField.h"
	#include <math.h>

	// double anymean(FieldIndex ftipo);
	// void   susum(FieldIndex from, FieldIndex to);
	// void   mulmul(FieldIndex from, FieldIndex to);
	// void   axby(FieldIndex from, FieldIndex to, double a, double b);

	char*	chosechar	(Scalar *field, PadIndex start){
	switch (start){
		case PFIELD_M:
			return static_cast<char*>(field->mCpu());
			break;
		case PFIELD_MS:
			return static_cast<char*>(field->mStart());
			break;
		case PFIELD_V:
			return static_cast<char*>(field->vCpu());
			break;
		case PFIELD_VS:
			return static_cast<char*>(field->vStart());
			break;
		case PFIELD_M2:
			return static_cast<char*>(field->m2Cpu());
			break;
		case PFIELD_M2S:
			return static_cast<char*>(field->m2Start());
			break;
		case PFIELD_M2H:
			return static_cast<char*>(field->m2half());
			break;
		case PFIELD_M2HS:
			return static_cast<char*>(field->m2hStart());
			break;
		}
	}

	void	mulmul(Scalar *axion, PadIndex ftipo1, PadIndex ftipo2)
	{
		/* Multiplies something 1 into something 2*/
		LogMsg(VERB_NORMAL,"[SOP] multiply field%d times field%d ",ftipo2,ftipo1);
		void* meandro  = static_cast<void *> (chosechar(axion,ftipo1));
	  void* meandro2 = static_cast<void *> (chosechar(axion,ftipo2));
	  size_t nData = axion->DataSize()/axion->Precision();

		switch(axion->Precision()){
			case FIELD_DOUBLE:{
				double *m = static_cast<double*>(meandro);
				double *d = static_cast<double*>(meandro2);
				#pragma omp parallel default(shared)
				{
					#pragma omp for schedule(static)
					for (size_t idx = 0; idx < axion->Size()*nData; idx++)
					d[idx] *= m[idx];
				}
			} break;
			case FIELD_SINGLE:{
				float *m = static_cast<float*>(meandro);
				float *d = static_cast<float*>(meandro2);
				#pragma omp parallel
				{
					#pragma omp for schedule(static)
					for (size_t idx = 0; idx < axion->Size()*nData; idx++)
					d[idx] *= m[idx];
				}
			} break;
			}

		return;
	} //end mulmul

	void	axby(Scalar *axion, PadIndex ftipo1, PadIndex ftipo2, double a, double b)
	{
		/* Multiplies something 1 into something 2
	    could be very nicely vectorised */
		LogMsg(VERB_NORMAL,"[SOP] axby (f%d) = (f%d)x%.e + (f%d)x%.e",ftipo2,ftipo2,b,ftipo1,a);
		void* meandro  = static_cast<void *> (chosechar(axion,ftipo1));
	  void* meandro2 = static_cast<void *> (chosechar(axion,ftipo2));
	  size_t nData = axion->DataSize()/axion->Precision();

		switch(axion->Precision()){
			case FIELD_DOUBLE:{
				double *m = static_cast<double*>(meandro);
				double *d = static_cast<double*>(meandro2);
				#pragma omp parallel default(shared)
				{
					#pragma omp for schedule(static)
					for (size_t idx = 0; idx < axion->Size()*nData; idx++)
					d[idx] = b*d[idx] + a*m[idx];
				}
			} break;
			case FIELD_SINGLE:{
				float *m = static_cast<float*>(meandro);
				float *d = static_cast<float*>(meandro2);
				#pragma omp parallel
				{
					#pragma omp for schedule(static)
					for (size_t idx = 0; idx < axion->Size()*nData; idx++)
					d[idx] = ((float) b)*d[idx] + ((float) a)*m[idx];
				}
			} break;
			}

		return;
	} //end mulmul


	void	unMoor(Scalar *axion, PadIndex ftipo1)
	{
		/* Converts Moore theta into standard (unmended) */
		LogMsg(VERB_NORMAL,"[SOP] unMoor (f%d)",ftipo1);
		void* meandro  = static_cast<void *> (chosechar(axion,ftipo1));
	  size_t nData = axion->DataSize()/axion->Precision();

	  double R = *axion->RV();
		switch(axion->Precision()){
			case FIELD_DOUBLE:{
				double *m = static_cast<double*>(meandro);
	      double twoPi = 2.0 * M_PI;
				#pragma omp parallel default(shared)
				{
					#pragma omp for schedule(static)
					for (size_t idx = 0; idx < axion->Size()*nData; idx++)
	          m[idx]= ( m[idx]+M_PI - twoPi * floor((m[idx] + M_PI)/twoPi) - M_PI)*R;
				}
			} break;
			case FIELD_SINGLE:{
				float *m = static_cast<float*>(meandro);
	      float twoPi = 2.0 * M_PI;
				#pragma omp parallel
				{
					#pragma omp for schedule(static)
					for (size_t idx = 0; idx < axion->Size()*nData; idx++)
				     m[idx]= ( m[idx]+M_PI - twoPi * floor((m[idx] + M_PI)/twoPi) - M_PI)*R;
				}
			} break;
			}

		return;
	} //end unMoor



	template <typename cFloat>
	void expandField(Scalar *axion, PadIndex ftipo1)
	{
		size_t nx = axion->rLength();
		size_t nz = axion->rDepth();
		size_t Nx = axion->Length();
		size_t Nz = axion->Depth();

		LogMsg(VERB_NORMAL,"[ex] Expanding field (padtype %d) Nx %d Nz %d nx %d nz %d\n", ftipo1, Nx, Nz, nx, nz);
		if (!axion->Reduced()){
			LogError("Error: Scalar not reduced!"); exit(0);}

		// Float *m = static_cast<Float>(axion->mStart());
		// Float *v = static_cast<Float>(axion->vStart());
		cFloat  *s = static_cast<cFloat*> (static_cast<void*> (chosechar(axion,ftipo1)));
		cFloat  *m2 = static_cast<cFloat*> (axion->m2Start());

		/* migrate to m2 the reduced grid */
		memcpy(&(m2[0]), &(s[0]), nx*nx*nz*axion->DataSize());

		/* We need to exchange ghosts but we are reduced, move last slice to unreduced ghost zone
		not critical: we use only bachghost, which comes from slice 0 */
		memcpy(&(m2[Nx*Nx*(Nz-1)]), &(m2[nx*nx*(nz-1)]), nx*nx*axion->DataSize());
		axion->exchangeGhosts(FIELD_M2);

		/* form a compact reduced field sandwiched between ghosts */
		cFloat *mr = static_cast<cFloat*>(axion->m2Cpu()) + Nx*Nx*axion->getNg() - nx*nx;
		memmove(mr, axion->m2Cpu(), nx*nx*axion->DataSize()); //not critical
		memmove(&(m2[nx*nx*nz]), axion->m2BackGhost(), nx*nx*axion->DataSize()); //relevant

		/* for each point in the reduced grid calculate all points of
		the extended grid in m2 and move
		rx = 0,...,nx-1
		ix = 0,...,N-1
		allow irregular grids 9 from 4, for instance 4/9 =
		n < L? N/N' < n+1
		Lmin = ceil(n*N'/N)
		Lmax = floor((n+1)*N'/N)
		*/
		double r = ((double) nx)/((double) Nx);

		#pragma omp parallel default(shared)
		{
			int thread = omp_get_thread_num ();
			cFloat xyz, xyZ, xYz, Xyz, xYZ, XyZ, XYz, XYZ;
			size_t rpx,rpy, xm, xM, ym, yM, zm, zM;
			double nx_d = (double) nx;
			double nz_d = (double) nz;
			size_t ix0, iy0, iz0, ixM, iyM, izM;
			double dx, dy, dz;
			#pragma omp for schedule(static)
			for (size_t riz = 0; riz < nz; riz++) {
			 for (size_t riy = 0; riy < nx; riy++) {
				for (size_t rix = 0; rix < nx; rix++) {
					rpx = (rix+1) % nx;
					rpy = (riy+1) % nx;
					xyz = m2[rix+nx*(riy+nx*riz)];
					xyZ = m2[rix+nx*(riy+nx*(riz+1))];
					xYz = m2[rix+nx*(rpy+nx*riz)];
					Xyz = m2[rpx+nx*(riy+nx*riz)];
					xYZ = m2[rix+nx*(rpy+nx*(riz+1))];
					XyZ = m2[rpx+nx*(riy+nx*(riz+1))];
					XYz = m2[rpx+nx*(rpy+nx*riz)];
					XYZ = m2[rpx+nx*(rpy+nx*(riz+1))];
					// iz0 = riz*Nz/nz;
					// izM = ((riz+1)*Nz)/nz;
					// iy0 = riy*Nx/nx;
					// iyM = ((riy+1)*Nx)/nx;
					// ix0 = rix*Nx/nx;
					// ixM = ((rix+1)*Nx)/nx;
					iz0 = riz*Nz/nz;
					if (riz*Nz % nz) ix0++;
					iy0 = riy*Nx/nx;
					if (riy*Nx % nx) iy0++;
					ix0 = rix*Nx/nx;
					if (rix*Nx % nx) ix0++;
					izM = ((riz+1)*Nz)/nz;
					if ((riz+1)*Nz % nz ==0) izM--;
					iyM = ((riy+1)*Nx)/nx;
					if ((riy+1)*Nx % nx ==0) iyM--;
					ixM = ((rix+1)*Nx)/nx;
					if ((rix+1)*Nz % nx ==0) ixM--;
	// if (thread==0)
	// 	LogOut("rz(%d) %d->%d ry(%d) %d->%d rx(%d) %d->%d\n", riz, iz0, izM, riy, iy0, iyM, rix, ix0, ixM);
					for (size_t iz = iz0; iz <= izM; iz++) {
						for (size_t iy = iy0; iy <= iyM; iy++) {
							for (size_t ix = ix0; ix <= ixM; ix++) {
	// if (thread==0)
	// 	LogOut("%d %d %d, %d %d, %d %d %d \n", riz,riy,rix,rpx,rpy, iz, iy, ix);
								dx = r*ix - ((double) rix);
								dy = r*iy - ((double) riy);
								dz = r*iz - ((double) riz);
	// if (thread==0	)
	// 	LogOut("dx %f dy %f dz %f \n", dx,dy,dz);
								s[ix+Nx*(iy+Nx*iz)] =
										xyz*((cFloat) ((1.-dx)*(1.-dy)*(1.-dz))) +
										Xyz*((cFloat) (     dx*(1.-dy)*(1.-dz))) +
										xYz*((cFloat) ((1.-dx)*    dy *(1.-dz))) +
										XYz*((cFloat) (dx     *    dy *(1.-dz))) +
									  xyZ*((cFloat) ((1.-dx)*(1.-dy)*    dz )) +
										XyZ*((cFloat) (dx     *(1.-dy)*    dz )) +
										xYZ*((cFloat) ((1.-dx)*    dy *    dz )) +
										XYZ*((cFloat) (dx     *    dy *    dz ));
									}
								}
							} //end 3 fors
				}
			 }
		 } //end reduced volume loop
		} //end parallel region
		commSync();
		return;
	}

	void	expandField(Scalar *axion)
	{
		/* Expands Nx, Nx, Nz into size(), size(), depth()
		by linear extrapolation */
		LogMsg(VERB_NORMAL,"[ex] Expanding field");
		switch(axion->Field())
		{
			case FIELD_SAXION:
				if (axion->Precision() == FIELD_SINGLE) {
					expandField<std::complex<float>>(axion, PFIELD_MS);
					expandField<std::complex<float>>(axion, PFIELD_V); }
				else {
					expandField<std::complex<double>>(axion, PFIELD_MS);
					expandField<std::complex<double>>(axion, PFIELD_V); }
			break;

			case FIELD_AXION:
			if (axion->Precision() == FIELD_SINGLE) {
				expandField<float>(axion, PFIELD_MS);
				expandField<float>(axion, PFIELD_V); }
			else {
				expandField<double>(axion, PFIELD_MS);
				expandField<double>(axion, PFIELD_V); }
			break;
		}
		return;
	}


#endif
