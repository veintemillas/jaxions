#ifndef	__SIMPLOPS__
	#define	__SIMPLOPS__

	#include "enum-field.h"
	#include "scalar/scalarField.h"
	#include <math.h>

	using namespace std;
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
	void expandFieldGeneral(Scalar *axion, PadIndex ftipo1)
	{
		size_t nx = axion->rNX();
		size_t ny = axion->rNY();
		size_t nz = axion->rNZ();
		size_t Nx = axion->NX();
		size_t Ny = axion->NY();
		size_t Nz = axion->NZ();

		LogMsg(VERB_NORMAL,
		       "[ex] Expanding field (padtype %d) Nx %zu Ny %zu Nz %zu nx %zu ny %zu nz %zu\n",
		       ftipo1, Nx, Ny, Nz, nx, ny, nz);
		LogFlush();

		if (!axion->Reduced()) {
			LogError("Error: Scalar not reduced!");
			exit(0);
		}

		if (nx == 0 || ny == 0 || nz == 0 || Nx == 0 || Ny == 0 || Nz == 0) {
			LogError("Error: zero-sized dimension in expandFieldGeneral!");
			exit(0);
		}

		cFloat *s  = static_cast<cFloat*>(static_cast<void*>(chosechar(axion, ftipo1)));
		cFloat *m2 = static_cast<cFloat*>(axion->m2Start());

		const size_t srcPlane = nx * ny;
		const size_t dstPlane = Nx * Ny;

		// Copy reduced data into m2
		memcpy(&(m2[0]), &(s[0]), srcPlane * nz * axion->DataSize());

		// Put last physical slice into the unreduced ghost zone if needed for exchange
		memcpy(&(m2[dstPlane * (Nz - 1)]),
		       &(m2[srcPlane * (nz - 1)]),
		       srcPlane * axion->DataSize());

		bool wasGPU = false;
		if (axion->Device() == DEV_GPU) {
			wasGPU = true;
			axion->setDev(DEV_CPU);
		}

		axion->exchangeGhosts(FIELD_M2);

		// Compact reduced field + back ghost
		cFloat *mr = static_cast<cFloat*>(axion->m2Cpu()) + dstPlane * axion->getNg() - srcPlane;
		memmove(mr, axion->m2Cpu(), srcPlane * axion->DataSize());
		memmove(&(m2[srcPlane * nz]), axion->m2BackGhost(), srcPlane * axion->DataSize());

		auto idxSrc = [nx, ny](size_t x, size_t y, size_t z) {
			return x + nx * (y + ny * z);
		};

		auto idxDst = [Nx, Ny](size_t x, size_t y, size_t z) {
			return x + Nx * (y + Ny * z);
		};

		#pragma omp parallel default(shared)
		{
			cFloat xyz, xyZ, xYz, Xyz, xYZ, XyZ, XYz, XYZ;
			size_t rpx, rpy;
			size_t ix0, iy0, iz0, ixM, iyM, izM;
			double dx, dy, dz;

			#pragma omp for schedule(static)
			for (size_t riz = 0; riz < nz; riz++) {

				if (nz == 1) {
					iz0 = 0;
					izM = Nz - 1;
				} else {
					iz0 = (riz * Nz) / nz;
					izM = ((riz + 1) * Nz) / nz;
					if (((riz + 1) * Nz) % nz == 0) izM--;
				}

				for (size_t riy = 0; riy < ny; riy++) {

					if (ny == 1) {
						iy0 = 0;
						iyM = Ny - 1;
					} else {
						iy0 = (riy * Ny) / ny;
						iyM = ((riy + 1) * Ny) / ny;
						if (((riy + 1) * Ny) % ny == 0) iyM--;
					}

					for (size_t rix = 0; rix < nx; rix++) {

						if (nx == 1) {
							ix0 = 0;
							ixM = Nx - 1;
						} else {
							ix0 = (rix * Nx) / nx;
							ixM = ((rix + 1) * Nx) / nx;
							if (((rix + 1) * Nx) % nx == 0) ixM--;
						}

						rpx = (nx == 1) ? 0 : (rix + 1) % nx;
						rpy = (ny == 1) ? 0 : (riy + 1) % ny;

						// z+1 is always valid because the extra slice was appended
						xyz = m2[idxSrc(rix, riy, riz)];
						Xyz = m2[idxSrc(rpx, riy, riz)];
						xYz = m2[idxSrc(rix, rpy, riz)];
						XYz = m2[idxSrc(rpx, rpy, riz)];

						xyZ = m2[idxSrc(rix, riy, riz + 1)];
						XyZ = m2[idxSrc(rpx, riy, riz + 1)];
						xYZ = m2[idxSrc(rix, rpy, riz + 1)];
						XYZ = m2[idxSrc(rpx, rpy, riz + 1)];

						for (size_t iz = iz0; iz <= izM; iz++) {
							if (nz == 1) dz = 0.0;
							else          dz = (static_cast<double>(iz) * static_cast<double>(nz)) / static_cast<double>(Nz)
							                 - static_cast<double>(riz);

							for (size_t iy = iy0; iy <= iyM; iy++) {
								if (ny == 1) dy = 0.0;
								else          dy = (static_cast<double>(iy) * static_cast<double>(ny)) / static_cast<double>(Ny)
								                 - static_cast<double>(riy);

								for (size_t ix = ix0; ix <= ixM; ix++) {
									if (nx == 1) dx = 0.0;
									else          dx = (static_cast<double>(ix) * static_cast<double>(nx)) / static_cast<double>(Nx)
									                 - static_cast<double>(rix);

									s[idxDst(ix, iy, iz)] =
										  xyz * cFloat((1.0 - dx) * (1.0 - dy) * (1.0 - dz))
										+ Xyz * cFloat(       dx  * (1.0 - dy) * (1.0 - dz))
										+ xYz * cFloat((1.0 - dx) *        dy  * (1.0 - dz))
										+ XYz * cFloat(       dx  *        dy  * (1.0 - dz))
										+ xyZ * cFloat((1.0 - dx) * (1.0 - dy) *        dz )
										+ XyZ * cFloat(       dx  * (1.0 - dy) *        dz )
										+ xYZ * cFloat((1.0 - dx) *        dy  *        dz )
										+ XYZ * cFloat(       dx  *        dy  *        dz );
								}
							}
						}
					}
				}
			}
		}

		commSync();

		if (wasGPU)
			axion->setDev(DEV_GPU);
	}

	template <typename cFloat>
	void expandField(Scalar *axion, PadIndex ftipo1)
	{
		size_t nx = axion->rNX();
		size_t ny = axion->rNY();
		size_t nz = axion->rNZ();
		size_t Nx = axion->NX();
		size_t Ny = axion->NY();
		size_t Nz = axion->NZ();

		LogMsg(VERB_NORMAL,"[ex] Expanding field (padtype %d) Nx %d Nz %d nx %d nz %d\n", ftipo1, Nx, Nz, nx, nz);
		if (!axion->Reduced()){
			LogError("Error: Scalar not reduced!"); exit(0);}

		cFloat  *s = static_cast<cFloat*> (static_cast<void*> (chosechar(axion,ftipo1)));
		cFloat  *m2 = static_cast<cFloat*> (axion->m2Start());

		/* migrate to m2 the reduced grid */
		memcpy(&(m2[0]), &(s[0]), nx*nx*nz*axion->DataSize());

		/* We need to exchange ghosts but we are reduced, move last slice to unreduced ghost zone
		not critical: we use only bachghost, which comes from slice 0 */
		memcpy(&(m2[Nx*Nx*(Nz-1)]), &(m2[nx*nx*(nz-1)]), nx*nx*axion->DataSize());

		bool wasGPU = false;
		if (axion->Device() == DEV_GPU){
			wasGPU = true;
			axion->setDev(DEV_CPU); // otherwise GPU data will overwrite CPU
		}
		axion->exchangeGhosts(FIELD_M2);



		/* form a compact reduced field sandwiched between ghosts */
		cFloat *mr = static_cast<cFloat*>(axion->m2Cpu()) + Nx*Ny*axion->getNg() - nx*ny;
		memmove(mr, axion->m2Cpu(), nx*ny*axion->DataSize()); //not critical
		memmove(&(m2[nx*ny*nz]), axion->m2BackGhost(), nx*ny*axion->DataSize()); //relevant


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
			 iz0 = (riz*Nz)/nz;
			 izM = ((riz+1)*Nz)/nz;
			 if ((riz+1)*Nz % nz ==0) izM--;

			 for (size_t riy = 0; riy < nx; riy++) {
				iy0 = (riy*Nx)/nx;
				iyM = ((riy+1)*Nx)/nx;
                                if ((riy+1)*Nx % nx ==0) iyM--;
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
					//iz0 = (riz*Nz)/nz;
					// if (riz*Nz % nz) iz0++;
					//iy0 = (riy*Nx)/nx;
					// if (riy*Nx % nx) iy0++;
					ix0 = (rix*Nx)/nx;
					// if (rix*Nx % nx) ix0++;
					//izM = ((riz+1)*Nz)/nz;
					//if ((riz+1)*Nz % nz ==0) izM--;
					//iyM = ((riy+1)*Nx)/nx;
					//if ((riy+1)*Nx % nx ==0) iyM--;
					ixM = ((rix+1)*Nx)/nx;
					if ((rix+1)*Nz % nx ==0) ixM--;


					for (size_t iz = iz0; iz <= izM; iz++) {
						for (size_t iy = iy0; iy <= iyM; iy++) {
							for (size_t ix = ix0; ix <= ixM; ix++) {


								dx = r*ix - ((double) rix);
								dy = r*iy - ((double) riy);
								dz = r*iz - ((double) riz);


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

		if (wasGPU)
			axion->setDev(DEV_GPU);

		return;
	}

	void	expandField(Scalar *axion)
	{
		/* Expands Nx, Nx, Nz into size(), size(), depth()
		by linear extrapolation */
		LogMsg(VERB_NORMAL,"[ex] Expanding field");LogFlush();
		switch(axion->Field())
		{
			case FIELD_SAXION:
				if (axion->Precision() == FIELD_SINGLE) {
					expandFieldGeneral<std::complex<float>>(axion, PFIELD_MS);
					expandFieldGeneral<std::complex<float>>(axion, PFIELD_V); }
				else {
					expandFieldGeneral<std::complex<double>>(axion, PFIELD_MS);
					expandFieldGeneral<std::complex<double>>(axion, PFIELD_V); }
			break;

			case FIELD_AXION:
			if (axion->Precision() == FIELD_SINGLE) {
				expandFieldGeneral<float>(axion, PFIELD_MS);
				expandFieldGeneral<float>(axion, PFIELD_V); }
			else {
				expandFieldGeneral<double>(axion, PFIELD_MS);
				expandFieldGeneral<double>(axion, PFIELD_V); }
			break;
		}
		return;
	}

	/* Converts a conformal theta configuration into conformal saxion */
	void	ctheta2csaxion(Scalar *axion)
	{
	/* ctheta = theta R
	  cvtheta = theta' R + theta R'
		cphi    = R exp(i theta) = R exp(i ctheta/R)
		cvphi   = R' exp(i theta) + R i theta' exp(i theta)
		        = (R'/R  + i theta') cphi
						= (R'/R  + i (cvtheta/R - theta R'/R)) cphi
						= (R'/R (1-i ctheta/R) + i cvtheta/R ) cphi
						*/

		LogMsg(VERB_NORMAL,"[SOP] c_theta to c_saxion");
		memmove(axion->m2Cpu(),axion->mStart(), axion->Size()*axion->Precision());
		memmove(axion->m2half(),axion->vStart(), axion->Size()*axion->Precision());



		switch(axion->Precision()){
			case FIELD_DOUBLE:{
				double R = *axion->RV();
				double Rp = axion->BckGnd()->Rp(*axion->zV());
				complex<double> II = complex<double>{0,1};

				double *m = static_cast<double*>(axion->m2Cpu());
				double *v = static_cast<double*>(axion->m2half());
				complex <double> *mc = static_cast<complex<double>*>(axion->mCpu())+axion->getNg()*axion->Surf();
				complex <double> *vc = static_cast<complex<double>*>(axion->vCpu());

				#pragma omp parallel
				{
					#pragma omp for schedule(static)
					for (size_t idx = 0; idx < axion->Size(); idx++){
						double theta = m[idx]/R;
						mc[idx] = R * std::exp(II*theta);
						vc[idx] = (Rp * (1.0 - II*theta) + II * v[idx]/R )* mc[idx];
					}
				}
			} break;
			case FIELD_SINGLE:{
				float R = *axion->RV();
				float Rp = axion->BckGnd()->Rp(*axion->zV());
				complex<float> II = complex<float>{0,1};

				float *m = static_cast<float*>(axion->m2Cpu());
				float *v = static_cast<float*>(axion->m2half());
				complex <float> *mc = static_cast<complex<float>*>(axion->mCpu())+axion->getNg()*axion->Surf();
				complex <float> *vc = static_cast<complex<float>*>(axion->vCpu());

				#pragma omp parallel
				{
					#pragma omp for schedule(static)
					for (size_t idx = 0; idx < axion->Size(); idx++){
						float theta = m[idx]/R;
						mc[idx] = R * std::exp(II*theta);
						vc[idx] = (Rp * (1.f - II*theta) + II * v[idx]/R )* mc[idx];
					}
				}
			} break;
			}

		axion->setField(FIELD_SAXION);
		return;
	} //end ctheta2csaxion





#endif
