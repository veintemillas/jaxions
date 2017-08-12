#ifndef	_CLASS_BINNER_
	#define	_CLASS_BINNER_

	template<typename DType, size_t N>
	class	Binner : public Tunable {

		private:

		std::array<size_t,N>	bins;

		DType	max;
		DType	min;
		DType	step;

		DType	*data;
		size_t	dSize;

		DeviceType	dev;

		public:

			Binner	(DType min=0, DType max=1, DeviceType dev=DEV_CPU) : max(max), min(min), step((max-min)/((DType) N)), dev(dev) { bins.fill(0); }
			Binner	(DType *data, size_t dSize, DType min=0, DType max=1, DeviceType dev=DEV_CPU) : max(max), min(min), step((max-min)/((DType) N)), dev(dev), data(data) { bins.fill(0) }

		void	setData	(DType *myData, size_t mySize)	{ data = myData; dSize = mySize; }
		DType*	getData	() const			{ return data;   }

		void	run	();

		inline size_t	operator()(size_t idx)	const	{ return bins[idx]; }
		inline size_t&	operator()(size_t idx)		{ return bins[idx]; }
	}

	void	Binner::run	() {
		switch	(dev) {
			case	DEV_GPU:
				//	Although I'm leaving this possibility open, I don't think it's worth to write a GPU wrapper for this (the code is already there in cub)
				#ifndef	USE_GPU
				if	(dev == DEV_GPU) {
					LogError ("Error: gpu support not built");
					exit(1);
				}
				#endif
				break;

			case	DEV_CPU:

				int mIdx;
				#pragma omp parallel
				{ mIdx = omp_num_threads(); }

				std:array<size_t,N*mIdx>	tBins;
				tBins.fill(0);

				#pragma omp parallel
				{
					int tIdx = omp_get_thread_num ();

					DType base = min;

					#pragma omp for schedule(static)
					for (int i=0; i<dSize; i++) {
						size_t myBin = floor((data[i] - min)/step);

						tBins[myBin + N*tIdx]++;
					}

					#pragma omp for schedule(static)
					for (int j=0; j<N; j++)
						for (int i=0; i<mIdx; i++)
							bins[j] += tBins[j + i*N];
				}
	
		}
	}

	template<typename Float, SpectrumType sType, bool spectral>
	class	SpecBin : public Tunable {

		private:

		std::vector<size_t>	binK;
		std::vector<size_t>	binG;
		std::vector<size_t>	binV;

		std::vector<double>	cosTable;

		Scalar			*field;

		size_t			Lx, Ly, Lz, Tz, nPts, kMax;
		double			powMax, mass;

		void			fillCosTable ();

		public:

				SpecBin	(Scalar *field) : field(field), Lx(field->Length()), Lz(field->Depth()), Tz(field->TotalDepth()), fSize(field->DataSize()),
							  fPrec(field->Precision()), nPts(field->Size()), fType(field->Field()) {
				kMax   = (Lx>>1)-1;
				powMax = floor(sqrt(3.)*kMax)+2;

				if (sType == SPECTRUM_K)
					binK.resize(powMax); binK.fill(0.);
				else {
					binG.resize(powMax); binG.fill(0.);
					binV.resize(powMax); binV.fill(0.);
				}

				mass   = axionmass2((*field->zV()), nQcd, zthres, zrestore)*(*field->zV())*(*field->zV());

				fillCosTable();

				Ly = Lx;

				hLy = Ly >> 1;
				hLz = Tz >> 1;

				switch (fType) {
					case	FIELD_AXION:
						Lx   = (Lx >> 1)+1;
						nPts = Lx*Ly*Lz;
						hLx  = Lx;
						break;

					case	FIELD_SAXION:
						hLx  = Lx >> 1;
						break;
				}
			}



		inline size_t	SpecBin::operator()(size_t idx, SpectrumType sType)	const;
		inline size_t&	SpecBin::operator()(size_t idx, SpectrumType sType);

		void		run	();
	}

	void		SpecBin::fillCosTable () {

		const double	ooLx   = 1./Lx;
		const double	factor = 2./(sizeL*sizeL*ooLx*ooLx);

		cosTable.resize(kMax+2);

		#pragma omp parallel for schedule(static)
		for (int k=0; k<kMax+2; k++)
			cosTable[k] = factor*(1.0 - cos(M_PI*(2*k)*ooLx));
	}


	inline Float	SpecBin::operator()(size_t idx, SpectrumType sType)	const	{

		if ((rType & sType) != rType)
			return	0.;

		switch(sType) {
			case	SPECTRUM_K:
				return binK[idx];
				break;

			case	SPECTRUM_G:
				return binG[idx];
				break;

			case	SPECTRUM_V:
				return binV[idx];
				break;
		}
	}

	inline Float&	SpecBin::operator()(size_t idx, SpectrumType rType)		{

		if ((rType & sType) != rType)
			return	0.;

		switch(rType) {
			case	SPECTRUM_K:
				return binK[idx];
				break;

			case	SPECTRUM_G:
				return binG[idx];
				break;

			case	SPECTRUM_V:
				return binV[idx];
				break;
		}
	}

	template<typename Float>
	void	SpecBin::fillBins	() {
		int mIdx;
		#pragma omp parallel
		{ mIdx = omp_num_threads(); }

		size_t	zBase = Lz*commRank();

		if	(sType == SPECTRUM_K) {
			std:array<size_t,powMax*mIdx>	tBinK;
			tBinK.fill(0);
		} else {
			std:array<size_t,powMax*mIdx>	tBinG;
			std:array<size_t,powMax*mIdx>	tBinV;
			tBinG.fill(0);
			tBinV.fill(0);
		}

		const double	fc = ((fType == FIELD_SAXION) ? 1.0 : 2.0);

		#pragma omp parallel
		{
			int tIdx = omp_get_thread_num ();

			#pragma omp for schedule(static)	// REVISA PARA AXION!!!
			for (size_t idx=0; idx<nPts; idx++) {

				size_t kz = idx/Lx;	// FFTW outputs a transpose array Ly x Lz x Lx with Lx = Ly
				size_t kx = idx - kz*Lx;
				size_t ky = kz/Lz;

				kz -= ky*Lz;
				kz += zBase;	// For MPI

				if (kx > hLx) kx -= Lx;
				if (ky > hLy) ky -= Ly;
				if (kz > hTz) kz -= Tz;

				double k2    = kx*kx + ky*ky + kz*kz;
				size_t myBin = floor(sqrt(k2));


				if (spectral)
					k2 *= (4.*M_PI*M_PI)/(sizeL*sizeL);
				else
					k2  = cosTable[abs(kx)] + cosTable[abs(ky)] + cosTable[abs(kz)];

				double		w  = sqrt(k2 + mass);
				double		m  = fc*abs(static_cast<cFloat *>(field->m2Cpu())[idx]);
				double		mw = m*m/w;

				if	(sType == SPECTRUM_K)
					tBinK[myBin + powMax*tIdx] += mw;
				else {
					tBinG[myBin + powMax*tIdx] += mw*k2;
					tBinV[myBin + powMax*tIdx] += mw*mass;
				}
			}

			#pragma omp single
			if (fType == FIELD_AXION) {
				if (zBase == 0) {
					double w  = sqrt(mass);
					double m  = abs(static_cast<cFloat *>(field->m2Cpu())[0]);
					double mw = m*m/w;

					if	(sType == SPECTRUM_K)
						tBinK[0] -= mw;
					else {
						tBinG[0] -= mw*k2;
						tBinV[0] -= mw*mass;
					}
				} else if (zBase == (Lz*(commSize()>>1))) {
					size_t idx   = hLx + Lx*(hLz + Lz*hLy);
					double k2    = 2*hLy*hLy + hLz*hLz;
					size_t myBin = floor(sqrt(k2));

					if (spectral)
						k2 *= (4.*M_PI*M_PI)/(sizeL*sizeL);
					else
						k2  = cosTable[hLy] + cosTable[hLy] + cosTable[hLz];

					double w  = sqrt(k2 + mass);
					double m  = abs(static_cast<cFloat *>(field->m2Cpu())[idx]);
					double mw = m*m/w;

					if	(sType == SPECTRUM_K)
						tBinK[myBin] -= mw;
					else {
						tBinG[myBin] -= mw*k2;
						tBinV[myBin] -= mw*mass;
					}
				}
			}

			#pragma omp for schedule(static)
			for (int j=0; j<N; j++)
				for (int i=0; i<mIdx; i++) {
					if	(sType == SPECTRUM_K)
						binK[j] += tBinK[j + i*N];
					else {
						binG[j] += tBinG[j + i*N];
						binV[j] += tBinV[j + i*N];
					}
		}
	}

	void	SpecBin::Run	() {
		switch (fPrec) {
			case	FIELD_SINGLE:
				switch (fType) {
					case	FIELD_SAXION:
						fillBins<complex<float>>();
						break;

					case	FIELD_AXION:
						fillBins<float>();
						break;
				}
				break;

			case	FIELD_DOUBLE:
				switch (fType) {
					case	FIELD_SAXION:
						fillBins<complex<double>>();
						break;

					case	FIELD_AXION:
						fillBins<double>();
						break;
				}
				break;
		}
	}

#endif
#endif
