#ifndef	_CLASS_BINNER_
	#define	_CLASS_BINNER_

	template<typename DType, size_t N>
	class	Binner {

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
				break;	
		}
	}

#endif
