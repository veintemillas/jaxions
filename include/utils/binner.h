#ifndef	_CLASS_BINNER_
	#define	_CLASS_BINNER_

	#include <array>

	template<typename DType, size_t N>
	class	Binner {

		private:

		std::array<size_t,N>	bins;

		DType	maxVal;
		DType	minVal;
		DType	step;

		DType	*inData;
		size_t	dSize;

		public:

			Binner	(DType minVal=0, DType maxVal=1) : maxVal(maxVal), minVal(minVal), step((maxVal-minVal)/((DType) N)) { bins.fill(0); }
			Binner	(DType *inData, size_t dSize, DType minVal=0, DType maxVal=1) : maxVal(maxVal), minVal(minVal), step((maxVal-minVal)/((DType) N)), inData(inData) { bins.fill(0); }

		void	setData	(DType *myData, size_t mySize)	{ inData = myData; dSize = mySize; }
		DType*	getData	() const			{ return inData;   }

		inline       size_t*	data	()		{ return bins.data();   }
		inline const size_t*	data	() const	{ return bins.data();   }

		void	run	();

		inline size_t	operator()(size_t idx)	const	{ return bins[idx]; }
		inline size_t&	operator()(size_t idx)		{ return bins[idx]; }
	};

	template<typename DType, size_t N>
	void	Binner<DType,N>::run	() {
		int mIdx = commThreads();
		std::vector<size_t>	tBins(N*mIdx);
		tBins.assign(N*mIdx, 0);

		#pragma omp parallel
		{
			int tIdx = omp_get_thread_num ();

			DType base = minVal;

			#pragma omp for schedule(static)
			for (int i=0; i<dSize; i++) {
				size_t myBin = floor((inData[i] - minVal)/step);

				tBins[myBin + N*tIdx]++;
			}

			#pragma omp for schedule(static)
			for (int j=0; j<N; j++)
				for (int i=0; i<mIdx; i++)
					bins[j] += tBins[j + i*N];
		}
	}

#endif
