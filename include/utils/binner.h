#ifndef	_CLASS_BINNER_
	#define	_CLASS_BINNER_

	#include <array>
	#include <algorithm>
	#include <mpi.h>

	template<typename DType, size_t N>
	class	Binner {

		private:

		std::array<double,N>	bins;

		DType	maxVal;
		DType	minVal;
		DType	step;

		DType	*inData;
		size_t	dSize;

		public:

			Binner	(DType minVal=0, DType maxVal=1) : maxVal(maxVal), minVal(minVal), step((maxVal-minVal)/((DType) N)) { bins.fill(0.); }
			Binner	(DType *inData, size_t dSize, DType minVal=0, DType maxVal=1) : maxVal(maxVal), minVal(minVal), dSize(dSize), step((maxVal-minVal)/((DType) N)), inData(inData) { bins.fill(0.); }

		void	setData	(DType *myData, size_t mySize)	{ inData = myData; dSize = mySize; }
		DType*	getData	() const			{ return inData;   }

		inline       double*	data	()		{ return bins.data();   }
		inline const double*	data	() const	{ return bins.data();   }

		void	run	();

		inline double	operator()(size_t idx)	const	{ return bins[idx]; }
		inline double&	operator()(size_t idx)		{ return bins[idx]; }
	};

	template<typename DType, size_t N>
	void	Binner<DType,N>::run	() {
		int mIdx = commThreads();
		std::vector<size_t>	tBins(N*mIdx);
		tBins.assign(N*mIdx, 0);

		double	tSize = static_cast<double>(dSize*commSize());

		#pragma omp parallel
		{
			int tIdx = omp_get_thread_num ();

			DType base = minVal;

			#pragma omp for schedule(static)
			for (size_t i=0; i<dSize; i++) {
				if ((inData[i] < minVal) || (inData[i] >= maxVal))
					continue;

				size_t myBin = floor((inData[i] - minVal)/step);

				tBins[myBin + N*tIdx]++;
			}

			#pragma omp for schedule(static)
			for (int j=0; j<N; j++)
				for (int i=0; i<mIdx; i++)
					bins[j] += static_cast<double>(tBins[j + i*N])/tSize;
		}

		std::array<double,N>    tmp;

		std::copy_n(bins.begin(), N, tmp.begin());
		MPI_Allreduce(tmp.data(), bins.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	}

#endif
