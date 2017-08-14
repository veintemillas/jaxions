#ifndef	_CLASS_BINNER_
	#define	_CLASS_BINNER_

	#include <array>
	#include <algorithm>
	#include <mpi.h>

	template<FindType fType, typename cFloat, bool sign>
	cFloat	find	(cFloat *data, size_t size) {

		if ((data == nullptr) || (size == 0))
			return cFloat(0);

		auto	cur = (sign ? data[0] : abs(data[0]));

		switch (fType) {
			case	FIND_MAX: {
				#pragma omp parallel for reduction(max:cur) schedule(static)
				for (int idx=1; idx<size; idx++)
					if (sign) {
						if (cur < data[idx])
							cur = data[idx];
					} else {
						if (abs(cur) < abs(data[idx]))
							cur = data[idx];
					}
			}
			break;

			case	FIND_MIN: {
				#pragma omp parallel for reduction(min:cur) schedule(static)
				for (int idx=1; idx<size; idx++)
					if (sign) {
						if (cur > data[idx])
							cur = data[idx];
					} else {
						if (abs(cur) > abs(data[idx]))
							cur = data[idx];
					}
			}
			break;
		}

		return	cur;
	}

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

			Binner	() { bins.fill(0.); }
			Binner	(DType *inData, size_t dSize) : dSize(dSize), step((maxVal-minVal)/((DType) N)), inData(inData) {
			bins.fill(0.);
			maxVal = find<FIND_MAX,DType,false> (inData, dSize);
			minVal = find<FIND_MIN,DType,false> (inData, dSize);
		}

		DType*	getData	() const			{ return inData;   }
		void	setData	(DType *myData, size_t mySize)	{ inData = myData; dSize = mySize;
								  maxVal = find<FIND_MAX,DType,false> (inData, dSize);
								  minVal = find<FIND_MIN,DType,false> (inData, dSize); }

		inline       double*	data	()		{ return bins.data();   }
		inline const double*	data	() const	{ return bins.data();   }

		void	run	();

		inline double	operator()(DType  val)	const	{ size_t idx = (val - minVal)/step; if (idx > 0 || idx < N) { return bins[idx]; } else { return 0; } }
		inline double&	operator()(DType  val)		{ size_t idx = (val - minVal)/step; if (idx > 0 || idx < N) { return bins[idx]; } else { return bins[0]; } }
		inline double	operator[](size_t idx)	const	{ return bins[idx]; }
		inline double&	operator[](size_t idx)		{ return bins[idx]; }
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
