#ifndef	_CLASS_BINNER_
	#define	_CLASS_BINNER_

	#include <array>
	#include <algorithm>
	#include <cmath>
	#include <functional>
	#include <string>
	#include <mpi.h>

	template<FindType fType, typename cFloat>
	double	find	(cFloat *data, size_t size, std::function<double(cFloat)> filter) {
		LogMsg (VERB_NORMAL, "Called Find");

		if ((data == nullptr) || (size == 0))
			return 0.0;

		auto	cur = filter(data[0]);
		auto	ret = cur;

		switch (fType) {
			case	FIND_MAX: {
				#pragma omp parallel for reduction(max:cur) schedule(static)
				for (size_t idx=1; idx<size; idx++) {
					auto tmp = filter(data[idx]);

					if (cur < tmp)
						cur = tmp;
				}
				MPI_Allreduce (&cur, &ret, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			}
			break;

			case	FIND_MIN: {
				#pragma omp parallel for reduction(min:cur) schedule(static)
				for (size_t idx=1; idx<size; idx++) {
					auto tmp = filter(data[idx]);

					if (cur > tmp)
						cur = tmp;
				}
				MPI_Allreduce (&cur, &ret, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
			}
			break;
		}


		return	ret;
	}

	template<size_t N, typename DType>
	class	Binner {

		private:

		size_t	dSize;
		DType	*inData;

		std::array<double,N> bins;
		std::function<double(DType)> filter;

		double	maxVal;
		double	minVal;
		double	step;

		double	baseVal;

		public:

			Binner	() { bins.fill(0.); maxVal = 0.; minVal = 0.; step = 0.; baseVal = 0.; dSize = 0; inData = nullptr; }
			Binner	(DType *inData, size_t dSize, std::function<double(DType)> myFilter = [] (DType x) -> double { return (double) x; }) :
				 dSize(dSize), inData(inData), filter(myFilter) {
			bins.fill(0.);
			maxVal = (find<FIND_MAX,DType> (inData, dSize, filter));
			minVal = (find<FIND_MIN,DType> (inData, dSize, filter));

			LogMsg (VERB_NORMAL, "Binner found %f min, %f max", minVal, maxVal);

			if (std::abs(maxVal - minVal) < 1e-10) { LogError ("Error: max-min too close!"); bins.fill(maxVal); return; }
			step    = (maxVal-minVal)/((double) (N-1));
			baseVal = minVal - step*0.5;

		}

		DType*	getData	() const			{ return inData;   }
		void	setData	(DType *myData, size_t mySize)	{ inData = myData; dSize = mySize;
								  double tMaxVal = (find<FIND_MAX,DType> (inData, dSize, filter));
								  double tMinVal = (find<FIND_MIN,DType> (inData, dSize, filter));

								  MPI_Allreduce (&maxVal, &tMaxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
								  MPI_Allreduce (&minVal, &tMinVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

								  step    = (maxVal-minVal)/((double) (N-1));
								  baseVal = minVal - step*0.5;
								  if (maxVal <= minVal) { LogError ("Error: max value can't be lower or equal than min"); return; } }

		inline       double*	data	()		{ return bins.data();   }
		inline const double*	data	() const	{ return bins.data();   }

		void	run	();

		/*	idx is unsigned, we only check one end		*/
		inline double	operator()(DType  val)	const	{ size_t idx = (filter(val) - baseVal)/step; if (idx < N) { return bins[idx]; } else { return 0; } }
		inline double&	operator()(DType  val)		{ size_t idx = (filter(val) - baseVal)/step; if (idx < N) { return bins[idx]; } else { return bins[0]; } }
		inline double	operator[](size_t idx)	const	{ if (idx < N) { return bins[idx]; } else { return 0; } }
		inline double&	operator[](size_t idx)		{ if (idx < N) { return bins[idx]; } else { return bins[0]; } }

		inline double	max()			const	{ return maxVal; }
		inline double	min()			const	{ return minVal; }
	};

	template<size_t N, typename DType>
	void	Binner<N,DType>::run	() {
		size_t mIdx = commThreads();
		std::vector<size_t>	tBins(N*mIdx);
		tBins.assign(N*mIdx, 0);

		if (std::abs(maxVal - minVal) < 1.e-10) {
			LogMsg (VERB_NORMAL, "Running binner with %d threads, %llu bins, %f step, %f min, %f max", mIdx, N, step, minVal, maxVal);
			LogError ("Error: max value can't be lower or equal than min"); bins.fill(maxVal); return; }

		LogMsg (VERB_NORMAL, "Running binner with %d threads, %llu bins, %f step, %f min, %f max", mIdx, N, step, minVal, maxVal);
		double	tSize = static_cast<double>(dSize*commSize())*step;

		#pragma omp parallel
		{
			int tIdx = omp_get_thread_num ();

			#pragma omp for schedule(static)
			for (size_t i=0; i<dSize; i++) {
				auto cVal = filter(inData[i]);

				if (std::abs(cVal - baseVal) < step/100.) {
					tBins[N*tIdx]++;
				} else {
					size_t myBin = floor((cVal - baseVal)/step);

					if (myBin < N)	// Comparison with NaN will always return false
						tBins.at(myBin + N*tIdx)++;
					else
						LogError ("Warning: Binner class found value out of range %f (interval [%f, %f], assigned bin %lu of %lu)", cVal, baseVal, maxVal+0.5*step, myBin, N);
				}
			}

			#pragma omp for schedule(static)
			for (size_t j=0; j<N; j++)
				for (size_t i=0; i<mIdx; i++)
					bins[j] += static_cast<double>(tBins[j + i*N])/tSize;
		}

		std::array<double,N>    tmp;

		std::copy_n(bins.begin(), N, tmp.begin());
		MPI_Allreduce(tmp.data(), bins.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	}

#endif
