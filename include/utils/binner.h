#ifndef	_CLASS_BINNER_
	#define	_CLASS_BINNER_

	#include <array>
	#include <algorithm>
	#include <functional>
	#include <string>
	#include <mpi.h>

	template<FindType fType, typename cFloat, typename Float>
	float	find	(Float *data, size_t size, std::function<Float(const cFloat &)> filter) {
		LogMsg (VERB_NORMAL, "Called Find");

		if ((data == nullptr) || (size == 0))
			return Float(0);

		auto	cur = filter(data[0]);

		switch (fType) {
			case	FIND_MAX: {
				#pragma omp parallel for reduction(max:cur) schedule(static)
				for (size_t idx=1; idx<size; idx++) {
					auto tmp = filter(data[idx]);

					if (cur < tmp)
						cur = tmp;
				}
			}
			break;

			case	FIND_MIN: {
				#pragma omp parallel for reduction(min:cur) schedule(static)
				for (size_t idx=1; idx<size; idx++) {
					auto tmp = filter(data[idx]);

					if (cur > tmp)
						cur = tmp;
				}
			}
			break;
		}

		return	cur;
	}

	template<typename DType, size_t N, typename CType = DType>
	class	Binner {

		private:

		std::array<double,N> bins;
		std::function<DType(const CType &)> filter;

		DType	maxVal;
		DType	minVal;
		DType	step;
		double	zVal;

		DType	*inData;
		size_t	dSize;

		public:

			Binner	() { bins.fill(0.); zVal = 1.0; }
			Binner	(DType *inData, size_t dSize, double zIn, std::function<DType(const CType &)> myFilter = [] (CType x) -> DType { return (DType) x; }) :
				 dSize(dSize), inData(inData), zVal(zIn), filter(myFilter) {
			bins.fill(0.);
			double tMaxVal = (find<FIND_MAX,CType,DType> (inData, dSize, filter))/zVal;
			double tMinVal = (find<FIND_MIN,CType,DType> (inData, dSize, filter))/zVal;

			double t1 = 0., t2 = 0.;

			MPI_Allreduce (&tMaxVal, &t1, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			MPI_Allreduce (&tMinVal, &t2, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

			maxVal = t1; minVal = t2;
			step   = (maxVal-minVal)/((DType) N);
			if (maxVal < minVal) { LogError ("Error: max value can't be lower than min"); return; }
		}

		void	setZ	(DType zIn)			{ maxVal *= zVal/zIn; minVal *= zVal/zIn; zVal = zIn; }

		DType*	getData	() const			{ return inData;   }
		void	setData	(DType *myData, size_t mySize)	{ inData = myData; dSize = mySize;
								  double tMaxVal = (find<FIND_MAX,CType,DType> (inData, dSize, filter))/zVal;
								  double tMinVal = (find<FIND_MIN,CType,DType> (inData, dSize, filter))/zVal;

								  double t1, t2;

								  MPI_Allreduce (&t1, &tMaxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
								  MPI_Allreduce (&t2, &tMinVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

								  maxVal = t1; minVal = t2;

								  step   = (maxVal-minVal)/((DType) N);
								  if (maxVal < minVal) { LogError ("Error: max value can't be lower than min"); return; } }

		inline       double*	data	()		{ return bins.data();   }
		inline const double*	data	() const	{ return bins.data();   }

		void	run	();

		inline double	operator()(DType  val)	const	{ size_t idx = (val - minVal)/step; if (idx > 0 || idx < N) { return bins[idx]; } else { return 0; } }
		inline double&	operator()(DType  val)		{ size_t idx = (val - minVal)/step; if (idx > 0 || idx < N) { return bins[idx]; } else { return bins[0]; } }
		inline double	operator[](size_t idx)	const	{ return bins[idx]; }
		inline double&	operator[](size_t idx)		{ return bins[idx]; }

		inline double	max()			const	{ return maxVal; }
		inline double	min()			const	{ return minVal; }
	};

	template<typename DType, size_t N, typename CType>
	void	Binner<DType,N,CType>::run	() {
		int mIdx = commThreads();
		std::vector<size_t>	tBins(N*mIdx);
		tBins.assign(N*mIdx, 0);

		LogMsg (VERB_NORMAL, "Running binner with %d threads, %llu bins, %f step, %f min, %f max @ z %lf", mIdx, N, step, minVal, maxVal, zVal);

		double	tSize = static_cast<double>(dSize*commSize())*step;

		#pragma omp parallel
		{
			int tIdx = omp_get_thread_num ();

			DType base = minVal;

			#pragma omp for schedule(static)
			for (size_t i=0; i<dSize; i++) {
				auto cVal = filter(inData[i])/zVal;

				if (fabs(cVal - minVal) < step/100.) {
					tBins[N*tIdx]++;
				} else {
					size_t myBin = floor((cVal - minVal)/step);

					if (myBin >= N)
						LogError ("Warning: Binner class found value out of range %f (interval [%f, %f])", cVal, minVal, maxVal);
					else
						tBins[myBin + N*tIdx]++;
				}
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
