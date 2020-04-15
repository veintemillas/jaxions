#ifndef	_CLASS_BINNER_
	#define	_CLASS_BINNER_

	#include <array>
	#include <algorithm>
	#include <cmath>
	#include <functional>
	#include <string>
	#include <mpi.h>

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

		size_t	PaddedLength;
		size_t	DataLength;
		bool	setmaxmin;

		bool	unphysicalmaxmin;

		public:

			Binner	() { bins.fill(0.); maxVal = 0.; minVal = 0.; step = 0.; baseVal = 0.; dSize = 0; inData = nullptr; }
			Binner	(DType *inData, size_t dSize, std::function<double(DType)> myFilter = [] (DType x) -> double { return (double) x; }) :
				 dSize(dSize), inData(inData), filter(myFilter) {
			bins.fill(0.);

			DataLength = PaddedLength =1;
			setmaxmin = false;
			unphysicalmaxmin = false;
			// maxVal = (find<FIND_MAX,DType> (inData, dSize, filter));
			// minVal = (find<FIND_MIN,DType> (inData, dSize, filter));
			//
			// LogMsg (VERB_NORMAL, "Binner found %f min, %f max", minVal, maxVal);
			//
			// if (std::isinf(maxVal) || std::isinf(minVal))	{ LogError ("Error: infinite value found");			 return; }
			// if (std::isnan(maxVal) || std::isnan(minVal))	{ LogError ("Error: NaN found");				 return; }
			// if (std::abs(maxVal - minVal) < 1e-10)		{ LogError ("Error: max-min too close!");   bins.fill(maxVal);   return; }
			// step    = (maxVal-minVal)/((double) (N-1));
			// baseVal = minVal - step*0.5;

		}

		DType*	getData	() const			{ return inData;   }
		// void	setData	(DType *myData, size_t mySize)	{ inData = myData; dSize = mySize;
		// 						  double tMaxVal = (find<FIND_MAX,DType> (inData, dSize, filter));
		// 						  double tMinVal = (find<FIND_MIN,DType> (inData, dSize, filter));
		//
		// 						  MPI_Allreduce (&maxVal, &tMaxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		// 						  MPI_Allreduce (&minVal, &tMinVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		//
		// 						  step    = (maxVal-minVal)/((double) (N-1));
		// 						  baseVal = minVal - step*0.5;
		// 							if (std::isinf(maxVal) || std::isinf(minVal))	{ LogError ("Error: infinite value found");			 return; }
		// 						  if (std::isnan(maxVal) || std::isnan(minVal))	{ LogError ("Error: NaN found");				 return; }
		// 						  if (std::abs(maxVal - minVal) < 1e-10)	{ LogError ("Error: max-min too close!");   bins.fill(maxVal);   return; }
		// 							if (maxVal <= minVal) { LogError ("Error: max value can't be lower or equal than min"); return; } }

		inline       double*	data	()		{ return bins.data();   }
		inline const double*	data	() const	{ return bins.data();   }

		void	setpad	(size_t PadLin,size_t DatLin) { PaddedLength = PadLin; DataLength = DatLin;};

		void	find	();
		void	run	();

		/*	idx is unsigned, we only check one end		*/
		inline double	operator()(DType  val)	const	{ size_t idx = (filter(val) - baseVal)/step; if (idx < N) { return bins[idx]; } else { return 0; } }
		inline double&	operator()(DType  val)		{ size_t idx = (filter(val) - baseVal)/step; if (idx < N) { return bins[idx]; } else { return bins[0]; } }
		inline double	operator[](size_t idx)	const	{ if (idx < N) { return bins[idx]; } else { return 0; } }
		inline double&	operator[](size_t idx)		{ if (idx < N) { return bins[idx]; } else { return bins[0]; } }

		inline double	max()			const	{ return maxVal; }
		inline double	min()			const	{ return minVal; }
	};


	template<size_t N,typename DType>
	void	Binner<N,DType>::find	() {

		if (PaddedLength == DataLength)
			LogMsg (VERB_NORMAL, "Called Find");
		else
			LogMsg (VERB_NORMAL, "Called Find PadLength/DataLength = %d/%d",PaddedLength,DataLength);
		LogFlush();

		if ((inData == nullptr) || (dSize == 0))
			return ;

		auto	car = filter(inData[0]);
		auto	cir = filter(inData[0]);
		int	anf =0;
		int Ninf = 0;

		if (PaddedLength == DataLength) {

			#pragma omp parallel for schedule(static) reduction(max:car) reduction(min:cir) reduction(+:anf)
			for (size_t idx=1; idx<dSize; idx++) {

				double tmp = filter(inData[idx]);

				if (std::isinf(tmp)) {
					anf++;
					continue;
				}

				if (tmp > car)
					car = tmp;
				if (tmp < cir)
					cir = tmp;
			}

		} else if (PaddedLength > DataLength) {
			LogMsg (VERB_NORMAL, "Padded loop");LogFlush();
			#pragma omp parallel for schedule(static) reduction(max:car) reduction(min:cir) reduction(+:anf)
			for (size_t idx=1; idx<dSize; idx++) {

				if (idx%PaddedLength >= DataLength)
					continue;

				double tmp = filter(inData[idx]);

				if (std::isinf(tmp)) {
					anf++;
					continue;
				}

				if (tmp > car)
					car = tmp;
				if (tmp < cir)
					cir = tmp;
			}
		}

		MPI_Allreduce (&car, &maxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		MPI_Allreduce (&cir, &minVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce (&anf, &Ninf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		// if (std::isinf(maxVal) || std::isinf(minVal))	{ LogError ("Error: infinite value found");	unphysicalmaxmin = true; }
		// if (std::isnan(maxVal) || std::isnan(minVal))	{ LogError ("Error: NaN found");						unphysicalmaxmin = true; }
		// if (std::abs(maxVal - minVal) < 1e-10)        { LogError ("Error: max-min too close!");   bins.fill(maxVal); unphysicalmaxmin = true; }

		if (Ninf > 0)	{ LogMsg (VERB_NORMAL, "Error: infinite values found but ignored (%d of them)",Ninf); }
		/* I think cannot happen */
		if (std::isnan(maxVal) || std::isnan(minVal)) { LogError ("Error: NaN found");				     unphysicalmaxmin = true; }
		/* This yes  */
		if (std::abs(maxVal - minVal) < 1e-10)        { LogError ("Error: max-min too close!");   bins.fill(maxVal); unphysicalmaxmin = true; }


		step    = (maxVal-minVal)/((double) (N-1));

		baseVal = minVal - step*0.5;

		setmaxmin = true;

		LogMsg (VERB_NORMAL, "Binner found min %.5e , max %.5e, step %.5e, baseVal %.5e, setmaxmin %d ", minVal, maxVal, step, baseVal,setmaxmin);
	}





	template<size_t N, typename DType>
	void	Binner<N,DType>::run	() {
		LogMsg (VERB_DEBUG, "Called Bin run");LogFlush();

		if (!setmaxmin) {
			LogMsg (VERB_NORMAL, "Call find()");LogFlush();
			find();
		}

		if (unphysicalmaxmin) {
			LogMsg (VERB_NORMAL, "There was an unphysical max or min, skip");LogFlush();
			return;
		}

		size_t mIdx = commThreads();
		std::vector<size_t>	tBins(N*mIdx);
		tBins.assign(N*mIdx, 0);

		if (std::isinf(maxVal) || std::isinf(minVal))	{ LogError ("Error: can't run binner with infinities");	 return; }
		if (std::isnan(maxVal) || std::isnan(minVal))	{ LogError ("Error: cen't run binner with Nan values");	 return; }

		if (std::abs(maxVal - minVal) < 1.e-10) {
			LogMsg (VERB_NORMAL, "Running binner with %d threads, %llu bins, %f step, %f min, %f max", mIdx, N, step, minVal, maxVal);
			LogError ("Error: max value can't be lower or equal than min");
			bins.fill(maxVal);
			return;
		}

		LogMsg (VERB_NORMAL, "Running binner with %d threads, %llu bins, %e step, %e min, %e max", mIdx, N, step, minVal, maxVal);LogFlush();


		int unbins = 0;
		if (PaddedLength == DataLength) {

			double	tSize = static_cast<double>(dSize*commSize())*step;

			/* contiguous data */
			#pragma omp parallel
			{
				int tIdx = omp_get_thread_num ();

				#pragma omp for schedule(static) reduction(+:unbins)
				for (size_t i=0; i<dSize; i++) {

					double cVal = filter(inData[i]);

					if ((cVal >= minVal) && (cVal <= maxVal)) {

						if (std::abs(cVal - baseVal) < step/100.)
							tBins[N*tIdx]++;
						else {
							size_t myBin = floor((cVal - baseVal)/step);

							if (0 <= myBin && myBin < N)	// Comparison with NaN will always return false
								tBins.at(myBin + N*tIdx)++;
							else {
								LogMsg (VERB_DEBUG,"Warning: (th%d) Value out of range data[%lu]=%e > %e (interval [%f, %f], assigned bin %lu of %lu)",
								tIdx, i, inData[i], cVal, baseVal, maxVal+0.5*step, myBin, N);LogFlush();
								unbins++;
							}
						}
					} else
						unbins++;

				}

				#pragma omp for schedule(static)
				for (size_t j=0; j<N; j++)
					for (size_t i=0; i<mIdx; i++)
						bins[j] += static_cast<double>(tBins[j + i*N])/tSize;
			}
		} else if (PaddedLength > DataLength) {
			/* data padded with zeros */
			/* corrected for padding */
			size_t unpaddedSize = (dSize/PaddedLength)*DataLength;
			double tSize        = static_cast<double>(unpaddedSize*commSize())*step;

			#pragma omp parallel
			{
				int tIdx = omp_get_thread_num ();

				#pragma omp for schedule(static) reduction(+:unbins)
				for (size_t i=0; i<dSize; i++) {

					if (i%PaddedLength >= DataLength)
						continue;

					double cVal = filter(inData[i]);

					if ((cVal >= minVal) && (cVal <= maxVal)) {

						if (std::abs(cVal - baseVal) < step/100.) {
							tBins[N*tIdx]++;
						} else {
							size_t myBin = floor((cVal - baseVal)/step);

							if (0 <= myBin && myBin < N)	// Comparison with NaN will always return false
								tBins.at(myBin + N*tIdx)++;
							else {
								LogMsg (VERB_DEBUG,"Warning: (th%d) Value out of range data[%lu]=%e > %e (interval [%f, %f], assigned bin %lu of %lu)",
								tIdx, i, inData[i], cVal, baseVal, maxVal+0.5*step, myBin, N);LogFlush();
								unbins++;
							}
						}
					} else
						unbins++;
				}

				#pragma omp for schedule(static)
				for (size_t j=0; j<N; j++)
					for (size_t i=0; i<mIdx; i++)
						bins[j] += static_cast<double>(tBins[j + i*N])/tSize;
			}
		}

		LogMsg (VERB_DEBUG, "Binner Loop done (exceptions %d) ", unbins);LogFlush();

		std::array<double,N>    tmp;

		std::copy_n(bins.begin(), N, tmp.begin());
		MPI_Allreduce(tmp.data(), bins.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


		commSync();
		int unbin = 0 ;
		MPI_Allreduce (&unbins, &unbin, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		LogMsg (VERB_NORMAL, "Binner done (%d exceptions not binned)", unbin);LogFlush();
	}






	/* min max finder not using the class */

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


#endif
