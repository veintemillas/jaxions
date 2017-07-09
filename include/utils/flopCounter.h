#ifndef	_FLOP_COUNTER_
	#define	_FLOP_COUNTER_

	class	FlopCounter
	{
		private:

		bool	started;

		double	gFlops;
		double	gBytes;
		double	dTime;

		public:

			FlopCounter	() { started = false; gFlops = 0; gBytes = 0; dTime = 0; }
			~FlopCounter	() {};

		void	addFlops(double fl, double bt) { gFlops += fl; gBytes += bt; }
		void	addTime	(double tm) { started = true; dTime += tm; }

		bool	Started	() { return started; }

		double	GFlops	() { return gFlops/dTime; }
		double	GBytes	() { return gBytes/dTime; }
		double	DTime	() { return dTime; }
	};
#endif
