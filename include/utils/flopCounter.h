#ifndef	_FLOP_COUNTER_
	#define	_FLOP_COUNTER_

	class	FlopCounter
	{
		private:

		double	gFlops;
		double	gBytes;
		double	dTime;

		public:

			FlopCounter	() { gFlops = 0; gBytes = 0; dTime = 0; }
			~FlopCounter	() {};

		void	addFlops(double fl, double bt) { gFlops += fl; gBytes += bt; }
		void	addTime	(double tm) { dTime += tm; }
		void	reset	() { gFlops = 0; gBytes = 0; dTime = 0; }

		double	GFlops	() { return gFlops/dTime; }
		double	GBytes	() { return gBytes/dTime; }
		double	DTime	() { return dTime; }
	};
#endif
