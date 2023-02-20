#ifndef	_PROPBASE_
	#define	_PROPBASE_

	#include "enum-field.h"
	#include "utils/utils.h"

	class	PropBase : public Tunable
	{
		protected:

		std::string	baseName;

		bool	gravity;

		public:

			 PropBase() {};
		virtual	~PropBase() {};

		inline void	setBaseName(const char *bName) { baseName.assign(bName); }
		inline void	getBaseName() 		       { name = baseName; }

		inline void	SetGravity (const bool guav) {gravity = guav; };

		virtual void	sRunCpu	(const double) = 0;	// Saxion propagator
		virtual void	sRunGpu	(const double) = 0;
		virtual void	sSpecCpu(const double) = 0;	// Saxion spectral propagator
		virtual void	sFpecCpu(const double) = 0;	// Saxion full spectral propagator


		virtual void	tRunCpu	(const double) = 0;	// Axion propagator
		virtual void	tRunGpu	(const double) = 0;
		virtual void	tSpecCpu(const double) = 0;	// Axion spectral propagator
		virtual void	tFpecCpu(const double) = 0;	// Axion spectral propagator

		virtual void	nRunCpu	(const double) = 0;	// Naxion propagator

		virtual void	pRunCpu	(const double) = 0;	// Paxion propagator

		virtual void	lowCpu	(const double) = 0;	// Lowmem only available for saxion
		virtual void	lowGpu	(const double) = 0;
		virtual void	tlowGpu	(const double) = 0;

		virtual double	cFlops	(const PropcType) = 0;
		virtual double	cBytes	(const PropcType) = 0;

		std::function<void(const double)>	propAxion;
		std::function<void(const double)>	propSaxion;
		std::function<void(const double)>	propNaxion;
		std::function<void(const double)>	propPaxion;



	};
#endif
