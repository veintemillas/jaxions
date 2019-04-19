#ifndef	_PROPBASE_
	#define	_PROPBASE_

	#include "enum-field.h"
	#include "utils/utils.h"

	class	PropBase : public Tunable
	{
		protected:

		std::string	baseName;

		public:

			 PropBase() {};
		virtual	~PropBase() {};

		inline void	setBaseName(const char *bName) { baseName.assign(bName); }
		inline void	getBaseName() 		       { name = baseName; }

		virtual void	sRunCpu	(const double) = 0;	// Saxion propagator
		virtual void	sRunGpu	(const double) = 0;
//		virtual void	sRunXeon() = 0;

		virtual void	sSpecCpu(const double) = 0;	// Saxion spectral propagator
		virtual void	sFpecCpu(const double) = 0;	// Saxion full spectral propagator

		virtual void	tRunCpu	(const double) = 0;	// Axion propagator
		virtual void	tRunGpu	(const double) = 0;
//		virtual void	tRunXeon() = 0;

		virtual void	tSpecCpu(const double) = 0;	// Axion spectral propagator

		virtual void	lowCpu	(const double) = 0;	// Lowmem only available for saxion
		virtual void	lowGpu	(const double) = 0;
//		virtual void	lowXeon	() = 0;

		virtual double	cFlops	(const PropcType) = 0;
		virtual double	cBytes	(const PropcType) = 0;

		std::function<void(const double)>	propAxion;
		std::function<void(const double)>	propSaxion;

	};
#endif
