#ifndef	_TUNABLE_
	#define	_TUNABLE_
	#include <string>

	using namespace std;

	class	Tunable {
		private:

		string	name;

		double		gFlops;
		double		gBytes;

		public:

			Tunable() noexcept : gFlops(0.), gBytes(0.), name("") {}

		double	GFlops () const noexcept { return gFlops; }
		double	GBytes () const noexcept { return gBytes; }
		string	Name   () const noexcept { return name; }

		void	reset  ()                     { gFlops = 0.; gBytes = 0.; }
		void	add    (double GF, double GB) { gFlops += GF; gBytes += GB; }

		void	setName   (const char * newName) { name.assign(newName); }
		void	appendName(const char * appName) { name += string(appName); }
	};
#endif
