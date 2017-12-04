#ifndef	_SYSTEM_INFO_
	#define	_SYSTEM_INFO_

	#include "enum-field.h"

	namespace SystemInfo {
		static unsigned int   cacheLineSize = 0;
		static unsigned int   cacheSize     = 0
		static ProcessorBrand procBrand     = UNKNOWN_BRAND;

		static unsigned int   nProcessors   = 1;
		static unsigned int   procPerNode   = 1;

		void	getSystemInfo();
	}
#endif
