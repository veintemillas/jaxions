#include <unistd.h>
#include <linux/module.h>
#include "utils/system.h"

namespace SystemInfo {
	constexpr int cacheTypeMask      = 0b00000000000000000000000000011111;
	constexpr int cacheLevelMask     = 0b00000000000000000000000011100000;
	constexpr int cacheLineMask      = 0b00000000000000000000111111111111;
	constexpr int cacheLineMaskAMD   = 0b00000000000000000000000011111111;
	constexpr int cachePartitionMask = 0b00000000001111111111000000000000;
	constexpr int cacheWaysMask      = 0b11111111110000000000000000000000;
	constexpr int cacheThreadsMask   = 0b00000011111111111100000000000000;
	constexpr int packageAPICMask    = 0b11111100000000000000000000000000;
	constexpr int hyperthreadingMask = 0b00010000000000000000000000000000;
	constexpr int addressableIDMask  = 0b00000000111111110000000000000000;

	void	getSystemInfo	() {

		unsigned int eax, ebx, ecx, edx, iCache = 0;
		bool	     cachePending = true;

		unsigned int threadsPerCore, nCores = 4096;

		nProcessors = sysconf(_SC_NPROCESSORS_ONLN);
		procPerNode = nProcessors/commRanksPerNode();

		printf ("Detected %d processors, each rank will use %u processors\n", nProcessors, procPerNode);

		eax = 0x00;

		asm("cpuid" :"=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "0" (eax));

		char name[16];

		static_cast<unsigned int*>(name)[0] = ebx;
		static_cast<unsigned int*>(name)[1] = edx;
		static_cast<unsigned int*>(name)[2] = ecx;

		name[13] = '\0'

		printf("%s\n", name);

		if (!strcmp(name, "GenuineIntel")) {
			procBrand = GENUINE_INTEL;
		} else if (!strcmp(name, "AuthenticAMD")) {
			procBrand = AUTHENTIC_AMD;
		} else {
			LogMsg(VERB_NORMAL, "Unrecognized processor %s, cache tuning is disabled", name);
			return;
		}

		eax = 0x01;

		asm("cpuid" :"=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "0" (eax));

		bool htEnabled = ((edx & hyperthreadingMask) != 0);

		if (htEnabled) {
			threadsPerCore = ((ebx & addressableIDMask) >> 16);
			threadsPerCore--;
			threadsPerCore |= threadsPerCore >> 1;
			threadsPerCore |= threadsPerCore >> 2;
			threadsPerCore |= threadsPerCore >> 4;
			threadsPerCore |= threadsPerCore >> 8;
			threadsPerCore |= threadsPerCore >> 16;
			threadsPerCore++; 
		} else
			threadsPerCore = 1;

		cacheLineSize = ((ebx & 0b00000000000000001111111100000000) >> 8);

		printf ("Initial test clflush %u, threads %u\n", cacheLineSize, threadsPerCore);

		switch (procBrand) {
			case	GENIUNE_INTEL:
			do {
				cachePending = false;

				eax = 0x04;
				ecx = iCache;

				asm("cpuid" :"=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "0" (eax), "2" (ecx));

				unsigned int cacheType  = ((eax & cacheTypeMask));
				unsigned int cacheLevel = ((eax & cacheLevelMask) >> 5);

				if (cacheType != 0) {
					unsigned int cSets       = ecx;
					unsigned int cLine       = ((ebx & cacheLineMask));
					unsigned int cPartitions = ((ebx & cachePartitionMask) >> 12);
					unsigned int cWays       = ((ebx & cacheWaysMask)      >> 22);
					unsigned int cThreads    = ((eax & cacheThreadsMask)   >> 14) + 1;
					unsigned int nAPIC       = ((eax & packageAPICMask)    >> 26) + 1;

					cacheLineSize = cLine + 1;
					cacheSize     = (cWays + 1)*(cPartitions + 1)*cacheLineSize*(cSets + 1);

					printf ("APIC IDs %u\n", nAPIC);
					printf ("Cache level %u type %u is shared among %u threads and has %u total size and %u line size\n", cacheLevel, cacheType, cThreads, cacheSize, cacheLineSize);

					iCache++;
					ecx = iCache;
					cachePending = true;
				}
			}	while(cachePending);
			break;

			case	AUTHENTIC_AMD:

			eax = 0x80000006;

			asm("cpuid" :"=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "0" (eax));

			cacheLineSize = ((ecx & cacheLineMaskAMD)); 
			cacheSize     = ((ecx & cacheSizeMaskAMD) >> 18)*512*1024;
			printf ("Cache level 3 has %u total size and %u line size\n", cacheSize, cacheLineSize);
			break; 
		}

		return;
	}
}

