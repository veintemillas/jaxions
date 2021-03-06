#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <errno.h>
#include "enum-field.h"
#include "utils/logger.h"

static std::unordered_map<void *, std::pair<AllocType, size_t>> allocTable;
static size_t trackAlignMem = 0;
static size_t trackAllocMem = 0;

void	alignAlloc (void **ptr, size_t align, size_t size)
{
	int	out = posix_memalign (ptr, align, size);

	switch (out)
	{
		case 0:
		LogMsg (VERB_HIGH, "Memory allocated correctly (%lu bytes, %lu align). Registering pointer %p", size, align, *ptr);
		trackAlignMem += size;
		allocTable.insert(std::make_pair(*ptr, std::make_pair(ALLOC_ALIGN, size)));
		break;

		case EINVAL:
		LogError ("Error aligning memory: size (%lu) must be a multiple of align (%lu)", size, align);
		exit   (1);
		break;

		case ENOMEM:
		LogError ("Not enough memory. Requested %lu bytes with %lu alignment", size, align);
		exit   (1);
		break;

		default:
		LogError ("Unknown error");
		exit   (1);
		break;
	}

}

void	trackFree (void *ptr)
{
	AllocType aType = allocTable[ptr].first;
	size_t    bytes = allocTable[ptr].second;
	free (ptr);

	LogMsg (VERB_HIGH, "Memory freed correctly (%lu bytes). Deregistering pointer %p", bytes, ptr);

	if (aType == ALLOC_ALIGN)
		trackAlignMem -= bytes;
	else
		trackAllocMem -= bytes;

	allocTable.erase(ptr);
	ptr = nullptr;
}

void	trackAlloc (void **ptr, size_t size)
{
	if (((*ptr) = malloc(size)) == nullptr)
	{
		LogError ("Error allocating %lu bytes of unaligned memory", size);
		exit (1);
	}

	LogMsg (VERB_HIGH, "Memory allocated correctly (%lu bytes). Registering pointer %p", size, *ptr);

	allocTable.insert(std::make_pair(*ptr, std::make_pair(ALLOC_TRACK, size)));
	trackAllocMem += size;
}

void	printMemStats	()
{
	LogMsg (VERB_NORMAL, "Total allocated aligned   memory %lu", trackAlignMem);
	LogMsg (VERB_NORMAL, "Total allocated unaligned memory %lu", trackAllocMem);
	LogMsg (VERB_NORMAL, "");

	LogMsg (VERB_NORMAL, "Current pointers in memory:");
	LogMsg (VERB_NORMAL, "\tAligned");

	for (auto &data : allocTable) {
		void *ptr   = data.first;
		size_t size = data.second.second;
		if (data.second.first == ALLOC_ALIGN)
			LogMsg (VERB_NORMAL, "Pointer %p\tSize %lu", ptr, size);
	}

	LogMsg (VERB_NORMAL, "");
	LogMsg (VERB_NORMAL, "\tUnaligned");

	//for (data = allocTable[ALLOC_TRACK].begin(); data != allocTable[ALLOC_TRACK].end(); data++)
	for (auto &data : allocTable) {
		void *ptr   = data.first;
		size_t size = data.second.second;
		if (data.second.first == ALLOC_TRACK)
		LogMsg (VERB_NORMAL, "Pointer %p\tSize %lu", ptr, size);
	}
}
