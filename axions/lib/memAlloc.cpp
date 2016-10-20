#include <cstdio>
#include <cstdlib>
#include <map>
#include <errno.h>
#include "enum-field.h"

static std::map<void *, size_t> allocTable[2];
static size_t trackAlignMem = 0;
static size_t trackAllocMem = 0;

void	alignAlloc (void **ptr, size_t align, size_t size)
{
	int	out = posix_memalign (ptr, align, size);

	switch (out)
	{
		case 0:
		printf ("Memory allocated correctly (%lu bytes, %lu align). Registering pointer %p\n", size, align, *ptr);
		fflush (stdout);
		trackAlignMem += size;
		allocTable[ALLOC_ALIGN].insert(std::make_pair(*ptr, size));
		break;

		case EINVAL:
		printf ("Error aligning memory: size (%lu) must be a multiple of align (%lu)\n", size, align);
		exit   (1);
		break;

		case ENOMEM:
		printf ("Not enough memory. Requested %lu bytes with %lu alignment\n", size, align);
		exit   (1);
		break;

		default:
		printf ("Unknown error\n");
		exit   (1);
		break;
	}

}

void	trackFree (void **ptr, AllocType aType)
{
	size_t bytes = allocTable[aType][*ptr];
	free (*ptr);

	printf ("Memory freed correctly (%lu bytes). Deregistering pointer %p\n", bytes, *ptr);
	fflush (stdout);

	if (aType == ALLOC_ALIGN)
		trackAlignMem -= bytes;
	else
		trackAllocMem -= bytes;

	allocTable[aType].erase(*ptr);
	ptr = NULL;
}

void	trackAlloc (void **ptr, size_t size)
{

	if (((*ptr) = malloc(size)) == NULL)
	{
		printf ("Error allocating %lu bytes of unaligned memory\n", size);
		exit (1);
	}

	printf ("Memory allocated correctly (%lu bytes). Registering pointer %p\n", size, *ptr);

	fflush (stdout);
	allocTable[ALLOC_TRACK].insert(std::make_pair(*ptr, size));
	trackAllocMem += size;
	fflush (stdout);
}

void	printMemStats	()
{
	printf ("Total allocated aligned   memory %lu\n", trackAlignMem);
	printf ("Total allocated unaligned memory %lu\n", trackAllocMem);
	printf ("\nCurrent pointers in memory:\n");
	printf ("\tAligned\n");

	std::map<void *, size_t>::iterator data;

	for (data = allocTable[ALLOC_ALIGN].begin(); data != allocTable[ALLOC_ALIGN].end(); data++)
	{
		void *ptr   = data->first;
		size_t size = data->second;
		printf ("Pointer %p\tSize %lu\n", ptr, size);
	}

	printf ("\n\tUnaligned\n");

	for (data = allocTable[ALLOC_TRACK].begin(); data != allocTable[ALLOC_TRACK].end(); data++)
	{
		void *ptr   = data->first;
		size_t size = data->second;
		printf ("Pointer %p\tSize %lu\n", ptr, size);
	}
}
