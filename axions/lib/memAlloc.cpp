#include <cstdlib>
#include "enum_field.h"

static std::map<void *, size_t> allocTable[2];
static size_t trackAlignMem = 0;

void	alignAlloc (void **ptr, size_t align, size_t size)
{
	int	out = posix_memalign (ptr, align, size);

	switch (out)
	{
		case 0:
		trackAlignMem += size;
		allocTable[ALLOC_ALIGN].insert(std::make_pair(ptr, size));
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

void	trackFree (void *ptr, AllocType aType)
{
	free (ptr)
	trackAlignMem -= allocTable[aType][ptr];
	allocTable[aType].erase(ptr);
	ptr = NULL;
}

void	trackAlloc (void *ptr, size_t size)
{
	if ((ptr = malloc(size)) == NULL)
	{
		printf ("Error allocating %lu bytes of unaligned memory\n", size);
		exit (1);
	}

	allocTable[ALLOC_TRACK].insert(std::make_pair(ptr, size));
	trackAllocMem += size;
}

void	printMemStats	()
{
	printf ("Total allocated aligned   memory %lu\n", trackAlignMem);
	printf ("Total allocated unaligned memory %lu\n", trackAllocMem);
	printf ("\nCurrent pointers in memory:"\n);
	printf ("\tAligned\n");

	std::map<void *, size_t>::iterator it = allocTable[ALLOC_ALIGN].begin();

	while(it != allocTable[ALLOC_ALIGN].end())
	{
		std::cout<<it->first<<" :: "<<it->second<<std::endl;
		it++;
	}

	printf ("\n\tUnaligned\n");
	
	while(it != allocTable[ALLOC_TRACK].end())
	{
		std::cout<<it->first<<" :: "<<it->second<<std::endl;
		it++;
	}
}
