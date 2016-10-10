#ifndef	_XEON_DEFS_
	#define	_XEON_DEFS_

	#define	AllocX alloc_if(1) free_if(0)
	#define	ReUseX alloc_if(0) free_if(0)
	#define	FreeX alloc_if(0) free_if(1)
	#define	UseX alloc_if(1) free_if(1)

	extern __declspec(target(mic)) char *mX, *vX, *m2X;
#endif

