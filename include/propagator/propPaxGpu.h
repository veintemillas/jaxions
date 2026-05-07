#ifndef	_PROP_PAX_GPU_
	#define	_PROP_PAX_GPU_

template<KickDriftType kidi>
void propagatePaxGPU(
    void *m,
    void *v,
    PropParms ppar,
    const double dt,
    const size_t Vo,
    const size_t Vf,
    const FieldPrecision precision,
    const size_t xBlock,
    const size_t yBlock,
    const size_t zBlock,
    cudaStream_t stream
);
#endif