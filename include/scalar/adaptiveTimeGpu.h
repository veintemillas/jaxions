#ifndef ADAPTIVE_TIME_GPU_H
#define ADAPTIVE_TIME_GPU_H

#include <cstddef>

void adaptiveLineMaxGpu(const void *field, const void *velocity, size_t length,
                        bool singlePrecision, void *stream,
                        double &phi2Max, double &velocityMax);

#endif
