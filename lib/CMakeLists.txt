set(OBJS fft/fftCuda.cpp comms/comms.cpp fft/fftCode.cpp utils/parse.cpp io/readWrite.cpp scalar/scalarField.cpp propagator/propagator.cpp propagator/propXeon.cpp propagator/propGpu.cu
propagator/propSimple.cpp propagator/propThetaXeon.cpp propagator/propThetaGpu.cu energy/energy.cpp energy/energyXeon.cpp energy/energyThetaXeon.cpp energy/energyGpu.cu
energy/energyThetaGpu.cu strings/strings.cpp strings/stringCpu.cpp strings/stringXeon.cpp strings/stringGpu.cu map/maps.cpp utils/memAlloc.cpp powerCpusimple.cpp gen/genConf.cpp
gen/momConf.cpp gen/momGpu.cu gen/randXeon.cpp gen/randGpu.cu gen/smoothXeon.cpp gen/smoothKernel.cu scalar/scaleField.cpp scalar/scaleXeon.cpp scalar/scaleGpu.cu scalar/normField.cpp
scalar/normXeon.cpp scalar/normGpu.cu scalar/normCore.cpp scalar/normCoreXeon.cpp scalar/normCoreGpu.cu scalar/thetaGpu.cu scalar/thetaXeon.cpp scalar/thetaScalar.cpp scalar/folder.cpp
scalar/varNQCD.cpp utils/logger.cpp utils/profiler.cpp utils/misc.cpp WKB/WKB.cpp propagator/sPropagator.cpp propagator/sPropXeon.cpp)

include_directories(.)
if(USE_GPU)
	cuda_add_library(Axions STATIC ${OBJS})
else()
	add_library(Axions STATIC ${OBJS})
endif()


#ADD_CUSTOM_COMMAND(TARGET axions POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
