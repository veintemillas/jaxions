set(OBJS fft/fftCuda.cpp comms/comms.cpp fft/fftCode.cpp utils/parse.cpp io/readWrite.cpp scalar/scalarField.cpp propagator/propagator.cpp propagator/propXeon.cpp propagator/propXeonV2.cpp
propagator/propGpu.cu propagator/propSimple.cpp propagator/propThetaXeon.cpp propagator/propThetaGpu.cu propagator/propTheta.cpp energy/energy.cpp energy/energyXeon.cpp
energy/energyXeonV2.cpp energy/energyGpu.cu energy/energyThetaGpu.cu strings/strings.cpp strings/stringCpu.cpp strings/stringXeon.cpp strings/stringGpu.cu map/maps.cpp utils/memAlloc.cpp
powerCpusimple.cpp gen/genConf.cpp gen/momConf.cpp gen/randXeon.cpp gen/smoothXeon.cpp gen/smoothKernel.cu scalar/scaleField.cpp scalar/scaleXeon.cpp scalar/scaleGpu.cu scalar/normField.cpp
scalar/normXeon.cpp scalar/normGpu.cu scalar/normCore.cpp scalar/normCoreXeon.cpp scalar/normCoreGpu.cu energy/energyMap.cpp energy/energyMapXeon.cpp energy/energyThetaXeon.cpp
energy/energyMapThetaXeon.cpp scalar/thetaGpu.cu scalar/thetaXeon.cpp scalar/thetaScalar.cpp scalar/folder.cpp scalar/varNQCD.cpp WKB/WKB.cpp)

include_directories(.)
if(USE_GPU)
	cuda_add_library(Axions STATIC ${OBJS})
else()
	add_library(Axions STATIC ${OBJS})
endif()


#ADD_CUSTOM_COMMAND(TARGET axions POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
