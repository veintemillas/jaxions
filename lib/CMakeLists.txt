set(OBJS fft/fftCuda.cpp comms/comms.cpp fft/fftCode.cpp utils/parse.cpp io/readWrite.cpp scalar/scalarField.cpp propagator/propagator.cpp
propagator/propXeon.cpp propagator/propGpu.cu energy/energy.cpp energy/energyXeon.cpp energy/energyGpu.cu strings/strings.cpp
strings/stringCpu.cpp strings/stringXeon.cpp strings/stringGpu.cu propagator/propSimple.cpp map/map.cpp utils/memAlloc.cpp
powerCpusimple.cpp gen/genConf.cpp gen/momConf.cpp gen/randXeon.cpp gen/smoothXeon.cpp scalar/scaleField.cpp scalar/scaleXeon.cpp
scalar/normField.cpp scalar/normXeon.cpp scalar/normCore.cpp scalar/normCoreXeon.cpp energy/energyMap.cpp energy/energyMapXeon.cpp)

include_directories(.)
if(USE_GPU)
	cuda_add_library(Axions STATIC ${OBJS})
else()
	add_library(Axions STATIC ${OBJS})
endif()


#ADD_CUSTOM_COMMAND(TARGET axions POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
