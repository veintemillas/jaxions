set(OBJS fft/fftCuda.cpp comms/comms.cpp fft/fftCode.cpp utils/parse.cpp io/readWrite.pHdf5.cpp io/output_nyx.cpp scalar/scalarField.cpp propagator/propagator.cpp propagator/propGpu.cu
 propagator/propThetaGpu.cu energy/energy.cpp energy/energyXeon.cpp energy/energyThetaXeon.cpp energy/energyNaxionXeon.cpp energy/energyPaxionXeon.cpp energy/energyGpu.cu energy/energyThetaGpu.cu energy/dContrast.cpp
strings/strings.cpp strings/length.cpp strings/stringCpu.cpp strings/stringXeon.cpp strings/stringGpu.cu utils/memAlloc.cpp gen/genConf.cpp gen/momConf.cpp gen/momGpu.cu gen/randXeon.cpp gen/randGpu.cu
gen/smoothXeon.cpp gen/prepropa.cpp gen/smoothKernel.cu  meas/measa.cpp scalar/scaleField.cpp scalar/scaleXeon.cpp scalar/scaleGpu.cu scalar/theta2Cmplx.cpp scalar/theta2CmplxXeon.cpp scalar/theta2Naxion.cpp scalar/theta2NaxionXeon.cpp
scalar/theta2Paxion.cpp scalar/theta2PaxionXeon.cpp scalar/theta2CmplxGpu.cu axiton/tracker.cpp
scalar/normField.cpp scalar/normXeon.cpp scalar/normGpu.cu scalar/normCore.cpp scalar/normCoreXeon.cpp scalar/normCoreGpu.cu scalar/thetaGpu.cu scalar/thetaXeon.cpp scalar/thetaScalar.cpp scalar/folder.cpp
scalar/mendTheta.cpp scalar/mendThetaXeon.cpp scalar/mendThetaGpu.cu scalar/fourier.cpp utils/logger.cpp utils/profiler.cpp utils/misc.cpp utils/kgvops.cpp projector/projector.cpp propagator/laplacian.cpp reducer/reducer.cpp
spectrum/spectrum.cpp WKB/WKB.cpp cosmos/cosmos.cpp gravity/potential.cpp gadget/gadget_output.cpp)

include_directories(.)
if(USE_GPU)
	cuda_add_library(Axions STATIC ${OBJS})
else()
	add_library(Axions STATIC ${OBJS})
endif()


#ADD_CUSTOM_COMMAND(TARGET axions POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
