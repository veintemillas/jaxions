set(OBJS fftCuda.cpp comms.cpp fftCode.cpp parse.cpp readWrite.cpp scalarField.cpp propagator.cpp propXeon.cpp propGpu.cu map.cpp memAlloc.cpp)

include_directories(.)
if(USE_GPU)
	cuda_add_library(Axions STATIC ${OBJS})
else()
	add_library(Axions STATIC ${OBJS})
endif()


#ADD_CUSTOM_COMMAND(TARGET axions POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
