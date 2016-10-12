# basic setup for cmake
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_CXX_STANDARD_REQUIRED 11)
set(CMAKE_CXX_STANDARD 11)

set(DEFAULT_BUILD "RELEASE")
set(VALID_BUILDS DEVEL RELEASE DEBUG)
set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD}" CACHE STRING  "Choose the type of build, options are: ${VALID_BUILDS}")
string(TOUPPER "${CMAKE_BUILD_TYPE}" CHECK_BUILD)
LIST(FIND VALID_BUILDS "${CHECK_BUILD}" CURRENT_BUILD)

if(CURRENT_BUILD LESS 0)
  message(SEND_ERROR "Please specify a valid CMAKE_BUILD_TYPE type! Valid build types are: " "${VALID_BUILDS}")
endif()


#
# PROJECT is Axions
#
project("AXIONS")

#######################################################################
# Axions OPTIONS
# likely to be changed by users
#######################################################################

# build options
set(USE_GPU OFF CACHE BOOL "builds nVidia gpu support")
set(USE_XEON OFF CACHE BOOL "builds Xeon Phi support")
set(USE_AVX OFF CACHE BOOL "builds Avx support")
set(USE_AVX2 OFF CACHE BOOL "builds Avx2 support")

# do all the build definitions
#

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
#find_package(HDF5 REQUIRED)
#find_package (HDF5 NAMES hdf5 COMPONENTS C STATIC)
include_directories(SYSTEM ${HDF5_INCLUDE_DIR})

find_package(OpenMP REQUIRED)

find_library(FFTW_LIBRARY NAMES fftw3 fftw)
find_library(FFTW_MPI_LIBRARY NAMES fftw3_mpi fftw_mpi)
find_library(FFTWF_LIBRARY NAMES fftw3f fftwf)
find_library(FFTWF_MPI_LIBRARY NAMES fftw3f_mpi fftwf_mpi)
find_library(HDF5_LIBRARY NAMES hdf5)
if(${USE_GPU})
  add_definitions(-DUSE_GPU)
  set(GPU_ARCH sm_35 CACHE STRING "set the GPU architecture (sm_20, sm_21, sm_30, sm_35, sm_50, sm_52)")
  set(GPU_COMP compute_35 CACHE STRING "set the GPU compute capability (compute_20, compute_21, compute_30, compute_35, compute_50, compute_52)")
#  string(SUBSTRING "${GPU_ARCH}" 3 2 TMPARCH)
#  math(EXPR DEFARCH "${TMPARCH}*10")
  set(DEFARCH "350")
  find_package(CUDA REQUIRED)
  set(CUDA_HOST_COMPILER g++)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_NVCC_FLAGS "-arch=${GPU_COMP} -code=${GPU_ARCH} -ftz=true -prec-div=false -prec-sqrt=false -std=c++11 -DGPU_ARCH=${DEFARCH}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGPU_ARCH=${DEFARCH}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU_ARCH=${DEFARCH}")

  set(CUDA_NVCC_FLAGS_DEVEL "-O3 -lineinfo" CACHE STRING
      "Flags used by the CUDA compiler during regular development builds." FORCE )
  set(CUDA_NVCC_FLAGS_RELEASE "-O3 -w" CACHE STRING
    "Flags used by the C++ compiler during release builds." FORCE )
  set(CUDA_NVCC_FLAGS_HOSTDEBUG "-g -G" CACHE STRING
      "Flags used by the C++ compiler during debug builds." FORCE )
endif()

if(${USE_XEON})
  add_definitions(-DUSE_XEON)
endif()

#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
#  # using Visual Studio C++
if(${USE_AVX2})
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(AVX_FLAGS "-mavx2")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(AVX_FLAGS "-mavx")
  endif()
elseif(${USE_AVX})
#  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(AVX_FLAGS "-mavx")
#  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
#    set(AVX_FLAGS "-mavx")
#  endif()
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} ${AVX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${AVX_FLAGS} -std=c++11")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wa,-q -lc++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-q -lc++")
endif()

set(CMAKE_CXX_FLAGS_DEVEL "-O3 -Wall" CACHE STRING
    "Flags used by the CUDA compiler during regular development builds." FORCE )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -w" CACHE STRING
    "Flags used by the C++ compiler during release builds." FORCE )
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -fno-inline" CACHE STRING
    "Flags used by the C++ compiler during debug builds." FORCE )


include_directories(include)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})

### add tests and library
add_subdirectory(lib)
add_subdirectory(test)

