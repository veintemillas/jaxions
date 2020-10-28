# basic setup for cmake
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_CXX_STANDARD_REQUIRED 14)
set(CMAKE_CXX_STANDARD 14)

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
set(USE_AVX OFF CACHE BOOL "builds Avx support")
set(USE_AVX2 OFF CACHE BOOL "builds Avx2 support")
set(USE_AVX512 OFF CACHE BOOL "builds Avx-512 support")
set(USE_FMA OFF CACHE BOOL "builds Fma support")
set(HDF5_PARALLEL_COMPRESSION ON CACHE BOOL "uses experimental parallel compression, requires Hdf5 1.11.0")
set(USE_NYX_OUTPUT OFF CACHE BOOL "To produce Nyx-compatible files")
#set(USE_2D OFF CACHE BOOL "Uses only the Z=1 plane")


# do all the build definitions
#

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
#find_package(HDF5 REQUIRED)
#find_package (HDF5 NAMES hdf5 COMPONENTS C STATIC)
include_directories(SYSTEM ${HDF5_INCLUDE_DIR})


find_package(OpenMP REQUIRED)

find_library(FFTW_LIBRARY      NAMES fftw3 fftw           PATHS $FFTW_LIB)
find_library(FFTW_LIBRARY      NAMES fftw3 fftw           PATHS $FFTW_LIB)
find_library(FFTW_OMP_LIBRARY  NAMES fftw3_omp fftw_omp   PATHS $FFTW_LIB)
find_library(FFTW_MPI_LIBRARY  NAMES fftw3_mpi fftw_mpi   PATHS $FFTW_LIB)
find_library(FFTWF_LIBRARY     NAMES fftw3f fftwf         PATHS $FFTW_LIB)
find_library(FFTWF_OMP_LIBRARY NAMES fftw3f_omp fftwf_omp PATHS $FFTW_LIB)
find_library(FFTWF_MPI_LIBRARY NAMES fftw3f_mpi fftwf_mpi PATHS $FFTW_LIB)
find_library(HDF5_LIBRARY      NAMES hdf5                 PATHS $HDF5_LIB)
find_library(GSL_LIBRARY       NAMES gsl                  PATHS $GSL_HOME)
find_library(GSL_CBLAS_LIBRARY NAMES gslcblas             PATHS $GSL_HOME)

if(${USE_GPU})
  add_definitions(-DUSE_GPU)
  set(GPU_ARCH sm_35 CACHE STRING "set the GPU architecture (sm_30, sm_35, sm_50, sm_52, sm_60, sm_61, sm_62, sm_70)")
  set(GPU_COMP compute_35 CACHE STRING "set the GPU compute capability (compute_30, compute_35, compute_50, compute_52, compute_60, compute_61, compute_62, compute_70)")
  set(DEFARCH "350")
  find_package(CUDA REQUIRED)
  #set(CUDA_HOST_COMPILER gcc)
  set(CUDA_PROPAGATE_HOST_FLAGS ON)
  set(CUDA_NVCC_FLAGS "-arch=${GPU_COMP} -code=${GPU_ARCH} -ftz=true -prec-div=false -prec-sqrt=false -std=c++14 --expt-relaxed-constexpr")

  set(CUDA_NVCC_FLAGS_DEVEL "-O3 -lineinfo" CACHE STRING
      "Flags used by the CUDA compiler during regular development builds." FORCE )
  set(CUDA_NVCC_FLAGS_RELEASE "-O3 -w" CACHE STRING
    "Flags used by the C++ compiler during release builds." FORCE )
  set(CUDA_NVCC_FLAGS_HOSTDEBUG "-g -G" CACHE STRING
      "Flags used by the C++ compiler during debug builds." FORCE )
endif()

#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
#  # using Visual Studio C++
if(${USE_FMA})
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(FMA_FLAGS "-mfma")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(FMA_FLAGS "-fma")
  endif()
endif()

if(${USE_AVX512})
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(AVX_FLAGS "-mavx512f")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(AVX_FLAGS "-xcommon-avx512")
  endif()
elseif(${USE_AVX2})
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(AVX_FLAGS "-mavx2")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(AVX_FLAGS "-xcore-avx2")
  endif()
elseif(${USE_AVX})
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(AVX_FLAGS "-mavx")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(AVX_FLAGS "-xavx")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(SSE_FLAGS "-msse4.1")
endif()

# AMReX
if(${USE_NYX_OUTPUT})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{AMReX_DIR}/Tools/CMake")
  find_package(AMReX REQUIRED NAMES "AMReX_" )
  find_library(AMReX_LIBRARY NAMES amrex PATHS "$ENV{AMREX_LIBRARY_HOME}/lib")
  set(NYX_FLAGS "-DUSE_NYX_OUTPUT")
  include_directories("$ENV{AMREX_LIBRARY_HOME}/include")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} ${AVX_FLAGS} ${SSE_FLAGS} ${FMA_FLAGS} ${NYX_FLAGS} -mcmodel=medium -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${AVX_FLAGS} ${SSE_FLAGS} ${FMA_FLAGS} ${NYX_FLAGS} -mcmodel=medium -std=c++14")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wa,-q -lc++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-q -lc++")
endif()

set(CMAKE_CXX_FLAGS_DEVEL "-O3 -Wall" CACHE STRING
    "Flags used by the C++ compiler during regular development builds." FORCE )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -w" CACHE STRING
    "Flags used by the C++ compiler during release builds." FORCE )
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -fno-inline" CACHE STRING
    "Flags used by the C++ compiler during debug builds." FORCE )



include_directories("$ENV{FFTW_INC}")
include_directories("$ENV{HDF5_INC}")
include_directories(include)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})

### add tests and library
add_subdirectory(lib)
add_subdirectory(test)
