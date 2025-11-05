###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0" CACHE STRING "")

if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
    message(FATAL_ERROR "RAJA requires GCC 4.9 or greater!")
  endif ()
  if (RAJA_ENABLE_COVERAGE)
    if(NOT RAJA_ENABLE_CUDA)
      message(INFO "Coverage analysis enabled")
      set(CMAKE_CXX_FLAGS "-coverage ${CMAKE_CXX_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS "-coverage ${CMAKE_EXE_LINKER_FLAGS}")
    endif()
  endif()
endif()

set(RAJA_COMPILER "RAJA_COMPILER_${CMAKE_CXX_COMPILER_ID}")

if ( MSVC )
  if (NOT BUILD_SHARED_LIBS)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
endif()

if (RAJA_ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -restrict --expt-extended-lambda --expt-relaxed-constexpr -Xcudafe \"--display_error_number\"")

  if (NOT RAJA_HOST_CONFIG_LOADED)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Os")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo -O2")

    if(RAJA_ENABLE_COVERAGE)
      if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
        message(INFO "Coverage analysis enabled")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -coverage -Xlinker -coverage")
        set(CMAKE_EXE_LINKER_FLAGS "-coverage ${CMAKE_EXE_LINKER_FLAGS}")
      else()
        message(WARNING "Code coverage specified but not enabled -- GCC was not detected")
      endif()
    endif()
  endif()
endif()
# end RAJA_ENABLE_CUDA section

if (RAJA_ENABLE_HIP)
  set(RAJA_HIPCC_STD_FLAG -std=c++17)
  if (NOT RAJA_HOST_CONFIG_LOADED)
    #list(APPEND RAJA_EXTRA_HIPCC_FLAGS)

    set(RAJA_HIPCC_FLAGS_RELEASE -O2 CACHE STRING "")
    set(RAJA_HIPCC_FLAGS_DEBUG -g; -O0 CACHE STRING "")
    set(RAJA_HIPCC_FLAGS_MINSIZEREL -Os CACHE STRING "")
    set(RAJA_HIPCC_FLAGS_RELWITHDEBINFO -g; -O2 CACHE STRING "")

    if(RAJA_ENABLE_COVERAGE)
      set(RAJA_EXTRA_HIPCC_FLAGS ${RAJA_EXTRA_HIPCC_FLAGS}; -fcoverage-mapping)
      set(CMAKE_EXE_LINKER_FLAGS "-fcoverage-mapping ${CMAKE_EXE_LINKER_FLAGS}")
    endif()
  endif()
  set(RAJA_HIPCC_FLAGS ${RAJA_EXTRA_HIPCC_FLAGS} CACHE STRING "")
  set(HIP_HIPCC_FLAGS ${RAJA_HIPCC_STD_FLAG} ${RAJA_HIPCC_FLAGS})
  set(HIP_HIPCC_FLAGS_RELEASE ${RAJA_HIPCC_STD_FLAG} ${RAJA_HIPCC_FLAGS_RELEASE})
  set(HIP_HIPCC_FLAGS_DEBUG ${RAJA_HIPCC_STD_FLAG} ${RAJA_HIPCC_FLAGS_DEBUG})
  set(HIP_HIPCC_FLAGS_MINSIZEREL ${RAJA_HIPCC_STD_FLAG} ${RAJA_HIPCC_FLAGS_MINSIZEREL})
  set(HIP_HIPCC_FLAGS_RELWITHDEBINFO ${RAJA_HIPCC_STD_FLAG} ${RAJA_HIPCC_FLAGS_RELWITHDEBINFO})
endif()
# end RAJA_ENABLE_HIP section

set(RAJA_RANGE_ALIGN 4 CACHE STRING "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE STRING "")
set(RAJA_DATA_ALIGN 64 CACHE STRING "")
