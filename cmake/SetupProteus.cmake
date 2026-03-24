if (NOT CMAKE_CXX_COMPILER_ID MATCHES Clang)
  message(FATAL_ERROR "RAJA requires an LLVM based compiler in order to support JIT compilation, 
  but found ${CMAKE_CXX_COMPILER_ID}")
endif()
# We set the variable PROTEUS_HEADERS_DIR so that we can modify target include directories
# consistently regardless of whether Proteus was brought in via an installation (i.e. spack)
# or via source (submodule or FetchContent)
# If a user specifies an installation directory for Proteus, respect it.  Otherwise, if RAJA_ENABLE_JIT
# is specified, pull in Proteus using FetchContent.
if (DEFINED PROTEUS_INSTALL_DIR)
  find_package(proteus REQUIRED PATHS "${PROTEUS_INSTALL_DIR}")
  set(PROTEUS_HEADERS_DIR "${PROTEUS_INSTALL_DIR}/include" CACHE STRING "")
else()
  set(PROTEUS_ENABLE_HIP  "${RAJA_ENABLE_HIP}" CACHE BOOL "Enable JIT compilation of HIP kernels")
  set(PROTEUS_ENABLE_CUDA "${RAJA_ENABLE_CUDA}" CACHE BOOL "Enable JIT compilation of CUDA kernels")
  # Proteus shadow's RAJA's ENABLE_TESTS variable.  We don't want to build Proteus's tests, so we
  # force it off here, and re-enable it off of RAJA's internal variable below
  set(ENABLE_TESTS OFF)
  include(FetchContent)
  FetchContent_Declare(
    proteus
    GIT_REPOSITORY https://github.com/Olympus-HPC/proteus.git
    GIT_TAG        257707cf7e60452ed38161f6429421be303ddaf3
    )
  FetchContent_MakeAvailable(proteus)
  # Re-enable tests if specified by user.
  set(ENABLE_TESTS RAJA_ENABLE_TESTS)
  include(${proteus_SOURCE_DIR}/cmake/ProteusFunctions.cmake)
  set(PROTEUS_HEADERS_DIR "${proteus_SOURCE_DIR}/include" CACHE STRING "")
endif()
# We don't explicitly link against ProteusPass, but it is required to be
#available as an LLVM pass, so manually enforce order
add_dependencies(RAJA ProteusPass)