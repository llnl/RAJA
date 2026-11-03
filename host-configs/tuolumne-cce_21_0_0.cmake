#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.24.2/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/cce_21_0_0/install/__spack_path_placeholder__/__spack_path_placeholder__/__spack_pa/cce-21.0.0/blt-0.7.2-ylhwajnckvu5k2lxv5a6z4uttflgh5nx;/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/cce_21_0_0/install/__spack_path_placeholder__/__spack_path_placeholder__/__spack_pa/cce-21.0.0/camp-2026.07.1-2nhezjn23q3ghjw3ujjaxmxed6ywtoy5;/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/cce_21_0_0/install/__spack_path_placeholder__/__spack_path_placeholder__/__spack_pa/none-none/compiler-wrapper-1.1.0-5jwq47pydjx26rsxpbvkpax3pce5jsby;/usr/tce/packages/cce-tce/cce-21.0.0;/usr/tce/packages/cmake/cmake-3.24.2" CACHE STRING "")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH "ON" CACHE STRING "")

set(CMAKE_BUILD_RPATH "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/cce_21_0_0/install/__spack_path_placeholder__/__spack_path_placeholder__/__spack_pa/cce-21.0.0/raja-develop-szvvua5twyijbeafwr7jvsd4tntgb5fu/lib;/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/cce_21_0_0/install/__spack_path_placeholder__/__spack_path_placeholder__/__spack_pa/cce-21.0.0/raja-develop-szvvua5twyijbeafwr7jvsd4tntgb5fu/lib64;/opt/cray/pe/mpich/9.1.0/gtl/lib;/opt/cray/pe/cce/21.0.0/cce/x86_64/lib;/opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13" CACHE STRING "")

set(CMAKE_INSTALL_RPATH "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/cce_21_0_0/install/__spack_path_placeholder__/__spack_path_placeholder__/__spack_pa/cce-21.0.0/raja-develop-szvvua5twyijbeafwr7jvsd4tntgb5fu/lib;/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/cce_21_0_0/install/__spack_path_placeholder__/__spack_path_placeholder__/__spack_pa/cce-21.0.0/raja-develop-szvvua5twyijbeafwr7jvsd4tntgb5fu/lib64;/opt/cray/pe/mpich/9.1.0/gtl/lib;/opt/cray/pe/cce/21.0.0/cce/x86_64/lib;/opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13" CACHE STRING "")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: cce@21.0.0/7awpbq7y7ol23yws6dpvm2p6dl4bgf27
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/cce_21_0_0/install/__spack_path_placeholder__/__spack_path_placeholder__/__spack_pa/none-none/compiler-wrapper-1.1.0-5jwq47pydjx26rsxpbvkpax3pce5jsby/libexec/spack/cce/craycc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/cce_21_0_0/install/__spack_path_placeholder__/__spack_path_placeholder__/__spack_pa/none-none/compiler-wrapper-1.1.0-5jwq47pydjx26rsxpbvkpax3pce5jsby/libexec/spack/cce/case-insensitive/crayCC" CACHE PATH "")

  # No Fortran compiler defined in spec
else()

  set(CMAKE_C_COMPILER "/usr/tce/packages/cce-tce/cce-21.0.0/bin/craycc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/tce/packages/cce-tce/cce-21.0.0/bin/crayCC" CACHE PATH "")

  # No Fortran compiler defined in spec
endif()

set(BLT_EXE_LINKER_FLAGS "${BLT_EXE_LINKER_FLAGS} -Wl,-rpath,/opt/cray/pe/cce/21.0.0/cce-clang/x86_64/lib" CACHE STRING "Adds a missing rpath for libraries associated with the fortran compiler")

#------------------------------------------------------------------------------
# Hardware
#------------------------------------------------------------------------------

#------------------------------------------------
# Package custom hardware settings
#------------------------------------------------

set(ENABLE_OPENMP ON CACHE BOOL "")

set(ENABLE_CUDA OFF CACHE BOOL "")

set(ENABLE_HIP OFF CACHE BOOL "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(BLT_SOURCE_DIR "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/cce_21_0_0/install/__spack_path_placeholder__/__spack_path_placeholder__/__spack_pa/cce-21.0.0/blt-0.7.2-ylhwajnckvu5k2lxv5a6z4uttflgh5nx" CACHE PATH "")

set(camp_DIR "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/cce_21_0_0/install/__spack_path_placeholder__/__spack_path_placeholder__/__spack_pa/cce-21.0.0/camp-2026.07.1-2nhezjn23q3ghjw3ujjaxmxed6ywtoy5" CACHE PATH "")

#------------------------------------------------------------------------------
# Build Options
#------------------------------------------------------------------------------

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "")

set(RAJA_ENABLE_DESUL_ATOMICS OFF CACHE BOOL "")

set(RAJA_ENABLE_VECTORIZATION ON CACHE BOOL "")

set(RAJA_ENABLE_OPENMP_TASK OFF CACHE BOOL "")

set(RAJA_ENABLE_TARGET_OPENMP OFF CACHE BOOL "")

set(RAJA_ENABLE_SYCL OFF CACHE BOOL "")

set(RAJA_ENABLE_NV_TOOLS_EXT OFF CACHE BOOL "")

set(RAJA_ENABLE_ROCTX OFF CACHE BOOL "")

set(BLT_CXX_STD "c++20" CACHE STRING "")

set(RAJA_ENABLE_RUNTIME_PLUGINS OFF CACHE BOOL "")

set(RAJA_ENABLE_EXAMPLES ON CACHE BOOL "")

set(RAJA_ENABLE_EXERCISES ON CACHE BOOL "")

set(ENABLE_TESTS ON CACHE BOOL "")

set(RAJA_HOST_CONFIG_LOADED ON CACHE BOOL "")


