#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.24.2/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/rocmcc_7_2_1_hip/install/__spack_path_placeholder__/__spack_path_placeholder__/__sp/llvm-amdgpu-7.2.1/blt-0.7.2-fvmtvy3r2qezl2d5nwsovivxxi7gc62v;/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/rocmcc_7_2_1_hip/install/__spack_path_placeholder__/__spack_path_placeholder__/__sp/llvm-amdgpu-7.2.1/camp-2026.07.1-i3uorzkfetveu64vdxxubporhnlgtrpt;/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/rocmcc_7_2_1_hip/install/__spack_path_placeholder__/__spack_path_placeholder__/__sp/none-none/compiler-wrapper-1.1.0-5jwq47pydjx26rsxpbvkpax3pce5jsby;/opt/rocm-7.2.1;/opt/rocm-7.2.1;/opt/rocm-7.2.1;/usr/tce/packages/cmake/cmake-3.24.2;/opt/rocm-7.2.1" CACHE STRING "")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH "ON" CACHE STRING "")

set(CMAKE_BUILD_RPATH "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/rocmcc_7_2_1_hip/install/__spack_path_placeholder__/__spack_path_placeholder__/__sp/llvm-amdgpu-7.2.1/raja-develop-bharho6tu4thnlhke6q7ybithtmn4os4/lib;/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/rocmcc_7_2_1_hip/install/__spack_path_placeholder__/__spack_path_placeholder__/__sp/llvm-amdgpu-7.2.1/raja-develop-bharho6tu4thnlhke6q7ybithtmn4os4/lib64;/opt/rocm-7.2.1/lib;/opt/rocm-7.2.1/llvm/lib;/opt/cray/pe/mpich/9.1.0/gtl/lib;/opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13" CACHE STRING "")

set(CMAKE_INSTALL_RPATH "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/rocmcc_7_2_1_hip/install/__spack_path_placeholder__/__spack_path_placeholder__/__sp/llvm-amdgpu-7.2.1/raja-develop-bharho6tu4thnlhke6q7ybithtmn4os4/lib;/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/rocmcc_7_2_1_hip/install/__spack_path_placeholder__/__spack_path_placeholder__/__sp/llvm-amdgpu-7.2.1/raja-develop-bharho6tu4thnlhke6q7ybithtmn4os4/lib64;/opt/rocm-7.2.1/lib;/opt/rocm-7.2.1/llvm/lib;/opt/cray/pe/mpich/9.1.0/gtl/lib;/opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13" CACHE STRING "")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: llvm-amdgpu@7.2.1/t2ruvfyax3xvn7hchiuuxmhrmxylwez6
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/rocmcc_7_2_1_hip/install/__spack_path_placeholder__/__spack_path_placeholder__/__sp/none-none/compiler-wrapper-1.1.0-5jwq47pydjx26rsxpbvkpax3pce5jsby/libexec/spack/rocmcc/amdclang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/rocmcc_7_2_1_hip/install/__spack_path_placeholder__/__spack_path_placeholder__/__sp/none-none/compiler-wrapper-1.1.0-5jwq47pydjx26rsxpbvkpax3pce5jsby/libexec/spack/rocmcc/amdclang++" CACHE PATH "")

  # No Fortran compiler defined in spec
else()

  set(CMAKE_C_COMPILER "/opt/rocm-7.2.1/llvm/bin/amdclang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/opt/rocm-7.2.1/llvm/bin/amdclang++" CACHE PATH "")

  # No Fortran compiler defined in spec
endif()

#------------------------------------------------------------------------------
# Hardware
#------------------------------------------------------------------------------

#------------------------------------------------
# ROCm
#------------------------------------------------

set(ROCM_PATH "/opt/rocm-7.2.1" CACHE PATH "")

set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING "")

set(CMAKE_HIP_COMPILER "/opt/rocm-7.2.1/bin/amdclang++" CACHE FILEPATH "")

#------------------------------------------------
# Package custom hardware settings
#------------------------------------------------

set(ENABLE_OPENMP ON CACHE BOOL "")

set(ENABLE_CUDA OFF CACHE BOOL "")

set(ENABLE_HIP ON CACHE BOOL "")

set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath=/opt/rocm-7.2.1/llvm/lib/" CACHE STRING "")

set(HIP_HIPCC_FLAGS "-std=c++17" CACHE STRING "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(BLT_SOURCE_DIR "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/rocmcc_7_2_1_hip/install/__spack_path_placeholder__/__spack_path_placeholder__/__sp/llvm-amdgpu-7.2.1/blt-0.7.2-fvmtvy3r2qezl2d5nwsovivxxi7gc62v" CACHE PATH "")

set(camp_DIR "/usr/WS1/rajasa/rad-bat/raja/tuolumne/786323/rocmcc_7_2_1_hip/install/__spack_path_placeholder__/__spack_path_placeholder__/__sp/llvm-amdgpu-7.2.1/camp-2026.07.1-i3uorzkfetveu64vdxxubporhnlgtrpt" CACHE PATH "")

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


