#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.25.2/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/usr/WS1/rajasa/rad-bat/raja/corona/786324/clang_22_0_0_sycl_gcc_10_3_1_rocmcc_6_4_3/install/__spack_path_placeholder__/__spack_/llvm-22.0.0git/blt-0.7.2-5iofca2ohlymkjoaetev5dw53jc2qfo4;/usr/WS1/rajasa/rad-bat/raja/corona/786324/clang_22_0_0_sycl_gcc_10_3_1_rocmcc_6_4_3/install/__spack_path_placeholder__/__spack_/llvm-22.0.0git/camp-2026.07.1-tfh7tiirjkbazabp7nbr4tcremxuy6yk;/usr/WS1/rajasa/rad-bat/raja/corona/786324/clang_22_0_0_sycl_gcc_10_3_1_rocmcc_6_4_3/install/__spack_path_placeholder__/__spack_/none-none/compiler-wrapper-1.1.0-5jwq47pydjx26rsxpbvkpax3pce5jsby;/usr/tce/packages/cmake/cmake-3.25.2;/usr/workspace/raja-dev/clang_sycl_16b7bcb09915_hip_gcc10.3.1_rocm6.4.3" CACHE STRING "")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH "ON" CACHE STRING "")

set(CMAKE_BUILD_RPATH "/usr/WS1/rajasa/rad-bat/raja/corona/786324/clang_22_0_0_sycl_gcc_10_3_1_rocmcc_6_4_3/install/__spack_path_placeholder__/__spack_/llvm-22.0.0git/raja-develop-t6t4kydgn6qgssfnyzt5wrdm4p6zjs6y/lib;/usr/WS1/rajasa/rad-bat/raja/corona/786324/clang_22_0_0_sycl_gcc_10_3_1_rocmcc_6_4_3/install/__spack_path_placeholder__/__spack_/llvm-22.0.0git/raja-develop-t6t4kydgn6qgssfnyzt5wrdm4p6zjs6y/lib64;/usr/workspace/raja-dev/clang_sycl_16b7bcb09915_hip_gcc10.3.1_rocm6.4.3/lib;/usr/workspace/raja-dev/clang_sycl_16b7bcb09915_hip_gcc10.3.1_rocm6.4.3/lib64;/opt/rh/gcc-toolset-12/root/usr/lib/gcc/x86_64-redhat-linux/12" CACHE STRING "")

set(CMAKE_INSTALL_RPATH "/usr/WS1/rajasa/rad-bat/raja/corona/786324/clang_22_0_0_sycl_gcc_10_3_1_rocmcc_6_4_3/install/__spack_path_placeholder__/__spack_/llvm-22.0.0git/raja-develop-t6t4kydgn6qgssfnyzt5wrdm4p6zjs6y/lib;/usr/WS1/rajasa/rad-bat/raja/corona/786324/clang_22_0_0_sycl_gcc_10_3_1_rocmcc_6_4_3/install/__spack_path_placeholder__/__spack_/llvm-22.0.0git/raja-develop-t6t4kydgn6qgssfnyzt5wrdm4p6zjs6y/lib64;/usr/workspace/raja-dev/clang_sycl_16b7bcb09915_hip_gcc10.3.1_rocm6.4.3/lib;/usr/workspace/raja-dev/clang_sycl_16b7bcb09915_hip_gcc10.3.1_rocm6.4.3/lib64;/opt/rh/gcc-toolset-12/root/usr/lib/gcc/x86_64-redhat-linux/12" CACHE STRING "")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: llvm@22.0.0git/uyclvnmnuwzdx4ukqdw2bl6taly3bq4y
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS1/rajasa/rad-bat/raja/corona/786324/clang_22_0_0_sycl_gcc_10_3_1_rocmcc_6_4_3/install/__spack_path_placeholder__/__spack_/none-none/compiler-wrapper-1.1.0-5jwq47pydjx26rsxpbvkpax3pce5jsby/libexec/spack/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS1/rajasa/rad-bat/raja/corona/786324/clang_22_0_0_sycl_gcc_10_3_1_rocmcc_6_4_3/install/__spack_path_placeholder__/__spack_/none-none/compiler-wrapper-1.1.0-5jwq47pydjx26rsxpbvkpax3pce5jsby/libexec/spack/clang/clang++" CACHE PATH "")

  # No Fortran compiler defined in spec
else()

  set(CMAKE_C_COMPILER "/usr/workspace/raja-dev/clang_sycl_16b7bcb09915_hip_gcc10.3.1_rocm6.4.3/bin/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/workspace/raja-dev/clang_sycl_16b7bcb09915_hip_gcc10.3.1_rocm6.4.3/bin/clang++" CACHE PATH "")

  # No Fortran compiler defined in spec
endif()

set(CMAKE_CXX_FLAGS "-w -fsycl -fsycl-unnamed-lambda -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906" CACHE STRING "")

#------------------------------------------------------------------------------
# Hardware
#------------------------------------------------------------------------------

#------------------------------------------------
# Package custom hardware settings
#------------------------------------------------

set(ENABLE_OPENMP OFF CACHE BOOL "")

set(ENABLE_CUDA OFF CACHE BOOL "")

set(ENABLE_HIP OFF CACHE BOOL "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(BLT_SOURCE_DIR "/usr/WS1/rajasa/rad-bat/raja/corona/786324/clang_22_0_0_sycl_gcc_10_3_1_rocmcc_6_4_3/install/__spack_path_placeholder__/__spack_/llvm-22.0.0git/blt-0.7.2-5iofca2ohlymkjoaetev5dw53jc2qfo4" CACHE PATH "")

set(camp_DIR "/usr/WS1/rajasa/rad-bat/raja/corona/786324/clang_22_0_0_sycl_gcc_10_3_1_rocmcc_6_4_3/install/__spack_path_placeholder__/__spack_/llvm-22.0.0git/camp-2026.07.1-tfh7tiirjkbazabp7nbr4tcremxuy6yk" CACHE PATH "")

#------------------------------------------------------------------------------
# Build Options
#------------------------------------------------------------------------------

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "")

set(RAJA_ENABLE_DESUL_ATOMICS OFF CACHE BOOL "")

set(RAJA_ENABLE_VECTORIZATION ON CACHE BOOL "")

set(RAJA_ENABLE_OPENMP_TASK OFF CACHE BOOL "")

set(RAJA_ENABLE_TARGET_OPENMP OFF CACHE BOOL "")

set(RAJA_ENABLE_SYCL ON CACHE BOOL "")

set(RAJA_ENABLE_NV_TOOLS_EXT OFF CACHE BOOL "")

set(RAJA_ENABLE_ROCTX OFF CACHE BOOL "")

set(BLT_CXX_STD "c++20" CACHE STRING "")

set(RAJA_ENABLE_RUNTIME_PLUGINS OFF CACHE BOOL "")

set(RAJA_ENABLE_EXAMPLES ON CACHE BOOL "")

set(RAJA_ENABLE_EXERCISES ON CACHE BOOL "")

set(ENABLE_TESTS ON CACHE BOOL "")

set(RAJA_HOST_CONFIG_LOADED ON CACHE BOOL "")


