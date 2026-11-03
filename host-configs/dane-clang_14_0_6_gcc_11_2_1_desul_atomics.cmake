#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.25.2/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/usr/WS1/rajasa/rad-bat/raja/dane/786322/clang_14_0_6_gcc_11_2_1_desul_atomics/install/__spack_path_placeholder__/__spack_path_p/llvm-14.0.6/blt-0.7.2-m4ibyqnil26mj3gmpjucbyiz4a2kzu5c;/usr/WS1/rajasa/rad-bat/raja/dane/786322/clang_14_0_6_gcc_11_2_1_desul_atomics/install/__spack_path_placeholder__/__spack_path_p/llvm-14.0.6/camp-2026.07.1-wqjuiuzihzdtkvwedvwtpevirhg477p4;/usr/WS1/rajasa/rad-bat/raja/dane/786322/clang_14_0_6_gcc_11_2_1_desul_atomics/install/__spack_path_placeholder__/__spack_path_p/none-none/compiler-wrapper-1.1.0-pbd2ayub7z6lkuzpvgsnzfsycl3xbuy3;/usr/tce/packages/cmake/cmake-3.25.2;/usr/tce/packages/clang/clang-14.0.6" CACHE STRING "")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH "ON" CACHE STRING "")

set(CMAKE_BUILD_RPATH "/usr/WS1/rajasa/rad-bat/raja/dane/786322/clang_14_0_6_gcc_11_2_1_desul_atomics/install/__spack_path_placeholder__/__spack_path_p/llvm-14.0.6/raja-develop-uzmemvjwyq26drrsl6gzoc33n3yc375l/lib;/usr/WS1/rajasa/rad-bat/raja/dane/786322/clang_14_0_6_gcc_11_2_1_desul_atomics/install/__spack_path_placeholder__/__spack_path_p/llvm-14.0.6/raja-develop-uzmemvjwyq26drrsl6gzoc33n3yc375l/lib64;;/opt/rh/gcc-toolset-10/root/usr/lib/gcc/x86_64-redhat-linux/10" CACHE STRING "")

set(CMAKE_INSTALL_RPATH "/usr/WS1/rajasa/rad-bat/raja/dane/786322/clang_14_0_6_gcc_11_2_1_desul_atomics/install/__spack_path_placeholder__/__spack_path_p/llvm-14.0.6/raja-develop-uzmemvjwyq26drrsl6gzoc33n3yc375l/lib;/usr/WS1/rajasa/rad-bat/raja/dane/786322/clang_14_0_6_gcc_11_2_1_desul_atomics/install/__spack_path_placeholder__/__spack_path_p/llvm-14.0.6/raja-develop-uzmemvjwyq26drrsl6gzoc33n3yc375l/lib64;;/opt/rh/gcc-toolset-10/root/usr/lib/gcc/x86_64-redhat-linux/10" CACHE STRING "")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: llvm@14.0.6/6xln4fa5ivmbd5p6t2qaxvfoaigbo4tn
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS1/rajasa/rad-bat/raja/dane/786322/clang_14_0_6_gcc_11_2_1_desul_atomics/install/__spack_path_placeholder__/__spack_path_p/none-none/compiler-wrapper-1.1.0-pbd2ayub7z6lkuzpvgsnzfsycl3xbuy3/libexec/spack/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS1/rajasa/rad-bat/raja/dane/786322/clang_14_0_6_gcc_11_2_1_desul_atomics/install/__spack_path_placeholder__/__spack_path_p/none-none/compiler-wrapper-1.1.0-pbd2ayub7z6lkuzpvgsnzfsycl3xbuy3/libexec/spack/clang/clang++" CACHE PATH "")

  # No Fortran compiler defined in spec
else()

  set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-14.0.6/bin/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-14.0.6/bin/clang++" CACHE PATH "")

  # No Fortran compiler defined in spec
endif()

set(CMAKE_C_FLAGS "--gcc-toolchain=/usr/tce/packages/gcc/gcc-11.2.1" CACHE STRING "")

set(CMAKE_CXX_FLAGS "--gcc-toolchain=/usr/tce/packages/gcc/gcc-11.2.1" CACHE STRING "")

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

set(BLT_SOURCE_DIR "/usr/WS1/rajasa/rad-bat/raja/dane/786322/clang_14_0_6_gcc_11_2_1_desul_atomics/install/__spack_path_placeholder__/__spack_path_p/llvm-14.0.6/blt-0.7.2-m4ibyqnil26mj3gmpjucbyiz4a2kzu5c" CACHE PATH "")

set(camp_DIR "/usr/WS1/rajasa/rad-bat/raja/dane/786322/clang_14_0_6_gcc_11_2_1_desul_atomics/install/__spack_path_placeholder__/__spack_path_p/llvm-14.0.6/camp-2026.07.1-wqjuiuzihzdtkvwedvwtpevirhg477p4" CACHE PATH "")

#------------------------------------------------------------------------------
# Build Options
#------------------------------------------------------------------------------

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "")

set(RAJA_ENABLE_DESUL_ATOMICS ON CACHE BOOL "")

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


