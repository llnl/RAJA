#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.25.2/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/usr/WS1/rajasa/rad-bat/raja/matrix/786321/clang_18_1_8_cuda_13_1_1/install/__spack_path_placeholder__/__spack_path_placeholder_/llvm-18.1.8/blt-0.7.2-loafhbr4vhol4yzdgowheh3g5uxxscnj;/usr/WS1/rajasa/rad-bat/raja/matrix/786321/clang_18_1_8_cuda_13_1_1/install/__spack_path_placeholder__/__spack_path_placeholder_/llvm-18.1.8/camp-2026.07.1-b2nlhqltemt4tbcyo6wmkmwlgl6g7kdg;/usr/WS1/rajasa/rad-bat/raja/matrix/786321/clang_18_1_8_cuda_13_1_1/install/__spack_path_placeholder__/__spack_path_placeholder_/none-none/compiler-wrapper-1.1.0-pbd2ayub7z6lkuzpvgsnzfsycl3xbuy3;/usr/tce/packages/cmake/cmake-3.25.2;/usr/tce/packages/cuda/cuda-13.1.1;/usr/tce/packages/clang/clang-18.1.8" CACHE STRING "")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH "ON" CACHE STRING "")

set(CMAKE_BUILD_RPATH "/usr/WS1/rajasa/rad-bat/raja/matrix/786321/clang_18_1_8_cuda_13_1_1/install/__spack_path_placeholder__/__spack_path_placeholder_/llvm-18.1.8/raja-develop-gxinh5vvey6dnbg4msenzrunawe6inej/lib;/usr/WS1/rajasa/rad-bat/raja/matrix/786321/clang_18_1_8_cuda_13_1_1/install/__spack_path_placeholder__/__spack_path_placeholder_/llvm-18.1.8/raja-develop-gxinh5vvey6dnbg4msenzrunawe6inej/lib64;;/usr/tce/backend/installations/linux-rhel8-x86_64/gcc-13.3.1/llvm-18.1.8-wqli5dfxio2tmhdkqs4jxsmtajy7goya/lib/x86_64-unknown-linux-gnu;/opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13" CACHE STRING "")

set(CMAKE_INSTALL_RPATH "/usr/WS1/rajasa/rad-bat/raja/matrix/786321/clang_18_1_8_cuda_13_1_1/install/__spack_path_placeholder__/__spack_path_placeholder_/llvm-18.1.8/raja-develop-gxinh5vvey6dnbg4msenzrunawe6inej/lib;/usr/WS1/rajasa/rad-bat/raja/matrix/786321/clang_18_1_8_cuda_13_1_1/install/__spack_path_placeholder__/__spack_path_placeholder_/llvm-18.1.8/raja-develop-gxinh5vvey6dnbg4msenzrunawe6inej/lib64;;/usr/tce/backend/installations/linux-rhel8-x86_64/gcc-13.3.1/llvm-18.1.8-wqli5dfxio2tmhdkqs4jxsmtajy7goya/lib/x86_64-unknown-linux-gnu;/opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13" CACHE STRING "")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: llvm@18.1.8/ti6swfngqmczqzjji3f5wtzg4gv7dmps
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS1/rajasa/rad-bat/raja/matrix/786321/clang_18_1_8_cuda_13_1_1/install/__spack_path_placeholder__/__spack_path_placeholder_/none-none/compiler-wrapper-1.1.0-pbd2ayub7z6lkuzpvgsnzfsycl3xbuy3/libexec/spack/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS1/rajasa/rad-bat/raja/matrix/786321/clang_18_1_8_cuda_13_1_1/install/__spack_path_placeholder__/__spack_path_placeholder_/none-none/compiler-wrapper-1.1.0-pbd2ayub7z6lkuzpvgsnzfsycl3xbuy3/libexec/spack/clang/clang++" CACHE PATH "")

  # No Fortran compiler defined in spec
else()

  set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-18.1.8/bin/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-18.1.8/bin/clang++" CACHE PATH "")

  # No Fortran compiler defined in spec
endif()

#------------------------------------------------------------------------------
# Hardware
#------------------------------------------------------------------------------

#------------------------------------------------
# Cuda
#------------------------------------------------

set(CUDAToolkit_ROOT "/usr/tce/packages/cuda/cuda-13.1.1" CACHE PATH "")

set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc" CACHE PATH "")

set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-13.1.1" CACHE PATH "")

set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "")

set(CMAKE_CUDA_FLAGS "" CACHE STRING "")

#------------------------------------------------
# Package custom hardware settings
#------------------------------------------------

set(ENABLE_OPENMP OFF CACHE BOOL "")

set(ENABLE_CUDA ON CACHE BOOL "")

set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "")

set(ENABLE_HIP OFF CACHE BOOL "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(BLT_SOURCE_DIR "/usr/WS1/rajasa/rad-bat/raja/matrix/786321/clang_18_1_8_cuda_13_1_1/install/__spack_path_placeholder__/__spack_path_placeholder_/llvm-18.1.8/blt-0.7.2-loafhbr4vhol4yzdgowheh3g5uxxscnj" CACHE PATH "")

set(camp_DIR "/usr/WS1/rajasa/rad-bat/raja/matrix/786321/clang_18_1_8_cuda_13_1_1/install/__spack_path_placeholder__/__spack_path_placeholder_/llvm-18.1.8/camp-2026.07.1-b2nlhqltemt4tbcyo6wmkmwlgl6g7kdg" CACHE PATH "")

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


