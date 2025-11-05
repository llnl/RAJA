//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "RAJA/RAJA.hpp"

#include "camp/resource.hpp"

#include "memoryManager.hpp"

/*
 *  RAJA::kernel execution policies
 *
 *  In this exercise, you will use a variety of nested-loop execution
 *  policies to initalize entries in a three-dimensional tensor. The
 *  goal of the exercise is to gain familiarity with RAJA::kernel
 *  execution policies for various RAJA execution back-ends.
 *
 *  RAJA features you will use:
 *    - `RAJA::kernel` kernel execution template method and exec policies
 *    - Simple RAJA View/Layout
 *    - RAJA Range segment
 *
 * If CUDA is enabled, CUDA unified memory is used.
 * If HIP is enabled, HIP global device memory is used, with explicit
 * host-device mem copy operations.
 */

#if defined(RAJA_ENABLE_CUDA)
// _cuda_tensorinit_kernel_start
template< int i_block_size, int j_block_size, int k_block_size >
__launch_bounds__(i_block_size*j_block_size*k_block_size)
__global__ void nested_init(double* a, double c, int N)
{
  int i = blockIdx.x * i_block_size + threadIdx.x;
  int j = blockIdx.y * j_block_size + threadIdx.y;
  int k = blockIdx.z;

  if ( i < N && j < N && k < N ) {
    a[i+N*(j+N*k)] = c * i * j * k ;
  }
}
// _cuda_tensorinit_kernel_end
#endif

//
// Function to check result.
//
void checkResult(double* a, double* aref, const int n);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nExercise: RAJA::kernel execution policies tensor init...\n";

// _init_define_start
//
// 3D tensor has N^3 entries
//
  constexpr int N = 100;
  constexpr int N_tot = N * N * N;
  constexpr double c = 0.0001;
  double* a = memoryManager::allocate<double>(N_tot);
  double* a_ref = memoryManager::allocate<double>(N_tot);
// _init_define_end

//----------------------------------------------------------------------------//
// C-style sequential variant establishes reference solution to compare with.
//----------------------------------------------------------------------------//

  std::cout << "\n Running C-style sequential tensor init: create reference solution ...\n";

// _cstyle_tensorinit_seq_start
  for (int k = 0; k < N; ++k ) {
    for (int j = 0; j < N; ++j ) {
      for (int i = 0; i < N; ++i ) {
        a_ref[i+N*(j+N*k)] = c * i * j * k ;
      }
    }
  }
// _cstyle_tensorinit_seq_end


//----------------------------------------------------------------------------//
// We introduce a RAJA View to wrap the tensor data pointer and simplify
// multi-dimensional indexing.
// We use this in the rest of the examples in this file.
//----------------------------------------------------------------------------//

  std::cout << "\n Running C-style sequential tensor init...\n";

// _3D_raja_view_start
  RAJA::View< double, RAJA::Layout<3, int> > aView(a, N, N, N);
// _3D_raja_view_end

// _cstyle_tensorinit_view_seq_start
  for (int k = 0; k < N; ++k ) {
    for (int j = 0; j < N; ++j ) {
      for (int i = 0; i < N; ++i ) {
        aView(i, j, k) = c * i * j * k ;
      }
    }
  }
// _cstyle_tensorinit_view_seq_end

  checkResult(a, a_ref, N_tot);

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential tensor init...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement a sequential RAJA::kernel based version of the
  ///           the tensor initialization kernel. Hint: recall the
  ///           kernelintro-nested-loop-reorder.cpp exercise file used in
  ///           the previous tutorial section.
  ///

  checkResult(a, a_ref, N_tot);

#if defined(RAJA_ENABLE_OPENMP)

//----------------------------------------------------------------------------//
// C-style and RAJA OpenMP multithreading variants.
//----------------------------------------------------------------------------//

  std::cout << "\n Running C-style OpenMP tensor init...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));

  // _cstyle_tensorinit_omp_outer_start
  #pragma omp parallel for
  for (int k = 0; k < N; ++k ) {
    for (int j = 0; j < N; ++j ) {
      for (int i = 0; i < N; ++i ) {
        aView(i, j, k) = c * i * j * k ;
      }
    }
  }
// _cstyle_tensorinit_omp_outer_end

  checkResult(a, a_ref, N_tot);

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA OpenMP tensor init...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));

// _raja_tensorinit_omp_outer_start
  using EXEC_POL2 =
    RAJA::KernelPolicy<
      RAJA::statement::For<2, RAJA::omp_parallel_for_exec,    // k
        RAJA::statement::For<1, RAJA::seq_exec,              // j
          RAJA::statement::For<0, RAJA::seq_exec,            // i
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

  RAJA::kernel<EXEC_POL2>( 
    RAJA::make_tuple( RAJA::TypedRangeSegment<int>(0, N),
                      RAJA::TypedRangeSegment<int>(0, N),
                      RAJA::TypedRangeSegment<int>(0, N) ),

    [=]( int i, int j, int k) {
       aView(i, j, k) = c * i * j * k ;
    }
  );
// _raja_tensorinit_omp_outer_end

  checkResult(a, a_ref, N_tot);

//----------------------------------------------------------------------------//

  std::cout << "\n Running C-style OpenMP collapse (3) tensor init...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));

  // _cstyle_tensorinit_omp_collapse_start
  #pragma omp parallel for collapse(3)
  for (int k = 0; k < N; ++k ) {
    for (int j = 0; j < N; ++j ) {
      for (int i = 0; i < N; ++i ) {
        aView(i, j, k) = c * i * j * k ;
      }
    }
  }
// _cstyle_tensorinit_omp_collapse_end

  checkResult(a, a_ref, N_tot);

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA OpenMP collapse(3) tensor init...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));

// _raja_tensorinit_omp_collapse_start
  using EXEC_POL3 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                RAJA::ArgList<2, 1, 0>,  // k, j, i
        RAJA::statement::Lambda<0>
      >
    >;

  RAJA::kernel<EXEC_POL3>( 
    RAJA::make_tuple( RAJA::TypedRangeSegment<int>(0, N),
                      RAJA::TypedRangeSegment<int>(0, N),
                      RAJA::TypedRangeSegment<int>(0, N) ),

    [=]( int i, int j, int k) {
       aView(i, j, k) = c * i * j * k ;
    }
  );
// _raja_tensorinit_omp_collapse_end

  checkResult(a, a_ref, N_tot);

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA OpenMP collapse(2) tensor init...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an OpenMP RAJA::kernel based version of the
  ///           kernel that collapses the outer two (k, j) loops and
  ///           runs the inner 'i' loop sequentially. Hint: adjust the
  ///           entries in the 'ArgList' above and insert a 'For' statement
  ///           statement to execute the inner loop.
  ///

  checkResult(a, a_ref, N_tot);

#endif // if defined(RAJA_ENABLE_OPENMP)


#if defined(RAJA_ENABLE_CUDA)

//----------------------------------------------------------------------------//
// C-style and RAJA CUDA GPU variants.
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA CUDA tensor init...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));

// _raja_tensorinit_cuda_start
  using EXEC_POL5 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::For<2, RAJA::cuda_thread_z_loop,      // k
          RAJA::statement::For<1, RAJA::cuda_thread_y_loop,    // j
            RAJA::statement::For<0, RAJA::cuda_thread_x_loop,  // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

  RAJA::kernel<EXEC_POL5>(
    RAJA::make_tuple( RAJA::TypedRangeSegment<int>(0, N),
                      RAJA::TypedRangeSegment<int>(0, N),
                      RAJA::TypedRangeSegment<int>(0, N) ),

    [=] __device__ ( int i, int j, int k) {
       aView(i, j, k) = c * i * j * k ;
    }
  );
// _raja_tensorinit_cuda_end

  checkResult(a, a_ref, N_tot);

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA CUDA tensor init tiled-direct...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));

  //
  // Define total thread-block size and size of each block dimension
  //
// _cuda_blockdim_start
  constexpr int block_size = 256;
  constexpr int i_block_sz = 32;
  constexpr int j_block_sz = block_size / i_block_sz;
  constexpr int k_block_sz = 1;
// _cuda_blockdim_end

// _raja_tensorinit_cuda_tiled_direct_start
  using EXEC_POL6 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernelFixed< i_block_sz * j_block_sz * k_block_sz,
        RAJA::statement::Tile<1, RAJA::tile_fixed<j_block_sz>,
                                 RAJA::cuda_block_y_direct,
          RAJA::statement::Tile<0, RAJA::tile_fixed<i_block_sz>,
                                   RAJA::cuda_block_x_direct,
            RAJA::statement::For<2, RAJA::cuda_block_z_direct,      // k
              RAJA::statement::For<1, RAJA::cuda_thread_y_direct,   // j
                RAJA::statement::For<0, RAJA::cuda_thread_x_direct, // i
                  RAJA::statement::Lambda<0>
                >
              >
            >
          >
        >
      >
    >;

  RAJA::kernel<EXEC_POL6>(
    RAJA::make_tuple( RAJA::TypedRangeSegment<int>(0, N),
                      RAJA::TypedRangeSegment<int>(0, N),
                      RAJA::TypedRangeSegment<int>(0, N) ),

    [=] __device__ ( int i, int j, int k) {
       aView(i, j, k) = c * i * j * k ;
    }
  );
// _raja_tensorinit_cuda_tiled_direct_end

  checkResult(a, a_ref, N_tot);

//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA tensor init tiled-direct...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));

// _cuda_tensorinit_tiled_direct_start
  dim3 nthreads_per_block(i_block_sz, j_block_sz, k_block_sz);
  static_assert(i_block_sz*j_block_sz*k_block_sz == block_size,
                "Invalid block_size");

  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N, i_block_sz)),
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N, j_block_sz)),
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N, k_block_sz)));

  nested_init<i_block_sz, j_block_sz, k_block_sz>
    <<<nblocks, nthreads_per_block>>>(a, c, N);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
// _cuda_tensorinit_tiled_direct_end

  checkResult(a, a_ref, N_tot);

#endif // if defined(RAJA_ENABLE_CUDA)


#if defined(RAJA_ENABLE_HIP)

//----------------------------------------------------------------------------//
// RAJA HIP GPU variants.
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA HIP tensor init...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));
  double *d_a = memoryManager::allocate_gpu<double>(N_tot);

// _3D_raja_device_view_start
  RAJA::View< double, RAJA::Layout<3, int> > d_aView(d_a, N, N, N);
// _3D_raja_device_view_end

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_a, a, N_tot * sizeof(double), hipMemcpyHostToDevice);

// _raja_tensorinit_hip_start
  using EXEC_POL7 =
    RAJA::KernelPolicy<
      RAJA::statement::HipKernel<
        RAJA::statement::For<2, RAJA::hip_thread_z_loop,      // k
          RAJA::statement::For<1, RAJA::hip_thread_y_loop,    // j
            RAJA::statement::For<0, RAJA::hip_thread_x_loop,  // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

  RAJA::kernel<EXEC_POL7>(
    RAJA::make_tuple( RAJA::TypedRangeSegment<int>(0, N),
                      RAJA::TypedRangeSegment<int>(0, N),
                      RAJA::TypedRangeSegment<int>(0, N) ),

    [=] __device__ ( int i, int j, int k) {
       d_aView(i, j, k) = c * i * j * k ;
    }
  );
// _raja_tensorinit_hip_end

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, a, d_a, N_tot * sizeof(double), hipMemcpyDeviceToHost);
  checkResult(a, a_ref, N_tot);

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA HIP tensor init tiled-direct...\n";

  //
  // Define total thread-block size and size of each block dimension
  //
  constexpr int block_size = 256;
  constexpr int i_block_sz = 32;
  constexpr int j_block_sz = block_size / i_block_sz;
  constexpr int k_block_sz = 1;

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_a, a, N_tot * sizeof(double), hipMemcpyHostToDevice);

// _raja_tensorinit_hip_tiled_direct_start
  using EXEC_POL8 =
    RAJA::KernelPolicy<
      RAJA::statement::HipKernelFixed< i_block_sz * j_block_sz * k_block_sz,
        RAJA::statement::Tile<1, RAJA::tile_fixed<j_block_sz>,
                                 RAJA::hip_block_y_direct,
          RAJA::statement::Tile<0, RAJA::tile_fixed<i_block_sz>,
                                   RAJA::hip_block_x_direct,
            RAJA::statement::For<2, RAJA::hip_block_z_direct,      // k
              RAJA::statement::For<1, RAJA::hip_thread_y_direct,   // j
                RAJA::statement::For<0, RAJA::hip_thread_x_direct, // i
                  RAJA::statement::Lambda<0>
                >
              >
            >
          >
        >
      >
    >;

  RAJA::kernel<EXEC_POL8>(
    RAJA::make_tuple( RAJA::TypedRangeSegment<int>(0, N),
                      RAJA::TypedRangeSegment<int>(0, N),
                      RAJA::TypedRangeSegment<int>(0, N) ),

    [=] __device__ ( int i, int j, int k) {
       d_aView(i, j, k) = c * i * j * k ;
    }
  );
// _raja_tensorinit_hip_tiled_direct_end

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, a, d_a, N_tot * sizeof(double), hipMemcpyDeviceToHost);
  checkResult(a, a_ref, N_tot);

  memoryManager::deallocate_gpu(d_a);

#endif // if defined(RAJA_ENABLE_HIP)

//----------------------------------------------------------------------------//

  // Clean up...
  memoryManager::deallocate(a);
  memoryManager::deallocate(a_ref);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to compare result to reference and print result P/F.
//
void checkResult(double* a, double* aref, const int n)
{
  bool correct = true;

  int i = 0;
  while ( correct && (i < n) ) {
    correct = std::abs(a[i] - aref[i]) < 10e-12;
    i++;
  }

  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}
