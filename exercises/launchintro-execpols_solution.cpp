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
 *  RAJA::Launch execution policies
 *
 *  In this exercise, you will use a variety of nested-loop execution
 *  policies to initalize entries in a three-dimensional tensor. The
 *  goal of the exercise is to gain familiarity with RAJA::Launch
 *  execution policies for various RAJA execution back-ends.
 *
 *  RAJA features you will use:
 *    - `RAJA::Launch` kernel execution template method and exec policies
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

// _raja_tensorinit_seq_start
  using loop_policy_1 = RAJA::LoopPolicy<RAJA::seq_exec>;
  using launch_policy_1 = RAJA::LaunchPolicy<RAJA::seq_launch_t>;

  RAJA::launch<launch_policy_1>
    (RAJA::LaunchParams(), //LaunchParams may be empty when running on the host
    [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

      RAJA::loop<loop_policy_1>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int k) {
         RAJA::loop<loop_policy_1>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int j) {
            RAJA::loop<loop_policy_1>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int i) {

                aView(i, j, k) = c * i * j * k ;

            });
         });
      });
  });
// _raja_tensorinit_seq_end

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
  using omp_policy_2 = RAJA::LoopPolicy<RAJA::omp_for_exec>;
  using loop_policy_2 = RAJA::LoopPolicy<RAJA::seq_exec>;
  using launch_policy_2 = RAJA::LaunchPolicy<RAJA::omp_launch_t>;

  RAJA::launch<launch_policy_2>
    (RAJA::LaunchParams(), //LaunchParams may be empty when running on the host
    [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

      RAJA::loop<omp_policy_2>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int k) {
         RAJA::loop<loop_policy_2>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int j) {
            RAJA::loop<loop_policy_2>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int i) {

                aView(i, j, k) = c * i * j * k ;

            });
         });
      });
  });
// _raja_tensorinit_omp_outer_end

  checkResult(a, a_ref, N_tot);
#endif
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  //
  // Define total thread-block size and size of each block dimension
  //
// _cuda_blockdim_start
  constexpr int block_size = 256;
  constexpr int i_block_sz = 32;
  constexpr int j_block_sz = block_size / i_block_sz;
  constexpr int k_block_sz = 1;

  const int n_blocks_i = RAJA_DIVIDE_CEILING_INT(N,i_block_sz);
  const int n_blocks_j = RAJA_DIVIDE_CEILING_INT(N,j_block_sz);
  const int n_blocks_k = RAJA_DIVIDE_CEILING_INT(N,k_block_sz);
// _cuda_blockdim_end

//----------------------------------------------------------------------------//
// C-style and RAJA CUDA GPU variants.
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA CUDA tensor init...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));

// _raja_tensorinit_cuda_start
  using cuda_teams_z_3 = RAJA::LoopPolicy<RAJA::cuda_block_z_direct>;
  using cuda_global_thread_y_3 = RAJA::LoopPolicy<RAJA::cuda_global_thread_y>;
  using cuda_global_thread_x_3 = RAJA::LoopPolicy<RAJA::cuda_global_thread_x>;

  const bool async_3 = false;
  using launch_policy_3 = RAJA::LaunchPolicy<RAJA::cuda_launch_t<async_3>>;

  RAJA::launch<launch_policy_3>
    (RAJA::LaunchParams(RAJA::Teams(n_blocks_i ,n_blocks_j, n_blocks_k),
                      RAJA::Threads(i_block_sz, j_block_sz, k_block_sz)),
    [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

      RAJA::loop<cuda_teams_z_3>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int k) {
        RAJA::loop<cuda_global_thread_y_3>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int j) {
          RAJA::loop<cuda_global_thread_x_3>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int i) {

            aView(i, j, k) = c * i * j * k ;

          });
        });
      });
  });

// _raja_tensorinit_cuda_end

  checkResult(a, a_ref, N_tot);

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA CUDA tensor init tiled-direct...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));

// _raja_tensorinit_cuda_tiled_direct_start
  using cuda_teams_z_4 = RAJA::LoopPolicy<RAJA::cuda_block_z_direct>;
  using cuda_teams_y_4 = RAJA::LoopPolicy<RAJA::cuda_block_y_direct>;
  using cuda_teams_x_4 = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;

  using cuda_threads_y_4 = RAJA::LoopPolicy<RAJA::cuda_thread_y_direct>;
  using cuda_threads_x_4 = RAJA::LoopPolicy<RAJA::cuda_thread_x_direct>;

  const bool async_4 = false;
  using launch_policy_4 = RAJA::LaunchPolicy<RAJA::cuda_launch_t<async_4>>;

  RAJA::launch<launch_policy_4>
    (RAJA::LaunchParams(RAJA::Teams(n_blocks_i, n_blocks_j, n_blocks_k),
                        RAJA::Threads(i_block_sz, j_block_sz, k_block_sz)),
    [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

      RAJA::loop<cuda_teams_z_4>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int k) {

        RAJA::tile<cuda_teams_y_4>
          (ctx, j_block_sz, RAJA::TypedRangeSegment<int>(0, N), [&] (RAJA::TypedRangeSegment<int> const &j_tile) {

          RAJA::tile<cuda_teams_x_4>
            (ctx, i_block_sz, RAJA::TypedRangeSegment<int>(0, N), [&] (RAJA::TypedRangeSegment<int> const &i_tile) {

            RAJA::loop<cuda_threads_y_4>(ctx, j_tile, [&] (int j) {
                RAJA::loop<cuda_threads_x_4>(ctx, i_tile, [&] (int i) {

                    aView(i, j, k) = c * i * j * k ;

                  });
              });

            });
          });

      });
    });
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

  //
  // Define total thread-block size and size of each block dimension
  //
  constexpr int block_size = 256;
  constexpr int i_block_sz = 32;
  constexpr int j_block_sz = block_size / i_block_sz;
  constexpr int k_block_sz = 1;

  const int n_blocks_i = RAJA_DIVIDE_CEILING_INT(N,i_block_sz);
  const int n_blocks_j = RAJA_DIVIDE_CEILING_INT(N,j_block_sz);
  const int n_blocks_k = RAJA_DIVIDE_CEILING_INT(N,k_block_sz);

//----------------------------------------------------------------------------//
// RAJA HIP GPU variants.
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA HIP tensor init...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));
  double *d_a = memoryManager::allocate_gpu<double>(N_tot);

// _3D_raja_device_view_start
  RAJA::View< double, RAJA::Layout<3, int> > d_aView(d_a, N, N, N);
// _3D_raja_deviceview_end

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_a, a, N_tot * sizeof(double), hipMemcpyHostToDevice);

// _raja_tensorinit_hip_start
  using hip_teams_z_5 = RAJA::LoopPolicy<RAJA::hip_block_z_direct>;
  using hip_global_thread_y_5 = RAJA::LoopPolicy<RAJA::hip_global_thread_y>;
  using hip_global_thread_x_5 = RAJA::LoopPolicy<RAJA::hip_global_thread_x>;

  const bool async_5 = false;
  using launch_policy_5 = RAJA::LaunchPolicy<RAJA::hip_launch_t<async_5>>;

  RAJA::launch<launch_policy_5>
    (RAJA::LaunchParams(RAJA::Teams(n_blocks_i, n_blocks_j, n_blocks_k),
                        RAJA::Threads(i_block_sz, j_block_sz, k_block_sz)),
    [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

       RAJA::loop<hip_teams_z_5>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int k) {
           RAJA::loop<hip_global_thread_y_5>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int j) {
               RAJA::loop<hip_global_thread_x_5>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int i) {

                   d_aView(i, j, k) = c * i * j * k ;

           });
         });
       });

  });
// _raja_tensorinit_hip_end

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, a, d_a, N_tot * sizeof(double), hipMemcpyDeviceToHost);
  checkResult(a, a_ref, N_tot);

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA HIP tensor init tiled-direct...\n";

  // set tensor data to zero to ensure we initializing it correctly.
  std::memset(a, 0, N_tot * sizeof(double));
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_a, a, N_tot * sizeof(double), hipMemcpyHostToDevice);

// _raja_tensorinit_hip_tiled_direct_start
  using hip_teams_z_6 = RAJA::LoopPolicy<RAJA::hip_block_z_direct>;
  using hip_teams_y_6 = RAJA::LoopPolicy<RAJA::hip_block_y_direct>;
  using hip_teams_x_6 = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

  using hip_threads_y_6 = RAJA::LoopPolicy<RAJA::hip_thread_y_direct>;
  using hip_threads_x_6 = RAJA::LoopPolicy<RAJA::hip_thread_x_direct>;

  const bool async_6 = false;
  using launch_policy_6 = RAJA::LaunchPolicy<RAJA::hip_launch_t<async_6>>;

  RAJA::launch<launch_policy_6>
    (RAJA::LaunchParams(RAJA::Teams(n_blocks_i, n_blocks_j, n_blocks_k),
                      RAJA::Threads(i_block_sz, j_block_sz, k_block_sz)),
    [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

      RAJA::loop<hip_teams_z_6>(ctx, RAJA::TypedRangeSegment<int>(0, N), [&] (int k) {

        RAJA::tile<hip_teams_y_6>
          (ctx, j_block_sz, RAJA::TypedRangeSegment<int>(0, N), [&] (RAJA::TypedRangeSegment<int> const &j_tile) {

          RAJA::tile<hip_teams_x_6>
            (ctx, i_block_sz, RAJA::TypedRangeSegment<int>(0, N), [&] (RAJA::TypedRangeSegment<int> const &i_tile) {

            RAJA::loop<hip_threads_y_6>(ctx, j_tile, [&] (int j) {
                RAJA::loop<hip_threads_x_6>(ctx, i_tile, [&] (int i) {

                    d_aView(i, j, k) = c * i * j * k ;

                  });
              });

            });
          });

      });
    });
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
