//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Matrix Multiplication Examples using RAJA Launch
 *
 *  Example computes the product of two square matrices and introduces
 *  RAJA Launch loop capabilities via a sequence of implementations.
 *
 *  RAJA features shown:
 *    - Index range segment
 *    - View abstraction
 *    - Basic usage of 'RAJA Launch' abstractions for nested loops
 *
 *  If CUDA is enabled, CUDA unified memory is used.
 */

/*
 *  Define number of threads in x and y dimensions in a RAJA thread team
 *  or in a CUDA/HIP thread blocks
*/
#define THREAD_SZ 16

/*
 * Define host/device launch policies
 */
using launch_policy = RAJA::LaunchPolicy<
    RAJA::seq_launch_t
#if defined(RAJA_ENABLE_CUDA)
    ,
    RAJA::cuda_launch_t<false>
#endif
#if defined(RAJA_ENABLE_HIP)
    ,
    RAJA::hip_launch_t<false>
#endif
    >;

using loop_policy = RAJA::seq_exec;

#if defined(RAJA_ENABLE_CUDA)
using gpu_block_x_policy = RAJA::cuda_block_x_direct;
using gpu_block_y_policy = RAJA::cuda_block_y_direct;
using gpu_thread_x_policy = RAJA::cuda_thread_x_loop;
using gpu_thread_y_policy = RAJA::cuda_thread_y_loop;
using gpu_global_thread_x_policy = RAJA::cuda_global_thread_x;
using gpu_global_thread_y_policy = RAJA::cuda_global_thread_y;
using gpu_global_thread_xy_policy = RAJA::cuda_global_thread_xy;
#endif

#if defined(RAJA_ENABLE_HIP)
using gpu_block_x_policy = RAJA::hip_block_x_direct;
using gpu_block_y_policy = RAJA::hip_block_y_direct;
using gpu_thread_x_policy = RAJA::hip_thread_x_loop;
using gpu_thread_y_policy = RAJA::hip_thread_y_loop;
using gpu_global_thread_x_policy = RAJA::hip_global_thread_x;
using gpu_global_thread_y_policy = RAJA::hip_global_thread_y;
using gpu_global_thread_xy_policy = RAJA::hip_global_thread_xy;
#endif

/*
  Define RAJA Team/Thread policies, if a device is available add
  a device policy.
*/
using teams_x = RAJA::LoopPolicy<loop_policy
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
                                 , gpu_block_x_policy
#endif
                                >;

using teams_y = RAJA::LoopPolicy<loop_policy
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
                                 , gpu_block_y_policy
#endif
                                >;

using threads_x = RAJA::LoopPolicy<loop_policy
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
                                   , gpu_thread_x_policy
#endif
                                  >;

using threads_y = RAJA::LoopPolicy<loop_policy
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
                                   , gpu_thread_y_policy
#endif
                                  >;

using global_thread_x = RAJA::LoopPolicy<loop_policy
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
                                         , gpu_global_thread_x_policy
#endif
                                        >;

using global_thread_y = RAJA::LoopPolicy<loop_policy
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
                                         , gpu_global_thread_y_policy
#endif
                                        >;

//
// Define dimensionality of matrices.
//
const int DIM = 2;

//
// Define macros to simplify row-col indexing (non-RAJA implementations only)
//
// _matmult_macros_start
#define A(r, c) A[c + N * r]
#define B(r, c) B[c + N * r]
#define C(r, c) C[c + N * r]
// _matmult_macros_end

/*
  Define CUDA/HIP matrix multiplication kernel for comparison to RAJA version
*/
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
__global__ void matMultKernel(int N, double* C, double* A, double* B)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ( row < N && col < N ) {
    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += A(row, k) * B(k, col);
    }

    C(row, col) = dot;
  }
}

__global__ void sharedMatMultKernel(int N, double* C, double* A, double* B)
{

  int Row = blockIdx.y*THREAD_SZ + threadIdx.y;
  int Col = blockIdx.x*THREAD_SZ + threadIdx.x;

  __shared__ double As[THREAD_SZ][THREAD_SZ];
  __shared__ double Bs[THREAD_SZ][THREAD_SZ];
  __shared__ double Cs[THREAD_SZ][THREAD_SZ];

  Cs[threadIdx.y][threadIdx.x] = 0.0;

  for (int k = 0; k < (THREAD_SZ + N - 1)/THREAD_SZ; k++) {

    if ( static_cast<int>(k*THREAD_SZ + threadIdx.x) < N && Row < N )
      As[threadIdx.y][threadIdx.x] = A[Row*N + k*THREAD_SZ + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;

    if ( static_cast<int>(k*THREAD_SZ + threadIdx.y) < N && Col < N)
      Bs[threadIdx.y][threadIdx.x] = B[(k*THREAD_SZ + threadIdx.y)*N + Col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int n = 0; n < THREAD_SZ; ++n)
      Cs[threadIdx.y][threadIdx.x] += As[threadIdx.y][n] * Bs[n][threadIdx.x];

    __syncthreads();
  }

  if (Row < N && Col < N)
    C[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
      (blockIdx.x * blockDim.x)+ threadIdx.x] = Cs[threadIdx.y][threadIdx.x];
}
#endif

//
// Functions for checking results
//
template <typename T>
void checkResult(T *C, int N);

template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N);

//
// Functions for printing results
//
template <typename T>
void printResult(T *C, int N);

template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA matrix multiplication example...\n";

//
// Define num rows/cols in matrix and number of teams based on
// number of threads in a dimension.
//
  const int N = 1000;
  const int NTeams = (N - 1)/THREAD_SZ + 1;

//
// Allocate and initialize matrix data.
//
  double *A = memoryManager::allocate<double>(N * N);
  double *B = memoryManager::allocate<double>(N * N);
  double *C = memoryManager::allocate<double>(N * N);

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      A(row, col) = row;
      B(row, col) = col;
    }
  }

//----------------------------------------------------------------------------//

  std::cout << "\n Running C-version of matrix multiplication...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // _matmult_cstyle_start
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += A(row, k) * B(k, col);
      }
      C(row, col) = dot;

    }
  }
  // _matmult_cstyle_end

  checkResult<double>(C, N);
//printResult<double>(C, N);


//----------------------------------------------------------------------------//

//
// We define RAJA range segments to define the ranges of
// row, column, and dot-product loops for RAJA variants
//
  // _matmult_ranges_start
  RAJA::RangeSegment row_range(0, N);
  RAJA::RangeSegment col_range(0, N);
#if defined(RAJA_ENABLE_CUDA)
  RAJA::RangeSegment dot_range(0, N);
#endif
  // _matmult_ranges_end

//----------------------------------------------------------------------------//

//
// For the RAJA implementations of matrix multiplication, we
// use RAJA 'View' objects to access the matrix data. A RAJA view
// holds a pointer to a data array and enables multi-dimensional indexing
// into that data, similar to the macros we defined above.
//
  // _matmult_views_start
  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Cview(C, N, N);
  // _matmult_views_end

//----------------------------------------------------------------------------//
//
// RAJA Team loops uses a RAJA::launch method to launch a kernel.
// These examples, illustrate the basic interface and mechanics.
//
// This is different than RAJA::forall and so a few points of exmplanation
// are in order:
//
// 1) RAJA Team loops execute inside a RAJA execution space (RAJA::launch)
//    execution is chosen at run time and we support running on the host
//    or device.
//
// 2) RAJA Team loops follows the thread/block programming models of CUDA/HIP
//    and considers programming using a group of threads in which we group into
//    teams. Number of threads and teams are defined inside the Resources struct.
//
// 3) Launch context is used synchronize threads within a team, an example of this
//    is presented further below.
//
// 4) Parallelism is expressed through RAJA loops. Hierarchical parallelism can be
//    expressed by mapping outer loops (up to 3) to gpu blocks (teams) and inner
//    loops to threads in a block (team).
//

  std::cout << "\n Running sequential mat-mult (RAJA-nested)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //As a starting point we demonstrate assigning each dot product
  //to a thread on a two dimensional compute grid. Rows are mapped
  //to threads in the x dimension, while Cols are mapped to threads
  //in the y dimension. On the host this mapping simplifies to executing
  //two for loops.

  // _matmult_basickernel_start
  RAJA::launch<launch_policy>(RAJA::ExecPlace::HOST,
   RAJA::LaunchParams(RAJA::Teams(NTeams,NTeams),
                         RAJA::Threads(THREAD_SZ,THREAD_SZ)),
       [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

   RAJA::loop<global_thread_y>(ctx, col_range, [&] (int col) {
       RAJA::loop<global_thread_x>(ctx, row_range, [&] (int row) {

          double dot = 0.0;
          for (int k = 0; k < N; ++k) {
            dot += Aview(row, k) * Bview(k, col);
          }
          Cview(row, col) = dot;
      });
    });

  });
  // _matmult_basickernel_end

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running OpenMP mat-mult (RAJA-nested - omp outer)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //RAJA Team loops currently only support a pair of policies at a time.
  //Switching between a sequential and OpenMP launch space requires
  //recompiling execution policies. When running exclusively on the host
  //the compute grid may be left uninitialized as loop methods get expanded to
  //standard C style loops.
  using omp_launch_policy = RAJA::LaunchPolicy<RAJA::omp_launch_t>;

  using omp_col_policy0 = RAJA::LoopPolicy<RAJA::omp_for_exec>;

  using omp_row_policy0 = RAJA::LoopPolicy<loop_policy>;

  RAJA::launch<omp_launch_policy>(RAJA::LaunchParams(),
       [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

   RAJA::loop<omp_col_policy0>(ctx, col_range, [&] (int col) {
       RAJA::loop<omp_row_policy0>(ctx, row_range, [&] (int row) {

          double dot = 0.0;
          for (int k = 0; k < N; ++k) {
            dot += Aview(row, k) * Bview(k, col);
          }
          Cview(row, col) = dot;
      });
    });

  });

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

  std::cout << "\n Running OpenMP mat-mult (RAJA-nested - collapse)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // This example collapses the row and col loops in an OpenMP parallel region.
  // This is the same as using an OpenMP 'parallel for' directive on the
  // outer loop with a 'collapse(2) clause.
  //
  using global_thread_xy = RAJA::LoopPolicy<RAJA::omp_for_exec>;

   RAJA::launch<omp_launch_policy>(RAJA::ExecPlace::HOST,
                                         RAJA::LaunchParams(),
   [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

     RAJA::expt::loop<global_thread_xy>(ctx, col_range, row_range, [&] (int col, int row) {

           double dot = 0.0;
           for (int k = 0; k < N; ++k) {
             dot += Aview(row, k) * Bview(k, col);
           }
           Cview(row, col) = dot;

     });

   });

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif // if RAJA_ENABLE_OPENMP

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running CUDA mat-mult (RAJA-nested)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // This example maps row indicies to RAJA teams (CUDA
  // thread blocks) and col indices are assigned to a threads within
  // each team.
  //
  // This is equivalent to launching a CUDA kernel with grid dimension N
  // and blocksize N; i.e., kernel<<<N, N>>> and defining row = blockIdx.x
  // and col = threadIdx.x in the kernel.
  //
  //
  RAJA::launch<launch_policy>(RAJA::ExecPlace::DEVICE,
    RAJA::LaunchParams(RAJA::Teams(N),
                          RAJA::Threads(N)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

    RAJA::loop<teams_x>(ctx, col_range, [&] (int col) {
        RAJA::loop<threads_x>(ctx, row_range, [&] (int row) {

           double dot = 0.0;
           for (int k = 0; k < N; ++k) {
             dot += Aview(row, k) * Bview(k, col);
           }
           Cview(row, col) = dot;
       });
     });

   });

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA tiled mat-mult ...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // This example takes the extents of the col and row loops and breaks
  // them down into `tiles`. Tile loops are used to generate RangeSegments of
  // fixed size, THREAD_SZ in this case. RAJA loops are then used to iterate
  // across the work within each tile. On the device, tiles are typically assigned
  // to teams, while RAJA loops are mapped to threads.
  //
  // The tiling capabilities in RAJA will also mask out of bounds iterations.
  //
  RAJA::launch<launch_policy>(RAJA::ExecPlace::DEVICE,
    RAJA::LaunchParams(RAJA::Teams(NTeams,NTeams),
                          RAJA::Threads(THREAD_SZ,THREAD_SZ)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

      RAJA::tile<teams_y>
        (ctx, THREAD_SZ, row_range, [&] (RAJA::RangeSegment const &row_tile) {
          RAJA::tile<teams_x>
            (ctx, THREAD_SZ, col_range, [&] (RAJA::RangeSegment const &col_tile) {

              RAJA::loop<threads_y>(ctx, row_tile, [&] (int col) {
                RAJA::loop<threads_x>(ctx, col_tile, [&] (int row) {

                    double dot = 0.0;
                    for (int k = 0; k < N; ++k) {
                      dot += Aview(row, k) * Bview(k, col);
                    }
                    Cview(row, col) = dot;

                  });
                });
            });
        });
   });

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

#endif // if RAJA_ENABLE_CUDA

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)

  double *d_A = memoryManager::allocate_gpu<double>(N * N);
  double *d_B = memoryManager::allocate_gpu<double>(N * N);
  double *d_C = memoryManager::allocate_gpu<double>(N * N);

  std::cout << "\n Running HIP mat-mult (RAJA-nested - POL4)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_A, A, N * N * sizeof(double), hipMemcpyHostToDevice);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_B, B, N * N * sizeof(double), hipMemcpyHostToDevice);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice);

  RAJA::View<double, RAJA::Layout<DIM>> d_Aview(d_A, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> d_Bview(d_B, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> d_Cview(d_C, N, N);

  //
  // This example maps row indicies to RAJA teams (HIP
  // thread blocks) and col indices are assigned to a threads within
  // each team.
  //
  // This is equivalent to launching a HIP kernel with grid dimension N
  // and blocksize N; i.e., kernel<<<N, N>>> and defining row = blockIdx.x
  // and col = threadIdx.x in the kernel.
  //
  //
  RAJA::launch<launch_policy>(RAJA::ExecPlace::DEVICE,
    RAJA::LaunchParams(RAJA::Teams(N),
                          RAJA::Threads(N)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

     RAJA::loop<teams_x>(ctx, col_range, [&] (int col) {
       RAJA::loop<threads_x>(ctx, row_range, [&] (int row) {

            double dot = 0.0;
            for (int k = 0; k < N; ++k) {
              dot += d_Aview(row, k) * d_Bview(k, col);
            }

            d_Cview(row, col) = dot;

        });
     });
  });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost);
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


//----------------------------------------------------------------------------//

  std::cout << "\n Running HIP tiled mat-mult ...\n";

  std::memset(C, 0, N*N * sizeof(double));
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice);

  //
  // This example takes the extents of the col and row loops and breaks
  // them down into `tiles`. Tile loops are used to generate RangeSegments of
  // fixed size, THREAD_SZ in this case. RAJA loops are then used to iterate
  // across the work within each tile. On the device tiles are typically assigned
  // to teams, while RAJA loops are mapped to threads.
  //
  // The tiling capabilities in RAJA will also mask out of bounds iterations.
  //
  RAJA::launch<launch_policy>(RAJA::ExecPlace::DEVICE,
    RAJA::LaunchParams(RAJA::Teams(NTeams,NTeams),
                          RAJA::Threads(THREAD_SZ,THREAD_SZ)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

      RAJA::tile<teams_y>
        (ctx, THREAD_SZ, row_range, [&] (RAJA::RangeSegment const &row_tile) {
          RAJA::tile<teams_x>
            (ctx, THREAD_SZ, col_range, [&] (RAJA::RangeSegment const &col_tile) {

              RAJA::loop<threads_y>(ctx, row_tile, [&] (int col) {
                RAJA::loop<threads_x>(ctx, col_tile, [&] (int row) {

                    double dot = 0.0;
                    for (int k = 0; k < N; ++k) {
                      dot += Aview(row, k) * Bview(k, col);
                    }
                    Cview(row, col) = dot;

                  });
                });
            });
        });
   });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost);
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif // if RAJA_ENABLE_HIP

//----------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running CUDA tiled mat-mult with shared memory ...\n";

  std::memset(C, 0, N*N * sizeof(double));

  using seq_loop =  RAJA::LoopPolicy<RAJA::seq_exec, RAJA::seq_exec>;

  //
  // This example builds on the RAJA tiling capabilies presented earlier
  // and introduced RAJA_TEAM_SHARED memory. Team shared memory is made
  // accessible to all threads within a given thread team.
  //
  // In this example tiles of the global matrix are loaded into shared
  // memory, and the solution is accumulated in a third tile.
  // This example also uses the teamSync() method in the launch context
  // to add a barrier ensuring all threads have loaded/read from shared memory
  //
  RAJA::launch<launch_policy>(RAJA::ExecPlace::DEVICE,
    RAJA::LaunchParams(RAJA::Teams(NTeams,NTeams),
                          RAJA::Threads(THREAD_SZ,THREAD_SZ)),
     [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
   //
   // Loop over teams
   //
   RAJA::tile<teams_y>
     (ctx, THREAD_SZ, row_range, [&] (RAJA::RangeSegment const &y_tile) {
     RAJA::tile<teams_x>
       (ctx, THREAD_SZ, col_range, [&] (RAJA::RangeSegment const &x_tile) {

         RAJA_TEAM_SHARED double As[THREAD_SZ][THREAD_SZ];
         RAJA_TEAM_SHARED double Bs[THREAD_SZ][THREAD_SZ];
         RAJA_TEAM_SHARED double Cs[THREAD_SZ][THREAD_SZ];

         RAJA::loop_icount<threads_y>(ctx, y_tile, [&](int row, int ty) {
             RAJA::loop_icount<threads_x>(ctx, x_tile, [&](int col, int tx) {
               Cs[ty][tx] = 0.0;
             });
         });

         RAJA::tile<seq_loop>
           (ctx, THREAD_SZ, dot_range, [&] (RAJA::RangeSegment const &k_tile) {

           RAJA::loop_icount<threads_y>(ctx, y_tile, [&](int row, int ty) {
               RAJA::loop_icount<threads_x>(ctx, k_tile, [&](int k_id, int tx) {
                   As[ty][tx] = Aview(row,k_id);
                 });
             });

           RAJA::loop_icount<threads_y>(ctx, k_tile, [&](int k_id, int ty) {
               RAJA::loop_icount<threads_x>(ctx, x_tile, [&](int col, int tx) {
                   Bs[ty][tx] = Bview(k_id,col);
               });
             });

           ctx.teamSync();

           RAJA::loop_icount<threads_y>(ctx, y_tile, [&](int row, int ty) {
               RAJA::loop_icount<threads_x>(ctx, x_tile, [&](int col, int tx) {

                   RAJA::loop_icount<seq_loop>(ctx, k_tile, [&] (int gid, int e) {
                       Cs[ty][tx] += As[ty][e] * Bs[e][tx];
                     });

                 });
             });

           ctx.teamSync();

         });  // slide across matrix

          RAJA::loop_icount<threads_y>(ctx, y_tile, [&](int row, int ty) {
               RAJA::loop_icount<threads_x>(ctx, x_tile, [&](int col, int tx) {
                   Cview(col,row) = Cs[ty][tx];
               });
           });
       });
     });
  });  // kernel

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif
//----------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running CUDA tiled mat-mult (no RAJA)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // Define thread block dimensions
  dim3 blockdim(THREAD_SZ, THREAD_SZ);
  // Define grid dimensions to match the RAJA version above
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N,blockdim.x),
               RAJA_DIVIDE_CEILING_INT(N,blockdim.y));

//printf("griddim = (%d,%d), blockdim = (%d,%d)\n", (int)griddim.x, (int)griddim.y, (int)blockdim.x, (int)blockdim.y);

  // Launch CUDA kernel defined near the top of this file.
  matMultKernel<<<griddim, blockdim>>>(N, C, A, B);

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  checkResult<double>(Cview, N);

  std::cout << "\n Running CUDA tiled mat-mult with shared memory (no RAJA)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  sharedMatMultKernel<<<griddim, blockdim>>>(N, C, A, B);

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

#endif // if RAJA_ENABLE_CUDA

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)

  std::cout << "\n Running HIP tiled mat-mult (no RAJA)...\n";

  // Define thread block dimensions
  dim3 blockdim(THREAD_SZ, THREAD_SZ);
  // Define grid dimensions to match the RAJA version above
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N,blockdim.x),
               RAJA_DIVIDE_CEILING_INT(N,blockdim.y));

//printf("griddim = (%d,%d), blockdim = (%d,%d)\n", (int)griddim.x, (int)griddim.y, (int)blockdim.x, (int)blockdim.y);

  std::memset(C, 0, N*N * sizeof(double));
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice);

  // Launch HIP kernel defined near the top of this file.
  hipLaunchKernelGGL((matMultKernel), dim3(griddim), dim3(blockdim), 0, 0, N, d_C, d_A, d_B);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipGetLastError);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost);
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

  std::cout << "\n Running HIP tiled mat-mult with shared memory (no RAJA)...\n";

  std::memset(C, 0, N*N * sizeof(double));
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice);

  // Launch HIP kernel defined near the top of this file.
  hipLaunchKernelGGL((sharedMatMultKernel), dim3(griddim), dim3(blockdim), 0, 0, N, d_C, d_A, d_B);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipGetLastError);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost);
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

  memoryManager::deallocate_gpu(d_A);
  memoryManager::deallocate_gpu(d_B);
  memoryManager::deallocate_gpu(d_C);
#endif // if RAJA_ENABLE_HIP

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Functions to check result and report P/F.
//
template <typename T>
void checkResult(T* C, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if ( std::abs( C(row, col) - row * col * N ) > 10e-12 ) {
        match = false;
      }
    }
  }
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if ( std::abs( Cview(row, col) - row * col * N ) > 10e-12 ) {
        match = false;
      }
    }
  }
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

//
// Functions to print result.
//
template <typename T>
void printResult(T* C, int N)
{
  std::cout << std::endl;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      std::cout << "C(" << row << "," << col << ") = "
                << C(row, col) << std::endl;
    }
  }
  std::cout << std::endl;
}

template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N)
{
  std::cout << std::endl;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      std::cout << "C(" << row << "," << col << ") = "
                << Cview(row, col) << std::endl;
    }
  }
  std::cout << std::endl;
}
