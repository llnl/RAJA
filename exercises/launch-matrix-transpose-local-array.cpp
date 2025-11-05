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

#include "RAJA/RAJA.hpp"
#include "memoryManager.hpp"

/*
 *  Matrix Transpose Example
 *
 *  In this example, an input matrix A of dimension N_r x N_c is
 *  transposed and returned as a second matrix At of size N_c x N_r.
 *
 *  This operation is carried out using a local memory tiling
 *  algorithm. The algorithm first loads matrix entries into an
 *  iteraion shared tile, a two-dimensional array, and then
 *  reads from the tile with row and column indices swapped for
 *  the output matrix.
 *
 *  The algorithm is expressed as a collection of ``outer``
 *  and ``inner`` for loops. Iterations of the inner loops will load/read
 *  data into the tile; while outer loops will iterate over the number
 *  of tiles needed to carry out the transpose.
 *
 *  RAJA variants of the example use RAJA_TEAM_SHARED as tile memory.
 *  Furthermore, the tiling pattern is handled by RAJA's tile methods.
 *  For CPU execution, RAJA_TEAM_SHARED are used to improve
 *  performance via cache blocking. For CUDA GPU execution,
 *  RAJA shared memory is mapped to CUDA shared memory which
 *  enables threads in the same thread block to share data.
 *
 *  RAJA features shown:
 *    - Basic usage of 'RAJA::launch' abstractions for nested loops
 *       - tile methods
 *       - loop_icount methods
 *       - RAJA_TEAM_SHARED
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

//
// Define dimensionality of matrices and tile size
//
const int DIM = 2;
#define TILE_DIM (16)  // #define to appease msvc

//
// Function for checking results
//
template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c);

//
// Function for printing results
//
template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA shared matrix transpose example...\n";

  //
  // Define num rows/cols in matrix, tile dimensions, and number of tiles
  //
  // _mattranspose_localarray_dims_start
  constexpr int N_r = 267;
  constexpr int N_c = 251;

  constexpr int outer_Dimc = (N_c - 1) / TILE_DIM + 1;
  constexpr int outer_Dimr = (N_r - 1) / TILE_DIM + 1;
  // _mattranspose_localarray_dims_end

  //
  // Allocate matrix data
  //
  int *A = memoryManager::allocate<int>(N_r * N_c);
  int *At = memoryManager::allocate<int>(N_r * N_c);

  //
  // In the following implementations of matrix transpose, we
  // use RAJA 'View' objects to access the matrix data. A RAJA view
  // holds a pointer to a data array and enables multi-dimensional indexing
  // into the data.
  //
  // _mattranspose_localarray_views_start
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_r, N_c);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_c, N_r);
  // _mattranspose_localarray_views_end

  //
  // Initialize matrix data
  //
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      Aview(row, col) = col;
    }
  }
  // printResult<int>(Aview, N_r, N_c);

  //----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of shared matrix transpose...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  // _mattranspose_localarray_cstyle_start
  //
  // (0) Outer loops to iterate over tiles
  //
  for (int by = 0; by < outer_Dimr; ++by) {
    for (int bx = 0; bx < outer_Dimc; ++bx) {

      // Stack-allocated local array for data on a tile
      int Tile[TILE_DIM][TILE_DIM];

      //
      // (1) Inner loops to read input matrix tile data into the array
      //
      //     Note: loops are ordered so that input matrix data access
      //           is stride-1.
      //
      for (int ty = 0; ty < TILE_DIM; ++ty) {
        for (int tx = 0; tx < TILE_DIM; ++tx) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Tile[ty][tx] = Aview(row, col);
          }
        }
      }

      //
      // (2) Inner loops to write array data into output array tile
      //
      //     Note: loop order is swapped from above so that output matrix
      //           data access is stride-1.
      //
      for (int tx = 0; tx < TILE_DIM; ++tx) {
        for (int ty = 0; ty < TILE_DIM; ++ty) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Atview(col, row) = Tile[ty][tx];
          }
        }
      }

    }
  }
  // _mattranspose_localarray_cstyle_end

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

  //----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA - sequential matrix transpose example ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  // _mattranspose_localarray_raja_start
  using loop_pol_1 = RAJA::LoopPolicy<RAJA::seq_exec>;
  using launch_policy_1 = RAJA::LaunchPolicy<RAJA::seq_launch_t>;

  RAJA::launch<launch_policy_1>(
    RAJA::LaunchParams(), //LaunchParams may be empty when only running on the cpu
    [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

      RAJA::tile<loop_pol_1>(ctx, TILE_DIM, RAJA::TypedRangeSegment<int>(0, N_r), [&] (RAJA::TypedRangeSegment<int> const &row_tile) {

        RAJA::tile<loop_pol_1>(ctx, TILE_DIM, RAJA::TypedRangeSegment<int>(0, N_c), [&] (RAJA::TypedRangeSegment<int> const &col_tile) {

          RAJA_TEAM_SHARED double Tile_Array[TILE_DIM][TILE_DIM];

          ///
          /// TODO ...
          ///
          /// Exercise Implement loop_icount methods to load tiles of the
          /// input matrix into the RAJA_TEAM_SHARED memory array
          ///

          RAJA::loop_icount<loop_pol_1>(ctx, col_tile, [&] (int col, int tx) {
            RAJA::loop_icount<loop_pol_1>(ctx, row_tile, [&] (int row, int ty) {

              Atview(col, row) = Tile_Array[ty][tx];

            });
          });

        });
      });

    });
  // _mattranspose_localarray_raja_end

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

#if defined(RAJA_ENABLE_OPENMP)
  //--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - OpenMP (parallel outer loop) matrix "
               "transpose example ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  //
  // This policy loops over tiles sequentially while exposing parallelism on
  // one of the inner loops.
  //

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an omp_pol_2 type that will distribute loop iterations
  ///           within the omp parallel region.
  ///

  //using loop_pol_2 = RAJA::LoopPolicy<RAJA::seq_exec>;
  using launch_policy_2 = RAJA::LaunchPolicy<RAJA::omp_launch_t>;

  RAJA::launch<launch_policy_2>(
    RAJA::LaunchParams(), //LaunchParams may be empty when only running on the cpu
    [=] RAJA_HOST_DEVICE (RAJA::LaunchContext /*ctx*/) {

      /*
      RAJA::tile<omp_pol_2>(ctx, TILE_DIM, RAJA::TypedRangeSegment<int>(0, N_r), [&] (RAJA::TypedRangeSegment<int> const &row_tile) {
          RAJA::tile<loop_pol_2>(ctx, TILE_DIM, RAJA::TypedRangeSegment<int>(0, N_c), [&] (RAJA::TypedRangeSegment<int> const &col_tile) {

            RAJA_TEAM_SHARED double Tile_Array[TILE_DIM][TILE_DIM];

            RAJA::loop_icount<loop_pol_2>(ctx, row_tile, [&] (int row, int ty) {
              RAJA::loop_icount<loop_pol_2>(ctx, col_tile, [&] (int col, int tx) {

                Tile_Array[ty][tx] = Aview(row, col);

              });
            });

            RAJA::loop_icount<loop_pol_2>(ctx, col_tile, [&] (int col, int tx) {
              RAJA::loop_icount<loop_pol_2>(ctx, row_tile, [&] (int row, int ty) {

                Atview(col, row) = Tile_Array[ty][tx];

                });
              });

          });
        });
    */
    });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);
#endif

  //--------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running RAJA - CUDA matrix transpose example ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  constexpr int c_block_sz = TILE_DIM;
  constexpr int r_block_sz = TILE_DIM;
  const int n_blocks_c = RAJA_DIVIDE_CEILING_INT(N_c, c_block_sz);
  const int n_blocks_r = RAJA_DIVIDE_CEILING_INT(N_r, r_block_sz);

  /// TODO...
  ///
  /// EXERCISE: Define loop policies to mapp loop iterations to blocks, threads directly
  ///

  const bool cuda_async = false;
  using cuda_launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<cuda_async>>;

  RAJA::launch<cuda_launch_policy>(
    RAJA::LaunchParams(RAJA::Teams(n_blocks_c, n_blocks_r),
                     RAJA::Threads(c_block_sz, r_block_sz)),
    [=] RAJA_HOST_DEVICE (RAJA::LaunchContext /*ctx*/) {
      /*
      RAJA::tile<cuda_teams_y>(ctx, TILE_DIM, RAJA::TypedRangeSegment<int>(0, N_r), [&] (RAJA::TypedRangeSegment<int> const &row_tile) {

        RAJA::tile<cuda_teams_x>(ctx, TILE_DIM, RAJA::TypedRangeSegment<int>(0, N_c), [&] (RAJA::TypedRangeSegment<int> const &col_tile) {

          RAJA_TEAM_SHARED double Tile_Array[TILE_DIM][TILE_DIM];

          RAJA::loop_icount<cuda_threads_y>(ctx, row_tile, [&] (int row, int ty) {
            RAJA::loop_icount<cuda_threads_x>(ctx, col_tile, [&] (int col, int tx) {

              Tile_Array[ty][tx] = Aview(row, col);

            });
          });

          RAJA::loop_icount<cuda_threads_x>(ctx, col_tile, [&] (int col, int tx) {
            RAJA::loop_icount<cuda_threads_y>(ctx, row_tile, [&] (int row, int ty) {

              Atview(col, row) = Tile_Array[ty][tx];

            });
          });

        });
      });
      */
  });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);
#endif

//--------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
  //--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - HIP matrix transpose example ...\n";

  int *d_A = memoryManager::allocate_gpu<int>(N_r * N_c);
  int *d_At = memoryManager::allocate_gpu<int>(N_r * N_c);

  //
  // In the following implementations of matrix transpose, we
  // use RAJA 'View' objects to access the matrix data. A RAJA view
  // holds a pointer to a data array and enables multi-dimensional indexing
  // into the data.
  //
  RAJA::View<int, RAJA::Layout<DIM>> d_Aview(d_A, N_r, N_c);
  RAJA::View<int, RAJA::Layout<DIM>> d_Atview(d_At, N_c, N_r);

  std::memset(At, 0, N_r * N_c * sizeof(int));
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_A, A, N_r * N_c * sizeof(int), hipMemcpyHostToDevice);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_At, At, N_r * N_c * sizeof(int), hipMemcpyHostToDevice);

  constexpr int c_block_sz = TILE_DIM;
  constexpr int r_block_sz = TILE_DIM;
  const int n_blocks_c = RAJA_DIVIDE_CEILING_INT(N_c, c_block_sz);
  const int n_blocks_r = RAJA_DIVIDE_CEILING_INT(N_r, r_block_sz);

  using hip_teams_y = RAJA::LoopPolicy<RAJA::hip_block_y_direct>;
  using hip_teams_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

  using hip_threads_y = RAJA::LoopPolicy<RAJA::hip_thread_y_direct>;
  using hip_threads_x = RAJA::LoopPolicy<RAJA::hip_thread_x_direct>;

  const bool hip_async = false;
  using hip_launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<hip_async>>;

  RAJA::launch<hip_launch_policy>
     (RAJA::LaunchParams(RAJA::Teams(n_blocks_c, n_blocks_r),
                     RAJA::Threads(c_block_sz, r_block_sz)),
    [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

      RAJA::tile<hip_teams_y>(ctx, TILE_DIM, RAJA::TypedRangeSegment<int>(0, N_r), [&] (RAJA::TypedRangeSegment<int> const &row_tile) {

        RAJA::tile<hip_teams_x>(ctx, TILE_DIM, RAJA::TypedRangeSegment<int>(0, N_c), [&] (RAJA::TypedRangeSegment<int> const &col_tile) {

          RAJA_TEAM_SHARED double Tile_Array[TILE_DIM][TILE_DIM];

          RAJA::loop_icount<hip_threads_y>(ctx, row_tile, [&] (int row, int ty) {
            RAJA::loop_icount<hip_threads_x>(ctx, col_tile, [&] (int col, int tx) {

              Tile_Array[ty][tx] = d_Aview(row, col);

            });
          });

          RAJA::loop_icount<hip_threads_x>(ctx, col_tile, [&] (int col, int tx) {
            RAJA::loop_icount<hip_threads_y>(ctx, row_tile, [&] (int row, int ty) {

              d_Atview(col, row) = Tile_Array[ty][tx];

            });
          });

        });
      });

    });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, At, d_At, N_r * N_c * sizeof(int), hipMemcpyDeviceToHost);
  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);
#endif

//--------------------------------------------------------------------------//

  return 0;
}


//
// Function to check result and report P/F.
//
template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c)
{
  bool match = true;
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      if (Atview(row, col) != row) {
        match = false;
      }
    }
  }
  if (match) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

//
// Function to print result.
//
template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c)
{
  std::cout << std::endl;
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      std::cout << "At(" << row << "," << col << ") = " << Atview(row, col)
                << std::endl;
    }
    std::cout << "" << std::endl;
  }
  std::cout << std::endl;
}
