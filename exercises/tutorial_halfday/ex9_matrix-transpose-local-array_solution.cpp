//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
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
 *  EXERCISE #9: Matrix Transpose with Local Array
 *
 *  In this exercise, you will use RAJA constructs to transpose a matrix
 *  using a loop tiling algorithm similar to exercise 8. However, this 
 *  exercise is different in that you will use a local array to write
 *  to and read from as each matrix tile is transposed. An input matrix 
 *  A of dimension N_r x N_c is provided. You will fill in the entries 
 *  of the transpose matrix At.
 *
 *  This file contains a C-style variant of the sequential matrix transpose.
 *  You will complete implementations of multiple RAJA variants by filling
 *  in missing elements of RAJA kernel API execution policies as well as the
 *  RAJA kernel implementation for each. Variants you will complete include
 *  sequential, OpenMP, and CUDA execution.
 *
 *  RAJA features shown:
 *    - Basic usage of 'RAJA::kernel' abstractions for nested loops
 *       - Multiple lambdas
 *       - Tile statement
 *       - ForICount statement
 *       - RAJA local arrays
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

//
// Define dimensionality of matrices
//
const int DIM = 2;

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

  std::cout << "\n\nExercise #9: RAJA local array matrix transpose...\n";

  //
  // Define num rows/cols in matrix
  //
  const int N_r = 267;
  const int N_c = 251;

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
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_r, N_c);

  //
  // Construct a permuted layout for At so that the column index has stride 1
  //
  std::array<RAJA::idx_t, 2> perm {{1, 0}};
  RAJA::Layout<2> perm_layout = RAJA::make_permuted_layout( {{N_c, N_r}},
                                                            perm );
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, perm_layout);

  //
  // Define size for each dimension of a square tile.
  //
  const int TILE_SZ = 16;

  // Calculate number of tiles (Needed for C++ version)
  const int outer_Dimc = (N_c - 1) / TILE_SZ + 1;
  const int outer_Dimr = (N_r - 1) / TILE_SZ + 1;

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
  std::cout << "\n Running C-version of local array matrix transpose...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  //
  // (0) Outer loops to iterate over tiles
  //
  for (int brow = 0; brow < outer_Dimr; ++brow) {
    for (int bcol = 0; bcol < outer_Dimc; ++bcol) {

      // Stack-allocated local array for data on a tile
      int Tile[TILE_SZ][TILE_SZ];

      //
      // (1) Inner loops to read input matrix tile data into array
      //
      //     Note: loops are ordered so that input matrix data access
      //           is stride-1.
      //
      for (int trow = 0; trow < TILE_SZ; ++trow) {
        for (int tcol = 0; tcol < TILE_SZ; ++tcol) {

          int col = bcol * TILE_SZ + tcol;  // Matrix column index
          int row = brow * TILE_SZ + trow;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Tile[trow][tcol] = Aview(row, col);
          }
        }
      }

      //
      // (2) Inner loops to write array data into output array tile
      //
      //     Note: loop order is swapped from above so that output matrix
      //           data access is stride-1.
      //
      for (int tcol = 0; tcol < TILE_SZ; ++tcol) {
        for (int trow = 0; trow < TILE_SZ; ++trow) {

          int col = bcol * TILE_SZ + tcol;  // Matrix column index
          int row = brow * TILE_SZ + trow;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Atview(col, row) = Tile[trow][tcol];
          }
        }
      }

    }
  }
  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

//----------------------------------------------------------------------------//

  //
  // The following RAJA variants will use the RAJA::kernel method to
  // perform the matrix transpose operation.
  //
  // Here, we define RAJA range segments to establish the iteration spaces.
  // Further partioning of the iteration space is carried out in the
  // tile_fixed statements. Iterations inside a RAJA loop is given by their
  // global iteration number.
  //
  RAJA::RangeSegment row_Range(0, N_r);
  RAJA::RangeSegment col_Range(0, N_c);

  // Next,  we define a RAJA local array type.
  // The array type is templated on
  // 1) Data type
  // 2) Index permutation
  // 3) Dimensions of the array
  //

  using TILE_MEM =
    RAJA::LocalArray<int, RAJA::Perm<0, 1>, RAJA::SizeList<TILE_SZ, TILE_SZ>>;

  // **NOTE** The LocalArray is created here, but it's memory is not yet
  //          allocated. This is done when the 'InitLocalMem' statement
  //          is reached in the kernel policy using the specified local
  //          memory policy.

  TILE_MEM RAJA_Tile;

//--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - sequential matrix transpose example ...\n";
  std::memset(At, 0, N_r * N_c * sizeof(int));

  using SEQ_EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::tile_fixed<TILE_SZ>,
                               RAJA::seq_exec,
        RAJA::statement::Tile<0, RAJA::tile_fixed<TILE_SZ>,
                                 RAJA::seq_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<2>,

          RAJA::statement::ForICount<1, RAJA::statement::Param<1>, 
                                        RAJA::seq_exec,
            RAJA::statement::ForICount<0, RAJA::statement::Param<0>, 
                                        RAJA::seq_exec,
              RAJA::statement::Lambda<0>
            >
          >,

          RAJA::statement::ForICount<0, RAJA::statement::Param<0>, 
                                        RAJA::seq_exec,
            RAJA::statement::ForICount<1, RAJA::statement::Param<1>, 
                                          RAJA::seq_exec,
              RAJA::statement::Lambda<1>
            >
          >

          >
        >
      >
    >;

  RAJA::kernel_param<SEQ_EXEC_POL>( RAJA::make_tuple(col_Range, row_Range),

    RAJA::make_tuple((int)0, (int)0, RAJA_Tile),

    [=](int col, int row, int tcol, int trow, TILE_MEM& RAJA_Tile) {

      RAJA_Tile(trow, tcol) = Aview(row, col);

    },

    [=](int col, int row, int tcol, int trow, TILE_MEM RAJA_Tile) {

      Atview(col, row) = RAJA_Tile(trow, tcol);

  });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

//--------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running RAJA - OpenMP (parallel outer loop) matrix "
               "transpose example ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  using OPENMP_EXEC_POL =
  RAJA::KernelPolicy<
    RAJA::statement::Tile<1, RAJA::tile_fixed<TILE_SZ>,
                             RAJA::omp_parallel_for_exec,
      RAJA::statement::Tile<0, RAJA::tile_fixed<TILE_SZ>,
                               RAJA::seq_exec,

        RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<2>,

          RAJA::statement::ForICount<1, RAJA::statement::Param<1>, 
                                        RAJA::seq_exec,
            RAJA::statement::ForICount<0, RAJA::statement::Param<0>, 
                                          RAJA::seq_exec,
               RAJA::statement::Lambda<0>
            >
          >,

          RAJA::statement::ForICount<0, RAJA::statement::Param<0>, 
                                        RAJA::seq_exec,
            RAJA::statement::ForICount<1, RAJA::statement::Param<1>, 
                                          RAJA::seq_exec,
              RAJA::statement::Lambda<1>
            >
          >
        >
      >
    >
   >;

  RAJA::kernel_param<OPENMP_EXEC_POL>( RAJA::make_tuple(col_Range, row_Range),

      RAJA::make_tuple((int)0, (int)0, RAJA_Tile),

      [=](int col, int row, int tcol, int trow, TILE_MEM& RAJA_Tile) {

        RAJA_Tile(trow, tcol) = Aview(row, col);

      },

      [=](int col, int row, int tcol, int trow, TILE_MEM RAJA_Tile) {

        Atview(col, row) = RAJA_Tile(trow, tcol);

      });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);
#endif


//--------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running RAJA - CUDA matrix transpose example ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  using CUDA_EXEC_POL =
  RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
      RAJA::statement::Tile<1, RAJA::tile_fixed<TILE_SZ>,
                               RAJA::cuda_block_y_loop,
        RAJA::statement::Tile<0, RAJA::tile_fixed<TILE_SZ>,
                                 RAJA::cuda_block_x_loop,

          RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<2>,

            RAJA::statement::ForICount<1, RAJA::statement::Param<1>, 
                                          RAJA::cuda_thread_y_direct,
              RAJA::statement::ForICount<0, RAJA::statement::Param<0>, 
                                            RAJA::cuda_thread_x_direct,
                RAJA::statement::Lambda<0>
              >
            >,

            RAJA::statement::CudaSyncThreads,

            RAJA::statement::ForICount<0, RAJA::statement::Param<0>, 
                                          RAJA::cuda_thread_y_direct,
              RAJA::statement::ForICount<1, RAJA::statement::Param<1>, 
                                            RAJA::cuda_thread_x_direct,
                RAJA::statement::Lambda<1>
              >
            >,

            RAJA::statement::CudaSyncThreads
          >
        >
      >
    >
  >;


  RAJA::kernel_param<CUDA_EXEC_POL>( RAJA::make_tuple(col_Range, row_Range),

      RAJA::make_tuple((int)0, (int)0, RAJA_Tile),

      [=] RAJA_DEVICE (int col, int row, int tcol, int trow, TILE_MEM& RAJA_Tile) {

        RAJA_Tile(trow, tcol) = Aview(row, col);

      },

      [=] RAJA_DEVICE(int col, int row, int tcol, int trow, TILE_MEM RAJA_Tile) {

        Atview(col, row) = RAJA_Tile(trow, tcol);

      });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);
#endif

  //--------------------------------------------------------------------------//

  //
  // Clean up.
  //
  memoryManager::deallocate(A);
  memoryManager::deallocate(At);

  std::cout << "\n DONE!...\n";

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
        match &= false;
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
