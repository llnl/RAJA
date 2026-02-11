//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"

#include "memoryManager.hpp"

/*
 *  EXERCISE #8: Tiled Matrix Transpose
 *
 *  In this exercise, you will use RAJA constructs to transpose a matrix 
 *  using a loop tiling algorithm. An input matrix A of dimension N_r x N_c
 *  is provided. You will fill in the entries of the transpose matrix At.
 *
 *  This file contains a C-style variant of the sequential matrix transpose.
 *  You will complete implementations of multiple RAJA variants by filling
 *  in missing elements of RAJA kernel API execution policies as well as the 
 *  RAJA kernel implementation for each. Variants you will complete include
 *  sequential, OpenMP, and CUDA execution.
 *
 *  RAJA features you will use:
 *    - Basic usage of 'RAJA::kernel' abstractions for nested loops
 *    - Tiling statement
 *
 * Note: The code for the RAJA execution policies code below are disabled
 *       so that the code can be compiled. As you work each part of the
 *       exercise, you must enable each policy so that the code will run.
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

  std::cout << "\n\nExercise #8: RAJA Tiled Matrix Transpose...\n";

  //
  // Define num rows/cols in matrix
  //
  const int N_r = 56;
  const int N_c = 75;

  //
  // Allocate matrix data
  //
  int *A = memoryManager::allocate<int>(N_r * N_c);
  int *At = memoryManager::allocate<int>(N_r * N_c);

  //
  // In the following implementations of tiled matrix transpose, we
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

  // Calculate number of tiles (needed for the c version)
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
  //printResult<int>(Aview, N_r, N_c);

//----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of tiled matrix transpose...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  //
  // (0) Outer loops to iterate over tiles
  //
  for (int by = 0; by < outer_Dimr; ++by) {
    for (int bx = 0; bx < outer_Dimc; ++bx) {

      //
      // (1) Loops to iterate over tile entries
      //
      //     Note: loops are ordered so that output matrix data access
      //           is stride-1.
      //
      for (int trow = 0; trow < TILE_SZ; ++trow) {
        for (int tcol = 0; tcol < TILE_SZ; ++tcol) {

          int col = bx * TILE_SZ + tcol;  // Matrix column index
          int row = by * TILE_SZ + trow;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Atview(col, row) = Aview(row, col);
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

// Note: this needs to be turned on for other back-ends when working the 
//       exercises (sequential, CUDA, etc.)
#if defined(RAJA_ENABLE_OPENMP)
  RAJA::RangeSegment row_Range(0, N_r);
  RAJA::RangeSegment col_Range(0, N_c);
#endif

//----------------------------------------------------------------------------//
  std::cout << "\n Running RAJA sequential tiled matrix transpose ...\n";
  std::memset(At, 0, N_r * N_c * sizeof(int));

  //
  // The following policy performs the matrix transpose operation
  // using sequential loops. The template parameter inside
  // tile_fixed corresponds to the dimension size of the tile.
  //
#if 0
  using KERNEL_EXEC_POL_SEQ =
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::tile_fixed<TILE_SZ>,
                               RAJA::seq_exec,
        RAJA::statement::Tile<0, RAJA::tile_fixed<TILE_SZ>,
                                 RAJA::seq_exec,
          // Fill in inner loop execution statements.... 
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;
#endif

  ///
  /// TODO...
  ///
  /// EXERCISE:
  ///
  ///   Implement the matrix tranpose kernel using the RAJA kernel API
  ///   and the kernel policy above. You will need to fill in inner
  ///   loop sequential execution statements where indicted above.
  ///

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

//----------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running RAJA openmp tiled matrix transpose -  parallel top inner loop...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  //
  // This policy loops over tiles sequentially while exposing parallelism on
  // one of the inner loops.
  //
#if 0
  using KERNEL_EXEC_POL_OMP =
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::tile_fixed<TILE_SZ>,
                               RAJA::seq_exec,
        RAJA::statement::Tile<0, RAJA::tile_fixed<TILE_SZ>,
                                 RAJA::seq_exec,
          // Fill in inner loop execution statements.... 
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;
#endif

  ///
  /// TODO...
  ///
  /// EXERCISE:
  ///
  ///   Implement the matrix tranpose kernel using the RAJA kernel API
  ///   and the kernel policy above. You will need to fill in inner
  ///   loop execution statements where indicted above so that the top
  ///   inner loop is parallel using OpenMP and the innermost loop is
  ///   sequential.
  ///

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA openmp tiled matrix transpose - collapsed inner loops...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  //
  // This policy loops over tiles sequentially while collapsing inner loops
  // into a single OpenMP loop enabling parallel loads/reads
  // to/from the tile.
  //
  using KERNEL_EXEC_POL_OMP2 =
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::tile_fixed<TILE_SZ>,
                               RAJA::seq_exec,
        RAJA::statement::Tile<0, RAJA::tile_fixed<TILE_SZ>,
                                 RAJA::seq_exec,
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    RAJA::ArgList<0, 1>,
            RAJA::statement::Lambda<0>
          > //closes collapse
        > // closes Tile 0
      > // closes Tile 1
    >; // closes policy list

  RAJA::kernel<KERNEL_EXEC_POL_OMP2>(
                        RAJA::make_tuple(col_Range, row_Range),
                        [=](int col, int row) {

    Atview(col, row) = Aview(row, col);

  });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);
#endif

//----------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running RAJA CUDA tiled matrix transpose ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

#if 0
  using KERNEL_EXEC_POL_CUDA =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::Tile<1, RAJA::tile_fixed<TILE_SZ>,
                                 RAJA::cuda_block_y_loop,
          RAJA::statement::Tile<0, RAJA::tile_fixed<TILE_SZ>,
                                   RAJA::cuda_block_x_loop,
            // Fill in inner loop execution statements.... 
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >
    >;
#endif

  ///
  /// TODO...
  ///
  /// EXERCISE:
  ///
  ///   Implement the matrix tranpose kernel using the RAJA kernel API
  ///   and the kernel policy above. You will need to fill in inner
  ///   loop execution statements so that each loop in a tile is mapped
  ///   to CUDA threads to execute in parallel.
  ///

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
      // std::cout << "At(" << row << "," << col << ") = " << Atview(row, col)
      //                << std::endl;
      std::cout<<Atview(row, col)<<" ";
    }
    std::cout << "" << std::endl;
  }
  std::cout << std::endl;
}
