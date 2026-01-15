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

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Tiled Matrix Transpose Exercise
 *
 *  In this exercise, an input matrix A of dimension N_r x N_c is
 *  transposed and returned as a second matrix At.
 *
 *  RAJA features shown:
 *    - Basic usage of 'RAJA::launch' abstractions for nested loops
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

//
// Define dimensionality of matrices
//
constexpr int DIM = 2;

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

  std::cout << "\n\nRAJA matrix transpose exercise...\n";

  //
  // Define num rows/cols in matrix.
  //
  // _mattranspose_dims_start
  constexpr int N_r = 56;
  constexpr int N_c = 75;
  // _mattranspose_dims_end

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
  // _mattranspose_views_start
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_r, N_c);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_c, N_r);
  // _mattranspose_views_end

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
  std::cout << "\n Running C-version of matrix transpose...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  // _cstyle_mattranspose_start
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
        Atview(col, row) = Aview(row, col);
    }
  }
  // _cstyle_mattranspose_end

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

  //----------------------------------------------------------------------------//

  //
  // The following RAJA variants use the RAJA::kernel method to carryout the
  // transpose. 
  //
  // Here, we define RAJA range segments to establish the iteration spaces.
  // Iterations inside a RAJA loop is given by their global iteration number. 
  //
  RAJA::TypedRangeSegment<int> row_Range(0, N_r);
  RAJA::TypedRangeSegment<int> col_Range(0, N_c);

  //----------------------------------------------------------------------------//
  std::cout << "\n Running sequential matrix transpose ...\n";
  std::memset(At, 0, N_r * N_c * sizeof(int));

  //
  // The following policy carries out the transpose
  // using sequential loops. 
  //
  // _raja_mattranspose_start
  using loop_policy_seq = RAJA::LoopPolicy<RAJA::seq_exec>;
  using launch_policy_seq = RAJA::LaunchPolicy<RAJA::seq_launch_t>;

  RAJA::launch<launch_policy_seq>
   (RAJA::LaunchParams(), //LaunchParams may be empty when running on the host
    [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

      RAJA::loop<loop_policy_seq>(ctx, row_Range, [&] (int /*row*/) {
        RAJA::loop<loop_policy_seq>(ctx, col_Range, [&] (int /*col*/) {

	  /// TODO...
	  ///
	  /// EXERCISE: Implement the kernel body for the transpose operation
	  ///

        });
      });

  });
  // _raja_mattranspose_end

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);
//----------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running openmp matrix transpose -  parallel top inner loop...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  //
  // This policy loops sequentially while exposing parallelism on
  // one of the inner loops.
  
  //uncomment to use in example below
  //using loop_policy_omp = RAJA::LoopPolicy<RAJA::omp_for_exec>;
  using launch_policy_omp = RAJA::LaunchPolicy<RAJA::omp_launch_t>;

  RAJA::launch<launch_policy_omp>(RAJA::LaunchParams(), //LaunchParams may be empty when running on the host
    [=] RAJA_HOST_DEVICE (RAJA::LaunchContext /*ctx*/) {


      /// TODO...
      ///
      /// EXERCISE: Implement the loops to apply omp parallism and sequential
      ///           execution on the column and row loops respectively
      ///

      //Atview(col, row) = Aview(row, col);


  });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);
#endif
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running cuda matrix transpose ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));
  
  // _raja_mattranspose_cuda_start
  using cuda_thread_x = RAJA::LoopPolicy<RAJA::cuda_thread_x_loop>;
  using cuda_thread_y = RAJA::LoopPolicy<RAJA::cuda_thread_y_loop>;

  const bool async = false; //execute asynchronously
  using launch_policy_cuda = RAJA::LaunchPolicy<RAJA::cuda_launch_t<async>>;

  RAJA::launch<launch_policy_cuda>(
    RAJA::LaunchParams(RAJA::Teams(1), RAJA::Threads(16,16)),
    [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

      RAJA::loop<cuda_thread_y>(ctx, row_Range, [&] (int row) {
        RAJA::loop<cuda_thread_x>(ctx, col_Range, [&] (int col) {

            Atview(col, row) = Aview(row, col);

        });
      });

  });
  // _raja_mattranspose_cuda_end

  checkResult<int>(Atview, N_c, N_r);
  //printResult<int>(Atview, N_c, N_r);
#endif

//----------------------------------------------------------------------------//

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
      // std::cout << "At(" << row << "," << col << ") = " << Atview(row, col)
      //                << std::endl;
      std::cout<<Atview(row, col)<<" ";
    }
    std::cout << "" << std::endl;
  }
  std::cout << std::endl;
}
