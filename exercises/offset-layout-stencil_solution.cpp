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
 *  Offset Layout Stencil Exercise
 *
 *  This exercise applies a five-point stencil to the interior cells of a 
 *  lattice and stores the resulting sums in a second lattice of equal size.
 *  You can think of the lattice as representing the centers of cells on a 
 *  two-dimensional Cartesian mesh. 
 *
 *  The five-point stencil accumulates values of a cell and its four neighbors. 
 *  Assuming the cells of a lattice may be accessed through a row/col fashion, 
 *  the stencil may be expressed as the following sum:
 * 
 *  output(row, col) = input(row, col) +
 *                     input(row - 1, col) + input(row + 1, col) +
 *                     input(row, col - 1) + input(row, col + 1)
 *
 *  We assume a lattice has N_r x N_c interior nodes and a padded edge of zeros
 *  for a lattice of size (N_r + 2) x (N_c + 2).  
 *
 *  In the case of N_r = N_c = 3, the input lattice values are:
 *
 *  ---------------------
 *  | 0 | 0 | 0 | 0 | 0 |
 *  ---------------------
 *  | 0 | 1 | 1 | 1 | 0 |
 *  ---------------------
 *  | 0 | 1 | 1 | 1 | 0 |
 *  ---------------------
 *  | 0 | 1 | 1 | 1 | 0 |
 *  ---------------------
 *  | 0 | 0 | 0 | 0 | 0 |
 *  ---------------------
 *
 *  after the computation, we expect the output lattice to have values:
 *
 *  ---------------------
 *  | 0 | 0 | 0 | 0 | 0 |
 *  ---------------------
 *  | 0 | 3 | 4 | 3 | 0 |
 *  ---------------------
 *  | 0 | 4 | 5 | 4 | 0 |
 *  ---------------------
 *  | 0 | 3 | 4 | 3 | 0 |
 *  ---------------------
 *  | 0 | 0 | 0 | 0 | 0 |
 *  ---------------------
 *
 * In this exercise, we use RAJA::OffsetLayout and RAJA::View objects to 
 * simplify the indexing to perform the stencil calculation. For the 
 * purposes of discussion, we enumerate the lattice in the following manner:
 *
 *  --------------------------------------------------
 *  | (-1, 3) | (0, 3)  | (1, 3)  | (2, 3)  | (3, 3)  |
 *  --------------------------------------------------
 *  | (-1, 2) | (0, 2)  | (1, 2)  | (2, 2)  | (3, 2)  |
 *  --------------------------------------------------
 *  | (-1, 1) | (0, 1)  | (1, 1)  | (2, 1)  | (3, 1)  |
 *  --------------------------------------------------
 *  | (-1, 0) | (0, 0)  | (1, 0)  | (2, 0)  | (3, 0)  |
 *  ---------------------------------------------------
 *  | (-1,-1) | (0, -1) | (1, -1) | (2, -1) | (3, -1) |
 *  ---------------------------------------------------
 *
 *  Notably (0, 0) corresponds to the bottom left corner of the stencil
 *  interior region to which we apply stencil.
 *
 *  RAJA features shown:
 *    - RAJA::kernel kernel execution method and execution policies
 *    - RAJA::View 
 *    - RAJA::OffsetLayout
 *    - RAJA::make_offset_layout method
 *
 * For the CUDA implementation, we use unified memory to hold the lattice data.
 * For HIP, we use explicit host-device memory and manually copy data between
 * the two. 
 */

/*
 * Define number of threads in x and y dimensions of a GPU thread block
 */
#if defined(RAJA_ENABLE_CUDA)
#define CUDA_BLOCK_SIZE 16
#endif

#if defined(RAJA_ENABLE_HIP)
#define HIP_BLOCK_SIZE 16
#endif

//
// Functions for printing and checking results
//
void printLattice(int* lattice, int N_r, int N_c);
void checkResult(int* compLattice, int* refLattice, int totCells);

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nFive-point stencil example...\n";

// _stencil_define_start
//
// Define num of interior cells in row/cols in a lattice
//
  constexpr int N_r = 5;
  constexpr int N_c = 4;

//
// Define total num of cells in rows/cols in a lattice
//
  constexpr int totCellsInRow = N_r + 2;
  constexpr int totCellsInCol = N_c + 2;

//
// Define total num of cells in a lattice
//
  constexpr int totCells = totCellsInRow * totCellsInCol;
// _stencil_define_end

//
// Allocate and initialize lattice
//
  int* input = memoryManager::allocate<int>(totCells * sizeof(int));
  int* output = memoryManager::allocate<int>(totCells * sizeof(int));
  int* output_ref = memoryManager::allocate<int>(totCells * sizeof(int));

  std::memset(input, 0, totCells * sizeof(int));
  std::memset(output, 0, totCells * sizeof(int));
  std::memset(output_ref, 0, totCells * sizeof(int));

//
// C-Style intialization
//
// _stencil_input_init_start
  for (int row = 1; row <= N_r; ++row) {
    for (int col = 1; col <= N_c; ++col) {
      int id = col + totCellsInCol * row;
      input[id] = 1;
    }
  }
// _stencil_input_init_end

  std::cout << "\ninput lattice:\n"; 
  printLattice(input, totCellsInRow, totCellsInCol);

//
// Generate reference solution
//
// _stencil_output_ref_start
  for (int row = 1; row <= N_r; ++row) {
    for (int col = 1; col <= N_c; ++col) {

      int id = col + totCellsInCol * row;
      output_ref[id] = input[id] + input[id + 1]
                        + input[id - 1]
                        + input[id + totCellsInCol]
                        + input[id - totCellsInCol];
    }
  }
// _stencil_output_ref_end

  std::cout << "\noutput reference lattice:\n"; 
  printLattice(output_ref, totCellsInRow, totCellsInCol);

//----------------------------------------------------------------------------//

//
// The following code illustrates pairing an offset layout and a RAJA view
// object to simplify multidimensional indexing.
// An offset layout is constructed by using the make_offset_layout method.
// The first argument of the layout is an array object with the coordinates of
// the bottom left corner of the lattice, and the second argument is an array
// object of the coordinates of the top right corner plus 1.
// The example uses double braces to initiate the array object and its
// subobjects.
//
  // _offsetlayout_views_start
  const int DIM = 2;

  RAJA::OffsetLayout<DIM, int> layout =
      RAJA::make_offset_layout<DIM, int>({{-1, -1}}, {{N_r+1, N_c+1}});

  RAJA::View<int, RAJA::OffsetLayout<DIM, int>> inputView(input, layout);
  RAJA::View<int, RAJA::OffsetLayout<DIM, int>> outputView(output, layout);
  // _offsetlayout_views_end

//
// Create range segments used in kernels
//
  // _offsetlayout_ranges_start
  RAJA::TypedRangeSegment<int> col_range(0, N_c);
  RAJA::TypedRangeSegment<int> row_range(0, N_r);
  // _offsetlayout_ranges_end

//----------------------------------------------------------------------------//

  std::cout << "\n Running five-point stencil (RAJA-Kernel sequential)...\n";

  std::memset(output, 0, totCells * sizeof(int));

  // _offsetlayout_rajaseq_start
  using NESTED_EXEC_POL1 =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::seq_exec,    // row
        RAJA::statement::For<0, RAJA::seq_exec,  // col
          RAJA::statement::Lambda<0>
        >
      >  
    >;  

  RAJA::kernel<NESTED_EXEC_POL1>(RAJA::make_tuple(col_range, row_range),
                                 [=](int col, int row) {

                                   outputView(row, col) =
                                       inputView(row, col)
                                       + inputView(row - 1, col)
                                       + inputView(row + 1, col)
                                       + inputView(row, col - 1)
                                       + inputView(row, col + 1);

                                 });
  // _offsetlayout_rajaseq_end

  std::cout << "\noutput lattice:\n"; 
  printLattice(output, totCellsInRow, totCellsInCol);
  checkResult(output, output_ref, totCells);

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n Running five-point stencil (RAJA-Kernel OpenMP)...\n";

  std::memset(output, 0, totCells * sizeof(int));

  // _offsetlayout_rajaomp_start
  using NESTED_EXEC_POL2 = 
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                RAJA::ArgList<1, 0>,   // row, col
        RAJA::statement::Lambda<0>
      > 
    >;

  RAJA::kernel<NESTED_EXEC_POL2>(RAJA::make_tuple(col_range, row_range),
                                 [=](int col, int row) {

                                   outputView(row, col) =
                                       inputView(row, col)
                                       + inputView(row - 1, col)
                                       + inputView(row + 1, col)
                                       + inputView(row, col - 1)
                                       + inputView(row, col + 1);

                                 });
  // _offsetlayout_rajaomp_end

  std::cout << "\noutput lattice:\n"; 
  printLattice(output, totCellsInRow, totCellsInCol);
  checkResult(output, output_ref, totCells);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running five-point stencil (RAJA-Kernel CUDA)...\n";

  std::memset(output, 0, totCells * sizeof(int));

  // _offsetlayout_rajacuda_start
  using NESTED_EXEC_POL3 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::For<1, RAJA::cuda_block_x_loop, //row
          RAJA::statement::For<0, RAJA::cuda_thread_x_loop, //col
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;                                                     

  RAJA::kernel<NESTED_EXEC_POL3>(RAJA::make_tuple(col_range, row_range),
                                 [=] RAJA_DEVICE(int col, int row) {

                                   outputView(row, col) =
                                       inputView(row, col)
                                       + inputView(row - 1, col)
                                       + inputView(row + 1, col)
                                       + inputView(row, col - 1)
                                       + inputView(row, col + 1);

                                 });
  // _offsetlayout_rajacuda_end

  std::cout << "\noutput lattice:\n"; 
  printLattice(output, totCellsInRow, totCellsInCol);
  checkResult(output, output_ref, totCells);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)

  std::cout << "\n Running five-point stencil (RAJA-Kernel - "
               "hip)...\n";

  std::memset(output, 0, totCells * sizeof(int));

  int* d_input  = memoryManager::allocate_gpu<int>(totCells);
  int* d_output = memoryManager::allocate_gpu<int>(totCells);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_input, input, totCells * sizeof(int), hipMemcpyHostToDevice);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_output, output, totCells * sizeof(int), hipMemcpyHostToDevice);

  RAJA::View<int, RAJA::OffsetLayout<DIM, int>> d_inputView (d_input, layout);
  RAJA::View<int, RAJA::OffsetLayout<DIM, int>> d_outputView(d_output, layout);

  // _offsetlayout_rajahip_start
  using NESTED_EXEC_POL4 =
    RAJA::KernelPolicy<
      RAJA::statement::HipKernel<
        RAJA::statement::For<1, RAJA::hip_block_x_loop, //row
          RAJA::statement::For<0, RAJA::hip_thread_x_loop, //col
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

  RAJA::kernel<NESTED_EXEC_POL4>(RAJA::make_tuple(col_range, row_range),
                                 [=] RAJA_DEVICE(int col, int row) {

                                   d_outputView(row, col) =
                                         d_inputView(row, col)
                                       + d_inputView(row - 1, col)
                                       + d_inputView(row + 1, col)
                                       + d_inputView(row, col - 1)
                                       + d_inputView(row, col + 1);
                                 });
  // _offsetlayout_rajahip_end

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, output, d_output, totCells * sizeof(int), hipMemcpyDeviceToHost);

  std::cout << "\noutput lattice:\n"; 
  printLattice(output, totCellsInRow, totCellsInCol);
  checkResult(output, output_ref, totCells);

  memoryManager::deallocate_gpu(d_input);
  memoryManager::deallocate_gpu(d_output);
#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(input);
  memoryManager::deallocate(output);
  memoryManager::deallocate(output_ref);

  std::cout << "\n DONE!...\n";
  return 0;
}

//
// Print Lattice
//
void printLattice(int* lattice, int totCellsInRow, int totCellsInCol)
{
  std::cout << std::endl;
  for (int row = 0; row < totCellsInRow; ++row) {
    for (int col = 0; col < totCellsInCol; ++col) {

      const int id = col + totCellsInCol * row;
      std::cout << lattice[id] << " ";
    }
    std::cout << " " << std::endl;
  }
  std::cout << std::endl;
}

//
// Check Result
//
void checkResult(int* compLattice, int* refLattice, int totCells)
{
  bool correct = true;

  int i = 0;
  while ( correct && (i < totCells) ) {
    correct = (compLattice[i] == refLattice[i]);
    i++;
  }

  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}
