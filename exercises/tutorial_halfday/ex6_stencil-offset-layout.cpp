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
 *  EXERCISE #6: Offset layout stencil computation. 
 *
 *  In this exercise, you will use RAJA Layouts and Views to perform
 *  a simple 5-point stencil computation on a 2-dimensional Cartesian mesh.
 *  The exercise demonstrates the relative ease with which array data access
 *  can be done using multi-dimensional RAJA Views as compared to C-style
 *  pointer offset arithmetic.
 *
 *  The five-cell stencil accumulates values in a cell from itself and
 *  its four neighbors. Assuming the cells are indexed using (i,j) pairs on
 *  the two dimensional mesh, the stencil computation looks like:
 * 
 *  out(i, j) = in(i, j) + in(i - 1, j) + in(i + 1, j) +
 *              in(i, j - 1) + in(i, j + 1)
 *
 *  where 'in' is the input data array and 'out' is the result of
 *  the stencil computation. For simplicity, in the code examples, we refer 
 *  to the index tuples used to access input array entries as C (center), 
 *  W (west), E (east), S (south), and N (north).
 *
 *  We assume that the input array has an entry for N x M interior mesh cells 
 *  plus a one cell wide halo region around the mesh interior; i.e., the size
 *  of the input array is (N + 2) * (M + 2). The output array has an entry
 *  for N x M interior mesh cells only, so its size is N * M. Note that since
 *  the arrays have different sizes, C-style indexing requires different 
 *  offset values in the code for accessing a cell entry in each array.
 * 
 *  The input array is initialized so that the entry for each interior cell 
 *  is one and the entry for each halo cell is zero. So for the case where
 *  N = 3 and M = 2, the input array looks like:
 *
 *  ---------------------
 *  | 0 | 0 | 0 | 0 | 0 |
 *  ---------------------
 *  | 0 | 1 | 1 | 1 | 0 |
 *  ---------------------
 *  | 0 | 1 | 1 | 1 | 0 |
 *  ---------------------
 *  | 0 | 0 | 0 | 0 | 0 |
 *  ---------------------
 *
 *  And, after the stencil computation, the output array looks like:
 *
 *      -------------
 *      | 3 | 4 | 3 |
 *      -------------
 *      | 4 | 5 | 4 |
 *      -------------
 *      | 3 | 4 | 3 |
 *      -------------
 *
 *  You can think about indexing into this mesh as illustrated in the 
 *  following diagram:
 *
 *  ---------------------------------------------------
 *  | (-1, 2) | (0, 2)  | (1, 2)  | (2, 2)  | (3, 2)  |
 *  ---------------------------------------------------
 *  | (-1, 1) | (0, 1)  | (1, 1)  | (2, 1)  | (3, 1)  |
 *  ---------------------------------------------------
 *  | (-1, 0) | (0, 0)  | (1, 0)  | (2, 0)  | (3, 0)  |
 *  ---------------------------------------------------
 *  | (-1,-1) | (0, -1) | (1, -1) | (2, -1) | (3, -1) |
 *  ---------------------------------------------------
 *
 *  Notably (0, 0) corresponds to the bottom left corner of the interior 
 *  region, which extends to (2, 1), and (-1, -1) corresponds to the bottom 
 *  left corner of the halo region, which extends to (3, 2).
 *
 *  This file contains two C-style sequential implementations of stencil 
 *  computation. One (Part a) has column indexing as stride-1 with the outer 
 *  loop traversing the rows ('i' loop variable) and the inner loop traversing 
 *  the columns ('j' loop variable). The other (Part B) has row indexing as 
 *  stride-1 and reverses the order of the loops. This shows that a C-style 
 *  implementation requires two different implementations, one for each loop 
 *  order, since the array offset arithmetic is different in the two cases. 
 *  Where indicated by comments, you will fill in versions using 
 *  two-dimensional RAJA Views with offset layouts. One loop ordering requires 
 *  permutations, while the other does not. If done properly, you will see 
 *  that both RAJA versions have identical inner loop bodies, which is not the 
 *  case for the C-style variants.
 *
 *  Note that you will use the same for-loop patterns as the C-style loops. 
 *  In a later exercise, we will show you how to use RAJA's nested loop
 *  support, which allows you to write both RAJA variants with identical 
 *  source code.
 *
 *  RAJA features you will use:
 *    -  Offset-layouts and RAJA Views
 * 
 *  Since this exercise is done on a CPU only, we use C++ new and delete
 *  operators to allocate and deallocate the arrays we will use.
 */

//
// Functions for printing and checking results
//
// For array printing, 'stride1dim' indicates which mesh dimenstride is 
// stride-1 (Rows indicates each row is stride-1, 
//           Columns indicates each column is stride-1).
//
enum class Stride1
{
   Rows,
   Columns 
};
void printArrayOnMesh(int* v, int Nrows, int Ncols, Stride1 stride1dim);
void checkResult(int* A, int* A_ref, int Ntot);

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nExercise #6: Offset layout stencil computation...\n";

//
// Define number of rows and columns of cells in the 2D mesh.
//
  const int Nr_int = 5; 
  const int Nc_int = 8;

  const int Nr_tot = Nr_int + 2; 
  const int Nc_tot = Nc_int + 2;
  
  const int int_cells = Nr_int * Nc_int;
  const int tot_cells = Nr_tot * Nc_tot; 

//
// Allocate and initialize input array
//
  int* B = memoryManager::allocate<int>(tot_cells * sizeof(int));
  int* A = memoryManager::allocate<int>(int_cells * sizeof(int));
  int* A_ref = memoryManager::allocate<int>(int_cells * sizeof(int));


//----------------------------------------------------------------------------//
// Part A:
// 
// Variant of stencil computation with column indexing as stride-1.
//----------------------------------------------------------------------------//

  std::memset(B, 0, tot_cells * sizeof(int));

//
// We assume that for each cell id (i,j) that j is the stride-1 index.
//
  for (int i = 1; i <= Nc_int; ++i) {
    for (int j = 1; j <= Nr_int; ++j) {
      int idx = j + Nr_tot * i;
      B[idx] = 1;
    }
  }
//printArrayOnMesh(B, Nr_tot, Nc_tot, Stride1::Columns); 


//----------------------------------------------------------------------------//
// C-style stencil computation establishes reference solution to compare with.
//----------------------------------------------------------------------------//

  std::cout << "\n\n Running C-style stencil computation (reference soln)...\n";

  std::memset(A_ref, 0, int_cells * sizeof(int));

  for (int i = 0; i < Nc_int; ++i) {
    for (int j = 0; j < Nr_int; ++j) {

      int idx_out = j + Nr_int * i;
      int idx_in = (j + 1) + Nr_tot * (i + 1);

      A_ref[idx_out] = B[idx_in] +                                // C
                       B[idx_in - Nr_tot] + B[idx_in + Nr_tot] +  // W, E
                       B[idx_in - 1] + B[idx_in + 1];             // S, N

    }
  }

//printArrayOnMesh(A_ref, Nr_int, Nc_int, Stride1::Columns);


//----------------------------------------------------------------------------//
// Variant using RAJA Layouts and Views (no permutation).
//----------------------------------------------------------------------------//

  std::cout << "\n\n Running stencil computation with RAJA Views...\n";

  std::memset(A, 0, int_cells * sizeof(int));

  ///
  /// TODO...
  ///
  /// EXERCISE (Part A): 
  ///
  ///   Fill in the stencil computation below where you use RAJA::View 
  ///   objects for accessing entries in the A and B arrays. You will use
  ///   a RAJA::OffsetLayout for the B array and a RAJA::Layout for the
  ///   A array. The B array access requires an offset since the loops 
  //    iterate over the interior (i, j) indices. 
  ///
  ///   For this part (A) of the exercise, the column (j-loop) indexing 
  ///   has stride 1.
  ///


  for (int i = 0; i < Nc_int; ++i) {
    for (int j = 0; j < Nr_int; ++j) {

      // fill in the loop body

    }
  }

  checkResult(A, A_ref, int_cells);
//printArrayOnMesh(A, Nr_int, Nc_int, Stride1::Columns);


//----------------------------------------------------------------------------//
// Part B:
// 
// Variant of stencil computation with row indexing as stride-1.
//----------------------------------------------------------------------------//

  std::memset(B, 0, tot_cells * sizeof(int));

//
// We assume that for each cell id (i,j) that i is the stride-1 index.
//
  for (int j = 1; j <= Nr_int; ++j) {
    for (int i = 1; i <= Nc_int; ++i) {
      int idx = i + Nc_tot * j;
      B[idx] = 1;
    }
  }
//printArrayOnMesh(B, Nr_tot, Nc_tot, Stride1::Rows);


//----------------------------------------------------------------------------//
// C-style stencil computation establishes reference solution to compare with.
//----------------------------------------------------------------------------//

  std::cout << "\n\n Running C-style stencil computation (reference soln)...\n";

  std::memset(A_ref, 0, int_cells * sizeof(int));

  for (int j = 0; j < Nr_int; ++j) {
    for (int i = 0; i < Nc_int; ++i) {

      int idx_out = i + Nc_int * j;
      int idx_in = (i + 1) + Nc_tot * (j + 1);

      A_ref[idx_out] = B[idx_in] +                                // C
                       B[idx_in - Nc_tot] + B[idx_in + Nc_tot] +  // S, N
                       B[idx_in - 1] + B[idx_in + 1];             // W, E

    }
  }

//printArrayOnMesh(A_ref, Nr_int, Nc_int, Stride1::Rows);


//----------------------------------------------------------------------------//
// Variant using RAJA Layouts and Views (with permutation).
//----------------------------------------------------------------------------//

  std::cout << "\n\n Running stencil computation with RAJA Views (permuted)...\n";

  std::memset(A, 0, int_cells * sizeof(int));

  ///
  /// TODO...
  ///
  /// EXERCISE (Part B): 
  ///
  ///   Fill in the stencil computation below where you use RAJA::View 
  ///   objects for accessing entries in the A and B arrays. You will use
  ///   a RAJA::OffsetLayout for the B array and a RAJA::Layout for the
  ///   A array. The B array access requires an offset since the loops 
  //    iterate over the interior (i, j) indices.
  ///
  ///   For this part (A) of the exercise, the row (i-loop) indexing 
  ///   has stride 1. Thus, layouts for the A and B arrays require 
  ///   the same permutation.
  ///


  for (int j = 0; j < Nr_int; ++j) {
    for (int i = 0; i < Nc_int; ++i) {

      // fill in the loop body

    }
  }

  checkResult(A, A_ref, int_cells);
//printArrayOnMesh(A, Nr_int, Nc_int, Stride1::Rows);

//
// Clean up.
//
  memoryManager::deallocate(B);
  memoryManager::deallocate(A);
  memoryManager::deallocate(A_ref);

  std::cout << "\n DONE!...\n";
  return 0;
}


//
// For array printing, 'stride1dim' indicates which mesh dimenstride is 
// stride-1 (0 indicates each row is stride-1, 
//           1 indicates each column is stride-1).
//
void printArrayOnMesh(int* v, int Nrows, int Ncols, Stride1 stride1dim)
{
  std::cout << std::endl;
  for (int j = 0; j < Nrows; ++j) {
    for (int i = 0; i < Ncols; ++i) {
      int idx = 0;
      if ( stride1dim == Stride1::Columns ) {
        idx = j + Nrows * i;
      } else {
        idx = i + Ncols * j;
      }
      std::cout << v[idx] << " ";
    }
    std::cout << " " << std::endl;
  }
  std::cout << std::endl;
}

//
// Check Result
//
void checkResult(int* A, int* A_ref, int Ntot)
{
  bool pass = true;

  for (int i = 0; i < Ntot; ++i) {
    if ( pass && (A[i] != A_ref[i]) ) {
      pass = false;
    }
  }

  if (pass) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}
