//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
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
 *  View and Layout Exercise
 *
 *  Examples illustrate the use of RAJA View and Layout types.
 *
 *  RAJA features shown:
 *    - RAJA::View
 *    - RAJA::Layout
 *    - Layout permutations 
 *    - OffsetLayout
 *    - OffsetLayout permutations 
 *
 * NOTE: no RAJA kernel execution methods are used in these examples.
 */

//
// Functions to check and print arrays
//
template <typename T>
void checkResult(T* C, T* Cref, int N);

template <typename T>
void printValues(T* C, int N);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA view & layout exercises...\n";

//----------------------------------------------------------------------------//
//
// Matrix-matrix multiplication: default layout
//
//----------------------------------------------------------------------------//

  // _matmult_init_start
  //
  // Define the size of N x N of matrices.
  //
  constexpr int N = 4;

  //
  // Allocate storage for matrices and initialize matrix entries
  //
  double *A = new double[ N * N ];
  double *B = new double[ N * N ];
  double *C = new double[ N * N ];
  double *Cref = new double[ N * N ];

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      A[ col + N*row ] = row + 1;
      B[ col + N*row ] = col + 1;
      C[ col + N*row ] = 0.0;
      Cref[ col + N*row ] = 0.0;
    }
  }
  // _matmult_init_end

//printValues<double>(A, N*N); 
//printValues<double>(B, N*N); 
//printValues<double>(C, N*N); 
//printValues<double>(Cref, N*N); 

//----------------------------------------------------------------------------//

  std::cout << "\n Running matrix multiplication reference solution...\n";

  // _cstyle_matmult_start
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      for (int k = 0; k < N; ++k) {
        Cref[col + N*row] += A[k + N*row] * B[col + N*k];
      }
    }
  }
  // _cstyle_matmult_end

//printValues<double>(Cref, N*N);


//----------------------------------------------------------------------------//

  std::cout << "\n Running matrix multiplication w/Views...\n";

  // 
  // Define RAJA View objects to simplify access to the matrix entries.
  // 
  // Note: we use default Layout and specify unit stride
  //
  // _matmult_views_start
  RAJA::View< double, RAJA::Layout<2, int, 1> > Aview(A, N, N);
  RAJA::View< double, RAJA::Layout<2, int, 1> > Bview(B, N, N);
  RAJA::View< double, RAJA::Layout<2, int, 1> > Cview(C, N, N);
  // _matmult_views_end

  // _cstyle_matmult_views_start
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      for (int k = 0; k < N; ++k) {
        Cview(row, col) += Aview(row, k) * Bview(k, col);
      }
    }
  }
  // _cstyle_matmult_views_end

  checkResult<double>(C, Cref, N*N);
//printValues<double>(C, N*N);

//
// Clean up.
//
  delete [] A;
  delete [] B;
  delete [] C;
  delete [] Cref;

//----------------------------------------------------------------------------//
//
// Default layouts use row-major data ordering
//
//----------------------------------------------------------------------------//

  //
  // Define dimensions and allocate arrays
  //
  // _default_views_init_start
  constexpr int Nx = 3;
  constexpr int Ny = 5;
  constexpr int Nz = 2;
  constexpr int Ntot  = Nx*Ny*Nz;
  int* a = new int[ Ntot ];
  int* aref = new int[ Ntot ];

  for (int i = 0; i < Ntot; ++i)
  {
    aref[i] = i;
  }
  // _default_views_init_end

//printValues<int>(ref, Ntot);

//----------------------------------------//

  std::cout << "\n Running default layout view cases...\n";

  std::cout << "\n\t Running 1D view case...\n";
 
  std::memset(a, 0, Ntot * sizeof(int));
 
  // _default_view1D_start 
  RAJA::View< int, RAJA::Layout<1, int, 0> > view_1D(a, Ntot);

  for (int i = 0; i < Ntot; ++i) {
    view_1D(i) = i;
  }
  // _default_view1D_end 

  checkResult<int>(a, aref, Ntot);
//printValues<int>(a, Ntot);

//----------------------------------------//

  std::cout << "\n\t Running 2D default layout view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));
 
  // _default_view2D_start
  RAJA::View< int, RAJA::Layout<2, int, 1> > view_2D(a, Nx, Ny);

  int iter{0};
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      view_2D(i, j) = iter;
      ++iter;
    }
  }
  // _default_view2D_end

  checkResult<int>(a, aref, Nx*Ny);
//printValues<int>(a, Nx*Ny);

//----------------------------------------//

  std::cout << "\n\t Running 3D default layout view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));

  // _default_view3D_start    
  RAJA::View< int, RAJA::Layout<3, int, 2> > view_3D(a, Nx, Ny, Nz);

  iter = 0;
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      for (int k = 0; k < Nz; ++k) {
        view_3D(i, j, k) = iter;
        ++iter;
      }
    }
  }
  // _default_view3D_end

  checkResult<int>(a, aref, Nx*Ny*Nz);
//printValues<int>(a, Nx*Ny*Nz);

//----------------------------------------------------------------------------//
//
// Permuted layouts change the data striding order
//
//----------------------------------------------------------------------------//

  std::cout << "\n Running permuted layout cases...\n";

//----------------------------------------//

  std::cout << "\n\t Running 2D default permutation view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));

  // _default_perm_view2D_start
  std::array<RAJA::idx_t, 2> defperm2 {{0, 1}};
  RAJA::Layout< 2, int> defperm2_layout =
    RAJA::make_permuted_layout( {{Nx, Ny}}, defperm2);
  RAJA::View< int, RAJA::Layout<2, int, 1> > defperm_view_2D(a, defperm2_layout);

  iter = 0;
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      defperm_view_2D(i, j) = iter;
      ++iter;
    }
  }
  // _default_perm_view2D_end

  checkResult<int>(a, aref, Nx*Ny);
//printValues<int>(a, Nx*Ny);

//----------------------------------------//

  std::cout << "\n\t Running 3D default permutation view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));

  // _default_perm_view3D_start
  std::array<RAJA::idx_t, 3> defperm3 {{0, 1, 2}};
  RAJA::Layout< 3, int > defperm3_layout =
    RAJA::make_permuted_layout( {{Nx, Ny, Nz}}, defperm3);
  RAJA::View< int, RAJA::Layout<3, int, 2> > defperm_view_3D(a, defperm3_layout);

  iter = 0;
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      for (int k = 0; k < Nz; ++k) {
        defperm_view_3D(i, j, k) = iter;
        ++iter;
      }
    }
  }
  // _default_perm_view3D_end

  checkResult<int>(a, aref, Nx*Ny*Nz);
//printValues<int>(a, Nx*Ny*Nz);

//----------------------------------------//
//----------------------------------------//

  std::cout << "\n\t Running 2D permuted layout view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));

  // _perm_2D_start
  std::array<RAJA::idx_t, 2> perm2 {{1, 0}};
  RAJA::Layout< 2, int> perm2_layout =
    RAJA::make_permuted_layout( {{Nx, Ny}}, perm2);
  RAJA::View< int, RAJA::Layout<2, int, 0> > perm_view_2D(a, perm2_layout);

  iter = 0;
  for (int j = 0; j < Ny; ++j) {
    for (int i = 0; i < Nx; ++i) {
      perm_view_2D(i, j) = iter;
      ++iter;
    }
  }
  // _perm_2D_end

  checkResult<int>(a, aref, Nx*Ny);
//printValues<int>(a, Nx*Ny);

//----------------------------------------//

  std::cout << "\n\t Running 3D perma layout view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));

  // _perma_view3D_start
  std::array<RAJA::idx_t, 3> perm3a {{2, 1, 0}};
  RAJA::Layout< 3, int> perm3a_layout =
    RAJA::make_permuted_layout( {{Nx, Ny, Nz}}, perm3a);
  RAJA::View< int, RAJA::Layout<3, int, 0> > perm3a_view_3D(a, perm3a_layout);

  iter = 0;
  for (int k = 0; k < Nz; ++k) {
    for (int j = 0; j < Ny; ++j) {
      for (int i = 0; i < Nx; ++i) {
        perm3a_view_3D(i, j, k) = iter;
        ++iter;
      }
    }
  }
  // _perma_view3D_end

  checkResult<int>(a, aref, Nx*Ny*Nz);
//printValues<int>(a, Nx*Ny*Nz);

//----------------------------------------//

  std::cout << "\n\t Running 3D permb layout view case...\n";

  std::memset(a, 0, Ntot * sizeof(int));

  // _permb_view3D_start
  std::array<RAJA::idx_t, 3> perm3b {{1, 2, 0}};
  RAJA::Layout< 3, int > perm3b_layout =
    RAJA::make_permuted_layout( {{Nx, Ny, Nz}}, perm3b);
  RAJA::View< int, RAJA::Layout<3, int, 0> > perm3b_view_3D(a, perm3b_layout);

  iter = 0;
  for (int j = 0; j < Ny; ++j) {
    for (int k = 0; k < Nz; ++k) {
      for (int i = 0; i < Nx; ++i) {
        perm3b_view_3D(i, j, k) = iter;
        ++iter;
      }
    }
  }
  // _permb_view3D_end

  checkResult<int>(a, aref, Nx*Ny*Nz);
//printValues<int>(a, Nx*Ny*Nz);

//
// Clean up.
//
  delete [] a;
  delete [] aref;

//----------------------------------------------------------------------------//
//
// Layouts: multi-dimensional indices vs. linear indicies
//
// RAJA::Layout type has methods that can be used to convert between
// multi-dimensional and linear indices. We show these below using the
// three-dimensional layouts in the examples above. Recall the Nx, Ny, Nz
// sizes defined earlier:
//
//  constexpr int Nx = 3;
//  constexpr int Ny = 5;
//  constexpr int Nz = 2;
//
//----------------------------------------------------------------------------//

  std::cout << "\n Multi-dimensional indices to linear indices...\n";


  std::cout << "\nperm3a_layout...\n" << std::endl;

  int lin = -1;
  int i = -1;
  int j = -1;
  int k = -1;

  // _perm3d_layout_start
  lin = perm3a_layout(1, 2, 0);
  std::cout << "\tperm3a_layout(1, 2, 0) = " << lin << std::endl;
  std::cout << "\t  Should be 7 = 1 + 2 * Nx + 0 * Nx * Ny "
            << "(since perm is {2, 1, 0})" << std::endl;

  perm3a_layout.toIndices(7, i, j, k);
  std::cout << "\tperm3a_layout.toIndices(7, i, j, k) --> (i, j, k) = "
            << "(" << i << ", " << j << ", " << k << ")\n" << std::endl;
  // _perm3d_layout_end


  lin = perm3a_layout(2, 3, 1);
  std::cout << "\tperm3a_layout(2, 3, 1) = " << lin << std::endl;
  std::cout << "\t  Should be 26 = 2 + 3 * Nx + 1 * Nx * Ny "
            << "(since perm is {2, 1, 0})" << std::endl;

  perm3a_layout.toIndices(26, i, j, k);
  std::cout << "\tperm3a_layout.toIndices(26, i, j, k) --> (i, j, k) = "
            << "(" << i << ", " << j << ", " << k << ")\n" << std::endl;


  lin = perm3a_layout(0, 2, 1);
  std::cout << "\tperm3a_layout(0, 2, 1) = " << lin << std::endl;
  std::cout << "\t  Should be 21 = 0 + 2 * Nx + 1 * Nx * Ny "
            << "(since perm is {2, 1, 0})" << std::endl;

  perm3a_layout.toIndices(21, i, j, k);
  std::cout << "\tperm3a_layout.toIndices(21, i, j, k) --> (i, j, k) = "
            << "(" << i << ", " << j << ", " << k << ")\n" << std::endl;

//----------------------------------------------------------------------------//

  std::cout << "\nperm3b_layout...\n" << std::endl;

  lin = perm3b_layout(1, 2, 0);
  std::cout << "\tperm3b_layout(1, 2, 0) = " << lin << std::endl;
  std::cout << "\t  Should be 13 = 1 + 0 * Nx + 2 * Nx * Nz "
            << "(since perm is {1, 2, 0})" << std::endl;

  perm3b_layout.toIndices(13, i, j, k);
  std::cout << "\tperm3b_layout.toIndices(13, i, j, k) --> (i, j, k) = "
            << "(" << i << ", " << j << ", " << k << ")\n" << std::endl;


  lin = perm3b_layout(2, 3, 1);
  std::cout << "\tperm3b_layout(2, 3, 1) = " << lin << std::endl;
  std::cout << "\t  Should be 23 = 2 + 1 * Nx + 3 * Nx * Nz "
            << "(since perm is {1, 2, 0})" << std::endl;

  perm3b_layout.toIndices(23, i, j, k);
  std::cout << "\tperm3b_layout.toIndices(23, i, j, k) --> (i, j, k) = "
            << "(" << i << ", " << j << ", " << k << ")\n" << std::endl;


  lin = perm3b_layout(0, 2, 1);
  std::cout << "\tperm3b_layout(0, 2, 1) = " << lin << std::endl;
  std::cout << "\t  Should be 15 = 0 + 1 * Nx + 2 * Nx * Nz "
            << "(since perm is {1, 2, 0})" << std::endl;

  perm3b_layout.toIndices(15, i, j, k);
  std::cout << "\tperm3b_layout.toIndices(15, i, j, k) --> (i, j, k) = "
            << "(" << i << ", " << j << ", " << k << ")\n" << std::endl; 

//----------------------------------------------------------------------------//
//
// Offset layouts apply offsets to indices
//
//----------------------------------------------------------------------------//

  std::cout << "\n Running offset layout cases...\n";

  //
  // Define some dimensions, and allocate arrays
  //
  constexpr int Ntot_ao = 40;
  int* ao = new int[ Ntot_ao ];
  int* ao_ref = new int[ Ntot_ao ];

//----------------------------------------//

  std::cout << "\n\t Running 1D offset layout case...\n";

  //
  // Set reference solution to compare with
  //

  std::memset(ao_ref, 0, Ntot_ao * sizeof(int));

  // _cstyle_offlayout1D_start
  int imin = -5;
  int imax = 6;

  for (int ii = imin; ii < imax; ++ii) {
    ao_ref[ ii-imin ] = i;
  }
  // _cstyle_offlayout1D_end

//printValues<int>(ao_ref, imax-imin);

//----------------------------------------//

  std::memset(ao, 0, Ntot_ao * sizeof(int));

  // _raja_offlayout1D_start
  RAJA::OffsetLayout<1, int> offlayout_1D = 
    RAJA::make_offset_layout<1, int>( {{imin}}, {{imax}} ); 

  RAJA::View< int, RAJA::OffsetLayout<1, int> > aoview_1Doff(ao,
                                                             offlayout_1D);

  for (int ii = imin; ii < imax; ++ii) {
    aoview_1Doff(ii) = ii;
  }
  // _raja_offlayout1D_end

  checkResult<int>(ao, ao_ref, imax-imin);
//printValues<int>(ao, 11);

//----------------------------------------//

  std::cout << "\n\t Running 2D offset layout case...\n";

  //
  // Set reference solution to compare with
  //

  std::memset(ao_ref, 0, Ntot_ao * sizeof(int));

  // _cstyle_offlayout2D_start
  imin = -1;
  imax = 2;
  int jmin = -5;
  int jmax = 5;

  iter = 0;
  for (int ii = imin; ii < imax; ++ii) {
    for (int jj = jmin; jj < jmax; ++jj) {
      ao_ref[ (jj-jmin) + (ii-imin) * (jmax-jmin)  ] = iter;
      iter++;
    }
  }
  // _cstyle_offlayout2D_end

//printValues<int>(ao_ref, (imax-imin)*(jmax-jmin));

//----------------------------------------//

  std::memset(ao, 0, Ntot_ao * sizeof(int));

  // _raja_offlayout2D_start
  RAJA::OffsetLayout<2, int> offlayout_2D =
    RAJA::make_offset_layout<2, int>( {{imin, jmin}}, {{imax, jmax}} );

  RAJA::View< int, RAJA::OffsetLayout<2, int> > aoview_2Doff(ao,
                                                             offlayout_2D);
  iter = 0;
  for (int ii = imin; ii < imax; ++ii) {
    for (int jj = jmin; jj < jmax; ++jj) {
      aoview_2Doff(ii, jj) = iter;
      iter++;
    }
  }
  // _raja_offlayout2D_end

  checkResult<int>(ao, ao_ref, (imax-imin)*(jmax-jmin));
//printValues<int>(ao, (imax-imin)*(jmax-jmin)); 

//----------------------------------------//

  std::cout << "\n\t Running 2D permuted offset layout case...\n";

  //
  // Set reference solution to compare with
  //

  std::memset(ao, 0, Ntot_ao * sizeof(int));

  // _cstyle_permofflayout2D_start
  iter = 0;
  for (int jj = jmin; jj < jmax; ++jj) {
    for (int ii = imin; ii < imax; ++ii) {
      ao_ref[ (ii-imin) + (jj-jmin) * (imax-imin)  ] = iter; 
      iter++;
    }
  }
  // _cstyle_permofflayout2D_end

//printValues<int>(ao_ref, (imax-imin)*(jmax-jmin));

//----------------------------------------//

  std::memset(ao, 0, Ntot_ao * sizeof(int));

  // _raja_permofflayout2D_start
  std::array<RAJA::idx_t, 2> perm1D {{1, 0}};
  RAJA::OffsetLayout<2> permofflayout_2D =
    RAJA::make_permuted_offset_layout<2>( {{imin, jmin}}, 
                                          {{imax, jmax}},
                                          perm1D );

  RAJA::View< int, RAJA::OffsetLayout<2> > aoview_2Dpermoff(ao,
                                                            permofflayout_2D);

  iter = 0;
  for (int jj = jmin; jj < jmax; ++jj) {
    for (int ii = imin; ii < imax; ++ii) {
      aoview_2Dpermoff(ii, jj) = iter;
      iter++;
    }
  }
  // _raja_permofflayout2D_end

  checkResult<int>(ao, ao_ref, (imax-imin)*(jmax-jmin));
//printValues<int>(ao, (imax-imin)*(jmax-jmin));

//
// Clean up.
//
  delete [] ao;
  delete [] ao_ref;

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to check result and report P/F.
//
template <typename T>
void checkResult(T* C, T* Cref, int N)
{
  bool match = true;
  for (int i = 0; i < N; ++i) {
    if ( std::abs( C[i] - Cref[i] ) > 10e-12 ) {
      match = false;
    }
  }
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

template <typename T>
void printValues(T* C, int N)
{
  for (int i = 0; i < N; ++i) {
    std::cout << "array[" << i << "] = " << C[i] << std::endl;
    }
};
