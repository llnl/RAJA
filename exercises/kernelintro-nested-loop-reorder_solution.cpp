//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>

#include "RAJA/RAJA.hpp"

/*
 * Nested Loop Basics and Loop Reordering (RAJA::kernel)
 *
 *  In this exercise, we introduce basic RAJA::kernel mechanics for executing
 *  nested loop kernels, including using execution policies to permute the 
 *  order of loops in a loop nest. The exercise performs no actual 
 *  computation and just prints out loop indices to show different
 *  loop ordering. Also, to avoid difficulty in interpreting parallel 
 *  output, the execution policies use sequential execution.
 *
 *  RAJA features shown:
 *    - 'RAJA::kernel' loop abstractions and execution policies
 *    - 'RAJA::TypedRangeSegment' iteration spaces
 *    - Strongly-typed loop indices
 */

//
// Define three named loop index integer types used in the triply-nested loops.
// These will trigger compilation errors if lambda index argument ordering 
// and types do not match the typed range index ordering.  See final
// example in this file.
//
// _raja_typed_indices_start
RAJA_INDEX_VALUE_T(KIDX, int, "KIDX");
RAJA_INDEX_VALUE_T(JIDX, int, "JIDX"); 
RAJA_INDEX_VALUE_T(IIDX, int, "IIDX"); 
// _raja_typed_indices_end


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  // _range_min_max_start
  constexpr int imin = 0;
  constexpr int imax = 2;
  constexpr int jmin = 1;
  constexpr int jmax = 3;
  constexpr int kmin = 2;
  constexpr int kmax = 4;
  // _range_min_max_end

//
// The RAJA variants of the loop nest use the following typed range segments
// based on the typed indices defined above, outside of main().
//
  // _raja_typed_index_ranges_start
  RAJA::TypedRangeSegment<KIDX> KRange(kmin, kmax);
  RAJA::TypedRangeSegment<JIDX> JRange(jmin, jmax);
  RAJA::TypedRangeSegment<IIDX> IRange(imin, imax);
  // _raja_typed_index_ranges_end
 

  std::cout << "\n\nRAJA::kernel nested loop reorder example...\n";

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//

  std::cout << "\n Running C-style nested loop order: K-outer, J-middle, I-inner" 
            << "...\n\n" << " (I, J, K)\n" << " ---------\n";

  // _cstyle_kji_loops_start
  for (int k = kmin; k < kmax; ++k) {
    for (int j = jmin; j < jmax; ++j) {
      for (int i = imin; i < imax; ++i) {
        printf( " (%d, %d, %d) \n", i, j, k);
      }
    }
  }
  // _cstyle_kji_loops_end

//----------------------------------------------------------------------------//
 
  std::cout << "\n\n Running RAJA nested loop order (K-outer, J-middle, I-inner)"
            << "...\n\n" << " (I, J, K)\n" << " ---------\n";

  // _raja_kji_loops_start
  using KJI_EXECPOL = RAJA::KernelPolicy<
                        RAJA::statement::For<2, RAJA::seq_exec,    // k
                          RAJA::statement::For<1, RAJA::seq_exec,  // j
                            RAJA::statement::For<0, RAJA::seq_exec,// i 
                              RAJA::statement::Lambda<0>
                            > 
                          > 
                        > 
                      >;

  RAJA::kernel<KJI_EXECPOL>( RAJA::make_tuple(IRange, JRange, KRange),
  [=] (IIDX i, JIDX j, KIDX k) { 
     printf( " (%d, %d, %d) \n", (int)(*i), (int)(*j), (int)(*k));
  });
  // _raja_kji_loops_end

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//
 
  std::cout << "\n Running C-style nested loop order: J-outer, I-middle, K-inner" 
            << "...\n\n" << " (I, J, K)\n" << " ---------\n";

  // _cstyle_jik_loops_start
  for (int j = jmin; j < jmax; ++j) {
    for (int i = imin; i < imax; ++i) {
      for (int k = kmin; k < kmax; ++k) {
        printf( " (%d, %d, %d) \n", i, j, k);
      }
    }
  }
  // _cstyle_jik_loops_end

//----------------------------------------------------------------------------//
  std::cout << "\n Running RAJA nested loop order (J-outer, I-middle, K-inner)"
            << "...\n\n" << " (I, J, K)\n" << " ---------\n";

  // _raja_jik_loops_start
  using JIK_EXECPOL = RAJA::KernelPolicy<
                        RAJA::statement::For<1, RAJA::seq_exec,    // j
                          RAJA::statement::For<0, RAJA::seq_exec,  // i
                            RAJA::statement::For<2, RAJA::seq_exec,// k 
                              RAJA::statement::Lambda<0>
                            > 
                          > 
                        > 
                      >;

  RAJA::kernel<JIK_EXECPOL>( RAJA::make_tuple(IRange, JRange, KRange),
  [=] (IIDX i, JIDX j, KIDX k) { 
     printf( " (%d, %d, %d) \n", (int)(*i), (int)(*j), (int)(*k));
  });
  // _raja_jik_loops_end

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//
 
  std::cout << "\n Running C-style nested loop order: I-outer, K-middle, J-inner" 
            << "...\n\n" << " (I, J, K)\n" << " ---------\n";

  // _cstyle_ikj_loops_start
  for (int i = imin; i < imax; ++i) {
    for (int k = kmin; k < kmax; ++k) {
      for (int j = jmin; j < jmax; ++j) {
        printf( " (%d, %d, %d) \n", i, j, k);
      }
    }
  }
  // _cstyle_ikj_loops_end

//----------------------------------------------------------------------------//
 
  std::cout << "\n Running RAJA nested loop order (I-outer, K-middle, J-inner)"
            << "...\n\n" << " (I, J, K)\n" << " ---------\n";

  // _raja_ikj_loops_start
  using IKJ_EXECPOL = RAJA::KernelPolicy<
                        RAJA::statement::For<0, RAJA::seq_exec,    // i
                          RAJA::statement::For<2, RAJA::seq_exec,  // k
                            RAJA::statement::For<1, RAJA::seq_exec,// j 
                              RAJA::statement::Lambda<0>
                            > 
                          > 
                        > 
                      >;

  RAJA::kernel<IKJ_EXECPOL>( RAJA::make_tuple(IRange, JRange, KRange),
  [=] (IIDX i, JIDX j, KIDX k) {
     printf( " (%d, %d, %d) \n", (int)(*i), (int)(*j), (int)(*k));
  });
  // _raja_ikj_loops_end


//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//
 
#if 0  // Enable this code block to generate compiler error.
//----------------------------------------------------------------------------//
// The following demonstrates that code will not compile if lambda argument
// types/order do not match the types/order For statements in the execution
// policy. To see this, enable this code section and try to compile this file.
//----------------------------------------------------------------------------//

  // _raja_compile_error_start
  RAJA::kernel<IKJ_EXECPOL>( RAJA::make_tuple(IRange, JRange, KRange),
  [=] (JIDX i, IIDX j, KIDX k) {
     printf( " (%d, %d, %d) \n", (int)(*i), (int)(*j), (int)(*k));
  });
  // _raja_compile_error_end

#endif

  std::cout << "\n DONE!...\n";

  return 0;
}

