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
#include <vector>
#include <algorithm>

#include "RAJA/RAJA.hpp"

#include "camp/resource.hpp"

/*
 *  Segments and Index Sets exercise
 *
 *  In this exercise, you will learn how to create RAJA segments and index sets
 *  and use them to execute kernels. There are no computations performed in the
 *  exercises and no parallel execution. The kernels contain only print 
 *  statements to illustrate various iteration patterns. Thus, all kernels
 *  look the same. The only thing that changes in these versions is the object 
 *  passed to the 'forall' method that defines the iteration space.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  TypedRangeSegment iteration space
 *    -  TypedRangeStrideSegment iteration space
 *    -  TypedListSegment iteration space
 *    -  TypedIndexSet segment container
 *    -  Hierarchical execution policies
 */

//----------------------------------------------------------------------------//
// Define aliases for types used in the exercises
// (so example code is less verbose)
//----------------------------------------------------------------------------//
// _raja_segment_type_start
using IdxType = int;
using RangeSegType = RAJA::TypedRangeSegment<IdxType>;
using RangeStrideSegType = RAJA::TypedRangeStrideSegment<IdxType>;
using ListSegType = RAJA::TypedListSegment<IdxType>;
using IndexSetType = RAJA::TypedIndexSet< RangeSegType, ListSegType >;
// _raja_segment_type_end


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA segments index sets and index sets...\n";

// Resource object used to construct list segment objects with indices
// living in host (CPU) memory.
  camp::resources::Resource host_res{camp::resources::Host()};


//----------------------------------------------------------------------------//
// Stride-1 iteration spaces
//----------------------------------------------------------------------------//

  std::cout << "\n Running C-version range kernel...\n";

// _cstyle_range1_start
  for (IdxType i = 0; i < 20; i++) {
    std::cout << i << "  "; 
  }
// _cstyle_range1_end

  std::cout << std::endl;

//----------------------------------//

  std::cout << "\n Running RAJA range kernel...\n";

  // _raja_range1_start
  RAJA::forall<RAJA::seq_exec>(RangeSegType(0, 20), [=] (IdxType i) {
    std::cout << i << "  ";
  });
  // _raja_range1_end

  std::cout << std::endl;

//----------------------------------//

  std::cout << "\n Running RAJA stride-1 range kernel...\n";

  // _raja_striderange1_start
  RAJA::forall<RAJA::seq_exec>(RangeStrideSegType(0, 20, 1), [=] (IdxType i) {
    std::cout << i << "  ";
  });
  // _raja_striderange1_end

  std::cout << std::endl;

//----------------------------------//

  std::cout << "\n Running RAJA stride-1 list kernel...\n";

  // _raja_list1_start
  //
  // Collect indices in a vector to create list segment
  //
  std::vector<IdxType> idx;
  for (IdxType i = 0; i < 20; ++i) {
    idx.push_back(i); 
  } 

  ListSegType idx_list1( idx, host_res );

  RAJA::forall<RAJA::seq_exec>(idx_list1, [=] (IdxType i) {
    std::cout << i << "  ";
  });
  // _raja_list1_end

  std::cout << std::endl;

//----------------------------------//

  std::cout << "\n Running C-style stride-1 list kernel...\n";

  // _cstyle_list1_start
  IdxType iis = static_cast<IdxType>(idx.size());  // to avoid compiler warning
  for (IdxType ii = 0; ii < iis; ++ii) { 
    std::cout << idx[ ii ] << "  ";
  }
  // _cstyle_list1_end

  std::cout << std::endl;

//----------------------------------------------------------------------------//
// Negative stride iteration spaces
//----------------------------------------------------------------------------//

  std::cout << "\n Running C-version negative stride kernel...\n";

  // _cstyle_negstriderange1_start
  for (IdxType i = 19; i > -1; i--) {
    std::cout << i << "  ";
  }
  // _cstyle_negstriderange1_end

  std::cout << std::endl;

//----------------------------------//

  std::cout << "\n Running RAJA negative stride kernel...\n";

  // _raja_negstriderange1_start
  RAJA::forall<RAJA::seq_exec>(RangeStrideSegType(19, -1, -1), [=] (IdxType i) {
    std::cout << i << "  ";
  });
  // _raja_negstriderange1_end

  std::cout << std::endl;

//----------------------------------//
// List variant
//----------------------------------//

  std::cout << "\n Running RAJA negative stride list kernel...\n";

  // _raja_negstridelist1_start
  //
  // Reverse the order of indices in the vector
  //
  std::reverse( idx.begin(), idx.end() );
  ListSegType idx_list1_reverse( &idx[0], idx.size(), host_res );

  RAJA::forall<RAJA::seq_exec>(idx_list1_reverse, [=] (IdxType i) {
    std::cout << i << "  ";
  });
  // _raja_negstridelist1_end

  std::cout << std::endl;

//----------------------------------------------------------------------------//
// Non-unit uniform stride iteration spaces
//----------------------------------------------------------------------------//

  std::cout << "\n Running C-version stride-2 range kernel...\n";

  // _cstyle_range2_start
  for (IdxType i = 0; i < 20; i += 2) {
    std::cout << i << "  ";
  }
  // _cstyle_range2_end

  std::cout << std::endl;

//----------------------------------//

  std::cout << "\n Running RAJA stride-2 range kernel...\n";

  // _raja_range2_start
  RAJA::forall<RAJA::seq_exec>(RangeStrideSegType(0, 20, 2), [=] (IdxType i) {
    std::cout << i << "  ";
  });
  // _raja_range2_end

  std::cout << std::endl;

//----------------------------------//

  std::cout << "\n Running RAJA stride-3 range kernel...\n";

  // _raja_range3_start
  RAJA::forall<RAJA::seq_exec>(RangeStrideSegType(0, 20, 3), [=] (IdxType i) {
    std::cout << i << "  ";
  });
  // _raja_range3_end

  std::cout << std::endl;

//----------------------------------------------------------------------------//
// IndexSets: complex iteration spaces
//----------------------------------------------------------------------------//

//
// Sequential index set execution policy used in several of the following
// example implementations.
//

  std::cout << "\n Running RAJA index set (2 RangeSegments) kernel...\n";

  // _raja_indexset_2ranges_start
  using SEQ_ISET_EXECPOL = RAJA::ExecPolicy<RAJA::seq_segit,
                                            RAJA::seq_exec>;

  IndexSetType is2;
  is2.push_back( RangeSegType(0, 10) );
  is2.push_back( RangeSegType(15, 20) );
  
  RAJA::forall<SEQ_ISET_EXECPOL>(is2, [=] (IdxType i) {
    std::cout << i << "  ";
  });
  // _raja_indexset_2ranges_end

  std::cout << std::endl;

//----------------------------------//

  std::cout << "\n Running C-version of two segment kernel...\n";

  // _cstyle_2ranges_start
  for (IdxType i = 0; i < 10; ++i) {
    std::cout << i << "  ";
  }
  for (IdxType i = 15; i < 20; ++i) {
    std::cout << i << "  ";
  }
  // _cstyle_2ranges_end

  std::cout << std::endl;

//----------------------------------//

  std::cout << "\n Running RAJA index set (3 segments) kernel...\n";

  // _raja_indexset_3segs_start
  IndexSetType is3;

  is3.push_back( RangeSegType(0, 8) );

  IdxType indx[ ] = {10, 11, 14, 20, 22};
  ListSegType list2( indx, 5, host_res );
  is3.push_back( list2 );

  is3.push_back( RangeSegType(24, 28) );
 
  RAJA::forall<SEQ_ISET_EXECPOL>(is3, [=] (IdxType i) {
    std::cout << i << "  ";
  });
  // _raja_indexset_3segs_end

  std::cout << std::endl;

//----------------------------------------------------------------------------//

  std::cout << "\n DONE!...\n";
 
  return 0;
}

