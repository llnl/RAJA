//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Methods to construct index sets for RAJA tests.
//

#ifndef __TEST_FORALL_INDEXSET_BUILD_HPP__
#define __TEST_FORALL_INDEXSET_BUILD_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/resource.hpp"

#include <random>

//
// Utility routine to construct index set with mix of Range, RangeStride, 
// and List segments to use in various tests.
//
template <typename INDEX_TYPE,
          typename RANGE_TYPE,
          typename RANGESTRIDE_TYPE,
          typename LIST_TYPE>
void buildIndexSet( 
  RAJA::TypedIndexSet< RANGE_TYPE, RANGESTRIDE_TYPE, LIST_TYPE >& iset, 
  std::vector<INDEX_TYPE>& indices_out,
  camp::resources::Resource working_res )
{
  //
  //  Build vector of integers for creating List segments.
  //
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<INDEX_TYPE> lindices;
  INDEX_TYPE idx = 0;
  while (lindices.size() < 3000) {
    double dval = dist(gen);
    if (dval > 0.3) {
      lindices.push_back(idx);
    }
    idx++;
  }

  //
  // Construct a mix of Range, RangeStride, and List segments 
  // and add them to index set
  //
  INDEX_TYPE rbeg = 0;
  INDEX_TYPE rend = 0;
  INDEX_TYPE stride = 0;
  INDEX_TYPE last_idx = 0;
  INDEX_TYPE lseg_len = static_cast<INDEX_TYPE>( lindices.size() );
  std::vector<INDEX_TYPE> lseg(lseg_len);
  std::vector<INDEX_TYPE> lseg_vec(lseg_len);

  indices_out.clear(); 

  // Create empty Range segment
  rbeg = 1;
  rend = 1;
  iset.push_back(RANGE_TYPE(rbeg, rend));
  last_idx = rend;

  // Create Range segment
  rbeg = 1;
  rend = 1578;
  iset.push_back(RANGE_TYPE(rbeg, rend));
  for (INDEX_TYPE i = rbeg; i < rend; ++i) { 
    indices_out.push_back( i ); 
  }
  last_idx = rend;

  // Create List segment
  for (INDEX_TYPE i = 0; i < lseg_len; ++i) {
    lseg[i] = lindices[i] + last_idx + 3;
    indices_out.push_back( lseg[i] );
  }
  iset.push_back(LIST_TYPE(&lseg[0], lseg_len, working_res));
  last_idx = lseg[lseg_len - 1];

  // Create List segment using alternate ctor
  for (INDEX_TYPE i = 0; i < lseg_len; ++i) {
    lseg_vec[i] = lindices[i] + last_idx + 3;
    indices_out.push_back( lseg_vec[i] );
  }
  iset.push_back(LIST_TYPE(lseg_vec, working_res));
  last_idx = lseg_vec[lseg_len - 1];

  // Create Range-stride segment
  rbeg = last_idx + 16;
  rend = rbeg + 2040;
  stride = 3;
  iset.push_back(RANGESTRIDE_TYPE(rbeg, rend, stride));
  for (INDEX_TYPE i = rbeg; i < rend; i += stride) { 
    indices_out.push_back( i ); 
  }
  last_idx = rend;

  // Create Range segment
  rbeg = last_idx + 4;
  rend = rbeg + 2759;
  iset.push_back(RANGE_TYPE(rbeg, rend));
  for (INDEX_TYPE i = rbeg; i < rend; ++i) { 
    indices_out.push_back( i ); 
  }
  last_idx = rend;

  // Create List segment
  for (INDEX_TYPE i = 0; i < lseg_len; ++i) {
    lseg[i] = lindices[i] + last_idx + 5;
    indices_out.push_back( lseg[i] );
  }
  iset.push_back(LIST_TYPE(&lseg[0], lseg_len, working_res));
  last_idx = lseg[lseg_len - 1];

  // Create Range segment
  rbeg = last_idx + 1;
  rend = rbeg + 320;
  iset.push_back(RANGE_TYPE(rbeg, rend));
  for (INDEX_TYPE i = rbeg; i < rend; ++i) { 
    indices_out.push_back( i ); 
  }
  last_idx = rend;

  // Create List segment using alternate ctor
  for (INDEX_TYPE i = 0; i < lseg_len; ++i) {
    lseg_vec[i] = lindices[i] + last_idx + 7;
    indices_out.push_back( lseg_vec[i] );
  }
  iset.push_back(LIST_TYPE(lseg_vec, working_res));
  last_idx = lseg_vec[lseg_len - 1];
}

#endif  // __TEST_FORALL_INDEXSET_BUILD_HPP__
