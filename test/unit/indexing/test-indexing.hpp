//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_INDEXING_UTILS_HPP__
#define __TEST_INDEXING_UTILS_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"

#include "RAJA_unit-test-for3d3d.hpp"


//
// List of named_dims
//
using NamedDimensionTypeList =
    camp::list<
                camp::integral_constant<RAJA::named_dim, RAJA::named_dim::x>,
                camp::integral_constant<RAJA::named_dim, RAJA::named_dim::y>,
                camp::integral_constant<RAJA::named_dim, RAJA::named_dim::z>
              >;

//
// List of sizes
//
using SizeTypeList =
    camp::list<
                camp::integral_constant<int, RAJA::named_usage::ignored>,
                camp::integral_constant<int, RAJA::named_usage::unspecified>,
                camp::integral_constant<int, 1>,
                camp::integral_constant<int, 7>
              >;

//
// Holder for indexing templates
//
template < template < RAJA::named_dim, int, int > class T >
struct indexing_holder
{
  template < RAJA::named_dim dim, int BLOCK_SIZE, int GRID_SIZE >
  using type = T<dim, BLOCK_SIZE, GRID_SIZE>;
};

//
// List of indexing holder types
//
#if defined(RAJA_ENABLE_CUDA)
using CudaIndexingHolderList = camp::list< indexing_holder<RAJA::cuda::IndexGlobal> >;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipIndexingHolderList = camp::list< indexing_holder<RAJA::hip::IndexGlobal> >;
#endif

#endif  // __TEST_INDEXING_UTILS_HPP__
