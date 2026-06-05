//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing Sorter classes for sort tests
///

#ifndef __TEST_UNIT_ALGORITHM_SORT_HPP__
#define __TEST_UNIT_ALGORITHM_SORT_HPP__

#include "test-algorithm-sort-utils.hpp"

template < typename policy >
struct PolicySort
  : PolicySynchronize<policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;
  using supports_resource = std::true_type;

  std::string m_name;

  PolicySort()
    : m_name("RAJA::sort<unknown>")
  { }

  PolicySort(std::string const& policy_name)
    : m_name(std::string("RAJA::sort<") + policy_name + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::sort<policy>(std::forward<Args>(args)...);
  }
};

template < typename policy >
struct PolicySortPairs
  : PolicySynchronize<policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;
  using supports_resource = std::true_type;

  std::string m_name;

  PolicySortPairs()
    : m_name("RAJA::sort<unknown>[pairs]")
  { }

  PolicySortPairs(std::string const& policy_name)
    : m_name(std::string("RAJA::sort<") + policy_name + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::sort_pairs<policy>(std::forward<Args>(args)...);
  }
};


using SequentialSortSorters =
  camp::list<
              PolicySort<RAJA::seq_exec>,
              PolicySortPairs<RAJA::seq_exec>
            >;

#if defined(RAJA_ENABLE_OPENMP)

using OpenMPSortSorters =
  camp::list<
              PolicySort<RAJA::omp_parallel_for_exec>,
              PolicySortPairs<RAJA::omp_parallel_for_exec>
            >;

#endif

#if defined(RAJA_ENABLE_CUDA)

using CudaSortSorters =
  camp::list<
              PolicySort<RAJA::cuda_exec<128>>,
              PolicySortPairs<RAJA::cuda_exec<128>>,
              PolicySort<RAJA::cuda_exec_explicit<128, 2>>
            >;

#endif

#if defined(RAJA_ENABLE_HIP)

using HipSortSorters =
  camp::list<
              PolicySort<RAJA::hip_exec<128>>,
              PolicySortPairs<RAJA::hip_exec<128>>
            >;

#endif

#endif //__TEST_UNIT_ALGORITHM_SORT_HPP__

