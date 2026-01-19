//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing Reducer classes for util reduce tests
///

#ifndef __TEST_ALGORITHM_UTIL_REDUCE_HPP__
#define __TEST_ALGORITHM_UTIL_REDUCE_HPP__

#include "test-algorithm-reduce-utils.hpp"


template < typename test_policy >
using ForoneSynchronize = PolicySynchronize<test_equivalent_exec_policy<test_policy>>;


template < typename test_policy, typename platform = test_platform<test_policy> >
struct BinaryTreeReduce;

template < typename test_policy, typename platform = test_platform<test_policy> >
struct Accumulate;

template < typename test_policy, typename platform = test_platform<test_policy> >
struct KahanSum;


template < typename test_policy >
struct BinaryTreeReduce<test_policy, RunOnHost>
  : ForoneSynchronize<test_policy>
{
  using reduce_category = unordered_reduce_tag;
  using reduce_interface = reduce_interface_tag;
  using reduce_types = any_types_tag;

  const char* name()
  {
    return "RAJA::binary_tree_reduce";
  }

  template < typename T, typename... Args >
  void operator()(T* reduced_value, Args&&... args)
  {
    *reduced_value = RAJA::binary_tree_reduce(std::forward<Args>(args)...);
  }
};

template < typename test_policy >
struct Accumulate<test_policy, RunOnHost>
  : ForoneSynchronize<test_policy>
{
  using reduce_category = left_fold_reduce_tag;
  using reduce_interface = reduce_interface_tag;
  using reduce_types = any_types_tag;

  const char* name()
  {
    return "RAJA::accumulate";
  }

  template < typename T, typename... Args >
  void operator()(T* reduced_value, Args&&... args)
  {
    *reduced_value = RAJA::accumulate(std::forward<Args>(args)...);
  }
};

template < typename test_policy >
struct KahanSum<test_policy, RunOnHost>
  : ForoneSynchronize<test_policy>
{
  using reduce_category = unordered_reduce_tag;
  using reduce_interface = sum_interface_tag;
  using reduce_types = floating_point_types_tag;

  const char* name()
  {
    return "RAJA::kahan_sum";
  }

  template < typename T, typename... Args >
  void operator()(T* reduced_value, Args&&... args)
  {
    *reduced_value = RAJA::kahan_sum(std::forward<Args>(args)...);
  }
};

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)

template < typename test_policy >
struct BinaryTreeReduce<test_policy, RunOnDevice>
  : ForoneSynchronize<test_policy>
{
  using reduce_category = unordered_reduce_tag;
  using reduce_interface = reduce_interface_tag;
  using reduce_types = any_types_tag;

  std::string m_name;

  BinaryTreeReduce()
    : m_name(std::string("RAJA::binary_tree_reduce<") + test_policy_info<test_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename T, typename Container >
  void operator()(T* reduced_value, Container&& c)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      *reduced_value = RAJA::binary_tree_reduce(c);
    });
  }

  template < typename T, typename Container >
  void operator()(T* reduced_value, Container&& c, RAJA::detail::ContainerVal<Container> init)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      *reduced_value = RAJA::binary_tree_reduce(c, init);
    });
  }

  template < typename T, typename Container, typename BinaryOp >
  void operator()(T* reduced_value, Container&& c, RAJA::detail::ContainerVal<Container> init, BinaryOp op)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      *reduced_value = RAJA::binary_tree_reduce(c, init, op);
    });
  }
};

template < typename test_policy >
struct Accumulate<test_policy, RunOnDevice>
  : ForoneSynchronize<test_policy>
{
  using reduce_category = left_fold_reduce_tag;
  using reduce_interface = reduce_interface_tag;
  using reduce_types = any_types_tag;

  std::string m_name;

  Accumulate()
    : m_name(std::string("RAJA::accumulate<") + test_policy_info<test_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename T, typename Container >
  void operator()(T* reduced_value, Container&& c)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      *reduced_value = RAJA::accumulate(c);
    });
  }

  template < typename T, typename Container >
  void operator()(T* reduced_value, Container&& c, RAJA::detail::ContainerVal<Container> init)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      *reduced_value = RAJA::accumulate(c, init);
    });
  }

  template < typename T, typename Container, typename BinaryOp >
  void operator()(T* reduced_value, Container&& c, RAJA::detail::ContainerVal<Container> init, BinaryOp op)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      *reduced_value = RAJA::accumulate(c, init, op);
    });
  }
};


template < typename test_policy >
struct KahanSum<test_policy, RunOnDevice>
  : ForoneSynchronize<test_policy>
{
  using reduce_category = unordered_reduce_tag;
  using reduce_interface = sum_interface_tag;
  using reduce_types = floating_point_types_tag;

  std::string m_name;

  KahanSum()
    : m_name(std::string("RAJA::kahan_sum<") + test_policy_info<test_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename T, typename Container >
  void operator()(T* reduced_value, Container&& c)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      *reduced_value = RAJA::kahan_sum(c);
    });
  }

  template < typename T, typename Container >
  void operator()(T* reduced_value, Container&& c, RAJA::detail::ContainerVal<Container> init)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      *reduced_value = RAJA::kahan_sum(c, init);
    });
  }
};

#endif


using SequentialBinaryTreeReduceReducers =
  camp::list<
              BinaryTreeReduce<test_seq>
            >;

using SequentialAccumulateReduceReducers =
  camp::list<
              Accumulate<test_seq>
            >;

using SequentialKahanReduceReducers =
  camp::list<
              KahanSum<test_seq>
            >;

#if defined(RAJA_ENABLE_CUDA)

using CudaBinaryTreeReduceReducers =
  camp::list<
              BinaryTreeReduce<test_cuda>
            >;

using CudaAccumulateReduceReducers =
  camp::list<
              Accumulate<test_cuda>
            >;

using CudaKahanReduceReducers =
  camp::list<
              KahanSum<test_cuda>
            >;

#endif

#if defined(RAJA_ENABLE_HIP)

using HipBinaryTreeReduceReducers =
  camp::list<
              BinaryTreeReduce<test_hip>
            >;

using HipAccumulateReduceReducers =
  camp::list<
              Accumulate<test_hip>
            >;

using HipKahanReduceReducers =
  camp::list<
              KahanSum<test_hip>
            >;

#endif

#endif //__TEST_ALGORITHM_UTIL_REDUCE_HPP__

