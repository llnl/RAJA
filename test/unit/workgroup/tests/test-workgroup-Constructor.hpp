//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup constructors.
///

#ifndef __TEST_WORKGROUP_CONSTRUCTOR__
#define __TEST_WORKGROUP_CONSTRUCTOR__

#include "RAJA_test-workgroup.hpp"


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename DispatchTyper,
          typename IndexType,
          typename Allocator
          >
struct testWorkGroupConstructorSingle {
template < typename ... Xargs >
void operator()(RAJA::xargs<Xargs...>) const
{
  bool success = true;

  using DispatchPolicy = typename DispatchTyper::template type<>;

  {
    RAJA::WorkPool<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy, DispatchPolicy>,
                    IndexType,
                    RAJA::xargs<Xargs...>,
                    Allocator
                  >
        pool(Allocator{});

    ASSERT_EQ(pool.num_loops(), (size_t)0);
    ASSERT_EQ(pool.storage_bytes(), (size_t)0);

    RAJA::WorkGroup<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy, DispatchPolicy>,
                    IndexType,
                    RAJA::xargs<Xargs...>,
                    Allocator
                  >
        group = pool.instantiate();

    ASSERT_EQ(pool.num_loops(), (size_t)0);
    ASSERT_EQ(pool.storage_bytes(), (size_t)0);

    RAJA::WorkSite<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy, DispatchPolicy>,
                    IndexType,
                    RAJA::xargs<Xargs...>,
                    Allocator
                  >
        site = group.run(Xargs{}...);

    using resource_type = typename RAJA::WorkPool<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy, DispatchPolicy>,
                    IndexType,
                    RAJA::xargs<Xargs...>,
                    Allocator
                  >::resource_type;
    auto e = resource_type::get_default().get_event();
    e.wait();

    pool.clear();
    group.clear();
    site.clear();

    ASSERT_EQ(pool.num_loops(), (size_t)0);
    ASSERT_EQ(pool.storage_bytes(), (size_t)0);
  }

  ASSERT_TRUE(success);
}
};


#if defined(RAJA_ENABLE_HIP) && !defined(RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL)

/// leave unsupported types untested
template <size_t BLOCK_SIZE, bool Async,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator
          >
struct testWorkGroupConstructorSingle<RAJA::hip_work<BLOCK_SIZE, Async>,
                                      RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average,
                                      StoragePolicy,
                                      detail::indirect_function_call_dispatch_typer,
                                      IndexType,
                                      Allocator> {
template < typename ... Xargs >
void operator()(RAJA::xargs<Xargs...>) const
{ }
};
///
template <size_t BLOCK_SIZE, bool Async,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator
          >
struct testWorkGroupConstructorSingle<RAJA::hip_work<BLOCK_SIZE, Async>,
                                      RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average,
                                      StoragePolicy,
                                      detail::indirect_virtual_function_dispatch_typer,
                                      IndexType,
                                      Allocator> {
template < typename ... Xargs >
void operator()(RAJA::xargs<Xargs...>) const
{ }
};

#endif


template <typename T>
class WorkGroupBasicConstructorSingleUnitTest : public ::testing::Test
{
};


TYPED_TEST_SUITE_P(WorkGroupBasicConstructorSingleUnitTest);

TYPED_TEST_P(WorkGroupBasicConstructorSingleUnitTest, BasicWorkGroupConstructorSingle)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using DispatchTyper = typename camp::at<TypeParam, camp::num<3>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<4>>::type;
  using Xargs = typename camp::at<TypeParam, camp::num<5>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<6>>::type;

  testWorkGroupConstructorSingle< ExecPolicy, OrderPolicy, StoragePolicy, DispatchTyper, IndexType, Allocator >{}(Xargs{});
}

#endif  //__TEST_WORKGROUP_CONSTRUCTOR__
