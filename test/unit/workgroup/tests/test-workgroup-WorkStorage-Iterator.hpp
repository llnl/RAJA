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

#ifndef __TEST_WORKGROUP_WORKSTORAGEITERATOR__
#define __TEST_WORKGROUP_WORKSTORAGEITERATOR__

#include "RAJA_test-workgroup.hpp"
#include "test-util-workgroup-WorkStorage.hpp"

#include <random>
#include <array>
#include <cstddef>


template <typename StoragePolicy,
          typename DispatchTyper,
          typename Allocator
          >
void testWorkGroupWorkStorageIterator()
{
  bool success = true;

  using callable = TestCallable<int>;

  static constexpr auto platform = RAJA::Platform::host;
  using DispatchPolicy = typename DispatchTyper::template type<callable>;
  using Dispatcher_type = RAJA::detail::Dispatcher<
      platform, DispatchPolicy, void, void*, bool*, bool*>;
  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      Dispatcher_type
                                                    >;


  const Dispatcher_type* dispatcher = RAJA::detail::get_Dispatcher<
      callable, Dispatcher_type>(RAJA::seq_work{});

  {
    WorkStorage_type container(Allocator{});

    ASSERT_EQ(container.end()-container.begin(), (std::ptrdiff_t)0);
    ASSERT_FALSE(container.begin() < container.end());
    ASSERT_FALSE(container.begin() > container.end());
    ASSERT_TRUE(container.begin() == container.end());
    ASSERT_FALSE(container.begin() != container.end());
    ASSERT_TRUE(container.begin() <= container.end());
    ASSERT_TRUE(container.begin() >= container.end());

    container.template emplace<callable>(dispatcher, callable{-1});

    ASSERT_EQ(container.end()-container.begin(), (std::ptrdiff_t)1);
    ASSERT_TRUE(container.begin() < container.end());
    ASSERT_FALSE(container.begin() > container.end());
    ASSERT_FALSE(container.begin() == container.end());
    ASSERT_TRUE(container.begin() != container.end());
    ASSERT_TRUE(container.begin() <= container.end());
    ASSERT_FALSE(container.begin() >= container.end());

    {
      auto iter = container.begin();

      ASSERT_EQ(&*iter, &iter[0]);

      ASSERT_EQ(iter++, container.begin());
      ASSERT_EQ(iter--, container.end());
      ASSERT_EQ(++iter, container.end());
      ASSERT_EQ(--iter, container.begin());

      ASSERT_EQ(iter+1, container.end());
      ASSERT_EQ(1+iter, container.end());
      ASSERT_EQ(++iter, container.end());
      ASSERT_EQ(iter-1, container.begin());
      ASSERT_EQ(iter-=1, container.begin());
      ASSERT_EQ(iter+=1, container.end());
    }
  }

  ASSERT_TRUE(success);
}


template <typename T>
class WorkGroupBasicWorkStorageIteratorUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicWorkStorageIteratorUnitTest);


TYPED_TEST_P(WorkGroupBasicWorkStorageIteratorUnitTest, BasicWorkGroupWorkStorageIterator)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using DispatchTyper = typename camp::at<TypeParam, camp::num<1>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<2>>::type;

  testWorkGroupWorkStorageIterator< StoragePolicy, DispatchTyper, Allocator >();
}

#endif  //__TEST_WORKGROUP_WORKSTORAGEITERATOR__
