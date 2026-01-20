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

#ifndef __TEST_WORKGROUP_WORKSTORAGEINSERTCALL__
#define __TEST_WORKGROUP_WORKSTORAGEINSERTCALL__

#include "RAJA_test-workgroup.hpp"
#include "test-util-workgroup-WorkStorage.hpp"

#include <random>
#include <array>
#include <cstddef>


template <typename StoragePolicy,
          typename DispatchTyper,
          typename Allocator
          >
void testWorkGroupWorkStorageInsertCall()
{
  bool success = true;

  using callable = TestCallable<double>;

  static constexpr auto platform = RAJA::Platform::host;
  using DispatchPolicy = typename DispatchTyper::template type<callable>;
  using Dispatcher_type = RAJA::detail::Dispatcher<
      platform, DispatchPolicy, void, void*, bool*, bool*>;
  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      Dispatcher_type
                                                    >;
  using WorkStruct_type = typename WorkStorage_type::value_type;

  const Dispatcher_type* dispatcher = RAJA::detail::get_Dispatcher<
      callable, Dispatcher_type>(RAJA::seq_work{});

  {
    auto test_empty = [&](WorkStorage_type& container) {

      ASSERT_EQ(container.size(), (size_t)(0));
      ASSERT_EQ(container.storage_size(), (size_t)0);
    };

    auto fill_contents = [&](WorkStorage_type& container, double init_val) {

      callable c(init_val);

      ASSERT_FALSE(c.move_constructed);
      ASSERT_FALSE(c.moved_from);

      container.template emplace<callable>(dispatcher, std::move(c));

      ASSERT_FALSE(c.move_constructed);
      ASSERT_TRUE(c.moved_from);

      ASSERT_EQ(container.size(), (size_t)1);
      ASSERT_TRUE(container.storage_size() >= sizeof(callable));
    };

    auto test_contents = [&](WorkStorage_type& container, double init_val) {

      ASSERT_EQ(container.size(), (size_t)1);
      ASSERT_TRUE(container.storage_size() >= sizeof(callable));

      auto iter = container.begin();

      double test_val = -1;
      bool move_constructed = false;
      bool moved_from = true;
      WorkStruct_type::host_call(&*iter, (void*)&test_val, &move_constructed, &moved_from);

      ASSERT_EQ(test_val, init_val);
      ASSERT_TRUE(move_constructed);
      ASSERT_FALSE(moved_from);
    };


    WorkStorage_type container(Allocator{});

    test_empty(container);

    container.clear();

    test_empty(container);
    fill_contents(container, 1.23456789);
    test_contents(container, 1.23456789);


    WorkStorage_type container2(std::move(container));

    test_empty(container);
    test_contents(container2, 1.23456789);


    WorkStorage_type container3(Allocator{});
    container3 = std::move(container2);

    test_empty(container2);
    test_contents(container3, 1.23456789);


    WorkStorage_type container4(Allocator{});

    fill_contents(container4, 2.34567891);
    test_contents(container4, 2.34567891);

    container4 = std::move(container3);

    test_empty(container3);
    test_contents(container4, 1.23456789);
  }

  ASSERT_TRUE(success);
}


template <typename T>
class WorkGroupBasicWorkStorageInsertCallUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicWorkStorageInsertCallUnitTest);


TYPED_TEST_P(WorkGroupBasicWorkStorageInsertCallUnitTest, BasicWorkGroupWorkStorageInsertCall)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using DispatchTyper = typename camp::at<TypeParam, camp::num<1>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<2>>::type;

  testWorkGroupWorkStorageInsertCall< StoragePolicy, DispatchTyper, Allocator >();
}

#endif  //__TEST_WORKGROUP_WORKSTORAGEINSERTCALL__
