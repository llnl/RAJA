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

#ifndef __TEST_WORKGROUP_WORKSTORAGECONSTRUCTOR__
#define __TEST_WORKGROUP_WORKSTORAGECONSTRUCTOR__

#include "RAJA_test-workgroup.hpp"
#include "test-util-workgroup-WorkStorage.hpp"

#include <random>
#include <array>
#include <cstddef>


template <typename StoragePolicy,
          typename DispatchTyper,
          typename Allocator
          >
void testWorkGroupWorkStorageConstructor()
{
  bool success = true;

  static constexpr auto platform = RAJA::Platform::host;
  using DispatchPolicy = typename DispatchTyper::template type<>;
  using Dispatcher_type = RAJA::detail::Dispatcher<
      platform, DispatchPolicy, void, void*, bool*, bool*>;
  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      Dispatcher_type
                                                    >;

  {
    auto test_empty = [&](WorkStorage_type& container) {

      ASSERT_EQ(container.size(), (size_t)(0));
      ASSERT_EQ(container.storage_size(), (size_t)0);
    };

    WorkStorage_type container(Allocator{});

    test_empty(container);

    container.clear();

    test_empty(container);


    WorkStorage_type container2(std::move(container));

    test_empty(container);
    test_empty(container2);


    WorkStorage_type container3(Allocator{});
    container3 = std::move(container2);

    test_empty(container2);
    test_empty(container3);
  }

  ASSERT_TRUE(success);
}


template <typename T>
class WorkGroupBasicWorkStorageConstructorUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicWorkStorageConstructorUnitTest);


TYPED_TEST_P(WorkGroupBasicWorkStorageConstructorUnitTest, BasicWorkGroupWorkStorageConstructor)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using DispatchTyper = typename camp::at<TypeParam, camp::num<1>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<2>>::type;

  testWorkGroupWorkStorageConstructor< StoragePolicy, DispatchTyper, Allocator >();
}


#endif  //__TEST_WORKGROUP_WORKSTORAGECONSTRUCTOR__
