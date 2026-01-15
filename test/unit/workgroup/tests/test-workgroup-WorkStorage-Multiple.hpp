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

#ifndef __TEST_WORKGROUP_WORKSTORAGEMULTIPLE__
#define __TEST_WORKGROUP_WORKSTORAGEMULTIPLE__

#include "RAJA_test-workgroup.hpp"
#include "test-util-workgroup-WorkStorage.hpp"

#include <random>
#include <array>
#include <cstddef>


template <typename StoragePolicy,
          typename DispatchTyper,
          typename Allocator
          >
void testWorkGroupWorkStorageMultiple(
    const size_t num0, const size_t num1, const size_t num2)
{
  bool success = true;

  using type0 = double;
  using type1 = TestArray<double, 6>;
  using type2 = TestArray<double, 14>;

  auto make_type0 = [](double init_val, size_t i) {
    type0 obj(init_val - (double)i);
    return obj;
  };
  auto make_type1 = [](double init_val, size_t i) {
    type1 obj{};
    for (size_t j = 0; j < 6; ++j) {
      obj[j] = init_val + 10.0 * j + i;
    }
    return obj;
  };
  auto make_type2 = [](double init_val, size_t i) {
    type2 obj{};
    for (size_t j = 0; j < 14; ++j) {
      obj[j] = init_val + 10.0 * j + i;
    }
    return obj;
  };

  using callable0 = TestCallable<type0>;
  using callable1 = TestCallable<type1>;
  using callable2 = TestCallable<type2>;

  static constexpr auto platform = RAJA::Platform::host;
  using DispatchPolicy = typename DispatchTyper::template type<callable0, callable1, callable2>;
  using Dispatcher_type = RAJA::detail::Dispatcher<
      platform, DispatchPolicy, void, void*, bool*, bool*>;
  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      Dispatcher_type
                                                    >;
  using WorkStruct_type = typename WorkStorage_type::value_type;


  const Dispatcher_type* dispatcher0 = RAJA::detail::get_Dispatcher<
      callable0, Dispatcher_type>(RAJA::seq_work{});
  const Dispatcher_type* dispatcher1 = RAJA::detail::get_Dispatcher<
      callable1, Dispatcher_type>(RAJA::seq_work{});
  const Dispatcher_type* dispatcher2 = RAJA::detail::get_Dispatcher<
      callable2, Dispatcher_type>(RAJA::seq_work{});

  {
    auto test_empty = [&](WorkStorage_type& container) {

      ASSERT_EQ(container.size(), (size_t)(0));
      ASSERT_EQ(container.storage_size(), (size_t)0);
    };

    auto fill_contents = [&](WorkStorage_type& container, double init_val0, double init_val1, double init_val2) {

      std::vector<callable0> vec0;
      vec0.reserve(num0);
      for (size_t i = 0; i < num0; ++i) {
        vec0.emplace_back(make_type0(init_val0, i));
        ASSERT_FALSE(vec0[i].move_constructed);
        ASSERT_FALSE(vec0[i].moved_from);
        container.template emplace<callable0>(dispatcher0, std::move(vec0[i]));
        ASSERT_FALSE(vec0[i].move_constructed);
        ASSERT_TRUE (vec0[i].moved_from);
      }

      std::vector<callable1> vec1;
      vec1.reserve(num1);
      for (size_t i = 0; i < num1; ++i) {
        vec1.emplace_back(make_type1(init_val1, i));
        ASSERT_FALSE(vec1[i].move_constructed);
        ASSERT_FALSE(vec1[i].moved_from);
        container.template emplace<callable1>(dispatcher1, std::move(vec1[i]));
        ASSERT_FALSE(vec1[i].move_constructed);
        ASSERT_TRUE (vec1[i].moved_from);
      }

      std::vector<callable2> vec2;
      vec2.reserve(num2);
      for (size_t i = 0; i < num2; ++i) {
        vec2.emplace_back(make_type2(init_val2, i));
        ASSERT_FALSE(vec2[i].move_constructed);
        ASSERT_FALSE(vec2[i].moved_from);
        container.template emplace<callable2>(dispatcher2, std::move(vec2[i]));
        ASSERT_FALSE(vec2[i].move_constructed);
        ASSERT_TRUE (vec2[i].moved_from);
      }

      ASSERT_EQ(container.size(), num0+num1+num2);
      ASSERT_GE(container.storage_size(),
          num0*sizeof(callable0) +
          num1*sizeof(callable1) +
          num2*sizeof(callable2));
    };

    auto test_contents = [&](WorkStorage_type& container, double init_val0, double init_val1, double init_val2) {

      ASSERT_EQ(container.size(), num0+num1+num2);
      ASSERT_GE(container.storage_size(),
          num0*sizeof(callable0) +
          num1*sizeof(callable1) +
          num2*sizeof(callable2));

      {
        auto iter = container.begin();

        for (size_t i = 0; i < num0; ++i) {
          type0 val{};
          bool move_constructed = false;
          bool moved_from = true;
          WorkStruct_type::host_call(&*iter, (void*)&val, &move_constructed, &moved_from);

          type0 expected = make_type0(init_val0, i);
          ASSERT_EQ(val, expected);
          ASSERT_TRUE(move_constructed);
          ASSERT_FALSE(moved_from);

          ++iter;
        }

        for (size_t i = 0; i < num1; ++i) {
          type1 val{};
          bool move_constructed = false;
          bool moved_from = true;
          WorkStruct_type::host_call(&*iter, (void*)&val, &move_constructed, &moved_from);

          type1 expected = make_type1(init_val1, i);
          ASSERT_EQ(val, expected);
          ASSERT_TRUE(move_constructed);
          ASSERT_FALSE(moved_from);

          ++iter;
        }

        for (size_t i = 0; i < num2; ++i) {
          type2 val{};
          bool move_constructed = false;
          bool moved_from = true;
          WorkStruct_type::host_call(&*iter, (void*)&val, &move_constructed, &moved_from);

          type2 expected = make_type2(init_val2, i);
          ASSERT_EQ(val, expected);
          ASSERT_TRUE(move_constructed);
          ASSERT_FALSE(moved_from);

          ++iter;
        }

        ASSERT_EQ(iter, container.end());
      }
    };

    WorkStorage_type container(Allocator{});

    test_empty(container);
    fill_contents(container, 1.0, 100.0, 1000.0);

    container.clear();

    test_empty(container);
    fill_contents(container, 1.0, 100.0, 1000.0);
    test_contents(container, 1.0, 100.0, 1000.0);


    WorkStorage_type container2(std::move(container));

    test_empty(container);
    test_contents(container2, 1.0, 100.0, 1000.0);


    WorkStorage_type container3(Allocator{});
    container3 = std::move(container2);

    test_empty(container2);
    test_contents(container3, 1.0, 100.0, 1000.0);


    WorkStorage_type container4(Allocator{});

    fill_contents(container4, 1.5, 100.5, 1000.5);
    test_contents(container4, 1.5, 100.5, 1000.5);

    container4 = std::move(container3);

    test_empty(container3);
    test_contents(container4, 1.0, 100.0, 1000.0);

  }

  ASSERT_TRUE(success);
}


template <typename T>
class WorkGroupBasicWorkStorageMultipleUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicWorkStorageMultipleUnitTest);


TYPED_TEST_P(WorkGroupBasicWorkStorageMultipleUnitTest, BasicWorkGroupWorkStorageMultiple)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using DispatchTyper = typename camp::at<TypeParam, camp::num<1>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<2>>::type;

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, 128);

  testWorkGroupWorkStorageMultiple< StoragePolicy, DispatchTyper, Allocator >(
      dist(rng), dist(rng), dist(rng));
}

#endif  //__TEST_WORKGROUP_WORKSTORAGEMULTIPLE__
