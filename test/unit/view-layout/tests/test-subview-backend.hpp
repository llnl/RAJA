//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef TEST_SUBVIEW_BACKEND_HPP
#define TEST_SUBVIEW_BACKEND_HPP

#include "RAJA/RAJA.hpp"

#include "RAJA_unit-test-forone.hpp"
#include "RAJA_unit-test-policy.hpp"

template<typename TestPolicy>
class SubViewBackendTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P(SubViewBackendTest);

template<typename TestPolicy>
void testSubViewBackendReadWrite()
{
  constexpr RAJA::Index_type rows        = 3;
  constexpr RAJA::Index_type cols        = 6;
  constexpr RAJA::Index_type num_values  = rows * cols;
  constexpr RAJA::Index_type num_results = 14;

  auto host_resource    = get_test_resource<test_seq>();
  auto working_resource = get_test_resource<TestPolicy>();

  auto* data = host_resource.template allocate<RAJA::Index_type>(num_values);
  auto* results =
      host_resource.template allocate<RAJA::Index_type>(num_results);

  for (RAJA::Index_type i = 0; i < num_values; ++i)
  {
    data[i] = i + 1;
  }
  for (RAJA::Index_type i = 0; i < num_results; ++i)
  {
    results[i] = -1;
  }

  data = test_reallocate(working_resource, host_resource, data, num_values);
  results =
      test_reallocate(working_resource, host_resource, results, num_results);

  RAJA::View<RAJA::Index_type, RAJA::Layout<2>> view(
      data, RAJA::Layout<2>(rows, cols));

  forone<TestPolicy>([=] RAJA_HOST_DEVICE() {
    auto subview = RAJA::make_subview(view, RAJA::RangeSlice<> {1, 3},
                                      RAJA::StridedSlice<> {1, 6, 2});
    auto adapter = RAJA::SlicingAdapter(view, RAJA::RangeSlice<> {1, 3},
                                        RAJA::StridedSlice<> {1, 6, 2});

    results[0] = subview(0, 0);
    results[1] = subview(0, 1);
    results[2] = subview(0, 2);
    results[3] = subview(1, 0);
    results[4] = subview(1, 1);
    results[5] = subview(1, 2);

    auto const& sublayout = subview.get_layout();
    results[6]            = sublayout.template get_dim_stride<0>();
    results[7]            = sublayout.template get_dim_stride<1>();

    results[8]  = adapter(0, 0);
    results[9]  = adapter(0, 1);
    results[10] = adapter(0, 2);
    results[11] = adapter(1, 0);
    results[12] = adapter(1, 1);
    results[13] = adapter(1, 2);

    subview(0, 0) = 80;
    adapter(1, 2) = 180;
  });

  data = test_reallocate(host_resource, working_resource, data, num_values);
  results =
      test_reallocate(host_resource, working_resource, results, num_results);

  const RAJA::Index_type expected[] = {8, 10, 12, 14, 16, 18, 6,
                                       2, 8,  10, 12, 14, 16, 18};
  for (RAJA::Index_type i = 0; i < num_results; ++i)
  {
    EXPECT_EQ(results[i], expected[i]);
  }
  EXPECT_EQ(data[7], 80);
  EXPECT_EQ(data[17], 180);

  host_resource.deallocate(results);
  host_resource.deallocate(data);
}

TYPED_TEST_P(SubViewBackendTest, ReadWrite)
{
  testSubViewBackendReadWrite<TypeParam>();
}

REGISTER_TYPED_TEST_SUITE_P(SubViewBackendTest, ReadWrite);

#endif  // TEST_SUBVIEW_BACKEND_HPP
