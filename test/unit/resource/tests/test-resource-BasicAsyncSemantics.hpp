//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_RESOURCE_DEPENDS_HPP__
#define __TEST_RESOURCE_DEPENDS_HPP__

#include "RAJA_test-base.hpp"

template <typename WORKING_RES, typename EXEC_POLICY>
void ResourceBasicAsyncSemanticsTestImpl()
{
  constexpr std::size_t ARRAY_SIZE{10000000};
  using namespace RAJA;

  WORKING_RES dev;
  resources::Host host;

  int* d_array = resources::Resource{dev}.allocate<int>(ARRAY_SIZE);
  int* h_array  = host.allocate<int>(ARRAY_SIZE);

  forall<policy::sequential::seq_exec>(host, RangeSegment(0,ARRAY_SIZE),
    [=] RAJA_HOST_DEVICE (int i) {
      h_array[i] = i;
    }
  );

  dev.memcpy(d_array, h_array, sizeof(int) * ARRAY_SIZE);

  forall<EXEC_POLICY>(dev, RangeSegment(0,ARRAY_SIZE),
    [=] RAJA_HOST_DEVICE (int i) {
      d_array[i] = i + 2;
    }
  );

  dev.memcpy(h_array, d_array, sizeof(int) * ARRAY_SIZE);

  dev.wait();

  forall<policy::sequential::seq_exec>(host, RangeSegment(0,ARRAY_SIZE),
    [=] (int i) {
      ASSERT_EQ(h_array[i], i + 2); 
    }
  );

  dev.deallocate(d_array);
  host.deallocate(h_array);
  
}

TYPED_TEST_SUITE_P(ResourceBasicAsyncSemanticsTest);
template <typename T>
class ResourceBasicAsyncSemanticsTest : public ::testing::Test
{
};

TYPED_TEST_P(ResourceBasicAsyncSemanticsTest, ResourceBasicAsyncSemantics)
{
  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<1>>::type;

  ResourceBasicAsyncSemanticsTestImpl<WORKING_RES, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ResourceBasicAsyncSemanticsTest,
                            ResourceBasicAsyncSemantics);

#endif  // __TEST_RESOURCE_DEPENDS_HPP__
