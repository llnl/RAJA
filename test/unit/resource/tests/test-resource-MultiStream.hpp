//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_RESOURCE_MULTISTREAM_HPP__
#define __TEST_RESOURCE_MULTISTREAM_HPP__

#include "RAJA_test-base.hpp"

template <typename WORKING_RES, typename EXEC_POLICY>
void ResourceMultiStreamTestImpl()
{
  constexpr std::size_t ARRAY_SIZE{10000};
  using namespace RAJA;

  WORKING_RES dev1;
  WORKING_RES dev2;
  WORKING_RES dev3;
  resources::Host host;

  int* d_array = resources::Resource{dev1}.allocate<int>(ARRAY_SIZE);
  int* h_array  = host.allocate<int>(ARRAY_SIZE);

  resources::Event e1 = forall<EXEC_POLICY>(dev1, RangeSegment(0,ARRAY_SIZE),
    [=] RAJA_HOST_DEVICE (int i) {
      if (i % 3 == 0) {
        d_array[i] = i;
      }
  });

  resources::Event e2 = forall<EXEC_POLICY>(dev2, RangeSegment(0,ARRAY_SIZE),
    [=] RAJA_HOST_DEVICE (int i) {
      if (i % 3 == 1) {
        d_array[i] = i;
      }
  });

  resources::Event e3 = forall<EXEC_POLICY>(dev2, RangeSegment(0,ARRAY_SIZE),
    [=] RAJA_HOST_DEVICE (int i) {
      if (i % 3 == 2) {
        d_array[i] = i;
      }
  });

  dev1.wait_for(&e2);
  dev1.wait_for(&e3);

  dev1.memcpy(h_array, d_array, sizeof(int) * ARRAY_SIZE);

  dev1.wait();

  forall<policy::sequential::seq_exec>(host, RangeSegment(0,ARRAY_SIZE),
    [=] (int i) {
      ASSERT_EQ(h_array[i], i); 
    }
  );

  dev1.deallocate(d_array);
  host.deallocate(h_array);
}

TYPED_TEST_SUITE_P(ResourceMultiStreamTest);
template <typename T>
class ResourceMultiStreamTest : public ::testing::Test
{
};

TYPED_TEST_P(ResourceMultiStreamTest, ResourceMultiStream)
{
  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<1>>::type;

  ResourceMultiStreamTestImpl<WORKING_RES, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ResourceMultiStreamTest,
                            ResourceMultiStream);

#endif  // __TEST_RESOURCE_MULTISTREAM_HPP__
