//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_REGION_HPP__
#define __TEST_KERNEL_REGION_HPP__

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void KernelRegionTestImpl(INDEX_TYPE first, INDEX_TYPE last)
{
  camp::resources::Resource host_res{camp::resources::Host::get_default()};
  camp::resources::Resource work_res{WORKING_RES::get_default()};

  const INDEX_TYPE N = last - first;
  
  INDEX_TYPE* work_array1;
  INDEX_TYPE* work_array2;
  INDEX_TYPE* work_array3;

  INDEX_TYPE* check_array;

  allocRegionTestData(N,
                      work_res,
                      &work_array1, &work_array2, &work_array3,
                      host_res,
                      &check_array);

  work_res.memset( work_array1, 0, sizeof(INDEX_TYPE) * N );
  work_res.memset( work_array2, 0, sizeof(INDEX_TYPE) * N );
  work_res.memset( work_array3, 0, sizeof(INDEX_TYPE) * N );

  host_res.memset( check_array, 0, sizeof(INDEX_TYPE) * N );


  RAJA::TypedRangeSegment<INDEX_TYPE> rseg(first, last);

  RAJA::kernel<EXEC_POLICY>(

    RAJA::make_tuple(rseg),

    [=] (INDEX_TYPE i) {
      work_array1[i - first] = 50;
    },

    [=] (INDEX_TYPE i) {
      work_array2[i - first] = 100;
    },

    [=] (INDEX_TYPE i) {
      work_array3[i - first] = work_array1[i - first] + 
                               work_array2[i - first] + 1;
    }

  );
  
  work_res.memcpy(check_array, work_array3, sizeof(INDEX_TYPE) * N );
  work_res.wait();

  for (INDEX_TYPE i = 0; i < N; i++) {
    ASSERT_EQ(check_array[i], 151);
  }

  deallocRegionTestData(work_res,
                        work_array1, work_array2, work_array3,
                        host_res,
                        check_array);
}


TYPED_TEST_SUITE_P(KernelRegionTest);
template <typename T>
class KernelRegionTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelRegionTest, RegionKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  KernelRegionTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(0, 25);
  KernelRegionTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(1, 153);
  KernelRegionTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(3, 2556);
}

REGISTER_TYPED_TEST_SUITE_P(KernelRegionTest,
                            RegionKernel);

#endif  // __TEST_KERNEL_REGION_HPP__
