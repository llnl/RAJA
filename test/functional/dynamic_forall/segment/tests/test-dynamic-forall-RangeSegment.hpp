//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_DYANMIC_FORALL_RANGESEGMENT_HPP__
#define __TEST_DYANMIC_FORALL_RANGESEGMENT_HPP__

#include <numeric>
#include <iostream>

template <typename INDEX_TYPE, typename WORKING_RES, typename POLICY_LIST>
void DynamicForallRangeSegmentTestImpl(INDEX_TYPE first, INDEX_TYPE last, const int pol)
{

  RAJA::TypedRangeSegment<INDEX_TYPE> r1(RAJA::stripIndexType(first), RAJA::stripIndexType(last));
  INDEX_TYPE N = static_cast<INDEX_TYPE>(r1.end() - r1.begin());

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  size_t data_len = RAJA::stripIndexType(N);
  if ( data_len == 0 ) {
    data_len = 1;
  }

  allocateForallTestData<INDEX_TYPE>(data_len,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  if ( RAJA::stripIndexType(N) > 0 ) {

    const INDEX_TYPE rbegin = *r1.begin();

    std::iota(test_array, test_array + RAJA::stripIndexType(N), rbegin);

    RAJA::dynamic_forall<POLICY_LIST>(pol, r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
      working_array[RAJA::stripIndexType(idx - rbegin)] = idx;
    });

  } else { // zero-length segment 

    memset(static_cast<void*>(test_array), 0, sizeof(INDEX_TYPE) * data_len);

    working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * data_len);

    RAJA::dynamic_forall<POLICY_LIST>(pol, r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
      (void) idx;
      working_array[0]++;
    });

  }

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * data_len);

  for (INDEX_TYPE i = INDEX_TYPE(0); i < N; i++) {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}


TYPED_TEST_SUITE_P(DynamicForallRangeSegmentTest);
template <typename T>
class DynamicForallRangeSegmentTest : public ::testing::Test
{
};

TYPED_TEST_P(DynamicForallRangeSegmentTest, RangeSegmentForall)
{

  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using POLICY_LIST = typename camp::at<TypeParam, camp::num<2>>::type;

#if defined(RAJA_GPU_ACTIVE)
  constexpr int N = camp::size<POLICY_LIST>::value;
#endif

  //If N == 2 host, no openmp is available
  //If N == 3 host, openmp is available
  //If N == 4 host, device is available
  //If N == 5 host, openmp, device are on

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  bool is_on_host = working_res.get_platform() == camp::resources::Platform::host ? true : false;

  if(is_on_host) { 
    int host_range = 2;
#if defined(RAJA_ENABLE_OPENMP)
    host_range = 3; 
#endif      
      //Loop through policy list
      for(int pol=0; pol<host_range; ++pol) 
        {
          DynamicForallRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, POLICY_LIST>
            (INDEX_TYPE(0), INDEX_TYPE(27), pol);
        }
  }
#if defined(RAJA_GPU_ACTIVE)
  else
  {
    int device_start = 2;
#if defined(RAJA_ENABLE_OPENMP)
    device_start = 3; 
#endif      
    for(int pol=device_start; pol<N; ++pol) 
    {
    DynamicForallRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, POLICY_LIST>
      (INDEX_TYPE(0), INDEX_TYPE(27), pol);
    }
  }
#endif


}

REGISTER_TYPED_TEST_SUITE_P(DynamicForallRangeSegmentTest,
                            RangeSegmentForall);

#endif  // __TEST_BASIC_SHARED_HPP__
