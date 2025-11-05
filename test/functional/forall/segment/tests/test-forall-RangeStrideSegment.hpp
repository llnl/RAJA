//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_RANGESTRIDESEGMENT_HPP__
#define __TEST_FORALL_RANGESTRIDESEGMENT_HPP__

#include <cstring>

template <typename INDEX_TYPE, typename DIFF_TYPE, 
          typename WORKING_RES, typename EXEC_POLICY>
void ForallRangeStrideSegmentTestImpl(INDEX_TYPE first, INDEX_TYPE last, 
                                      DIFF_TYPE stride)
{
  RAJA::TypedRangeStrideSegment<INDEX_TYPE> r1(RAJA::stripIndexType(first), RAJA::stripIndexType(last), stride);
  INDEX_TYPE N = INDEX_TYPE(r1.size());

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  camp::resources::Resource host_res{camp::resources::Host::get_default()};
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

  memset(static_cast<void*>(test_array), 0, sizeof(INDEX_TYPE) * data_len);

  working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * data_len);
  working_res.wait();

  if ( RAJA::stripIndexType(N) > 0 ) {

    INDEX_TYPE index = first;
    for (INDEX_TYPE i = INDEX_TYPE(0); i < N; ++i) {
      test_array[ RAJA::stripIndexType((index-first)/stride) ] = index;
      index += stride; 
    }

    RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
      working_array[ RAJA::stripIndexType((idx-first)/stride) ] = idx;
    });

  } else { // zero-length segment

    RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
      (void) idx;
      working_array[0]++;
    });

  }

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * data_len);
  working_res.wait();

  for (INDEX_TYPE i = INDEX_TYPE(0); i < N; i++) {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}


TYPED_TEST_SUITE_P(ForallRangeStrideSegmentTest);
template <typename T>
class ForallRangeStrideSegmentTest : public ::testing::Test
{
};

template <typename INDEX_TYPE, typename DIFF_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_unsigned<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeStrideTests()
{
}

template <typename INDEX_TYPE, typename DIFF_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeStrideTests()
{
  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(-10), INDEX_TYPE(-1), DIFF_TYPE(2));
  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(0), DIFF_TYPE(2));
  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(5), DIFF_TYPE(3));

// Test negative strides
  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(10), INDEX_TYPE(-1), DIFF_TYPE(-1));
  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(10), INDEX_TYPE(0), DIFF_TYPE(-2));
}


TYPED_TEST_P(ForallRangeStrideSegmentTest, RangeStrideSegmentForall)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;
  using DIFF_TYPE   = typename std::make_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::type;

  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(0), INDEX_TYPE(20), DIFF_TYPE(1));
  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(1), INDEX_TYPE(20), DIFF_TYPE(1));
  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(0), INDEX_TYPE(20), DIFF_TYPE(2));
  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(1), INDEX_TYPE(20), DIFF_TYPE(2));
  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(0), INDEX_TYPE(21), DIFF_TYPE(2));
  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(1), INDEX_TYPE(21), DIFF_TYPE(2));
  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(1), INDEX_TYPE(255), DIFF_TYPE(2));

// Test size zero segments
  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(0), INDEX_TYPE(20), DIFF_TYPE(-2));
  ForallRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(1), INDEX_TYPE(20), DIFF_TYPE(-2));

  runNegativeStrideTests<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ForallRangeStrideSegmentTest,
                            RangeStrideSegmentForall);

#endif  // __TEST_FORALL_RANGESTRIDESEGMENT_HPP__
