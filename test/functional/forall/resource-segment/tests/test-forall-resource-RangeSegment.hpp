//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_RESOURCE_RANGESEGMENT_HPP__
#define __TEST_FORALL_RESOURCE_RANGESEGMENT_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallResourceRangeSegmentTestImpl(INDEX_TYPE first, INDEX_TYPE last)
{
  RAJA::TypedRangeSegment<INDEX_TYPE> r1(RAJA::stripIndexType(first), RAJA::stripIndexType(last));
  INDEX_TYPE N = INDEX_TYPE(r1.end() - r1.begin());

  WORKING_RES working_res;
  camp::resources::Resource erased_working_res{working_res};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     erased_working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  const INDEX_TYPE rbegin = *r1.begin();

  std::iota(test_array, test_array + RAJA::stripIndexType(N), rbegin);

  RAJA::forall<EXEC_POLICY>(working_res, r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    working_array[RAJA::stripIndexType(idx - rbegin)] = idx;
  });

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));

  for (INDEX_TYPE i = INDEX_TYPE(0); i < N; i++) {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
  }

  deallocateForallTestData<INDEX_TYPE>(erased_working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}


TYPED_TEST_SUITE_P(ForallResourceRangeSegmentTest);
template <typename T>
class ForallResourceRangeSegmentTest : public ::testing::Test
{
};

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_unsigned<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeTests()
{
}

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeTests()
{
  ForallResourceRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(0));
  ForallResourceRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(5));
}


TYPED_TEST_P(ForallResourceRangeSegmentTest, ResourceRangeSegmentForall)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  ForallResourceRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(0), INDEX_TYPE(27));
  ForallResourceRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(1), INDEX_TYPE(2047));
  ForallResourceRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(1), INDEX_TYPE(32000));

  runNegativeTests<INDEX_TYPE, WORKING_RES, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ForallResourceRangeSegmentTest,
                            ResourceRangeSegmentForall);

#endif  // __TEST_FORALL_RESOURCE_RANGESEGMENT_HPP__
