//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_RESOURCE_LISTSEGMENT_HPP__
#define __TEST_FORALL_RESOURCE_LISTSEGMENT_HPP__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallResourceListSegmentTestImpl(INDEX_TYPE N)
{

  // Create and initialize indices in idx_array used to create list segment
  std::vector<INDEX_TYPE> idx_array;

  srand ( time(NULL) );

  for (INDEX_TYPE i = INDEX_TYPE(0); i < N; ++i) {
    INDEX_TYPE randval = INDEX_TYPE(rand() % RAJA::stripIndexType(N));
    if ( i < randval ) {
      idx_array.push_back(i);
    }     
  }

  size_t idxlen = idx_array.size();

  WORKING_RES working_res;
  camp::resources::Resource erased_working_res{working_res};

  // Create list segment for tests
  RAJA::TypedListSegment<INDEX_TYPE> lseg(&idx_array[0], idxlen, 
                                          erased_working_res);

  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     erased_working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  for (INDEX_TYPE i = INDEX_TYPE(0); i < N; i++) {
    test_array[RAJA::stripIndexType(i)] = INDEX_TYPE(0);
  }

  working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));

  for (size_t i = 0; i < idxlen; ++i) {
    test_array[ RAJA::stripIndexType(idx_array[i]) ] = idx_array[i];
  }

  RAJA::forall<EXEC_POLICY>(working_res, lseg, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    working_array[RAJA::stripIndexType(idx)] = idx;
  }); 

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));

  // 
  for (INDEX_TYPE i = INDEX_TYPE(0); i < N; i++) {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
  }

  deallocateForallTestData<INDEX_TYPE>(erased_working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}


TYPED_TEST_SUITE_P(ForallResourceListSegmentTest);
template <typename T>
class ForallResourceListSegmentTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallResourceListSegmentTest, ResourceListSegmentForall)
{
  using INDEX_TYPE       = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY      = typename camp::at<TypeParam, camp::num<2>>::type;

  ForallResourceListSegmentTestImpl<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(INDEX_TYPE(13));

  ForallResourceListSegmentTestImpl<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(INDEX_TYPE(2047));

  ForallResourceListSegmentTestImpl<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(INDEX_TYPE(32000));
}

REGISTER_TYPED_TEST_SUITE_P(ForallResourceListSegmentTest,
                            ResourceListSegmentForall);

#endif  // __TEST_FORALL_RESOURCE_LISTSEGMENT_HPP__
