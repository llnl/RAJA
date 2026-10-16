//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_REGION_HPP__
#define __TEST_FORALL_REGION_HPP__

#include <numeric>
#include <vector>

template <typename INDEX_TYPE, typename WORKING_RES, 
          typename REG_POLICY, typename EXEC_POLICY>
void ForallRegionTestImpl(INDEX_TYPE first, INDEX_TYPE last)
{
  camp::resources::Resource working_res{WORKING_RES::get_default()};

  //
  // Set some local variables and create some segments for using in tests
  //
  const INDEX_TYPE N = last - first;
  
  RAJA::TypedRangeSegment<INDEX_TYPE> rseg(first, last);

  std::vector<INDEX_TYPE> idx_array(RAJA::stripIndexType(N));
  std::iota(&idx_array[0], &idx_array[0] + RAJA::stripIndexType(N), first);

  RAJA::TypedListSegment<INDEX_TYPE> lseg(&idx_array[0], RAJA::stripIndexType(N),
                                          working_res);

  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  working_res.memset(working_array, 0,
                     sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));

  using layout_type =
      RAJA::TypedLayout<INDEX_TYPE, camp::tuple<INDEX_TYPE>>;
  using view_type = RAJA::View< INDEX_TYPE, layout_type >;

  view_type work_view(working_array, N);
  view_type check_view(check_array, N);

  RAJA::region<REG_POLICY>([=]() {

    RAJA::forall<EXEC_POLICY>(rseg, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
      work_view(idx - first) += 1;
    });

    RAJA::forall<EXEC_POLICY>(lseg, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
      work_view(idx - first) += 2;
    });

  });


  working_res.memcpy(check_array, working_array,
                     sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));

  for (INDEX_TYPE i {0}; i < N; i++) {
    ASSERT_EQ(check_view(i), 3);
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}


TYPED_TEST_SUITE_P(ForallRegionTest);
template <typename T>
class ForallRegionTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallRegionTest, RegionForall)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using REG_POLICY  = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  ForallRegionTestImpl<INDEX_TYPE, WORKING_RES, REG_POLICY, EXEC_POLICY>(
      INDEX_TYPE(0), INDEX_TYPE(25));
  ForallRegionTestImpl<INDEX_TYPE, WORKING_RES, REG_POLICY, EXEC_POLICY>(
      INDEX_TYPE(1), INDEX_TYPE(153));
  ForallRegionTestImpl<INDEX_TYPE, WORKING_RES, REG_POLICY, EXEC_POLICY>(
      INDEX_TYPE(3), INDEX_TYPE(2556));
}

REGISTER_TYPED_TEST_SUITE_P(ForallRegionTest,
                            RegionForall);

#endif  // __TEST_FORALL_REGION_HPP__
