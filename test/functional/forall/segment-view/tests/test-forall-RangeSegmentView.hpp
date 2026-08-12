//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_RANGESEGMENTVIEW_HPP__
#define __TEST_FORALL_RANGESEGMENTVIEW_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallRangeSegmentViewTestImpl(INDEX_TYPE first, INDEX_TYPE last)
{
  RAJA::TypedRangeSegment<INDEX_TYPE> r1(first, last);
  INDEX_TYPE N = r1.end() - r1.begin();

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  const INDEX_TYPE rbegin = *r1.begin();

  std::iota(test_array, test_array + RAJA::stripIndexType(N), rbegin);

  using layout_type =
      RAJA::TypedLayout<INDEX_TYPE, camp::tuple<INDEX_TYPE>>;
  using view_type = RAJA::View< INDEX_TYPE, layout_type >;
 
  view_type test_view(test_array, N);
  view_type work_view(working_array, N);
  view_type check_view(check_array, N);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    work_view( idx - rbegin ) = idx;
  }); 

  working_res.memcpy(check_array, working_array,
                     sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));

  for (INDEX_TYPE i = 0; i < N; i++) {
    ASSERT_EQ(test_view(i), check_view(i));
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallRangeSegmentOffsetViewTestImpl(INDEX_TYPE first, INDEX_TYPE last, 
                                          INDEX_TYPE offset)
{
  RAJA::TypedRangeSegment<INDEX_TYPE> r1(first+offset, last+offset);
  INDEX_TYPE N = r1.end() - r1.begin();

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  const INDEX_TYPE rbegin = *r1.begin();

  std::iota(test_array, test_array + RAJA::stripIndexType(N), rbegin);

  using layout_type =
      RAJA::TypedOffsetLayout<INDEX_TYPE, camp::tuple<INDEX_TYPE>>;
  using view_type = RAJA::View< INDEX_TYPE, layout_type >;

  using raw_index_type = RAJA::strip_index_type_t<INDEX_TYPE>;
  raw_index_type f_offset = RAJA::stripIndexType(first + offset);
  raw_index_type l_offset = RAJA::stripIndexType(last + offset);
  layout_type layout({{f_offset}}, {{l_offset}});
  view_type test_view(test_array, layout);
  view_type work_view(working_array, layout);
  view_type check_view(check_array, layout);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    work_view( idx ) = idx;
  });

  working_res.memcpy(check_array, working_array,
                     sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));

  for (INDEX_TYPE i = first + offset; i < last + offset; i++) {
    ASSERT_EQ(test_view(i), check_view(i));
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_unsigned<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeViewTests()
{
}

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeViewTests()
{
  ForallRangeSegmentViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(-5, 0);
  ForallRangeSegmentViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(-5, 5);

  ForallRangeSegmentOffsetViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(-5, 0, 1);
  ForallRangeSegmentOffsetViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(-5, 5, 2);
  ForallRangeSegmentOffsetViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(0, 10, -5);
}


TYPED_TEST_SUITE_P(ForallRangeSegmentViewTest);
template <typename T>
class ForallRangeSegmentViewTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallRangeSegmentViewTest, RangeSegmentForallView)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  ForallRangeSegmentViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(0, 5);
  ForallRangeSegmentViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(1, 5);
  ForallRangeSegmentViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(1, 255);

  ForallRangeSegmentOffsetViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(0, 5, 1);
  ForallRangeSegmentOffsetViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(1, 5, 2);
  ForallRangeSegmentOffsetViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(1, 255, 3);

  runNegativeViewTests<INDEX_TYPE, WORKING_RES, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ForallRangeSegmentViewTest,
                            RangeSegmentForallView);

#endif  // __TEST_FORALL_RANGESEGMENTVIEW_HPP__
