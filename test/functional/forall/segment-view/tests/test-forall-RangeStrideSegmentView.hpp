//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_RANGESTRIDESEGMENTVIEW_HPP__
#define __TEST_FORALL_RANGESTRIDESEGMENTVIEW_HPP__

template <typename INDEX_TYPE, typename DIFF_TYPE, 
          typename WORKING_RES, typename EXEC_POLICY>
void ForallRangeStrideSegmentViewTestImpl(INDEX_TYPE first, INDEX_TYPE last, 
                                          DIFF_TYPE stride)
{
  RAJA::TypedRangeStrideSegment<INDEX_TYPE> r1(RAJA::stripIndexType(first),
                                               RAJA::stripIndexType(last),
                                               stride);
  INDEX_TYPE N = INDEX_TYPE(r1.size());

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  memset( test_array, 0, sizeof(INDEX_TYPE) * RAJA::stripIndexType(N) );

  working_res.memcpy(working_array, test_array,
                     sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));

  INDEX_TYPE index = first;
  using layout_type =
      RAJA::TypedLayout<INDEX_TYPE, camp::tuple<INDEX_TYPE>>;
  using view_type = RAJA::View< INDEX_TYPE, layout_type >;

  view_type test_view(test_array, N);
  view_type work_view(working_array, N);
  view_type check_view(check_array, N);

  for (INDEX_TYPE i {0}; i < N; ++i) {
    test_view( (index-first)/stride ) = index;
    index += stride;
  }

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    work_view( (idx-first)/stride ) = idx;
  });

  working_res.memcpy(check_array, working_array,
                     sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));

  for (INDEX_TYPE i {0}; i < N; i++) {
    ASSERT_EQ(test_view(i), check_view(i));
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}

template <typename INDEX_TYPE, typename DIFF_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_unsigned<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeIndexViewTests()
{
}

template <typename INDEX_TYPE, typename DIFF_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeIndexViewTests()
{
  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(-10), INDEX_TYPE(-1), DIFF_TYPE(2));
  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(-5), INDEX_TYPE(0), DIFF_TYPE(2));
  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(-5), INDEX_TYPE(5), DIFF_TYPE(3));

  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(10), INDEX_TYPE(-1), DIFF_TYPE(-1));
  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(10), INDEX_TYPE(0), DIFF_TYPE(-2));
}


TYPED_TEST_SUITE_P(ForallRangeStrideSegmentViewTest);
template <typename T>
class ForallRangeStrideSegmentViewTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallRangeStrideSegmentViewTest, RangeStrideSegmentForallView)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;
  using DIFF_TYPE   =
      typename std::make_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::type;

  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(0), INDEX_TYPE(20), DIFF_TYPE(1));
  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(1), INDEX_TYPE(20), DIFF_TYPE(1));
  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(0), INDEX_TYPE(20), DIFF_TYPE(2));
  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(1), INDEX_TYPE(20), DIFF_TYPE(2));
  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(0), INDEX_TYPE(21), DIFF_TYPE(2));
  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(1), INDEX_TYPE(21), DIFF_TYPE(2));
  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(1), INDEX_TYPE(255), DIFF_TYPE(2));

// Test size zero segments
  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(0), INDEX_TYPE(20), DIFF_TYPE(-2));
  ForallRangeStrideSegmentViewTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(
      INDEX_TYPE(1), INDEX_TYPE(20), DIFF_TYPE(-2));

  runNegativeIndexViewTests<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ForallRangeStrideSegmentViewTest,
                            RangeStrideSegmentForallView);

#endif  // __TEST_FORALL_RANGESTRIDESEGMENTVIEW_HPP__
