//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_RANGESEGMENT2DVIEW_HPP__
#define __TEST_FORALL_RANGESEGMENT2DVIEW_HPP__

#include <iostream>
#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallRangeSegment2DViewTestImpl(INDEX_TYPE N)
{
  INDEX_TYPE lentot = N * N;
  const int NDIMS = 2;

  RAJA::TypedRangeSegment<INDEX_TYPE> r1(0, lentot);

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(lentot,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  std::iota(test_array, test_array + lentot, 0);

  using view_type = RAJA::View< INDEX_TYPE, RAJA::Layout<NDIMS> >;
  RAJA::Layout<NDIMS> layout(N, N);
  
  view_type work_view(working_array, layout);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    const INDEX_TYPE row = idx / N;
    const INDEX_TYPE col = idx % N;
    work_view(row, col) = row * N + col;
  });

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * lentot);

  for (INDEX_TYPE i = 0; i < lentot; i++) {
    ASSERT_EQ(test_array[i], check_array[i]);
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallRangeSegment2DOffsetViewTestImpl(INDEX_TYPE N)
{
  const INDEX_TYPE leninterior = N * N;
  const INDEX_TYPE lentot = (N + 2) * (N + 2);
  const int NDIMS = 2;

  RAJA::TypedRangeSegment<INDEX_TYPE> r1(0, leninterior);

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(lentot,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  memset( test_array, 0, sizeof(INDEX_TYPE) * lentot ); 

  working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * lentot);

  for (int row = 1; row < N + 1; ++row) {
    for (int col = 1; col < N + 1; ++col) {
      int idx = row * (N+2) + col;
      test_array[ idx ] = (row - 1) * N + (col - 1);
    }
  }

  using view_type = RAJA::View< INDEX_TYPE, RAJA::OffsetLayout<NDIMS> >;
  RAJA::OffsetLayout<NDIMS> layout =
    RAJA::make_offset_layout<NDIMS>( {{-1, -1}} , {{N+1, N+1}} );

  view_type work_view(working_array, layout);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    const INDEX_TYPE row = idx / N;
    const INDEX_TYPE col = idx % N;
    work_view(row, col) = idx;  
  });

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * lentot);

  for (INDEX_TYPE i = 0; i < lentot; i++) {
    ASSERT_EQ(test_array[i], check_array[i]);
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}

TYPED_TEST_SUITE_P(ForallRangeSegment2DViewTest);
template <typename T>
class ForallRangeSegment2DViewTest : public ::testing::Test
{
};

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_unsigned<INDEX_TYPE>::value>::type* = nullptr>
void runOffsetViewTests()
{
}

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_signed<INDEX_TYPE>::value>::type* = nullptr>
void runOffsetViewTests()
{
  ForallRangeSegment2DOffsetViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(4);
  ForallRangeSegment2DOffsetViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(100);
}


TYPED_TEST_P(ForallRangeSegment2DViewTest, RangeSegmentForall2DView)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  ForallRangeSegment2DViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(4);
  ForallRangeSegment2DViewTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(100);

  runOffsetViewTests<INDEX_TYPE, WORKING_RES, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ForallRangeSegment2DViewTest,
                            RangeSegmentForall2DView);

#endif  // __TEST_FORALL_RANGESEGMENT2DVIEW_HPP__
