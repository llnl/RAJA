//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTED_LOOPS_SEGMENT_TYPES_HPP__
#define __TEST_KERNEL_NESTED_LOOPS_SEGMENT_TYPES_HPP__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <vector>


template <typename IDX_TYPE, typename DATA_TYPE, typename EXEC_POLICY>
void KernelNestedLoopsSegmentTypesTestImpl(
  const RAJA::TypedRangeSegment<IDX_TYPE>& s1, 
  const std::vector<IDX_TYPE>& s1_idx,
  const RAJA::TypedRangeStrideSegment<IDX_TYPE>& s2,
  const std::vector<IDX_TYPE>& s2_idx,
  const RAJA::TypedListSegment<IDX_TYPE>& s3,
  const std::vector<IDX_TYPE>& s3_idx,
  camp::resources::Resource working_res,
  int perm)
{
  IDX_TYPE idx1_len = static_cast<IDX_TYPE>(s1_idx.size());
  IDX_TYPE idx2_len = static_cast<IDX_TYPE>(s2_idx.size());
  IDX_TYPE idx3_len = static_cast<IDX_TYPE>(s3_idx.size());

  bool zero_legth_segment = false;
  if ( RAJA::stripIndexType(idx1_len * idx2_len * idx3_len) == 0 ) {
    zero_legth_segment = true;
  }

  IDX_TYPE dim1 = 1;
  IDX_TYPE dim2 = 1;
  IDX_TYPE dim3 = 1;

  if ( !zero_legth_segment ) {
    dim1 = s1_idx[s1_idx.size() - 1] + 1;
    dim2 = s2_idx[s2_idx.size() - 1] + 1;
    dim3 = s3_idx[s3_idx.size() - 1] + 1;
  }

  IDX_TYPE data_len = dim1 * dim2 * dim3;

  DATA_TYPE* work_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  allocateForallTestData<DATA_TYPE>(data_len,
                                    working_res,
                                    &work_array,
                                    &check_array,
                                    &test_array);

  RAJA::View< DATA_TYPE, RAJA::Layout<3> > work_view(work_array, 
                                                     dim1, dim2, dim3);
  RAJA::View< DATA_TYPE, RAJA::Layout<3> > test_view(test_array, 
                                                     dim1, dim2, dim3);

  memset( static_cast<void*>(test_array), 0, 
          sizeof(DATA_TYPE) * RAJA::stripIndexType(data_len) );

  working_res.memcpy(work_array, test_array, 
                     sizeof(DATA_TYPE) * RAJA::stripIndexType(data_len));

  if ( !zero_legth_segment ) {
    for (IDX_TYPE i1 = 0; i1 < idx1_len; ++i1) {
      for (IDX_TYPE i2 = 0; i2 < idx2_len; ++i2) {
        for (IDX_TYPE i3 = 0; i3 < idx3_len; ++i3) {
          auto ii1 = RAJA::stripIndexType(i1);
          auto ii2 = RAJA::stripIndexType(i2);
          auto ii3 = RAJA::stripIndexType(i3);
          test_view( s1_idx[ii1], s2_idx[ii2], s3_idx[ii3] ) = 
            static_cast<DATA_TYPE>( RAJA::stripIndexType(
                                    s1_idx[ii1] + s2_idx[ii2] + s3_idx[ii3]) );
        }
      }
    }
  }

  if ( perm == 1 ) {
    RAJA::kernel<EXEC_POLICY>( RAJA::make_tuple( s1, s2, s3 ),
      [=] RAJA_HOST_DEVICE (IDX_TYPE i1, IDX_TYPE i2, IDX_TYPE i3) {
        work_view(i1, i2, i3) = 
          static_cast<DATA_TYPE>( RAJA::stripIndexType(i1 + i2 + i3) );
      }
    );
  }
 
  if ( perm == 2 ) {
    RAJA::kernel<EXEC_POLICY>( RAJA::make_tuple( s2, s3, s1 ),
      [=] RAJA_HOST_DEVICE (IDX_TYPE i2, IDX_TYPE i3, IDX_TYPE i1) {
        work_view(i1, i2, i3) = 
          static_cast<DATA_TYPE>( RAJA::stripIndexType(i1 + i2 + i3) );
      }
    );
  } 

  if ( perm == 3 ) {
    RAJA::kernel<EXEC_POLICY>( RAJA::make_tuple( s3, s1, s2 ),
      [=] RAJA_HOST_DEVICE (IDX_TYPE i3, IDX_TYPE i1, IDX_TYPE i2) {
        work_view(i1, i2, i3) = 
          static_cast<DATA_TYPE>( RAJA::stripIndexType(i1 + i2 + i3) );
      }
    );
  } 

  working_res.memcpy(check_array, work_array, 
                     sizeof(DATA_TYPE) * RAJA::stripIndexType(data_len));

  for (IDX_TYPE i = 0; i < data_len; ++i) {
    auto ii = RAJA::stripIndexType(i);
    ASSERT_EQ( test_array[ii], check_array[ii] );
  }

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      work_array,
                                      check_array,
                                      test_array);
}


TYPED_TEST_SUITE_P(KernelNestedLoopsSegmentTypesTest);
template <typename T>
class KernelNestedLoopsSegmentTypesTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelNestedLoopsSegmentTypesTest, NestedLoopsSegmentTypesKernel)
{
  using IDX_TYPE    = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  std::vector<IDX_TYPE> s1_idx;
  std::vector<IDX_TYPE> s2_idx;
  std::vector<IDX_TYPE> s3_idx;

// Create a segment of each basic type RAJA provides and test
// permutations of those segments in nested loops 

  RAJA::TypedRangeSegment<IDX_TYPE> s1( 0, 69 );
  RAJA::getIndices(s1_idx, s1);

  RAJA::TypedRangeStrideSegment<IDX_TYPE> s2( 3, 188, 2 );
  RAJA::getIndices(s2_idx, s2);

  IDX_TYPE last = IDX_TYPE(427);
  srand( time(NULL) );
  for (IDX_TYPE i = IDX_TYPE(0); i < last; ++i) {
    IDX_TYPE randval = IDX_TYPE( rand() % RAJA::stripIndexType(last) );
    if ( i < randval ) {
      s3_idx.push_back(i);
    }
  }
  RAJA::TypedListSegment<IDX_TYPE> s3( &s3_idx[0], s3_idx.size(),
                                       working_res );

  int perm = 1;
  KernelNestedLoopsSegmentTypesTestImpl<IDX_TYPE, int, EXEC_POLICY>(
                                        s1, s1_idx,    
                                        s2, s2_idx,    
                                        s3, s3_idx,
                                        working_res,
                                        perm);

  perm = 2;
  KernelNestedLoopsSegmentTypesTestImpl<IDX_TYPE, int, EXEC_POLICY>(
                                        s1, s1_idx,
                                        s2, s2_idx,
                                        s3, s3_idx,
                                        working_res,
                                        perm);

  perm = 3;
  KernelNestedLoopsSegmentTypesTestImpl<IDX_TYPE, int, EXEC_POLICY>(
                                        s1, s1_idx,
                                        s2, s2_idx,
                                        s3, s3_idx,
                                        working_res,
                                        perm);

// Test some zero-length segment combinations

// Zero-length range segment
  RAJA::TypedRangeSegment<IDX_TYPE> s4( 4, 4 );
  std::vector<IDX_TYPE> s4_idx;
  RAJA::getIndices(s4_idx, s4);

  perm = 1;
  KernelNestedLoopsSegmentTypesTestImpl<IDX_TYPE, int, EXEC_POLICY>(
                                        s4, s4_idx,
                                        s2, s2_idx,
                                        s3, s3_idx,
                                        working_res,
                                        perm);

  perm = 2;
  KernelNestedLoopsSegmentTypesTestImpl<IDX_TYPE, int, EXEC_POLICY>(
                                        s4, s4_idx,
                                        s2, s2_idx,
                                        s3, s3_idx,
                                        working_res,
                                        perm);

  perm = 3;
  KernelNestedLoopsSegmentTypesTestImpl<IDX_TYPE, int, EXEC_POLICY>(
                                        s4, s4_idx,
                                        s2, s2_idx,
                                        s3, s3_idx,
                                        working_res,
                                        perm);

// Zero-length range stride segment
  RAJA::TypedRangeStrideSegment<IDX_TYPE> s5( 3, 3, 2 );
  std::vector<IDX_TYPE> s5_idx;
  RAJA::getIndices(s5_idx, s5);

  perm = 1;
  KernelNestedLoopsSegmentTypesTestImpl<IDX_TYPE, int, EXEC_POLICY>(
                                        s1, s1_idx,
                                        s5, s5_idx,
                                        s3, s3_idx,
                                        working_res,
                                        perm);

  perm = 2;
  KernelNestedLoopsSegmentTypesTestImpl<IDX_TYPE, int, EXEC_POLICY>(
                                        s1, s1_idx,
                                        s5, s5_idx,
                                        s3, s3_idx,
                                        working_res,
                                        perm);

  perm = 3;
  KernelNestedLoopsSegmentTypesTestImpl<IDX_TYPE, int, EXEC_POLICY>(
                                        s1, s1_idx,
                                        s5, s5_idx,
                                        s3, s3_idx,
                                        working_res,
                                        perm);

// Zero-length list segment 
  std::vector<IDX_TYPE> s6_idx;
  RAJA::TypedListSegment<IDX_TYPE> s6( nullptr, s6_idx.size(),
                                       working_res );

  perm = 1;
  KernelNestedLoopsSegmentTypesTestImpl<IDX_TYPE, int, EXEC_POLICY>(
                                        s1, s1_idx,
                                        s2, s2_idx,
                                        s6, s6_idx,
                                        working_res,
                                        perm);

  perm = 2;
  KernelNestedLoopsSegmentTypesTestImpl<IDX_TYPE, int, EXEC_POLICY>(
                                        s1, s1_idx,
                                        s2, s2_idx,
                                        s6, s6_idx,
                                        working_res,
                                        perm);

  perm = 3;
  KernelNestedLoopsSegmentTypesTestImpl<IDX_TYPE, int, EXEC_POLICY>(
                                        s1, s1_idx,
                                        s2, s2_idx,
                                        s6, s6_idx,
                                        working_res,
                                        perm);
}

REGISTER_TYPED_TEST_SUITE_P(KernelNestedLoopsSegmentTypesTest,
                            NestedLoopsSegmentTypesKernel);

#endif  // __TEST_KERNEL_NESTED_LOOPS_SEGMENT_TYPES_HPP__
