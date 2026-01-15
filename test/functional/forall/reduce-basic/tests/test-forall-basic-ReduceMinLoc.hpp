//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_BASIC_REDUCEMINLOC_HPP__
#define __TEST_FORALL_BASIC_REDUCEMINLOC_HPP__

#include <cstdlib>
#include <ctime>
#include <numeric>
#include <vector>

template <typename IDX_TYPE, typename DATA_TYPE,
          typename SEG_TYPE,
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceMinLocBasicTestImpl(const SEG_TYPE& seg,
                                     const std::vector<IDX_TYPE>& seg_idx,
                                     camp::resources::Resource working_res)
{
  IDX_TYPE data_len = seg_idx[seg_idx.size() - 1] + 1;
  IDX_TYPE idx_len = static_cast<IDX_TYPE>( seg_idx.size() );

  DATA_TYPE* working_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  allocateForallTestData<DATA_TYPE>(data_len,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);

  const int modval = 100;
  const DATA_TYPE min_init = modval+1;
  const IDX_TYPE minloc_init = -1;
  const IDX_TYPE minloc_idx = seg_idx[ idx_len * 2/3 ];
  const DATA_TYPE small_min = -modval;
  const IDX_TYPE small_minloc = minloc_init;

  for (IDX_TYPE i = 0; i < data_len; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }
  test_array[minloc_idx] = static_cast<DATA_TYPE>(small_min);

  DATA_TYPE ref_min = min_init;
  IDX_TYPE ref_minloc = minloc_init;
  for (IDX_TYPE i = 0; i < idx_len; ++i) {
    if ( test_array[ seg_idx[i] ] < ref_min ) {
       ref_min = test_array[ seg_idx[i] ];
       ref_minloc = seg_idx[i];
    } 
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * data_len);


  RAJA::ReduceMinLoc<REDUCE_POLICY, DATA_TYPE, IDX_TYPE> mininit(small_min, minloc_init);
  RAJA::ReduceMinLoc<REDUCE_POLICY, DATA_TYPE, IDX_TYPE> min(min_init, minloc_init);

  RAJA::forall<EXEC_POLICY>(seg, RAJA::Name("Reduce Min Loc"), [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
    mininit.minloc( working_array[idx], idx );
    min.minloc( working_array[idx], idx );
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(mininit.get()), small_min);
  ASSERT_EQ(static_cast<IDX_TYPE>(mininit.getLoc()), small_minloc);
  ASSERT_EQ(static_cast<DATA_TYPE>(min.get()), ref_min);
  ASSERT_EQ(static_cast<IDX_TYPE>(min.getLoc()), ref_minloc);

  min.reset(min_init, minloc_init);
  ASSERT_EQ(static_cast<DATA_TYPE>(min.get()), min_init);
  ASSERT_EQ(static_cast<IDX_TYPE>(min.getLoc()), minloc_init);

  DATA_TYPE factor = 2;
  RAJA::forall<EXEC_POLICY>(seg, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
    min.minloc( working_array[idx] * factor, idx);
  });
  ASSERT_EQ(static_cast<DATA_TYPE>(min.get()), ref_min * factor);
  ASSERT_EQ(static_cast<IDX_TYPE>(min.getLoc()), ref_minloc);

  factor = 3;
  RAJA::forall<EXEC_POLICY>(seg, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) { 
    min.minloc( working_array[idx] * factor, idx);
  });
  ASSERT_EQ(static_cast<DATA_TYPE>(min.get()), ref_min * factor);
  ASSERT_EQ(static_cast<IDX_TYPE>(min.getLoc()), ref_minloc);
   

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}

TYPED_TEST_SUITE_P(ForallReduceMinLocBasicTest);
template <typename T>
class ForallReduceMinLocBasicTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallReduceMinLocBasicTest, ReduceMinLocBasicForall)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  std::vector<IDX_TYPE> seg_idx;

// Range segment tests
  RAJA::TypedRangeSegment<IDX_TYPE> r1( 0, 28 );
  RAJA::getIndices(seg_idx, r1);
  ForallReduceMinLocBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeSegment<IDX_TYPE>,
                                  EXEC_POLICY, REDUCE_POLICY>(
                                    r1, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r2( 3, 642 );
  RAJA::getIndices(seg_idx, r2);
  ForallReduceMinLocBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeSegment<IDX_TYPE>,
                                  EXEC_POLICY, REDUCE_POLICY>(
                                    r2, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r3( 0, 2057 );
  RAJA::getIndices(seg_idx, r3);
  ForallReduceMinLocBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeSegment<IDX_TYPE>,
                                  EXEC_POLICY, REDUCE_POLICY>(
                                    r3, seg_idx, working_res);

// Range-stride segment tests
  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r4( 0, 188, 2 );
  RAJA::getIndices(seg_idx, r4);
  ForallReduceMinLocBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeStrideSegment<IDX_TYPE>,
                                  EXEC_POLICY, REDUCE_POLICY>(
                                    r4, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r5( 3, 1029, 3 );
  RAJA::getIndices(seg_idx, r5);
  ForallReduceMinLocBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeStrideSegment<IDX_TYPE>,
                                  EXEC_POLICY, REDUCE_POLICY>(
                                    r5, seg_idx, working_res);

// List segment tests
  seg_idx.clear();
  IDX_TYPE last = 10567;
  srand( time(NULL) );
  for (IDX_TYPE i = 0; i < last; ++i) {
    IDX_TYPE randval = IDX_TYPE( rand() % RAJA::stripIndexType(last) );
    if ( i < randval ) {
      seg_idx.push_back(i);
    }
  }
  RAJA::TypedListSegment<IDX_TYPE> l1( &seg_idx[0], seg_idx.size(),
                                       working_res );
  ForallReduceMinLocBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedListSegment<IDX_TYPE>,
                                  EXEC_POLICY, REDUCE_POLICY>(
                                    l1, seg_idx, working_res);
}

REGISTER_TYPED_TEST_SUITE_P(ForallReduceMinLocBasicTest,
                            ReduceMinLocBasicForall);

#endif  // __TEST_FORALL_BASIC_REDUCEMINLOC_HPP__
