//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_BASIC_EXPT_REDUCEBITOR_HPP__
#define __TEST_FORALL_BASIC_EXPT_REDUCEBITOR_HPP__

#include <cstdlib>
#include <ctime>
#include <numeric>
#include <vector>

template <typename IDX_TYPE, typename DATA_TYPE,
          typename SEG_TYPE,
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceBitOrBasicTestImpl(const SEG_TYPE& seg,
                                     const std::vector<IDX_TYPE>& seg_idx,
                                     camp::resources::Resource working_res)
{
  using REF_BITOR = RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::bit_or>;

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

  //
  // First a simple non-trivial test that is mildly interesting
  //
  for (IDX_TYPE i = 0; i < data_len; ++i) {
    test_array[i] = 9;
  }
  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * data_len);

  DATA_TYPE simpor(5);

  RAJA::forall<EXEC_POLICY>(seg,
    RAJA::expt::Reduce<RAJA::operators::bit_or>(&simpor),
			    RAJA::Name("expt Reduce Bit Or"),
    [=] RAJA_HOST_DEVICE(IDX_TYPE idx, REF_BITOR & _simpor) {
      _simpor |= working_array[idx];
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(simpor), 13);

 
  //
  // And now a randomized test that pushes zeros around
  //

  const int modval = 100;

  for (IDX_TYPE i = 0; i < data_len; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }
  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * data_len);

  DATA_TYPE ref_or = 0;
  for (IDX_TYPE i = 0; i < idx_len; ++i) {
    ref_or |= test_array[ seg_idx[i] ];
  }

  DATA_TYPE redor(0);
  DATA_TYPE redor2(2);

  RAJA::forall<EXEC_POLICY>(seg,
    RAJA::expt::Reduce<RAJA::operators::bit_or>(&redor),
    RAJA::expt::Reduce<RAJA::operators::bit_or>(&redor2),
    RAJA::Name("RAJA Reduce BitOr"),
    [=] RAJA_HOST_DEVICE(IDX_TYPE idx, REF_BITOR &r1, REF_BITOR &r2) {
      r1 |= working_array[idx];
      r2 |= working_array[idx];
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(redor), ref_or);
  ASSERT_EQ(static_cast<DATA_TYPE>(redor2), ref_or);

  redor = 0;

  const int nloops = 3;
  for (int j = 0; j < nloops; ++j) {
    RAJA::forall<EXEC_POLICY>(seg,
      RAJA::expt::Reduce<RAJA::operators::bit_or>(&redor),
      [=] RAJA_HOST_DEVICE(IDX_TYPE idx, REF_BITOR &r1) {
        r1 |= working_array[idx];
    });
  }

  ASSERT_EQ(static_cast<DATA_TYPE>(redor), ref_or);
   

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}

TYPED_TEST_SUITE_P(ForallReduceBitOrBasicTest);
template <typename T>
class ForallReduceBitOrBasicTest : public ::testing::Test
{
};


TYPED_TEST_P(ForallReduceBitOrBasicTest, ReduceBitOrBasicForall)
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
  ForallReduceBitOrBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                 RAJA::TypedRangeSegment<IDX_TYPE>,
                                 EXEC_POLICY, REDUCE_POLICY>(
                                   r1, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r2( 3, 642 );
  RAJA::getIndices(seg_idx, r2);
  ForallReduceBitOrBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                 RAJA::TypedRangeSegment<IDX_TYPE>,
                                 EXEC_POLICY, REDUCE_POLICY>(
                                   r2, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r3( 0, 2057 );
  RAJA::getIndices(seg_idx, r3);
  ForallReduceBitOrBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                 RAJA::TypedRangeSegment<IDX_TYPE>,
                                 EXEC_POLICY, REDUCE_POLICY>(
                                   r3, seg_idx, working_res);

// Range-stride segment tests
  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r4( 0, 188, 2 );
  RAJA::getIndices(seg_idx, r4);
  ForallReduceBitOrBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                 RAJA::TypedRangeStrideSegment<IDX_TYPE>,
                                 EXEC_POLICY, REDUCE_POLICY>(
                                   r4, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r5( 3, 1029, 3 );
  RAJA::getIndices(seg_idx, r5);
  ForallReduceBitOrBasicTestImpl<IDX_TYPE, DATA_TYPE,
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
  ForallReduceBitOrBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                 RAJA::TypedListSegment<IDX_TYPE>,
                                 EXEC_POLICY, REDUCE_POLICY>(
                                   l1, seg_idx, working_res);
}

REGISTER_TYPED_TEST_SUITE_P(ForallReduceBitOrBasicTest,
                            ReduceBitOrBasicForall);

#endif  // __TEST_FORALL_BASIC_REDUCEBITOR_HPP__
