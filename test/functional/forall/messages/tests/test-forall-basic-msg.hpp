//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_BASIC_MSG_HPP__
#define __TEST_FORALL_BASIC_MSG_HPP__

#include <cstdlib>
#include <ctime>
#include <numeric>
#include <vector>

template <typename IDX_TYPE, typename DATA_TYPE,
          typename SEG_TYPE,
          typename EXEC_POLICY, typename WORKING_RES>
void ForallMsgBasicTestImpl(const SEG_TYPE& seg, 
                            const std::vector<IDX_TYPE>& seg_idx,
                            WORKING_RES working_res)
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

  for (IDX_TYPE i = 0; i < data_len; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }

  DATA_TYPE ref_max = 0;
  for (IDX_TYPE i = 0; i < idx_len; ++i) {
    ref_max = std::max(ref_max, test_array[ seg_idx[i] ]);
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * data_len);

  const std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
    RAJA::align_sz(sizeof(RAJA::MsgArgs<DATA_TYPE>));

  auto msg_manager  = RAJA::make_message_manager(msg_sz*idx_len, working_res);

  DATA_TYPE msg_max = 0;
  auto msg_queue    = msg_manager.template subscribe<RAJA::mpsc_queue>([&] (DATA_TYPE data) {
    msg_max = std::max(msg_max, data);
  });

  RAJA::forall<EXEC_POLICY>(seg, RAJA::Name("Test Messages"), [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
    msg_queue.try_post_message(working_array[idx]);
  });

  auto messages         = msg_manager.get_messages();
  ASSERT_EQ(static_cast<IDX_TYPE>(messages.size()), idx_len);

  msg_manager.handle_all(messages);
  ASSERT_EQ(msg_max, ref_max);

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}


TYPED_TEST_SUITE_P(ForallMsgBasicTest);
template <typename T>
class ForallMsgBasicTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallMsgBasicTest, MsgBasicForall)
{
  using IDX_TYPE     = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE    = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES  = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY  = typename camp::at<TypeParam, camp::num<3>>::type;

  auto working_res = WORKING_RES::get_default();

  std::vector<IDX_TYPE> seg_idx;

// Range segment tests
  RAJA::TypedRangeSegment<IDX_TYPE> r1( 0, 28 );
  RAJA::getIndices(seg_idx, r1);
  ForallMsgBasicTestImpl<IDX_TYPE, DATA_TYPE,
                               RAJA::TypedRangeSegment<IDX_TYPE>,
                               EXEC_POLICY>(
                                 r1, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r2( 3, 642 );
  RAJA::getIndices(seg_idx, r2);
  ForallMsgBasicTestImpl<IDX_TYPE, DATA_TYPE,
                               RAJA::TypedRangeSegment<IDX_TYPE>,
                               EXEC_POLICY>(
                                 r2, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r3( 0, 2057 );
  RAJA::getIndices(seg_idx, r3);
  ForallMsgBasicTestImpl<IDX_TYPE, DATA_TYPE,
                               RAJA::TypedRangeSegment<IDX_TYPE>,
                               EXEC_POLICY>(
                                 r3, seg_idx, working_res);

// Range-stride segment tests
  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r4( 0, 188, 2 );
  RAJA::getIndices(seg_idx, r4);
  ForallMsgBasicTestImpl<IDX_TYPE, DATA_TYPE,
                               RAJA::TypedRangeStrideSegment<IDX_TYPE>,
                               EXEC_POLICY>(
                                 r4, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r5( 3, 1029, 3 );
  RAJA::getIndices(seg_idx, r5);
  ForallMsgBasicTestImpl<IDX_TYPE, DATA_TYPE,
                               RAJA::TypedRangeStrideSegment<IDX_TYPE>,
                               EXEC_POLICY>(
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
  ForallMsgBasicTestImpl<IDX_TYPE, DATA_TYPE,
                               RAJA::TypedListSegment<IDX_TYPE>,
                               EXEC_POLICY>(
                                 l1, seg_idx, working_res);
}

REGISTER_TYPED_TEST_SUITE_P(ForallMsgBasicTest,
                            MsgBasicForall);

#endif  // __TEST_FORALL_BASIC_MSG_HPP__
