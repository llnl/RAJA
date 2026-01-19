//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for global indexing
///

#ifndef __TEST_INDEXING_GLOBAL__
#define __TEST_INDEXING_GLOBAL__

#include "../test-indexing.hpp"

template <typename T>
class IndexingUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( IndexingUnitTest );

template < typename test_policy,
           typename indexer_type,
           RAJA::named_dim dim_012,
           int BLOCK_SIZE,
           int GRID_SIZE >
void testBasicIndexing()
{
  dim3d3d expected_dim{{1,1,1}, {1,1,1}};
  if (BLOCK_SIZE != RAJA::named_usage::ignored) {
    if (BLOCK_SIZE == RAJA::named_usage::unspecified) {
      expected_dim.thread[static_cast<int>(dim_012)] = 3;
    } else {
      expected_dim.thread[static_cast<int>(dim_012)] = BLOCK_SIZE;
    }
  }

  if (GRID_SIZE != RAJA::named_usage::ignored) {
    if (GRID_SIZE == RAJA::named_usage::unspecified) {
      expected_dim.block[static_cast<int>(dim_012)] = 5;
    } else {
      expected_dim.block[static_cast<int>(dim_012)] = GRID_SIZE;
    }
  }

  const int total_global = expected_dim.product();

  auto host_res = get_test_resource<test_seq>();
  auto working_res = get_test_resource<test_policy>();

  int* actual_index = host_res.allocate<int>(total_global);
  int* actual_size = host_res.allocate<int>(total_global);

  for (int i = 0; i < total_global; ++i) {
    actual_index[i] = -1;
    actual_size[i] = -1;
  }

  actual_index = test_reallocate(working_res, host_res, actual_index, total_global);
  actual_size = test_reallocate(working_res, host_res, actual_size, total_global);

  for3d3d<test_policy>(expected_dim,
      [=] RAJA_HOST_DEVICE (dim3d3d idx, dim3d3d dim) {
    int i = index(idx, dim);
    actual_index[i] = indexer_type::template index<int>();
    actual_size[i] = indexer_type::template size<int>();
  });

  actual_index = test_reallocate(host_res, working_res, actual_index, total_global);
  actual_size = test_reallocate(host_res, working_res, actual_size, total_global);

  for (int i = 0; i < total_global; ++i) {
    ASSERT_EQ( actual_index[i], i );
    ASSERT_EQ( actual_size[i], total_global );
  }

  host_res.deallocate(actual_index);
  host_res.deallocate(actual_size);
}

TYPED_TEST_P( IndexingUnitTest, BasicIndexing )
{
  using test_policy = typename camp::at<TypeParam, camp::num<0>>::type;
  using indexer_holder_type = typename camp::at<TypeParam, camp::num<1>>::type;
  using dim_type = typename camp::at<TypeParam, camp::num<2>>::type;
  using threads_type = typename camp::at<TypeParam, camp::num<3>>::type;
  using blocks_type = typename camp::at<TypeParam, camp::num<4>>::type;

  using indexer_type = typename indexer_holder_type::template type<
      dim_type::value, threads_type::value, blocks_type::value>;

  testBasicIndexing< test_policy, indexer_type,
                     dim_type::value, threads_type::value, blocks_type::value >();
}

REGISTER_TYPED_TEST_SUITE_P( IndexingUnitTest,
                             BasicIndexing );

#endif  //__TEST_INDEXING_GLOBAL__
