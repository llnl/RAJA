//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA_test-base.hpp"

#include <vector>

namespace
{

using launch_policy = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
using flat_policy   = RAJA::launch_nd_flattened_policy<RAJA::seq_exec>;
using flat_left_policy =
    RAJA::launch_nd_flattened_policy<RAJA::seq_exec, RAJA::layout_left>;
using row_loop = RAJA::LoopPolicy<RAJA::seq_exec>;
using col_loop = RAJA::LoopPolicy<RAJA::seq_exec>;

}  // namespace

TEST(launch_nd, layout_right_default_resource)
{
  constexpr int n          = 3;
  constexpr int batch_size = 4;

  std::vector<int> values(n * batch_size, -1);
  auto rows    = RAJA::TypedRangeSegment<int>(0, n);
  auto batches = RAJA::TypedRangeSegment<int>(0, batch_size);

  RAJA::launch_nd(flat_policy {}, RAJA::nd_segments(rows, batches),
                  [&](int r, int b) {
                    values[b + batch_size * r] = 100 * r + b;
                  });

  for (int r = 0; r < n; ++r)
  {
    for (int b = 0; b < batch_size; ++b)
    {
      ASSERT_EQ(values[b + batch_size * r], 100 * r + b);
    }
  }
}

TEST(launch_nd, layout_left_resource)
{
  constexpr int n          = 3;
  constexpr int batch_size = 4;

  std::vector<int> values(n * batch_size, -1);
  auto rows    = RAJA::TypedRangeSegment<int>(0, n);
  auto batches = RAJA::TypedRangeSegment<int>(0, batch_size);

  RAJA::resources::Host host_res;
  RAJA::resources::Resource res(host_res);

  RAJA::launch_nd(res, flat_left_policy {}, RAJA::nd_segments(rows, batches),
                  [&](int r, int b) {
                    values[r + n * b] = 100 * r + b;
                  });

  for (int r = 0; r < n; ++r)
  {
    for (int b = 0; b < batch_size; ++b)
    {
      ASSERT_EQ(values[r + n * b], 100 * r + b);
    }
  }
}

TEST(launch_nd, grid_resource)
{
  constexpr int n          = 3;
  constexpr int batch_size = 4;

  std::vector<int> values(n * batch_size, -1);
  auto rows    = RAJA::TypedRangeSegment<int>(0, n);
  auto batches = RAJA::TypedRangeSegment<int>(0, batch_size);

  RAJA::resources::Host host_res;
  RAJA::resources::Resource res(host_res);

  RAJA::launch_nd(
      res,
      RAJA::launch_nd_grid_policy<launch_policy, row_loop, col_loop>(
          RAJA::LaunchParams()),
      RAJA::nd_segments(rows, batches), [&](int r, int b) {
        values[b + batch_size * r] = 100 * r + b;
      });

  for (int r = 0; r < n; ++r)
  {
    for (int b = 0; b < batch_size; ++b)
    {
      ASSERT_EQ(values[b + batch_size * r], 100 * r + b);
    }
  }
}

TEST(launch_nd, resource_place_mismatch_throws)
{
  RAJA::resources::Host host_res;
  RAJA::resources::Resource res(host_res);

  auto rows    = RAJA::TypedRangeSegment<int>(0, 1);
  auto batches = RAJA::TypedRangeSegment<int>(0, 1);

  // Need gtest death test to avoid complete failure due to eventual seg fault
#if defined(RAJA_ENABLE_TARGET_OPENMP)
  EXPECT_DEATH_IF_SUPPORTED(
      (RAJA::launch_nd(res, RAJA::ExecPlace::DEVICE,
                       RAJA::make_launch_nd_place_policy(flat_policy {},
                                                        flat_policy {}),
                       RAJA::nd_segments(rows, batches), [](int, int) {})),
      "");
#else
  EXPECT_THROW((RAJA::launch_nd(res, RAJA::ExecPlace::DEVICE,
                                RAJA::make_launch_nd_place_policy(flat_policy {},
                                                                 flat_policy {}),
                                RAJA::nd_segments(rows, batches),
                                [](int, int) {})),
               std::runtime_error);
#endif
}
