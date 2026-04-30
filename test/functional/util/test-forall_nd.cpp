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

using policy_list = camp::list<RAJA::seq_exec>;

}  // namespace

TEST(forall_nd, layout_right_default_resource)
{
  constexpr int n          = 3;
  constexpr int batch_size = 4;

  std::vector<int> values(n * batch_size, -1);
  auto rows    = RAJA::TypedRangeSegment<int>(0, n);
  auto batches = RAJA::TypedRangeSegment<int>(0, batch_size);

  RAJA::forall_nd<policy_list, RAJA::layout_right>(
      0, RAJA::segments(rows, batches), [&](int r, int b) {
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

TEST(forall_nd, layout_left_resource)
{
  constexpr int n          = 3;
  constexpr int batch_size = 4;

  std::vector<int> values(n * batch_size, -1);
  auto rows    = RAJA::TypedRangeSegment<int>(0, n);
  auto batches = RAJA::TypedRangeSegment<int>(0, batch_size);

  RAJA::resources::Host host_res;
  RAJA::resources::Resource res(host_res);

  RAJA::forall_nd<policy_list, RAJA::layout_left>(
      res, 0, RAJA::segments(rows, batches), [&](int r, int b) {
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
