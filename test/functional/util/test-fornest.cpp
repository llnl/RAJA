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

using COUNT_SUM = RAJA::expt::ValOp<int, RAJA::operators::plus>;

}  // namespace

TEST(fornest, static_2d_seq_exec)
{
  constexpr int n          = 3;
  constexpr int batch_size = 4;

  std::vector<int> values(n * batch_size, -1);
  auto rows    = RAJA::RangeSegment(0, n);
  auto batches = RAJA::RangeSegment(0, batch_size);

  int count = 0;

  RAJA::fornest<RAJA::seq_exec>(
      rows, batches, RAJA::expt::Reduce<RAJA::operators::plus>(&count),
      [&](int r, int b, COUNT_SUM& c) {
        values[b + batch_size * r] = 100 * r + b;
        c += 1;
      });

  ASSERT_EQ(count, n * batch_size);
  for (int r = 0; r < n; ++r)
  {
    for (int b = 0; b < batch_size; ++b)
    {
      ASSERT_EQ(values[b + batch_size * r], 100 * r + b);
    }
  }
}

TEST(fornest, collapsed_layout_left_seq_exec)
{
  using collapse_left_policy =
      RAJA::fornest_collapsed_policy<RAJA::seq_exec, RAJA::layout_left>;

  constexpr int n          = 3;
  constexpr int batch_size = 4;

  std::vector<int> values(n * batch_size, -1);
  auto rows    = RAJA::RangeSegment(0, n);
  auto batches = RAJA::RangeSegment(0, batch_size);

  RAJA::fornest(collapse_left_policy {}, rows, batches, [&](int r, int b) {
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

TEST(fornest, static_3d_seq_exec)
{
  constexpr int n          = 3;
  constexpr int batch_size = 4;
  constexpr int depth      = 2;

  std::vector<int> values(n * batch_size * depth, -1);
  auto rows    = RAJA::RangeSegment(0, n);
  auto batches = RAJA::RangeSegment(0, batch_size);
  auto depths  = RAJA::RangeSegment(0, depth);

  int count = 0;

  RAJA::fornest<RAJA::seq_exec>(
      rows, batches, depths, RAJA::expt::Reduce<RAJA::operators::plus>(&count),
      [&](int r, int b, int d, COUNT_SUM& c) {
        values[d + depth * (b + batch_size * r)] = 100 * r + 10 * b + d;
        c += 1;
      });

  ASSERT_EQ(count, n * batch_size * depth);
  for (int r = 0; r < n; ++r)
  {
    for (int b = 0; b < batch_size; ++b)
    {
      for (int d = 0; d < depth; ++d)
      {
        ASSERT_EQ(values[d + depth * (b + batch_size * r)],
                  100 * r + 10 * b + d);
      }
    }
  }
}

TEST(fornest, dynamic_switch_collapsed_vs_default)
{
  using policy_list = camp::list<RAJA::seq_exec,
                                 RAJA::fornest_collapsed_policy<RAJA::seq_exec>>;

  constexpr int n          = 3;
  constexpr int batch_size = 4;

  auto rows    = RAJA::RangeSegment(0, n);
  auto batches = RAJA::RangeSegment(0, batch_size);

  for (int pol = 0; pol < 2; ++pol)
  {
    std::vector<int> values(n * batch_size, -1);
    int count = 0;

    RAJA::fornest<policy_list>(
        pol, rows, batches, RAJA::expt::Reduce<RAJA::operators::plus>(&count),
        [&](int r, int b, COUNT_SUM& c) {
          values[b + batch_size * r] = 100 * r + b;
          c += 1;
        });

    ASSERT_EQ(count, n * batch_size);
    for (int r = 0; r < n; ++r)
    {
      for (int b = 0; b < batch_size; ++b)
      {
        ASSERT_EQ(values[b + batch_size * r], 100 * r + b);
      }
    }
  }
}

TEST(fornest, dynamic_resource_host)
{
  using policy_list = camp::list<RAJA::seq_exec,
                                 RAJA::fornest_collapsed_policy<RAJA::seq_exec>>;

  constexpr int n          = 3;
  constexpr int batch_size = 4;

  auto rows    = RAJA::RangeSegment(0, n);
  auto batches = RAJA::RangeSegment(0, batch_size);

  RAJA::resources::Host host_res;
  RAJA::resources::Resource res(host_res);

  for (int pol = 0; pol < 2; ++pol)
  {
    std::vector<int> values(n * batch_size, -1);
    int count = 0;

    (void)RAJA::fornest<policy_list>(
        res, pol, rows, batches,
        RAJA::expt::Reduce<RAJA::operators::plus>(&count),
        [&](int r, int b, COUNT_SUM& c) {
          values[b + batch_size * r] = 100 * r + b;
          c += 1;
        });

    ASSERT_EQ(count, n * batch_size);
    for (int r = 0; r < n; ++r)
    {
      for (int b = 0; b < batch_size; ++b)
      {
        ASSERT_EQ(values[b + batch_size * r], 100 * r + b);
      }
    }
  }
}

TEST(fornest, mapping_policy_host)
{
#if defined(RAJA_GPU_ACTIVE)
  using loop_pol = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::seq_exec>;
#else
  using loop_pol = RAJA::LoopPolicy<RAJA::seq_exec>;
#endif

  using mapping_pol = RAJA::fornest_mapping_policy<
      RAJA::seq_exec,
      loop_pol,
      loop_pol>;

  constexpr int n          = 3;
  constexpr int batch_size = 4;

  std::vector<int> values(n * batch_size, -1);
  auto rows    = RAJA::RangeSegment(0, n);
  auto batches = RAJA::RangeSegment(0, batch_size);

  RAJA::fornest<mapping_pol>(rows, batches, [&](int r, int b) {
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
