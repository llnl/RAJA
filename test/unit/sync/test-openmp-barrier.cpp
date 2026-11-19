//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA_test-base.hpp"

#include <array>
#include <omp.h>

TEST(SynchronizeTest, OpenMPBarrier)
{
  constexpr int requested_threads = 4;
  std::array<int, requested_threads> written{};
  std::array<int, requested_threads> observed{};
  int team_size = 0;

#pragma omp parallel num_threads(requested_threads) shared(written, observed, team_size)
  {
    const int thread = omp_get_thread_num();
    const int active_threads = omp_get_num_threads();
    if (thread == 0) {
      team_size = active_threads;
    }
    written[thread] = thread + 1;

    RAJA::synchronize<RAJA::omp_synchronize>();

    int sum = 0;
    for (int i = 0; i < active_threads; ++i) {
      sum += written[i];
    }
    observed[thread] = sum;
  }

  ASSERT_GE(team_size, 2) << "OpenMP barrier test requires multiple threads";
  const int expected = team_size * (team_size + 1) / 2;
  for (int i = 0; i < team_size; ++i) {
    EXPECT_EQ(observed[i], expected) << "thread " << i;
  }
}
