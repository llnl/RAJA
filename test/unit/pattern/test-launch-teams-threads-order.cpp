//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Unit tests for RAJA::launch Teams/Threads ordering helpers.
///

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"

namespace
{

constexpr RAJA::Teams t = RAJA::Teams::sycl_order(1, 2, 3);
static_assert(t.value[0] == 3, "Teams::sycl_order dim2 should map to x");
static_assert(t.value[1] == 2, "Teams::sycl_order dim1 should map to y");
static_assert(t.value[2] == 1, "Teams::sycl_order dim0 should map to z");

constexpr RAJA::Threads th = RAJA::Threads::sycl_order(4, 5, 6);
static_assert(th.value[0] == 6, "Threads::sycl_order dim2 should map to x");
static_assert(th.value[1] == 5, "Threads::sycl_order dim1 should map to y");
static_assert(th.value[2] == 4, "Threads::sycl_order dim0 should map to z");

constexpr RAJA::Threads th_out_of_order = RAJA::Threads::sycl_order(4, 6, 5);
static_assert(th_out_of_order.value[0] == 5,
              "Threads::sycl_order dim2 should map to x");
static_assert(th_out_of_order.value[1] == 6,
              "Threads::sycl_order dim1 should map to y");
static_assert(th_out_of_order.value[2] == 4,
              "Threads::sycl_order dim0 should map to z");

}  // namespace

TEST(LaunchTeamsThreadsOrder, compile_time_coverage) { SUCCEED(); }
