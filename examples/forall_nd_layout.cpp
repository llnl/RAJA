//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <vector>

#include "RAJA/RAJA.hpp"

/*
 *  Flattened N-D forall helper with layout-left/layout-right selection
 *
 *  This example shows the public RAJA::forall_nd wrapper, which executes a
 *  flattened 1-D kernel selected from a run-time policy list and reconstructs
 *  the multi-dimensional indices for the user lambda.
 *
 *  The public call shape is:
 *
 *    RAJA::forall_nd<policy_list, RAJA::layout_right>(
 *        res, pol, RAJA::segments(rows, batches),
 *        [=] RAJA_HOST_DEVICE(int r, int b) { ... });
 *
 *  Supported layouts:
 *    - RAJA::layout_right : right-most index varies fastest
 *    - RAJA::layout_left  : left-most index varies fastest
 */

namespace
{

using policy_list = camp::list<
    RAJA::seq_exec,
    RAJA::simd_exec
#if defined(RAJA_ENABLE_OPENMP)
    ,
    RAJA::omp_parallel_for_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
    ,
    RAJA::cuda_exec<256>
#endif
#if defined(RAJA_ENABLE_HIP)
    ,
    RAJA::hip_exec<256>
#endif
#if defined(RAJA_ENABLE_SYCL)
    ,
    RAJA::sycl_exec<256>
#endif
    >;

constexpr int host_policy_count =
    2
#if defined(RAJA_ENABLE_OPENMP)
    + 1
#endif
    ;

RAJA::resources::Resource get_resource_for_policy(int pol)
{
  if (pol < 0)
  {
    RAJA_ABORT_OR_THROW("Policy value out of range for this build");
  }

  RAJA::resources::Host host_res;
  if (pol < host_policy_count)
  {
    return RAJA::resources::Resource(host_res);
  }

#if defined(RAJA_ENABLE_CUDA)
  if (pol == host_policy_count)
  {
    return RAJA::resources::Resource(RAJA::resources::Cuda {});
  }
#endif

#if defined(RAJA_ENABLE_HIP)
  if (pol == host_policy_count
#if defined(RAJA_ENABLE_CUDA)
            + 1
#endif
  )
  {
    return RAJA::resources::Resource(RAJA::resources::Hip {});
  }
#endif

#if defined(RAJA_ENABLE_SYCL)
  if (pol == host_policy_count
#if defined(RAJA_ENABLE_CUDA)
            + 1
#endif
#if defined(RAJA_ENABLE_HIP)
            + 1
#endif
  )
  {
    return RAJA::resources::Resource(RAJA::resources::Sycl {});
  }
#endif

  RAJA_ABORT_OR_THROW("Policy value out of range for this build");
}

void print_linear_order(char const* label, std::vector<int> const& values)
{
  std::cout << "  " << label << ":";
  for (auto value : values)
  {
    std::cout << ' ' << value;
  }
  std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
  const int pol = (argc > 1) ? std::atoi(argv[1]) : 0;

  std::cout << "\nRAJA flattened N-D forall helper example...\n";
  std::cout << "Using policy #" << pol << '\n';

  constexpr int n          = 3;
  constexpr int batch_size = 4;
  constexpr int total_size = n * batch_size;

  auto res = get_resource_for_policy(pol);

  int* right_ptr = res.allocate<int>(total_size);
  int* left_ptr  = res.allocate<int>(total_size);

  res.memset(right_ptr, 0xff, sizeof(int) * total_size);
  res.memset(left_ptr, 0xff, sizeof(int) * total_size);

  auto rows    = RAJA::TypedRangeSegment<int>(0, n);
  auto batches = RAJA::TypedRangeSegment<int>(0, batch_size);

  // layout_right makes the right-most logical index ('b') unit stride.
  // _forall_nd_right_start
  RAJA::forall_nd<policy_list, RAJA::layout_right>(
      res, pol, RAJA::segments(rows, batches),
      [=] RAJA_HOST_DEVICE(int r, int b) {
        const int flat  = b + batch_size * r;
        right_ptr[flat] = 100 * r + b;
      });
  // _forall_nd_right_end

  // layout_left makes the left-most logical index ('r') unit stride.
  // _forall_nd_left_start
  RAJA::forall_nd<policy_list, RAJA::layout_left>(
      res, pol, RAJA::segments(rows, batches),
      [=] RAJA_HOST_DEVICE(int r, int b) {
        const int flat = r + n * b;
        left_ptr[flat] = 100 * r + b;
      });
  // _forall_nd_left_end

  std::vector<int> right(total_size);
  std::vector<int> left(total_size);
  std::vector<int> expect_right(total_size);
  std::vector<int> expect_left(total_size);

  res.wait();
  res.memcpy(right.data(), right_ptr, sizeof(int) * total_size);
  res.memcpy(left.data(), left_ptr, sizeof(int) * total_size);

  for (int r = 0; r < n; ++r)
  {
    for (int b = 0; b < batch_size; ++b)
    {
      expect_right[b + batch_size * r] = 100 * r + b;
      expect_left[r + n * b] = 100 * r + b;
    }
  }

  print_linear_order("layout_right", right);
  print_linear_order("layout_left ", left);

  const bool ok = (right == expect_right) && (left == expect_left);

  std::cout << "  result -- " << (ok ? "PASS" : "FAIL") << '\n';

  res.deallocate(right_ptr);
  res.deallocate(left_ptr);

  std::cout << "\nDONE!...\n";
  return ok ? 0 : 1;
}
