//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <iostream>
#include <string>
#include <vector>

#include "RAJA/RAJA.hpp"

/*
 *  RAJA::launch_nd mapping policies
 *
 *  This example shows how to select a launch_nd mapping at run time while
 *  keeping one logical 2-D loop body. Run with:
 *
 *    ./launch_nd flat
 *    ./launch_nd grid
 *
 *  The flattened policy takes a regular RAJA forall execution policy, such as
 *  cuda_exec<256> or hip_exec<256>, and RAJA maps it through launch internally.
 *  The grid policy takes an explicit launch policy, loop policies, and
 *  LaunchParams for a true 2-D launch mapping.
 */

namespace
{

constexpr int block_size_1d = 256;
constexpr int block_x       = 16;
constexpr int block_y       = 16;
constexpr int num_cells     = 257;
constexpr int num_comp      = 5;
constexpr int total         = num_cells * num_comp;

int ceil_div(int value, int divisor) { return (value + divisor - 1) / divisor; }

// _launch_nd_mapping_enum_start
enum class Mapping
{
  Flat,
  Grid
};
// _launch_nd_mapping_enum_end

// _launch_nd_policy_aliases_start
#if defined(RAJA_ENABLE_CUDA)
using launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<false>>;
using flat_exec     = RAJA::cuda_exec<block_size_1d>;
using cell_loop     = RAJA::LoopPolicy<RAJA::cuda_global_y_direct>;
using comp_loop     = RAJA::LoopPolicy<RAJA::cuda_global_x_direct>;
using resource_type = RAJA::resources::Cuda;

#elif defined(RAJA_ENABLE_HIP)
using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<false>>;
using flat_exec     = RAJA::hip_exec<block_size_1d>;
using cell_loop     = RAJA::LoopPolicy<RAJA::hip_global_y_direct>;
using comp_loop     = RAJA::LoopPolicy<RAJA::hip_global_x_direct>;
using resource_type = RAJA::resources::Hip;

#else
using launch_policy = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
using flat_exec     = RAJA::seq_exec;
using cell_loop     = RAJA::LoopPolicy<RAJA::seq_exec>;
using comp_loop     = RAJA::LoopPolicy<RAJA::seq_exec>;
using resource_type = RAJA::resources::Host;
#endif
// _launch_nd_policy_aliases_end

Mapping get_mapping(int argc, char** argv)
{
  if (argc < 2)
  {
    return Mapping::Flat;
  }

  const std::string arg = argv[1];
  if (arg == "flat" || arg == "flattened")
  {
    return Mapping::Flat;
  }
  if (arg == "grid")
  {
    return Mapping::Grid;
  }

  std::cout << "Unknown mapping '" << arg << "'. Use 'flat' or 'grid'.\n";
  return Mapping::Flat;
}

template<typename LaunchNdPolicy>
int run_mapping(RAJA::resources::Resource res,
                LaunchNdPolicy policy,
                char const* mapping_name)
{
  int* values_ptr = res.allocate<int>(total);
  res.memset(values_ptr, 0, sizeof(int) * total);

  auto cells = RAJA::TypedRangeSegment<int>(0, num_cells);
  auto comps = RAJA::TypedRangeSegment<int>(0, num_comp);

  // _launch_nd_call_start
  RAJA::launch_nd(res, policy, RAJA::segments(cells, comps),
                  [=] RAJA_HOST_DEVICE(int cell, int comp) {
                    const int idx   = comp + num_comp * cell;
                    values_ptr[idx] = 1000 * cell + comp;
                  });
  // _launch_nd_call_end

  std::vector<int> values(total);
  res.wait();
  res.memcpy(values.data(), values_ptr, sizeof(int) * total);
  res.wait();

  int errors = 0;
  for (int cell = 0; cell < num_cells; ++cell)
  {
    for (int comp = 0; comp < num_comp; ++comp)
    {
      const int idx      = comp + num_comp * cell;
      const int expected = 1000 * cell + comp;
      if (values[idx] != expected)
      {
        ++errors;
      }
    }
  }

  std::cout << "  mapping: " << mapping_name << '\n';
  std::cout << "  logical size: " << num_cells << " x " << num_comp << '\n';
  std::cout << "  result -- " << (errors == 0 ? "PASS" : "FAIL") << '\n';

  res.deallocate(values_ptr);
  return errors;
}

}  // namespace

int main(int argc, char** argv)
{
  std::cout << "\nRAJA launch_nd mapping example...\n";

  RAJA::resources::Resource res(resource_type {});
  const Mapping mapping = get_mapping(argc, argv);

  int errors = 0;
  // _launch_nd_runtime_select_start
  if (mapping == Mapping::Flat)
  {
    auto policy = RAJA::launch_nd_flattened_policy<flat_exec> {};
    errors      = run_mapping(res, policy, "flat");
    std::cout << "  flattened launch threads per block: " << block_size_1d
              << '\n';
  }
  else
  {
    auto policy =
        RAJA::launch_nd_grid_policy<launch_policy, cell_loop, comp_loop>(
            RAJA::LaunchParams(RAJA::Teams(ceil_div(num_comp, block_x),
                                           ceil_div(num_cells, block_y)),
                               RAJA::Threads(block_x, block_y)));
    errors = run_mapping(res, policy, "grid");
    std::cout << "  grid launch block shape: " << block_x << " x " << block_y
              << '\n';
  }
  // _launch_nd_runtime_select_end

  std::cout << "\nDONE!...\n";
  return errors == 0 ? 0 : 1;
}
