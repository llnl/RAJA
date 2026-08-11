//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "RAJA/RAJA.hpp"

/*
 * Minimal RAJA::fornest example (2-D)
 *
 * Demonstrates:
 *  - flattened mapping (1-D space + index reconstruction)
 *  - explicit per-dimension mapping via fornest_mapping_policy
 *  - Caliper labeling via RAJA::Name
 *
 * Run:
 *   ./fornest-basic flat
 *   ./fornest-basic map
 *
 * Caliper (built with RAJA_ENABLE_CALIPER=ON and RAJA_ENABLE_RUNTIME_PLUGINS=ON):
 *   RAJA_CALIPER=1 CALI_CONFIG=runtime-report ./fornest-basic flat
 *   RAJA_CALIPER=1 CALI_CONFIG=runtime-profile(output=fornest-basic.cali,output.format=cali) \
 *     ./fornest-basic map
 */

namespace
{

constexpr int n_rows = 8;
constexpr int n_cols = 4;

#if defined(RAJA_GPU_ACTIVE)
constexpr int block_size_1d = 256;
using exec_pol = RAJA::device_exec<block_size_1d>;
#else
using exec_pol = RAJA::seq_exec;
#endif

enum class Mapping
{
  Flat,
  Map
};

Mapping parse_mapping(const std::string& arg)
{
  if (arg == "flat" || arg == "flattened")
  {
    return Mapping::Flat;
  }
  if (arg == "map" || arg == "mapped")
  {
    return Mapping::Map;
  }
  throw std::runtime_error("expected 'flat' or 'map'");
}

// _fornest_policy_aliases_start
using flat_policy = RAJA::fornest_flattened_policy<exec_pol>;

// A portable "nested loops" mapping policy.
using map_policy = RAJA::fornest_mapping_policy<
    exec_pol,
    RAJA::LoopPolicy<RAJA::seq_exec, RAJA::seq_exec>,
    RAJA::LoopPolicy<RAJA::seq_exec, RAJA::seq_exec>>;

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
// CUDA/HIP can use explicit device mapping tags (global/block/thread spaces).
using map_global_policy = RAJA::fornest_mapping_policy<
    exec_pol,
    RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_x_direct>,
    RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_y_direct>>;
#endif
// _fornest_policy_aliases_end

}  // namespace

int main(int argc, char** argv)
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: " << argv[0] << " <flat|map>\n";
      return 2;
    }

    const Mapping mapping = parse_mapping(argv[1]);

    std::vector<int> values(n_rows * n_cols, -1);
    auto rows = RAJA::RangeSegment(0, n_rows);
    auto cols = RAJA::RangeSegment(0, n_cols);

    auto body = [&](int r, int c) {
      values[c + n_cols * r] = 100 * r + c;
    };

    // _fornest_runtime_select_start
    if (mapping == Mapping::Flat)
    {
      // _fornest_call_start
      RAJA::fornest(flat_policy {}, rows, cols, RAJA::Name("fornest_basic_flat"),
                    body);
      // _fornest_call_end
    }
    else
    {
#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
      RAJA::fornest(map_global_policy {}, rows, cols,
                    RAJA::Name("fornest_basic_map_global"),
                    body);
#else
      RAJA::fornest(map_policy {}, rows, cols, RAJA::Name("fornest_basic_map"),
                    body);
#endif
    }
    // _fornest_runtime_select_end

    int errors = 0;
    for (int r = 0; r < n_rows; ++r)
    {
      for (int c = 0; c < n_cols; ++c)
      {
        const int idx      = c + n_cols * r;
        const int expected = 100 * r + c;
        if (values[idx] != expected)
        {
          ++errors;
        }
      }
    }

    if (errors != 0)
    {
      std::cerr << "FAIL: " << errors << " errors\n";
      return 1;
    }

    std::cout << "PASS\n";
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
