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
 *  - collapsed mapping (1-D space + index reconstruction)
 *  - explicit per-dimension mapping via fornest_mapping_policy
 *  - basic mapping aliases
 *  - fixed, runtime, and auto tiling
 *  - dynamic_fornest policy selection
 *  - Caliper labeling via RAJA::Name
 *
 * Run:
 *   ./fornest-basic collapse
 *   ./fornest-basic map
 *   ./fornest-basic basic
 *   ./fornest-basic tile-fixed
 *   ./fornest-basic tile-runtime
 *   ./fornest-basic tile-auto
 *   ./fornest-basic dynamic
 *   ./fornest-basic omp-outer       (OpenMP builds)
 *   ./fornest-basic omp-collapse    (OpenMP host execution)
 *
 * Caliper (built with RAJA_ENABLE_CALIPER=ON and RAJA_ENABLE_RUNTIME_PLUGINS=ON):
 *   RAJA_CALIPER=1 CALI_CONFIG=runtime-report ./fornest-basic collapse
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
  Collapse,
  Map,
  Basic,
  TileFixed,
  TileRuntime,
  TileAuto,
  Dynamic,
#if defined(RAJA_ENABLE_OPENMP)
  OmpOuter,
  OmpCollapse,
#endif
};

/*
 * Parse the command-line mapping selector.
 *
 * Accepted values:
 *  - "collapse" / "collapsed": use fornest_collapsed_policy (1-D iteration space)
 *  - "map" / "mapped": use fornest_mapping_policy (per-dimension mapping)
 *  - "basic": use the convenience basic mapping alias
 *  - "tile-fixed": use compile-time tile sizes
 *  - "tile-runtime": use RAJA::TileSize runtime tile sizes
 *  - "tile-auto": let RAJA pick tile sizes
 *  - "dynamic": use dynamic_fornest to select a policy from a list
 *  - "omp-outer": use an outer OpenMP loop (OpenMP builds)
 *  - "omp-collapse": use OpenMP collapse on the host (OpenMP builds)
 *
 * Throws on unrecognized input so main() can report a clear error.
 */
Mapping parse_mapping(const std::string& arg)
{
  if (arg == "collapse" || arg == "collapsed")
  {
    return Mapping::Collapse;
  }
  if (arg == "map" || arg == "mapped")
  {
    return Mapping::Map;
  }
  if (arg == "basic")
  {
    return Mapping::Basic;
  }
  if (arg == "tile-fixed")
  {
    return Mapping::TileFixed;
  }
  if (arg == "tile-runtime")
  {
    return Mapping::TileRuntime;
  }
  if (arg == "tile-auto")
  {
    return Mapping::TileAuto;
  }
  if (arg == "dynamic")
  {
    return Mapping::Dynamic;
  }
#if defined(RAJA_ENABLE_OPENMP)
  if (arg == "omp-outer")
  {
    return Mapping::OmpOuter;
  }
  if (arg == "omp-collapse")
  {
    return Mapping::OmpCollapse;
  }
#endif
  throw std::runtime_error(
      "expected 'collapse', 'map', 'basic', 'tile-fixed', 'tile-runtime', "
      "'tile-auto', or 'dynamic'");
}

// _fornest_policy_aliases_start
using collapse_policy = RAJA::fornest_collapsed_policy<exec_pol>;
using basic_policy    = RAJA::fornest_basic_seq_2d<exec_pol>;
using tile_fixed_policy =
    RAJA::fornest_tiling_policy<exec_pol,
                                RAJA::fornest_tile_fixed<2>,
                                RAJA::fornest_tile_fixed<2>>;
using tile_runtime_policy =
    RAJA::fornest_tiling_policy<exec_pol,
                                RAJA::fornest_tile_runtime,
                                RAJA::fornest_tile_runtime>;
using tile_auto_policy =
    RAJA::fornest_tiling_policy<exec_pol,
                                RAJA::fornest_tile_auto,
                                RAJA::fornest_tile_auto>;
using dynamic_policy_list =
    camp::list<basic_policy, collapse_policy, tile_fixed_policy>;

// A portable "nested loops" mapping policy.
using map_policy = RAJA::fornest_mapping_policy<
    exec_pol,
    RAJA::LoopPolicy<RAJA::seq_exec>,
    RAJA::LoopPolicy<RAJA::seq_exec>>;

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
// CUDA/HIP can use unsized global device mapping tags.
using map_global_policy = RAJA::fornest_mapping_policy<
    exec_pol,
    RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_x_direct>,
    RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_y_direct>>;
#endif

#if defined(RAJA_GPU_ACTIVE)
// CUDA/HIP/SYCL can use device aliases for block/work-group and
// thread/work-item mappings.
using map_block_thread_policy = RAJA::fornest_mapping_policy<
    exec_pol,
    RAJA::LoopPolicy<RAJA::device_block_x_direct>,
    RAJA::LoopPolicy<RAJA::device_thread_x_loop>>;
#endif

#if defined(RAJA_ENABLE_OPENMP)
using omp_outer_policy = RAJA::fornest_basic_omp_outer_2d<exec_pol>;
using omp_collapse_policy = RAJA::fornest_omp_collapse_policy<RAJA::seq_exec>;
#endif
// _fornest_policy_aliases_end

template<typename Policy, typename Rows, typename Cols, typename Body>
void run_fornest(Policy policy,
                 Rows const& rows,
                 Cols const& cols,
                 const char* name,
                 Body const& body)
{
  RAJA::fornest(policy, rows, cols, RAJA::Name(name), body);
}

#if defined(RAJA_GPU_ACTIVE)
bool uses_device_memory(Mapping mapping)
{
#if defined(RAJA_ENABLE_OPENMP)
  if (mapping == Mapping::OmpCollapse)
  {
    return false;
  }
#endif
  RAJA_UNUSED_VAR(mapping);
  return true;
}
#endif

}  // namespace

int main(int argc, char** argv)
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: " << argv[0]
                << " <collapse|map|basic|tile-fixed|tile-runtime|tile-auto|"
                   "dynamic";
#if defined(RAJA_ENABLE_OPENMP)
      std::cerr << "|omp-outer";
      std::cerr << "|omp-collapse";
#endif
      std::cerr << ">\n";
      return 2;
    }

    const Mapping mapping = parse_mapping(argv[1]);

    std::vector<int> values(n_rows * n_cols, -1);
    auto rows = RAJA::RangeSegment(0, n_rows);
    auto cols = RAJA::RangeSegment(0, n_cols);

    int* values_ptr = values.data();

#if defined(RAJA_GPU_ACTIVE)
    const bool use_device_memory = uses_device_memory(mapping);
    auto device_res = RAJA::resources::get_default_resource<exec_pol>();
    int* values_d = nullptr;
    if (use_device_memory)
    {
      values_d   = device_res.allocate<int>(n_rows * n_cols);
      values_ptr = values_d;
    }
#endif

    auto body = [=] RAJA_HOST_DEVICE(int r, int c) {
      values_ptr[c + n_cols * r] = 100 * r + c;
    };

    // _fornest_runtime_select_start
    switch (mapping)
    {
    case Mapping::Collapse:
      // _fornest_call_start
      run_fornest(collapse_policy {}, rows, cols, "fornest_basic_collapse",
                  body);
      // _fornest_call_end
      break;
    case Mapping::Map:
#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
      run_fornest(map_global_policy {}, rows, cols, "fornest_basic_map_global",
                  body);
#elif defined(RAJA_GPU_ACTIVE)
      run_fornest(map_block_thread_policy {}, rows, cols,
                  "fornest_basic_map_block_thread", body);
#else
      run_fornest(map_policy {}, rows, cols, "fornest_basic_map", body);
#endif
      break;
    case Mapping::Basic:
      run_fornest(basic_policy {}, rows, cols, "fornest_basic_alias", body);
      break;
    case Mapping::TileFixed:
      run_fornest(tile_fixed_policy {}, rows, cols, "fornest_basic_tile_fixed",
                  body);
      break;
    case Mapping::TileRuntime:
      run_fornest(tile_runtime_policy {RAJA::TileSize(2), RAJA::TileSize(2)},
                  rows, cols, "fornest_basic_tile_runtime", body);
      break;
    case Mapping::TileAuto:
      run_fornest(tile_auto_policy {}, rows, cols, "fornest_basic_tile_auto",
                  body);
      break;
    case Mapping::Dynamic:
    {
      constexpr int policy_index = 2;
      RAJA::dynamic_fornest<dynamic_policy_list>(
          policy_index, rows, cols, RAJA::Name("fornest_basic_dynamic"), body);
      break;
    }
#if defined(RAJA_ENABLE_OPENMP)
    case Mapping::OmpOuter:
      run_fornest(omp_outer_policy {}, rows, cols, "fornest_basic_omp_outer",
                  body);
      break;
    case Mapping::OmpCollapse:
      run_fornest(omp_collapse_policy {}, rows, cols,
                  "fornest_basic_omp_collapse", body);
      break;
#endif
    }
    // _fornest_runtime_select_end

#if defined(RAJA_GPU_ACTIVE)
    if (use_device_memory)
    {
      device_res.memcpy(values.data(), values_d, sizeof(int) * values.size());
      device_res.wait();
      device_res.deallocate(values_d);
    }
#endif

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
