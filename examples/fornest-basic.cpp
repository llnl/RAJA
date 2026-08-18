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
#include <cctype>

#include "RAJA/RAJA.hpp"

/*
 * Minimal RAJA::fornest example (2-D)
 *
 * Demonstrates:
 *  - collapsed mapping (1-D space + index reconstruction)
 *  - explicit per-dimension mapping via fornest_mapping_policy
 *  - nested-loops mapping alias
 *  - fixed, runtime, and auto tiling
 *  - Caliper labeling via RAJA::Name
 *
 * Run:
 *   ./fornest-basic collapse
 *   ./fornest-basic map
 *   ./fornest-basic nested-loops
 *   ./fornest-basic tile-fixed
 *   ./fornest-basic tile-runtime
 *   ./fornest-basic tile-auto
#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
 *   ./fornest-basic map-global-sized
 *   ./fornest-basic map-block-thread-sized
#endif
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
using forall_exec_pol = RAJA::device_exec<block_size_1d>;
#else
using forall_exec_pol = RAJA::seq_exec;
#endif

#if defined(RAJA_GPU_ACTIVE)
// fornest mapping policies execute via RAJA::launch and derive Threads/Teams
// from the mapping tags and iteration space extents.
using mapping_exec_pol = RAJA::device_launch_t<false>;
#else
using mapping_exec_pol = RAJA::seq_launch_t;
#endif

enum class Mapping
{
  Collapse,
  Map,
  MapGlobalSized,
  MapBlockThreadSized,
  Basic,
  TileFixed,
  TileRuntime,
  TileAuto,
#if defined(RAJA_ENABLE_OPENMP)
  OmpOuter,
  OmpCollapse,
#endif
};

static void print_mapping_menu()
{
  int idx    = 0;
  auto print = [&](const char* name) {
    std::cout << "  " << idx++ << ": " << name << "\n";
  };

  print("collapse");
  print("map");
#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
  print("map-global-sized");
  print("map-block-thread-sized");
#endif
  print("nested-loops");
#if defined(RAJA_GPU_ACTIVE)
  print("tile-fixed (GPU: similar to map-global-sized)");
  print("tile-runtime (GPU: use RAJA::launch for runtime block shapes)");
  print("tile-auto (GPU: similar to map)");
#else
  print("tile-fixed");
  print("tile-runtime");
  print("tile-auto");
#endif
#if defined(RAJA_ENABLE_OPENMP)
  print("omp-outer");
  print("omp-collapse");
#endif
}

static bool parse_mapping_arg(const char* arg, Mapping& out)
{
  std::string s(arg ? arg : "");
  if (s.empty()) return false;

  // Numeric index support (like dynamic-fornest).
  {
    std::size_t pos = 0;
    if (s[0] == '+' || s[0] == '-') pos = 1;
    bool all_digits = (pos < s.size());
    for (; pos < s.size(); ++pos)
    {
      all_digits =
          all_digits && (std::isdigit(static_cast<unsigned char>(s[pos])) != 0);
    }
    if (all_digits)
    {
      const int sel = std::stoi(s);
      int idx       = 0;
      auto accept   = [&](Mapping m) {
        if (sel == idx++)
        {
          out = m;
          return true;
        }
        return false;
      };

      if (accept(Mapping::Collapse)) return true;
      if (accept(Mapping::Map)) return true;
#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
      if (accept(Mapping::MapGlobalSized)) return true;
      if (accept(Mapping::MapBlockThreadSized)) return true;
#endif
      if (accept(Mapping::Basic)) return true;
      if (accept(Mapping::TileFixed)) return true;
      if (accept(Mapping::TileRuntime)) return true;
      if (accept(Mapping::TileAuto)) return true;
#if defined(RAJA_ENABLE_OPENMP)
      if (accept(Mapping::OmpOuter)) return true;
      if (accept(Mapping::OmpCollapse)) return true;
#endif
      return false;
    }
  }

  if (s == "collapse" || s == "collapsed")
  {
    out = Mapping::Collapse;
    return true;
  }
  if (s == "map" || s == "mapped")
  {
    out = Mapping::Map;
    return true;
  }
  if (s == "map-global-sized")
  {
    out = Mapping::MapGlobalSized;
    return true;
  }
  if (s == "map-block-thread-sized")
  {
    out = Mapping::MapBlockThreadSized;
    return true;
  }
  if (s == "nested-loops" || s == "basic")
  {
    out = Mapping::Basic;
    return true;
  }
  if (s == "tile-fixed")
  {
    out = Mapping::TileFixed;
    return true;
  }
  if (s == "tile-runtime")
  {
    out = Mapping::TileRuntime;
    return true;
  }
  if (s == "tile-auto")
  {
    out = Mapping::TileAuto;
    return true;
  }
#if defined(RAJA_ENABLE_OPENMP)
  if (s == "omp-outer")
  {
    out = Mapping::OmpOuter;
    return true;
  }
  if (s == "omp-collapse")
  {
    out = Mapping::OmpCollapse;
    return true;
  }
#endif

  return false;
}

// _fornest_policy_aliases_start
// Note: collapsed/tiling policies use a forall-style exec policy
// (e.g., `device_exec<256>`), while mapping policies use a launch-style exec
// policy (e.g., `device_launch_t<false>`) and derive Threads/Teams from the
// mapping tags and segment extents.
using collapse_policy = RAJA::fornest_collapsed_policy<forall_exec_pol>;
using nested_loops_policy = RAJA::fornest_basic_seq_2d<forall_exec_pol>;
using tile_fixed_policy =
    RAJA::fornest_tiling_policy<forall_exec_pol,
                                RAJA::fornest_tile_fixed<2>,
                                RAJA::fornest_tile_fixed<2>>;
using tile_runtime_policy =
    RAJA::fornest_tiling_policy<forall_exec_pol,
                                RAJA::fornest_tile_runtime,
                                RAJA::fornest_tile_runtime>;
using tile_auto_policy =
    RAJA::fornest_tiling_policy<forall_exec_pol,
                                RAJA::fornest_tile_auto,
                                RAJA::fornest_tile_auto>;

// A portable "nested loops" mapping policy.
using map_policy = RAJA::fornest_mapping_policy<
    mapping_exec_pol,
    RAJA::seq_exec,
    RAJA::seq_exec>;

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
// CUDA/HIP can use unsized global device mapping tags.
//
// What this means for kernel launch configuration:
// - The logical dimensions map directly onto the CUDA/HIP global indices:
//     dim0 -> global_x, dim1 -> global_y
// - `mapping_exec_pol` is a launch-style policy (`device_launch_t<false>`), so
//   RAJA computes `LaunchParams(Teams, Threads)` from the mapping tags and the
//   segment extents.
// - Because these tags are *unsized* (no compile-time BLOCK_SIZE/GRID_SIZE),
//   RAJA chooses a reasonable 2D thread shape `(tx, ty)` from a thread budget
//   (defaults to 256 when the launch policy does not fix a thread count) and
//   the extents `(n_rows, n_cols)`.
// - Then RAJA sets:
//     Threads = (tx, ty)
//     Teams   = (ceil(n_rows/tx), ceil(n_cols/ty))
using map_global_policy = RAJA::fornest_mapping_policy<
    mapping_exec_pol,
    RAJA::device_global_x_direct,
    RAJA::device_global_y_direct>;

// A "fully specified" global mapping using compile-time sized tags.
// This configures:
// - Threads.x = n_rows, Threads.y = n_cols
// - Teams.x = ceil(n_rows / Threads.x) = 1
// - Teams.y = ceil(n_cols / Threads.y) = 1
using map_global_sized_policy = RAJA::fornest_mapping_policy<
    mapping_exec_pol,
    RAJA::device_global_size_x_direct<n_rows>,
    RAJA::device_global_size_y_direct<n_cols>>;

// A sized block/thread mapping:
// - `block_size_x_direct<n_rows>` requests Teams.x = n_rows (one block per row).
// - `thread_size_x_loop<128>` requests Threads.x = 128; each thread strides the
//   inner dimension (cols) by `threadIdx.x += blockDim.x`.
using map_block_thread_sized_policy = RAJA::fornest_mapping_policy<
    mapping_exec_pol,
    RAJA::device_block_size_x_direct<n_rows>,
    RAJA::device_thread_size_x_loop<128>>;
#endif

#if defined(RAJA_GPU_ACTIVE)
// CUDA/HIP/SYCL can use device aliases for block/work-group and
// thread/work-item mappings.
using map_block_thread_policy = RAJA::fornest_mapping_policy<
    mapping_exec_pol,
    RAJA::device_block_x_direct,
    RAJA::device_thread_x_loop>;
#endif

#if defined(RAJA_ENABLE_OPENMP)
using omp_outer_policy = RAJA::fornest_basic_omp_outer_2d<forall_exec_pol>;
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
      std::cerr << "Usage: " << argv[0] << " MAPPING\n";
      std::cerr << "Where MAPPING is a numeric index or a name:\n";
      print_mapping_menu();
      return 2;
    }

    Mapping mapping {};
    if (!parse_mapping_arg(argv[1], mapping))
    {
      std::cerr << "Unrecognized MAPPING '" << argv[1]
                << "'. Choose one of:\n";
      print_mapping_menu();
      return 2;
    }

    std::vector<int> values(n_rows * n_cols, -1);
    auto rows = RAJA::RangeSegment(0, n_rows);
    auto cols = RAJA::RangeSegment(0, n_cols);

    int* values_ptr = values.data();

#if defined(RAJA_GPU_ACTIVE)
    const bool use_device_memory = uses_device_memory(mapping);
    auto device_res = RAJA::resources::get_default_resource<forall_exec_pol>();
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
    case Mapping::MapGlobalSized:
#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
      run_fornest(map_global_sized_policy {}, rows, cols,
                  "fornest_basic_map_global_sized", body);
#else
      throw std::runtime_error("'map-global-sized' requires a CUDA or HIP build");
#endif
      break;
    case Mapping::MapBlockThreadSized:
#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
      run_fornest(map_block_thread_sized_policy {}, rows, cols,
                  "fornest_basic_map_block_thread_sized", body);
#else
      throw std::runtime_error(
          "'map-block-thread-sized' requires a CUDA or HIP build");
#endif
      break;
    case Mapping::Basic:
      run_fornest(nested_loops_policy {}, rows, cols, "fornest_basic_alias",
                  body);
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
