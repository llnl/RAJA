//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cctype>
#include <iostream>
#include <string>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  2D example with dynamic policy selection for RAJA::fornest
 *
 *  Selects a policy at runtime. The policy list contains both nested
 *  (exec/mapping policy) and collapsed (fornest_collapsed_policy) entries so
 *  the runtime selection can switch between mappings.
 */

//------------------------------------------------------------------------------
// Policies
//------------------------------------------------------------------------------
/*
 * Backend behavior (policy cheat sheet):
 * - `RAJA::fornest<policy_list>(pol, ...)` picks the `pol`-th TYPE in
 *   `policy_list` and calls `RAJA::fornest(selected_policy{}, ...)`.
 */

// `seq_nested = RAJA::seq_exec`:
// host backend: `fornest(ExecPolicy, ...)` overload; currently a collapsed 1D
// `forall(ni*nj)` with (i,j) reconstruction inside.
using seq_nested = RAJA::seq_exec;

// `seq_collapse = RAJA::fornest_collapsed_policy<RAJA::seq_exec>`:
// host backend: always flatten to 1D + reconstruct, scheduled with `seq_exec`.
using seq_collapse = RAJA::fornest_collapsed_policy<RAJA::seq_exec>;

// `simd_nested = RAJA::simd_exec`:
// host backend: same as `seq_nested`, scheduled with SIMD `forall`.
using simd_nested = RAJA::simd_exec;

// `simd_collapse = RAJA::fornest_collapsed_policy<RAJA::simd_exec>`:
// host backend: flattened 1D `forall`, scheduled with `simd_exec`.
using simd_collapse = RAJA::fornest_collapsed_policy<RAJA::simd_exec>;

#if defined(RAJA_ENABLE_OPENMP)
// `omp_nested = RAJA::omp_parallel_for_exec`:
// host backend: same `fornest(ExecPolicy, ...)` path, OpenMP-parallel `forall`.
using omp_nested = RAJA::omp_parallel_for_exec;

// `omp_collapse = fornest_collapsed_policy<omp_parallel_for_exec>`:
// host backend: flattened 1D `forall`, OpenMP-parallel.
using omp_collapse =
    RAJA::fornest_collapsed_policy<RAJA::omp_parallel_for_exec>;
#endif

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
// `dev256_nested = RAJA::device_exec<256>`:
// device backend default mapping: choose (tx,ty) fitting 256 threads, launch
// `Threads(tx,ty)`, `Teams(ceil(ni/tx), ceil(nj/ty))`, map `dim0->global_x`,
// `dim1->global_y`.
using dev256_nested = RAJA::device_exec<256>;

// `dev256_collapse = fornest_collapsed_policy<device_exec<256>>`:
// device backend: flattened 1D `forall` with 256-thread blocks, reconstruct
// (i,j).
using dev256_collapse = RAJA::fornest_collapsed_policy<dev256_nested>;

// `dev256_perfect = fornest_mapping_policy<device_launch_t<false>, (global_x, seq)>`:
// device backend: explicit `launch` mapping; outer dim is `global_x`, inner is
// seq.
//
// Launch configuration notes:
// - `device_launch_t<false>` is a launch-style policy; RAJA computes
//   `LaunchParams(Teams, Threads)` from the mapping tags and the segment extents.
// - Here, `global_x_direct` is unsized, so RAJA chooses `Threads.x` using a
//   default thread budget (currently 256) and the extent `ni`.
// - The `seq` inner dimension runs serially inside each (global_x,global_y)
//   location, so `Threads.y = 1` and `Teams.y = 1`.
using dev256_perfect = RAJA::fornest_mapping_policy<
    RAJA::device_launch_t<false>,
    RAJA::device_global_x_direct,
    RAJA::seq_exec>;

// `dev256_global_sized =
// fornest_mapping_policy<device_launch_t<false>, (global_size_x<32>, global_size_y<8>)>`:
// device backend: explicit `launch` mapping that fully determines `Threads` via
// compile-time sizes.
//
// This configures:
// - Threads = (32, 8)
// - Teams   = (ceil(ni/32), ceil(nj/8))
using dev256_global_sized = RAJA::fornest_mapping_policy<
    RAJA::device_launch_t<false>,
    RAJA::device_global_size_x_direct<32>,
    RAJA::device_global_size_y_direct<8>>;

// `dev256_block_thread =
// fornest_mapping_policy<device_launch_t<false>, (block_x, thread_x_loop)>`:
// device backend: explicit `launch` mapping; outer dim uses `blockIdx.x`, inner
// uses `threadIdx.x` with looping/striding semantics.
using dev256_block_thread = RAJA::fornest_mapping_policy<
    RAJA::device_launch_t<false>,
    RAJA::device_block_x_direct,
    RAJA::device_thread_x_loop>;

// `dev256_block_thread_sized =
// fornest_mapping_policy<device_launch_t<false>, (block_size_x<ni>, thread_size_x_loop<32>)>`:
//
// This configures:
// - Teams.x   = ni (one block per i)
// - Threads.x = 32 (threads stride j: j += blockDim.x)
using dev256_block_thread_sized = RAJA::fornest_mapping_policy<
    RAJA::device_launch_t<false>,
    RAJA::device_block_size_x_direct<262144>,
    RAJA::device_thread_size_x_loop<32>>;

// `dev512_nested`, `dev512_collapse`: same as the 256 variants but with
// 512-thread blocks.
using dev512_nested   = RAJA::device_exec<512>;
using dev512_collapse = RAJA::fornest_collapsed_policy<dev512_nested>;
#endif

using policy_list = camp::list<seq_nested,
                               seq_collapse,
                               simd_nested,
                               simd_collapse
#if defined(RAJA_ENABLE_OPENMP)
                               ,
                               omp_nested,
                               omp_collapse
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
                               ,
                               dev256_nested,
                               dev256_collapse,
                               dev256_perfect,
                               dev256_global_sized,
                               dev256_block_thread,
                               dev256_block_thread_sized,
                               dev512_nested,
                               dev512_collapse
#endif
                               >;

/*
 * Print the list of selectable policies and their numeric indices.
 *
 * This is used for the help/usage message. The ordering here is the "UI view"
 * of `policy_list`, so it must stay consistent with:
 *  - get_policy_name(): to print the selected policy label
 *  - parse_policy_arg(): to translate stable string aliases to indices
 */
static void print_policy_menu()
{
  int idx    = 0;
  auto print = [&](const char* name) {
    std::cout << "  " << idx++ << ": " << name << "\n";
  };

  print("seq nested");
  print("seq collapsed");
  print("simd nested");
  print("simd collapsed");
#if defined(RAJA_ENABLE_OPENMP)
  print("omp nested");
  print("omp collapsed");
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
  print("device_exec<256> nested (default device mapping)");
  print("device_exec<256> collapsed");
  print("device_launch_t<false> perfectly nested (global_x, seq)");
  print("device_launch_t<false> global sized (global_size_x<32>, global_size_y<8>)");
  print("device_launch_t<false> block/thread (block_x, thread_x_loop)");
  print("device_launch_t<false> block/thread sized (block_size_x<ni>, thread_size_x_loop<32>)");
  print("device_exec<512> nested (default device mapping)");
  print("device_exec<512> collapsed");
#endif
}

/*
 * Map a policy index to a human-readable name for logging.
 *
 * Returns "<unknown>" for out-of-range indices. The ordering must match
 * `print_policy_menu()` and `policy_list`.
 */
static const char* get_policy_name(int pol)
{
  int idx    = 0;
  auto match = [&](const char* name) -> const char* {
    return (pol == idx++) ? name : nullptr;
  };

  if (auto n = match("seq nested")) return n;
  if (auto n = match("seq collapsed")) return n;
  if (auto n = match("simd nested")) return n;
  if (auto n = match("simd collapsed")) return n;
#if defined(RAJA_ENABLE_OPENMP)
  if (auto n = match("omp nested")) return n;
  if (auto n = match("omp collapsed")) return n;
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
  if (auto n = match("device_exec<256> nested (default device mapping)"))
    return n;
  if (auto n = match("device_exec<256> collapsed")) return n;
  if (auto n = match("device_launch_t<false> perfectly nested (global_x, seq)"))
    return n;
  if (auto n = match("device_launch_t<false> global sized (global_size_x<32>, global_size_y<8>)"))
    return n;
  if (auto n = match("device_launch_t<false> block/thread (block_x, thread_x_loop)"))
    return n;
  if (auto n = match("device_launch_t<false> block/thread sized (block_size_x<ni>, thread_size_x_loop<32>)"))
    return n;
  if (auto n = match("device_exec<512> nested (default device mapping)"))
    return n;
  if (auto n = match("device_exec<512> collapsed")) return n;
#endif

  return "<unknown>";
}

/*
 * Parse the policy selector passed on the command line.
 *
 * Supports:
 *  - numeric indices (e.g., "0", "3", ...)
 *  - stable string aliases (e.g., "seq", "dev256_collapse", ...)
 *
 * On success, writes the selected policy index into `pol_out` and returns true.
 */
static bool parse_policy_arg(const char* arg, int& pol_out)
{
  std::string s(arg ? arg : "");
  if (s.empty()) return false;

  // Allow numeric indices.
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
      pol_out = std::stoi(s);
      return true;
    }
  }

  // Allow stable string names (more readable than indices).
  int idx     = 0;
  auto accept = [&](const char* canonical, const char* a = nullptr,
                    const char* b = nullptr) {
    if (s == canonical || (a && s == a) || (b && s == b))
    {
      pol_out = idx;
      return true;
    }
    ++idx;
    return false;
  };

  if (accept("seq", "seq_nested", "seq-nested")) return true;
  if (accept("seq_collapse", "seq_collapsed", "seq-collapse")) return true;
  if (accept("simd", "simd_nested", "simd-nested")) return true;
  if (accept("simd_collapse", "simd_collapsed", "simd-collapse")) return true;
#if defined(RAJA_ENABLE_OPENMP)
  if (accept("omp", "omp_nested", "omp-nested")) return true;
  if (accept("omp_collapse", "omp_collapsed", "omp-collapse")) return true;
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
  if (accept("dev256", "dev256_nested", "dev256-nested")) return true;
  if (accept("dev256_collapse", "dev256_collapsed", "dev256-collapse"))
    return true;
  if (accept("dev256_perfect", "dev256_perfectly_nested", "dev256-perfect"))
    return true;
  if (accept("dev256_global_sized", "dev256_global-sized", "dev256-global-sized"))
    return true;
  if (accept("dev256_block_thread", "dev256_block-thread",
             "dev256-block-thread"))
    return true;
  if (accept("dev256_block_thread_sized", "dev256_block-thread-sized",
             "dev256-block-thread-sized"))
    return true;
  if (accept("dev512", "dev512_nested", "dev512-nested")) return true;
  if (accept("dev512_collapse", "dev512_collapsed", "dev512-collapse"))
    return true;
#endif

  return false;
}

/*
 * Validate the kernel output against the expected reference values.
 *
 * This is a lightweight correctness check for the example: each output element
 * should equal `1000*i + j` at logical index (i, j).
 */
static void check_result(const int* out, int ni, int nj)
{
  int errors = 0;
  for (int i = 0; i < ni; ++i)
  {
    for (int j = 0; j < nj; ++j)
    {
      const int idx      = j + nj * i;
      const int expected = 1000 * i + j;
      if (out[idx] != expected) ++errors;
    }
  }

  std::cout << (errors == 0 ? "\n\t result -- PASS\n"
                            : "\n\t result -- FAIL\n");
}

int main(int argc, char* argv[])
{
  constexpr int num_policies = static_cast<int>(camp::size<policy_list>::value);

  if (argc != 2)
  {
    std::cerr << "Usage: ./dynamic-fornest POLICY\n";
    std::cerr << "Where POLICY is a numeric index or a name:\n";
    print_policy_menu();
    std::cerr << "Names: seq, seq_collapse, simd, simd_collapse"
#if defined(RAJA_ENABLE_OPENMP)
              << ", omp, omp_collapse"
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
              << ", dev256, dev256_collapse, dev256_perfect, dev256_global_sized, "
                 "dev256_block_thread, dev256_block_thread_sized, dev512, dev512_collapse"
#endif
              << "\n";
    return 1;
  }

  int pol = -1;
  if (!parse_policy_arg(argv[1], pol))
  {
    std::cerr << "Unrecognized POLICY '" << argv[1] << "'. Choose one of:\n";
    print_policy_menu();
    return 1;
  }
  if (pol < 0 || pol >= num_policies)
  {
    std::cerr << "Invalid policy index " << pol << ". Choose 0.."
              << (num_policies - 1) << "\n";
    print_policy_menu();
    return 1;
  }

  std::cout << "\n\nRAJA dynamic fornest example...\n";
  std::cout << "Policy: " << pol << ": " << get_policy_name(pol) << "\n";

  const int ni                 = 262144;
  const int nj                 = 8;
  const RAJA::Index_type total = static_cast<RAJA::Index_type>(ni) * nj;

  int* out = memoryManager::allocate<int>(total);

  auto iseg = RAJA::range(ni);
  auto jseg = RAJA::range(nj);

  int count       = 0;
  using COUNT_SUM = RAJA::expt::ValOp<int, RAJA::operators::plus>;

  // `RAJA::fornest<policy_list>(pol, ...)` selects the `pol`-th policy type
  // from `policy_list` and then invokes `RAJA::fornest(selected_policy{}, ...)`.
  // This is runtime selection among a compile-time list of instantiations.
  RAJA::fornest<policy_list>(pol, iseg, jseg,
                             RAJA::expt::Reduce<RAJA::operators::plus>(&count),
                             RAJA::Name("RAJA dynamic fornest"),
                             [=] RAJA_HOST_DEVICE(int i, int j, COUNT_SUM& c) {
                               out[j + nj * i] = 1000 * i + j;
                               c += 1;
                             });

  std::cout << "Count = " << count << ", expected count: " << (ni * nj) << "\n";
  check_result(out, ni, nj);

  memoryManager::deallocate(out);

  std::cout << "\n DONE!...\n";
  return 0;
}
