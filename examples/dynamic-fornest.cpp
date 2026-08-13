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
	using seq_nested = RAJA::seq_exec;
using seq_collapse = RAJA::fornest_collapsed_policy<RAJA::seq_exec>;

	using simd_nested = RAJA::simd_exec;
using simd_collapse = RAJA::fornest_collapsed_policy<RAJA::simd_exec>;

#if defined(RAJA_ENABLE_OPENMP)
	using omp_nested = RAJA::omp_parallel_for_exec;
using omp_collapse =
    RAJA::fornest_collapsed_policy<RAJA::omp_parallel_for_exec>;
#endif

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
	using dev256_nested = RAJA::device_exec<256>;
using dev256_collapse = RAJA::fornest_collapsed_policy<dev256_nested>;

// Perfectly nested loop structure: outer is global-parallel, inner is
// sequential.
using dev256_perfect = RAJA::fornest_mapping_policy<
    dev256_nested,
    RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_global_x_direct>,
    RAJA::LoopPolicy<RAJA::seq_exec, RAJA::seq_exec>>;

// Block/thread mapping: outer maps to blocks, inner maps to threads (loop).
using dev256_block_thread = RAJA::fornest_mapping_policy<
    dev256_nested,
    RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_block_x_direct>,
    RAJA::LoopPolicy<RAJA::seq_exec, RAJA::device_thread_x_loop>>;

	using dev512_nested = RAJA::device_exec<512>;
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
	                               dev256_block_thread,
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
	print("device_exec<256> perfectly nested (global_x, seq)");
	print("device_exec<256> block/thread (block_x, thread_x_loop)");
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
	if (auto n = match("device_exec<256> perfectly nested (global_x, seq)"))
		return n;
	if (auto n = match("device_exec<256> block/thread (block_x, thread_x_loop)"))
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
	if (accept("dev256_block_thread", "dev256_block-thread",
	           "dev256-block-thread"))
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
	          << ", dev256, dev256_collapse, dev256_perfect, dev256_block_thread, "
	             "dev512, dev512_collapse"
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
