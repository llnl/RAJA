//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <iostream>

#include "RAJA/RAJA.hpp"

/*
 * RAJA ranges example
 *
 * Demonstrates Python-like range helpers backed by RAJA segments:
 *   1. RAJA::range(N) for [0, N)
 *   2. RAJA::range(begin, end) for [begin, end)
 *   3. RAJA::range(begin, end, step) for a strided half-open interval
 *   4. Explicit storage types with RAJA::range<T>(...)
 *   5. Strong-index ranges using consistent strong types
 */

RAJA_INDEX_VALUE(CellIndex, "CellIndex");

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv))
{
  constexpr RAJA::Index_type N = 8;

  int values[N] = {};
  int typed_values[N] = {};
  int subrange_values[N] = {};
  int odd_values[N] = {};
  int strong_end_values[N] = {};
  int strong_stride_values[N] = {};

  // Equivalent to Python range(N): [0, N)
  RAJA::forall<RAJA::seq_exec>(RAJA::range(N), [&](RAJA::Index_type i) {
    values[i] = static_cast<int>(i * i);
  });

  // Equivalent to Python range(N), but preserving a strong index type.
  RAJA::forall<RAJA::seq_exec>(
      RAJA::range<CellIndex>(CellIndex {N}), [&](CellIndex i) {
    typed_values[*i] = static_cast<int>(*i + 10);
  });

  // Equivalent to Python range(2, 6): [2, 6)
  RAJA::forall<RAJA::seq_exec>(RAJA::range(2, 6), [&](int i) {
    subrange_values[i] = i;
  });

  // Equivalent to Python range(1, N, 2): odd indices in [1, N)
  RAJA::forall<RAJA::seq_exec>(RAJA::range(1, N, 2), [&](int i) {
    odd_values[i] = i;
  });

  // Strong-index range with consistent strong bounds.
  RAJA::forall<RAJA::seq_exec>(
      RAJA::range(CellIndex {1}, CellIndex {6}),
      [&](CellIndex i) { strong_end_values[*i] = static_cast<int>(*i * 10); });

  // Strong-index strided range with consistent strong bounds.
  RAJA::forall<RAJA::seq_exec>(
      RAJA::range(CellIndex {1}, CellIndex {N}, CellIndex {2}),
      [&](CellIndex i) {
        strong_stride_values[*i] = static_cast<int>(*i * 100);
      });

  std::cout << "range(N):";
  for (auto i : RAJA::range(N)) {
    std::cout << ' ' << values[i];
  }
  std::cout << '\n';

  std::cout << "range<CellIndex>(CellIndex{N}):";
  for (auto i : RAJA::range<CellIndex>(CellIndex {N})) {
    std::cout << ' ' << typed_values[*i];
  }
  std::cout << '\n';

  std::cout << "range(2, 6):";
  for (auto i : RAJA::range(2, 6)) {
    std::cout << ' ' << subrange_values[i];
  }
  std::cout << '\n';

  std::cout << "range(1, N, 2):";
  for (auto i : RAJA::range(1, N, 2)) {
    std::cout << ' ' << odd_values[i];
  }
  std::cout << '\n';

  std::cout << "range(CellIndex{1}, CellIndex{6}):";
  for (auto i : RAJA::range(CellIndex {1}, CellIndex {6})) {
    std::cout << ' ' << strong_end_values[*i];
  }
  std::cout << '\n';

  std::cout << "range(CellIndex{1}, CellIndex{N}, CellIndex{2}):";
  for (auto i : RAJA::range(CellIndex {1}, CellIndex {N}, CellIndex {2})) {
    std::cout << ' ' << strong_stride_values[*i];
  }
  std::cout << '\n';

  std::cout << "range(N - 1, -1, -2):";
  for (auto i : RAJA::range(N - 1, -1, -2)) {
    std::cout << ' ' << i;
  }
  std::cout << '\n';

  return 0;
}
