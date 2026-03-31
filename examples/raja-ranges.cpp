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
 *   1. RAJA::Range(N) for [0, N)
 *   2. RAJA::Range<T>(N) for a TypedRangeSegment<T> over [0, N)
 *   3. RAJA::Range(begin, end) for [begin, end)
 *   4. RAJA::Range(begin, end, step) for a strided half-open interval
 */

RAJA_INDEX_VALUE(CellIndex, "CellIndex");

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv))
{
  constexpr RAJA::Index_type N = 8;

  int values[N] = {};
  int typed_values[N] = {};
  int subrange_values[N] = {};
  int odd_values[N] = {};

  // Equivalent to Python range(N): [0, N)
  RAJA::forall<RAJA::seq_exec>(RAJA::Range(N), [&](RAJA::Index_type i) {
    values[i] = static_cast<int>(i * i);
  });

  // Equivalent to Python range(N), but preserving a strong index type.
  RAJA::forall<RAJA::seq_exec>(RAJA::Range<CellIndex>(N), [&](CellIndex i) {
    typed_values[*i] = static_cast<int>(*i + 10);
  });

  // Equivalent to Python range(2, 6): [2, 6)
  RAJA::forall<RAJA::seq_exec>(RAJA::Range(2, 6), [&](int i) {
    subrange_values[i] = i;
  });

  // Equivalent to Python range(1, N, 2): odd indices in [1, N)
  RAJA::forall<RAJA::seq_exec>(RAJA::Range(1, N, 2), [&](int i) {
    odd_values[i] = i;
  });

  std::cout << "Range(N):";
  for (auto i : RAJA::Range(N)) {
    std::cout << ' ' << values[i];
  }
  std::cout << '\n';

  std::cout << "Range<CellIndex>(N):";
  for (auto i : RAJA::Range<CellIndex>(N)) {
    std::cout << ' ' << typed_values[*i];
  }
  std::cout << '\n';

  std::cout << "Range(2, 6):";
  for (auto i : RAJA::Range(2, 6)) {
    std::cout << ' ' << subrange_values[i];
  }
  std::cout << '\n';

  std::cout << "Range(1, N, 2):";
  for (auto i : RAJA::Range(1, N, 2)) {
    std::cout << ' ' << odd_values[i];
  }
  std::cout << '\n';

  std::cout << "Range(N - 1, -1, -2):";
  for (auto i : RAJA::Range(N - 1, -1, -2)) {
    std::cout << ' ' << i;
  }
  std::cout << '\n';

  return 0;
}
