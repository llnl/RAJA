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
 * ZeroTo helper example
 *
 * Demonstrates two convenience forms for building the half-open interval
 * [0, N):
 *   1. RAJA::ZeroTo(N) for the default RAJA::Index_type-based RangeSegment
 *   2. RAJA::ZeroTo<T>(N) for a TypedRangeSegment<T>
 */

RAJA_INDEX_VALUE(CellIndex, "CellIndex");

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv))
{
  constexpr RAJA::Index_type N = 8;

  int values[N] = {};
  int* values_ptr = values;
  int typed_values[N] = {};
  int* typed_values_ptr = typed_values;

  // Build a RangeSegment over [0, N) using the default RAJA::Index_type.
  RAJA::forall<RAJA::seq_exec>(RAJA::ZeroTo(N), [=](RAJA::Index_type i) {
    values_ptr[i] = static_cast<int>(i * i);
  });

  // Build a TypedRangeSegment<CellIndex> over the same [0, N) interval.
  RAJA::forall<RAJA::seq_exec>(RAJA::ZeroTo<CellIndex>(N), [=](CellIndex i) {
    typed_values_ptr[*i] = static_cast<int>(*i + 10);
  });

  std::cout << "values:";
  for (auto i : RAJA::ZeroTo(N)) {
    std::cout << ' ' << values[i];
  }
  std::cout << '\n';

  std::cout << "typed values:";
  for (auto i : RAJA::ZeroTo<CellIndex>(N)) {
    std::cout << ' ' << typed_values[*i];
  }
  std::cout << '\n';

  return 0;
}
