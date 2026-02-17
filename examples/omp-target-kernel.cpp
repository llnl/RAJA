//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"

using namespace RAJA;
using namespace RAJA::statement;

int main(int /*argc*/, char** /*argv[]*/) {

  // using Pol = KernelPolicy<
  //               For<1, RAJA::seq_exec>,
  //               For<0, RAJA::omp_target_parallel_for_exec<1>, Lambda<0> >
  //             >;
  using Pol = KernelPolicy<
    Collapse<omp_target_parallel_collapse_exec, ArgList<0,1>, Lambda<0> > >;

  double* array = new double[25*25];

#pragma omp target enter data map(to: array[0:25*25])
#pragma omp target data use_device_ptr(array)

#if 1
  RAJA::kernel<Pol>(
      RAJA::make_tuple(
        RAJA::RangeSegment(0,25),
        RAJA::RangeSegment(0,25)),
      [=] (int /*i*/, int /*j*/) {
      //array[i + (25*j)] = i*j;
  //    int idx = i;
      //array[0] = i*j;
  });
#else
  RAJA::forall<RAJA::omp_target_parallel_for_exec<1>>(
      RAJA::RangeSegment(0,25),
      [=] (int i) {
      //
  });
#endif
}
