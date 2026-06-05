//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>


using namespace RAJA;
using namespace RAJA::statement;

TEST(SIMD, Align)
{

  int N = 1024;
  double c = 0.5;
  double *a =
      RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN, N * sizeof(double));
  double *b =
      RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN, N * sizeof(double));

  for (int i = 0; i < N; ++i) {
    a[i] = 0;
    b[i] = 2.0;
  }


  double *y = RAJA::align_hint(a);
  double *x = RAJA::align_hint(b);

  RAJA::forall<RAJA::simd_exec>(RAJA::RangeSegment(0, N),
                                [=](int i) { y[i] += x[i] * c; });

  for (int i = 0; i < N; ++i) {
    ASSERT_DOUBLE_EQ((double)y[i], (double)1.0);
  }

  RAJA::free_aligned(a);
  RAJA::free_aligned(b);
}

#if defined(RAJA_ENABLE_OPENMP)
TEST(SIMD, OMPAndSimd)
{

  using POL = RAJA::KernelPolicy<RAJA::statement::For<
      1,
      RAJA::omp_parallel_for_exec,
      RAJA::statement::For<0, RAJA::simd_exec, RAJA::statement::Lambda<0> > > >;

  const RAJA::Index_type N = 32;
  const RAJA::Index_type M = 32;

  double *a = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                  N * M * sizeof(double));
  double *b = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                  N * M * sizeof(double));
  double *c = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                  N * M * sizeof(double));

  for (int i = 0; i < N * M; ++i) {
    a[i] = 1;
    b[i] = 1;
    c[i] = 0.0;
  }

  RAJA::kernel<POL>(RAJA::make_tuple(RAJA::RangeSegment(0, N),
                                     RAJA::RangeSegment(0, M)),
                    [=](RAJA::Index_type i, RAJA::Index_type j) {
                      c[i + j * N] = a[i + j * N] + b[i + j * N];
                    });

  for (int i = 0; i < N * M; ++i) {
    ASSERT_DOUBLE_EQ((double)c[i], (double)2.0);
  }

  RAJA::free_aligned(a);
  RAJA::free_aligned(b);
  RAJA::free_aligned(c);
}

TEST(SIMD, OMPAndSimd_MultiLambda)
{

  using POL = RAJA::KernelPolicy<RAJA::statement::For<
      1,
      RAJA::omp_parallel_for_exec,
      RAJA::statement::For<0,
                           RAJA::simd_exec,
                           RAJA::statement::Lambda<0>,
                           RAJA::statement::Lambda<1> > > >;

  const RAJA::Index_type N = 32;
  const RAJA::Index_type M = 32;

  double *a = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                  N * M * sizeof(double));
  double *b = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                  N * M * sizeof(double));
  double *c = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                  N * M * sizeof(double));

  double *a2 = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                   N * M * sizeof(double));
  double *b2 = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                   N * M * sizeof(double));
  double *c2 = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                   N * M * sizeof(double));

  for (int i = 0; i < N * M; ++i) {
    a[i] = 1;
    b[i] = 1;
    c[i] = 0.0;
    a2[i] = 1;
    b2[i] = 1;
    c2[i] = 0.0;
  }

  RAJA::kernel<POL>(RAJA::make_tuple(RAJA::RangeSegment(0, N),
                                     RAJA::RangeSegment(0, M)),
                    [=](RAJA::Index_type i, RAJA::Index_type j) {
                      c[i + j * N] = a[i + j * N] + b[i + j * N];
                    },
                    [=](RAJA::Index_type i, RAJA::Index_type j) {
                      c2[i + j * N] = a2[i + j * N] + b2[i + j * N];
                    });

  for (int i = 0; i < N * M; ++i) {
    ASSERT_DOUBLE_EQ((double)c[i], (double)2.0);
    ASSERT_DOUBLE_EQ((double)c2[i], (double)2.0);
  }

  RAJA::free_aligned(a);
  RAJA::free_aligned(b);
  RAJA::free_aligned(c);

  RAJA::free_aligned(a2);
  RAJA::free_aligned(b2);
  RAJA::free_aligned(c2);
}
#endif
