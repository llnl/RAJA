//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <iostream>

#include "benchmark/benchmark.h"

#include "RAJA/RAJA.hpp"

#define N 10000000

static void benchmark_daxpy_raw(benchmark::State& state)
{
  double* a = new double[N];
  double* b = new double[N];

  for (int i = 0; i < N; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
  }
  double c = 3.14159;

  while (state.KeepRunning()) {
    RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N),
                                 [=](int i) { a[i] += b[i] * c; });
  }
}

static void benchmark_daxpy_host(benchmark::State& state)
{
  double* a = new double[N];
  double* b = new double[N];

  for (int i = 0; i < N; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
  }
  double c = 3.14159;

  while (state.KeepRunning()) {
    RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N),
                                 [=] __host__(int i) { a[i] += b[i] * c; });
  }
}

static void benchmark_daxpy_host_device(benchmark::State& state)
{
  double* a = new double[N];
  double* b = new double[N];

  for (int i = 0; i < N; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
  }
  double c = 3.14159;

  while (state.KeepRunning()) {
    RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N),
                                 [=] RAJA_HOST_DEVICE(int i) {
                                   a[i] += b[i] * c;
                                 });
  }
}

BENCHMARK(benchmark_daxpy_raw);
BENCHMARK(benchmark_daxpy_host);
BENCHMARK(benchmark_daxpy_host_device);

BENCHMARK_MAIN();
