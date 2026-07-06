//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing scan test infrastructure
///

#ifndef __TEST_UNIT_ALGORITHM_SCAN_HPP__
#define __TEST_UNIT_ALGORITHM_SCAN_HPP__

#include "RAJA_test-camp.hpp"

#include <algorithm>

constexpr int scan_len = 300;

template<typename ExecPolicy, typename Res>
void runScanCharToIntScanTest(Res res, bool inclusive)
{
  camp::resources::Host host_res = camp::resources::Host::get_default();

  char* work_in = res.template allocate<char>(scan_len);
  int* work_out = res.template allocate<int>(scan_len);
  char* host_in = host_res.template allocate<char>(scan_len);
  int* host_out = host_res.template allocate<int>(scan_len);

  std::fill_n(host_in, scan_len, char{1});
  std::fill_n(host_out, scan_len, -1);

  res.memcpy(work_in, host_in, scan_len * sizeof(char));
  res.wait();

  if (inclusive) {
    RAJA::inclusive_scan<ExecPolicy>(RAJA::make_span(
                                         static_cast<const char*>(work_in),
                                         scan_len),
                                     RAJA::make_span(work_out, scan_len),
                                     RAJA::operators::plus<int> {});
  } else {
    RAJA::exclusive_scan<ExecPolicy>(RAJA::make_span(
                                         static_cast<const char*>(work_in),
                                         scan_len),
                                     RAJA::make_span(work_out, scan_len),
                                     RAJA::operators::plus<int> {});
  }
  res.wait();

  res.memcpy(host_out, work_out, scan_len * sizeof(int));
  res.wait();

  for (int i = 0; i < scan_len; ++i) {
    ASSERT_EQ(host_out[i], inclusive ? (i + 1) : i)
        << "Mismatch in default-resource "
        << (inclusive ? "inclusive" : "exclusive") << " scan at index " << i;
  }

  std::fill_n(host_out, scan_len, -1);

  if (inclusive) {
    RAJA::inclusive_scan<ExecPolicy>(res,
                                     RAJA::make_span(
                                         static_cast<const char*>(work_in),
                                         scan_len),
                                     RAJA::make_span(work_out, scan_len),
                                     RAJA::operators::plus<int> {});
  } else {
    RAJA::exclusive_scan<ExecPolicy>(res,
                                     RAJA::make_span(
                                         static_cast<const char*>(work_in),
                                         scan_len),
                                     RAJA::make_span(work_out, scan_len),
                                     RAJA::operators::plus<int> {});
  }
  res.wait();

  res.memcpy(host_out, work_out, scan_len * sizeof(int));
  res.wait();

  for (int i = 0; i < scan_len; ++i) {
    ASSERT_EQ(host_out[i], inclusive ? (i + 1) : i)
        << "Mismatch in explicit-resource "
        << (inclusive ? "inclusive" : "exclusive") << " scan at index " << i;
  }

  res.deallocate(work_in);
  res.deallocate(work_out);
  host_res.deallocate(host_in);
  host_res.deallocate(host_out);
}

template<typename ExecPolicy, typename Res>
void runScanCharToIntExclusiveTest(Res res)
{
  runScanCharToIntScanTest<ExecPolicy>(res, false);
}

template<typename ExecPolicy, typename Res>
void runScanCharToIntInclusiveTest(Res res)
{
  runScanCharToIntScanTest<ExecPolicy>(res, true);
}

TYPED_TEST_SUITE_P(ScanUnitTest);

template<typename T>
class ScanUnitTest : public ::testing::Test
{
};

TYPED_TEST_P(ScanUnitTest, CharToIntExclusive)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<1>>::type;

  ResType res = ResType::get_default();
  runScanCharToIntExclusiveTest<ExecPolicy>(res);
}

TYPED_TEST_P(ScanUnitTest, CharToIntInclusive)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<1>>::type;

  ResType res = ResType::get_default();
  runScanCharToIntInclusiveTest<ExecPolicy>(res);
}

REGISTER_TYPED_TEST_SUITE_P(ScanUnitTest,
                            CharToIntExclusive,
                            CharToIntInclusive);

using SequentialScanExecPolicies = camp::list<RAJA::seq_exec>;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPScanExecPolicies = camp::list<RAJA::omp_parallel_for_exec>;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaScanExecPolicies =
  camp::list<RAJA::cuda_exec<128>, RAJA::cuda_exec_explicit<128, 2>>;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipScanExecPolicies = camp::list<RAJA::hip_exec<128>>;
#endif

#endif  // __TEST_UNIT_ALGORITHM_SCAN_HPP__
