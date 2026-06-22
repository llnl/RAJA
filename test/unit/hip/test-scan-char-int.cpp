//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <vector>

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#include "RAJA_gtest.hpp"

namespace
{

constexpr int scan_len = 300;

template <typename ExecPolicy>
void run_scan(const char* input, int* output, int len)
{
  RAJA::exclusive_scan<ExecPolicy>(RAJA::make_span(input, len),
                                   RAJA::make_span(output, len),
                                   RAJA::operators::plus<int> {});
}

void check_result(const std::vector<int>& result)
{
  for (int i = 0; i < scan_len; ++i) {
    ASSERT_EQ(result[i], i) << "Mismatch at index " << i;
  }
}

}  // namespace

GPU_TEST(ScanUnitTest, HIPCharToIntExclusive)
{
  std::vector<char> input(scan_len, 1);

  std::vector<int> seq_output(scan_len, -1);
  run_scan<RAJA::seq_exec>(input.data(), seq_output.data(), scan_len);
  check_result(seq_output);

  char* d_input = nullptr;
  int* d_output = nullptr;

  CAMP_HIP_API_INVOKE_AND_CHECK(hipSetDevice, 0);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMalloc, &d_input, scan_len * sizeof(char));
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMalloc, &d_output, scan_len * sizeof(int));
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_input, input.data(),
                                scan_len * sizeof(char),
                                hipMemcpyHostToDevice);

  run_scan<RAJA::hip_exec<256>>(d_input, d_output, scan_len);
  RAJA::synchronize<RAJA::hip_synchronize>();

  std::vector<int> hip_output(scan_len, -1);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, hip_output.data(), d_output,
                                scan_len * sizeof(int),
                                hipMemcpyDeviceToHost);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, d_input);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, d_output);

  check_result(hip_output);
}
