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

using namespace RAJA;

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running RAJA Resource Kernel on Multiple Streams...\n";

  constexpr int N = 10;
  constexpr int M = 1000000;

  RAJA::resources::Cuda def_cuda_res{RAJA::resources::Cuda::get_default()};
  RAJA::resources::Host def_host_res{RAJA::resources::Host::get_default()};
  int* d_array = def_cuda_res.allocate<int>(N*M);
  int* h_array = def_host_res.allocate<int>(N*M);

  RAJA::RangeSegment one_range(0, 1);
  RAJA::RangeSegment m_range(0, M);
  RAJA::RangeSegment n_range(0, N);

  using TEST_POL =
    RAJA::KernelPolicy<
      statement::CudaKernelAsync<
        statement::For<0, cuda_block_x_loop,
          statement::For<1, cuda_thread_x_loop,
            statement::Lambda<0>
          >
        >
      >
    >;

  RAJA::forall<RAJA::seq_exec>(def_host_res, n_range,
    [=, &def_cuda_res](int i){
      RAJA::resources::Cuda res_cuda; 

      RAJA::resources::Event e = RAJA::kernel_resource<TEST_POL>(
        RAJA::make_tuple(one_range,
                         m_range),

        res_cuda,

        [=] RAJA_DEVICE (int k, int j) {
          d_array[i*M + j] = i * M + j;  
        }
      );

      def_cuda_res.wait_for(&e);
    }
  );

  def_cuda_res.memcpy(h_array, d_array, sizeof(int) * N * M);

  int ec_count = 0;
  RAJA::forall<RAJA::seq_exec>( RAJA::RangeSegment(0, N*M),
    [=, &ec_count](int i){
      if (h_array[i] != i) ec_count++;
    }
  );

  std::cout << "    Result -- ";
  if (ec_count > 0) 
    std::cout << "FAIL : error count = " << ec_count << "\n";
  else
    std::cout << "PASS!\n";

#else
  std::cout << "Please build with CUDA to run this example ...\n";
#endif
  std::cout << "\n DONE!...\n";

  return 0;
}
