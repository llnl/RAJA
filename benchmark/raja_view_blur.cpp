//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <RAJA/RAJA.hpp>
#include "RAJA/util/Timer.hpp"
#include <iostream>

/*
 * RAJA view performance test
 * Kernel performs a 2D Gaussian blur
 *
 */

//Uncomment to specify variant
//#define RUN_HIP_VARIANT
//#define RUN_CUDA_VARIANT
//#define RUN_SYCL_VARIANT
//#define RUN_OPENMP_VARIANT
#define RUN_SEQ_VARIANT


using host_pol = RAJA::seq_exec;
using host_resources = RAJA::resources::Host;


#if defined(RAJA_ENABLE_HIP) && defined(RUN_HIP_VARIANT)
using device_pol = RAJA::hip_exec<256>;
using device_resources = RAJA::resource::Hip;

using kernel_pol = RAJA::KernelPolicy<
  RAJA::statement::HipKernelFixed<256,
    RAJA::statement::For<1, RAJA::hip_global_size_y_direct<16>,
      RAJA::statement::For<0, RAJA::hip_global_size_x_direct<16>,
         RAJA::statement::Lambda<0>
      >
    >
   >
  >;
#endif

#if defined(RAJA_ENABLE_CUDA) && defined(RUN_CUDA_VARIANT)
using device_pol = RAJA::cuda_exec<256>;
using device_resources = RAJA::resources::Cuda;

using kernel_pol = RAJA::KernelPolicy<
  RAJA::statement::CudaKernelFixed<256,
    RAJA::statement::For<1, RAJA::cuda_global_size_y_direct<16>,
      RAJA::statement::For<0, RAJA::cuda_global_size_x_direct<16>,
         RAJA::statement::Lambda<0>
      >
    >
   >
  >;
#endif

#if defined(RAJA_ENABLE_SYCL) && defined(RUN_SYCL_VARIANT)
using device_pol = RAJA::sycl_exec<256>;
using device_resources = RAJA::resources::Sycl;

using kernel_pol = RAJA::KernelPolicy<
  RAJA::statement::SyclKernel<
    RAJA::statement::For<1, RAJA::sycl_global_item_1,
      RAJA::statement::For<0, RAJA::sycl_global_item_2,
         RAJA::statement::Lambda<0>
      >
    >
   >
  >;
#endif

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP_VARIANT)
using device_pol = RAJA::omp_parallel_for_exec;
using device_resources = RAJA::resources::Host;

using kernel_pol = RAJA::KernelPolicy<
    RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
      RAJA::statement::For<0, RAJA::seq_exec,
         RAJA::statement::Lambda<0>
      >
    >
  >;
#endif

#if defined(RUN_SEQ_VARIANT)
using device_pol = RAJA::seq_exec;
using device_resources =  RAJA::resources::Host;

using kernel_pol = RAJA::KernelPolicy<
    RAJA::statement::For<1, RAJA::seq_exec,
      RAJA::statement::For<0, RAJA::seq_exec,
         RAJA::statement::Lambda<0>
      >
    >
  >;
#endif

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  const int N = 10000;
  const int K = 17;

  device_resources def_device_res{device_resources::get_default()};
  host_resources   def_host_res{host_resources::get_default()};

  auto timer = RAJA::Timer();

  //launch to intialize the stream
  RAJA::forall<device_pol>
    (RAJA::RangeSegment(0,1), [=] RAJA_HOST_DEVICE (int RAJA_UNUSED_ARG(i)) {
  });

  int * array      = def_host_res.allocate<int>(N * N);
  int * array_copy = def_host_res.allocate<int>(N * N);

  //big array, or image
  for (int i = 0; i < N * N; ++i) {
    array[i] = 1;
    array_copy[i] = 1;
  }

  //small array that acts as the blur
  int * kernel  = def_host_res.allocate<int>(K * K);
  for (int i = 0; i < K * K; ++i) {
    kernel[i] = 2;
  }

  // copying to gpu
  int* d_array      = def_device_res.allocate<int>(N * N);
  int* d_array_copy = def_device_res.allocate<int>(N * N);
  int* d_kernel     = def_device_res.allocate<int>(K * K);

  def_device_res.memcpy(d_array, array, N * N * sizeof(int));
  def_device_res.memcpy(d_array_copy, array_copy, N * N * sizeof(int));
  def_device_res.memcpy(d_kernel, kernel, K * K * sizeof(int));

  constexpr int DIM = 2;
  RAJA::View<int, RAJA::Layout<DIM, int, 1>> array_view(d_array, N, N);
  RAJA::View<int, RAJA::Layout<DIM, int, 1>> array_view_copy(d_array_copy, N, N);
  RAJA::View<int, RAJA::Layout<DIM, int, 1>> kernel_view(d_kernel, K, K);

  RAJA::RangeSegment range_i(0, N);
  RAJA::RangeSegment range_j(0, N);

  timer.start();

  RAJA::kernel<kernel_pol>
    (RAJA::make_tuple(range_i, range_j),
     [=] RAJA_HOST_DEVICE (int i, int j) {
      int sum = 0;

      //looping through the "blur"
      for (int m = 0; m < K; ++m) {
	for (int n = 0; n < K; ++n) {
	  int x = i + m;
	  int y = j + n;

	  // adding the "blur" to the "image" wherever the blur is located on the image
	  if (x < N && y < N) {
	    sum += kernel_view(m, n) * array_view(x, y);
	  }
	}
      }

      array_view(i, j) += sum;
    }
   );

  timer.stop();

  std::cout<<"Elapsed time with RAJA view : "<<timer.elapsed()<<std::endl;


 timer.reset();
 timer.start();

  RAJA::kernel<kernel_pol>
    (RAJA::make_tuple(range_i, range_j),
     [=] RAJA_HOST_DEVICE (int i, int j) {
      int sum = 0;

      // looping through the "blur"
      for (int m = 0; m < K; ++m) {
	for (int n = 0; n < K; ++n) {
	  int x = i + m;
	  int y = j + n;

	  // adding the "blur" to the "image" wherever the blur is located on the image
	  if (x < N && y < N) {
	    sum += d_kernel[m * K + n] * d_array_copy[x * N + y];
	  }
	}
      }

      d_array_copy[i * N + j] += sum;
    }
  );
  timer.stop();

  std::cout<<"Elapsed time with NO RAJA view : "<<timer.elapsed()<<std::endl;

  def_device_res.memcpy(array, d_array, N * N * sizeof(int));
  def_device_res.memcpy(array_copy, d_array_copy, N * N * sizeof(int));

  def_device_res.deallocate(d_array);
  def_device_res.deallocate(d_array_copy);
  def_device_res.deallocate(d_kernel);

  def_host_res.deallocate(array);
  def_host_res.deallocate(array_copy);
  def_host_res.deallocate(kernel);

  return 0;
}
