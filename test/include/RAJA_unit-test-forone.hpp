//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Header defining "for one" unit test utility so that constructs can be
// tested outside of standard RAJA kernel launch utilities (forall, kernel).
//

#ifndef __RAJA_test_forone_HPP__
#define __RAJA_test_forone_HPP__

#include "RAJA_unit-test-policy.hpp"

///
/// forone<test_policy>( [=] RAJA_HOST_DEVICE(){ /* code to test */ } );
///
template < typename test_policy, typename L >
inline void forone(L&& run);

// test_seq implementation
template < typename L >
inline void forone(test_seq, L&& run)
{
  std::forward<L>(run)();
}

#if defined(RAJA_ENABLE_OPENMP)

// test_openmp implementation
template < typename L >
inline void forone(test_openmp, L&& run)
{
  using BODY = camp::decay<L>;
  BODY input_body(std::forward<L>(run));

#pragma omp parallel num_threads(1)
  {
    auto body = input_body;
#pragma omp for
    for (int i = 0; i < 1; ++i) {
      RAJA_UNUSED_VAR(i);
      body();
    }
  }
}

#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)

// test_openmp_target implementation
template < typename L >
inline void forone(test_openmp_target, L&& run)
{
  using BODY = camp::decay<L>;
  BODY input_body(std::forward<L>(run));

#pragma omp target parallel for \
        map(to : input_body)
  for (int i = 0; i < 1; ++i) {
    RAJA_UNUSED_VAR(i);
    auto body = input_body;
    body();
  }
}

#endif

#if defined(RAJA_ENABLE_SYCL)

// test_sycl implementation
template < typename L >
inline void forone(test_sycl, L&& run)
{
  using BODY = camp::decay<L>;

  auto sycl_res = RAJA::resources::Sycl::get_default();
  ::sycl::queue* q = sycl_res.get_queue();
  BODY input_body(std::forward<L>(run));

  BODY* device_body = nullptr;
  if constexpr (!std::is_trivially_copyable_v<BODY>) {
    device_body = static_cast<BODY*>(::sycl::malloc_device(sizeof(BODY), *q));
    q->memcpy(device_body, &input_body, sizeof(BODY)).wait();
  }

  ::sycl::range<1> grid(1);
  ::sycl::range<1> block(1);
  q->submit([&](::sycl::handler& h) {
    if constexpr (std::is_trivially_copyable_v<BODY>) {
      h.parallel_for(::sycl::nd_range<1>{grid, block}, [=](::sycl::nd_item<1> item) {
        auto body = input_body;
        if (item.get_global_id(0) == 0) {
          body();
        }
      });
    } else {
      h.parallel_for(::sycl::nd_range<1>{grid, block}, [=](::sycl::nd_item<1> item) {
        auto body = *device_body;
        if (item.get_global_id(0) == 0) {
          body();
        }
      });
    }
  });
  q->wait();

  if constexpr (!std::is_trivially_copyable_v<BODY>) {
    ::sycl::free(device_body, *q);
  }
}

#endif

#if defined(RAJA_ENABLE_CUDA)

template <typename L>
__global__ void forone_cuda_global(L run)
{
  run();
}

// test_cuda implementation
template < typename L >
inline void forone(test_cuda, L&& run)
{
   forone_cuda_global<<<1,1>>>(std::forward<L>(run));
   CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
   CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
}

#endif

#if defined(RAJA_ENABLE_HIP)

template <typename L>
__global__ void forone_hip_global(L run)
{
  run();
}

// test_hip implementation
template < typename L >
inline void forone(test_hip, L&& run)
{
   hipLaunchKernelGGL(forone_hip_global<camp::decay<L>>, dim3(1), dim3(1), 0, 0, std::forward<L>(run));
   CAMP_HIP_API_INVOKE_AND_CHECK(hipGetLastError);
   CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);
}

#endif

template < typename test_policy, typename L >
void forone(L&& run)
{
  forone(test_policy{}, std::forward<L>(run));
}

#endif // RAJA_test_forone_HPP__
