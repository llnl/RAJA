//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//
// This test No-Ops in all cases except for when using cuda execution policies.

#ifndef __TEST_RESOURCE_ASYNC_HPP__
#define __TEST_RESOURCE_ASYNC_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA/util/Timer.hpp"

#if defined(RAJA_ENABLE_CUDA)
inline __host__ __device__ void
gpu_time_wait_for(float time, float clockrate) {
  clock_t time_in_clocks = time*clockrate;

  unsigned int start_clock = (unsigned int) clock();
  clock_t clock_offset = 0;
  while (clock_offset < time_in_clocks)
  {
    unsigned int end_clock = (unsigned int) clock();
    clock_offset = (clock_t)(end_clock - start_clock);
  }
}

int get_clockrate()
{
  int cuda_device = 0;
  cudaDeviceProp deviceProp;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetDevice, &cuda_device);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetDeviceProperties, &deviceProp, cuda_device);
  if ((deviceProp.concurrentKernels == 0))
  {
    printf("> GPU does not support concurrent kernel execution\n");
    printf("  CUDA kernel runs will be serialized\n");
    return -1;
  }
  //printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
  //    deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

#if defined(__arm__) || defined(__aarch64__)
  return deviceProp.clockRate/1000;
#else
  return deviceProp.clockRate;
#endif
}

template <typename WORKING_RES, typename EXEC_POL>
void ResourceAsyncTimeTestImpl(EXEC_POL&&) {}

template <typename WORKING_RES, size_t BLOCK_SIZE, bool Async>
void ResourceAsyncTimeTestImpl(RAJA::cuda_exec<BLOCK_SIZE, Async>&&)
{
  constexpr std::size_t ARRAY_SIZE{10000};
  using namespace RAJA;

  constexpr std::size_t NUM_STREAMS{8};
  WORKING_RES dev[NUM_STREAMS];
  resources::Host host;

  int clockrate{get_clockrate()};
  ASSERT_TRUE(clockrate != -1);

  using AsyncExecPol = RAJA::cuda_exec<BLOCK_SIZE, true>;
  using SyncExecPol = RAJA::cuda_exec<BLOCK_SIZE, false>;

  RAJA::Timer sync_timer;
  sync_timer.start();
  for (std::size_t stream = 0; stream < NUM_STREAMS; ++stream){
    forall<SyncExecPol>(dev[stream], RangeSegment(0,ARRAY_SIZE),
      [=] RAJA_HOST_DEVICE (int i) {
        gpu_time_wait_for(100, clockrate);
      }
    );
  }
  sync_timer.stop();
  RAJA::Timer::ElapsedType t_sync = sync_timer.elapsed();

  RAJA::Timer async_timer;
  async_timer.start();
  for (std::size_t stream = 0; stream < NUM_STREAMS; ++stream){
    forall<AsyncExecPol>(dev[stream], RangeSegment(0,ARRAY_SIZE),
      [=] RAJA_HOST_DEVICE (int i) {
        gpu_time_wait_for(100, clockrate);
      }
    );
  }
  async_timer.stop();
  RAJA::Timer::ElapsedType t_async = async_timer.elapsed();

  // We expect "total async time" to be roughly equal to "total sync time" / NUM_STREAMS.
  // For comparison tolerance, we multiple the latter by 2 in the check.
  ASSERT_LT(t_async, 2 * (t_sync / NUM_STREAMS));
}

template <typename WORKING_RES, typename EXEC_POLICY>
void ResourceAsyncTimeTestCall()
{
  ResourceAsyncTimeTestImpl<WORKING_RES>(EXEC_POLICY());
}

#else

template <typename WORKING_RES, typename EXEC_POLICY>
void ResourceAsyncTimeTestCall() {}

#endif

TYPED_TEST_SUITE_P(ResourceAsyncTimeTest);
template <typename T>
class ResourceAsyncTimeTest : public ::testing::Test
{
};

TYPED_TEST_P(ResourceAsyncTimeTest, ResourceAsyncTime)
{
  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<1>>::type;

  ResourceAsyncTimeTestCall<WORKING_RES, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ResourceAsyncTimeTest,
                            ResourceAsyncTime);

#endif  // __TEST_RESOURCE_ASYNC_HPP__
