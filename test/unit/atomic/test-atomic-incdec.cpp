//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for "wrapping" increment and decrement functions
///

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA_unit-test-forone.hpp"
#endif

using unsigned_types =
    ::testing::Types<
                      std::tuple<unsigned int, RAJA::builtin_atomic>,
                      std::tuple<unsigned int, RAJA::seq_atomic>,
                      std::tuple<unsigned long long int, RAJA::builtin_atomic>,
                      std::tuple<unsigned long long int, RAJA::seq_atomic>
#if defined(RAJA_ENABLE_OPENMP)
                      ,
                      std::tuple<unsigned int, RAJA::omp_atomic>,
                      std::tuple<unsigned long long int, RAJA::omp_atomic>
#endif
#if defined(RAJA_ENABLE_CUDA)
                      ,
                      std::tuple<unsigned int, RAJA::auto_atomic>,
                      std::tuple<unsigned int, RAJA::cuda_atomic>,
                      std::tuple<unsigned long long int, RAJA::auto_atomic>,
                      std::tuple<unsigned long long int, RAJA::cuda_atomic>
#endif
#if defined(RAJA_ENABLE_HIP)
                      ,
                      std::tuple<unsigned int, RAJA::auto_atomic>,
                      std::tuple<unsigned int, RAJA::hip_atomic>,
                      std::tuple<unsigned long long int, RAJA::auto_atomic>,
                      std::tuple<unsigned long long int, RAJA::hip_atomic>
#endif
                    >;

// Basic Inc Dec

template <typename T>
class AtomicBasicIncDecUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( AtomicBasicIncDecUnitTest );

TYPED_TEST_P( AtomicBasicIncDecUnitTest, BasicIncDecs )
{
  using T = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  // test "wrapping" increment
  // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc

  T inc_init = (T)0;
  T* inc_result = &inc_init;

  // oldval < val, increment oldval
  RAJA::atomicInc<AtomicPolicy>(inc_result, (T)1);
  ASSERT_EQ( inc_result[0], (T)1 );

  // oldval == val, wrap to 0
  RAJA::atomicInc<AtomicPolicy>(inc_result, (T)1);
  ASSERT_EQ( inc_result[0], (T)0 );

  // oldval > val, wrap to 0
  inc_result[0] = (T)2;
  RAJA::atomicInc<AtomicPolicy>(inc_result, (T)1);
  ASSERT_EQ( inc_result[0], (T)0 );

  // test "wrapping" decrement
  // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicdec

  T dec_init = (T)1;
  T* dec_result = &dec_init;

  // oldval > 0, decrement oldval
  RAJA::atomicDec<AtomicPolicy>(dec_result, (T)1);
  ASSERT_EQ( dec_result[0], (T)0 );

  // oldval == 0, wrap to val
  RAJA::atomicDec<AtomicPolicy>(dec_result, (T)1);
  ASSERT_EQ( dec_result[0], (T)1 );

  // oldval > val, wrap to val
  dec_result[0] = (T)3;
  RAJA::atomicDec<AtomicPolicy>(dec_result, (T)1);
  ASSERT_EQ( dec_result[0], (T)1 );
}

REGISTER_TYPED_TEST_SUITE_P( AtomicBasicIncDecUnitTest,
                             BasicIncDecs
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( BasicIncDecUnitTest,
                                AtomicBasicIncDecUnitTest,
                                unsigned_types
                              );


// Pure CUDA test.
#if defined(RAJA_ENABLE_CUDA)

using CUDA_unsigned_types =
    ::testing::Types<
                      std::tuple<unsigned int, RAJA::auto_atomic>,
                      std::tuple<unsigned int, RAJA::cuda_atomic>,
                      std::tuple<unsigned long long int, RAJA::auto_atomic>,
                      std::tuple<unsigned long long int, RAJA::cuda_atomic>
                    >;


template <typename T>
class AtomicCUDAIncDecUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( AtomicCUDAIncDecUnitTest );

GPU_TYPED_TEST_P( AtomicCUDAIncDecUnitTest, CUDAIncDecs )
{
  using T = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  T * inc_result = nullptr;
  T * dec_result = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void **)&inc_result, sizeof(T));
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void **)&dec_result, sizeof(T));
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  // test "wrapping" increment
  // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc

  inc_result[0] = (T)0;
  // oldval < val, increment oldval
  forone<test_cuda>( [=] __device__ () {RAJA::atomicInc<AtomicPolicy>(inc_result, (T)1);} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( inc_result[0], (T)1 );

  // oldval == val, wrap to 0
  forone<test_cuda>( [=] __device__ () {RAJA::atomicInc<AtomicPolicy>(inc_result, (T)1);} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( inc_result[0], (T)0 );

  // oldval > val, wrap to 0
  inc_result[0] = (T)2;
  forone<test_cuda>( [=] __device__ () {RAJA::atomicInc<AtomicPolicy>(inc_result, (T)1);} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( inc_result[0], (T)0 );

  // test "wrapping" decrement
  // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicdec

  dec_result[0] = (T)1;
  // oldval > 0, decrement oldval
  forone<test_cuda>( [=] __device__ () {RAJA::atomicDec<AtomicPolicy>(dec_result, (T)1);} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( dec_result[0], (T)0 );

  // oldval == 0, wrap to val
  forone<test_cuda>( [=] __device__ () {RAJA::atomicDec<AtomicPolicy>(dec_result, (T)1);} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( dec_result[0], (T)1 );

  // oldval > val, wrap to val
  dec_result[0] = (T)3;
  forone<test_cuda>( [=] __device__ () {RAJA::atomicDec<AtomicPolicy>(dec_result, (T)1);} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( dec_result[0], (T)1 );

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, inc_result);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, dec_result);
}

REGISTER_TYPED_TEST_SUITE_P( AtomicCUDAIncDecUnitTest,
                             CUDAIncDecs
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( CUDAIncDecUnitTest,
                                AtomicCUDAIncDecUnitTest,
                                CUDA_unsigned_types
                              );
#endif

