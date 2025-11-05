//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic bit methods
///

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA_unit-test-forone.hpp"
#endif

// Basic Bitwise

template <typename T>
class AtomicRefBasicBitwiseUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( AtomicRefBasicBitwiseUnitTest );

TYPED_TEST_P( AtomicRefBasicBitwiseUnitTest, BasicBitwises )
{
  using T = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  T theval = (T)1;
  T * memaddr = &theval;
  T result;

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test and/or
  result = test1.fetch_and( (T)0 );
  ASSERT_EQ( result, (T)1 );
  ASSERT_EQ( test1, (T)0 );

  result = test1.fetch_or( (T)1 );
  ASSERT_EQ( result, (T)0 );
  ASSERT_EQ( test1, (T)1 );

  result = (test1 &= (T)0);
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( result, (T)0 );

  result = (test1 |= (T)1);
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( result, (T)1 );

  // test xor
  result = test1.fetch_xor( (T)1 );
  ASSERT_EQ( result, (T)1 );
  ASSERT_EQ( test1, (T)0 );

  result = (test1 ^= (T)1);
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( result, (T)1 );
}

REGISTER_TYPED_TEST_SUITE_P( AtomicRefBasicBitwiseUnitTest,
                             BasicBitwises
                           );

using basic_types = 
    ::testing::Types<
                      std::tuple<int, RAJA::builtin_atomic>,
                      std::tuple<int, RAJA::seq_atomic>,
                      std::tuple<unsigned int, RAJA::builtin_atomic>,
                      std::tuple<unsigned int, RAJA::seq_atomic>,
                      std::tuple<unsigned long long int, RAJA::builtin_atomic>,
                      std::tuple<unsigned long long int, RAJA::seq_atomic>
#if defined(RAJA_ENABLE_OPENMP)
                      ,
                      std::tuple<int, RAJA::omp_atomic>,
                      std::tuple<unsigned int, RAJA::omp_atomic>,
                      std::tuple<unsigned long long int, RAJA::omp_atomic>
#endif
#if defined(RAJA_ENABLE_CUDA)
                      ,
                      std::tuple<int, RAJA::auto_atomic>,
                      std::tuple<int, RAJA::cuda_atomic>,
                      std::tuple<unsigned int, RAJA::auto_atomic>,
                      std::tuple<unsigned int, RAJA::cuda_atomic>,
                      std::tuple<unsigned long long int, RAJA::auto_atomic>,
                      std::tuple<unsigned long long int, RAJA::cuda_atomic>
#endif
                    >;

INSTANTIATE_TYPED_TEST_SUITE_P( BasicBitwiseUnitTest,
                                AtomicRefBasicBitwiseUnitTest,
                                basic_types
                              );


// Pure CUDA test.
#if defined(RAJA_ENABLE_CUDA)
// CUDA Accessors

template <typename T>
class AtomicRefCUDABitwiseUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( AtomicRefCUDABitwiseUnitTest );

GPU_TYPED_TEST_P( AtomicRefCUDABitwiseUnitTest, CUDABitwises )
{
  using T = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  T * memaddr = nullptr;
  T * result = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void **)&memaddr, sizeof(T));
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void **)&result, sizeof(T));
  memaddr[0] = (T)1;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test and/or
  forone<test_cuda>( [=] __device__ () {result[0] = test1.fetch_and( (T)0 );} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( result[0], (T)1 );
  ASSERT_EQ( test1, (T)0 );

  forone<test_cuda>( [=] __device__ () {result[0] = test1.fetch_or( (T)1 );} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( result[0], (T)0 );
  ASSERT_EQ( test1, (T)1 );

  forone<test_cuda>( [=] __device__ () {result[0] = (test1 &= (T)0);} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( result[0], (T)0 );

  forone<test_cuda>( [=] __device__ () {result[0] = (test1 |= (T)1);} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( result[0], (T)1 );

  // test xor
  forone<test_cuda>( [=] __device__ () {result[0] = test1.fetch_xor( (T)1 );} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( result[0], (T)1 );
  ASSERT_EQ( test1, (T)0 );

  forone<test_cuda>( [=] __device__ () {result[0] = (test1 ^= (T)1);} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( result[0], (T)1 );

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, memaddr);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, result);
}

REGISTER_TYPED_TEST_SUITE_P( AtomicRefCUDABitwiseUnitTest,
                             CUDABitwises
                           );

using CUDA_types = 
    ::testing::Types<
                      std::tuple<int, RAJA::auto_atomic>,
                      std::tuple<int, RAJA::cuda_atomic>,
                      std::tuple<unsigned int, RAJA::auto_atomic>,
                      std::tuple<unsigned int, RAJA::cuda_atomic>,
                      std::tuple<unsigned long long int, RAJA::auto_atomic>,
                      std::tuple<unsigned long long int, RAJA::cuda_atomic>
                    >;

INSTANTIATE_TYPED_TEST_SUITE_P( CUDABitwiseUnitTest,
                                AtomicRefCUDABitwiseUnitTest,
                                CUDA_types
                              );
#endif

