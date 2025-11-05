//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic add, subtract, inc, and dec methods
///

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA_unit-test-forone.hpp"
#endif

#include "test-atomic-ref.hpp"

// Basic AddSub

template <typename T>
class AtomicRefBasicAddSubUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( AtomicRefBasicAddSubUnitTest );

TYPED_TEST_P( AtomicRefBasicAddSubUnitTest, BasicAddSubs )
{
  using T = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  T theval = (T)0;
  T * memaddr = &theval;

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test inc ops
  T val2 = ++test1;
  T val3 = test1++;
  ASSERT_EQ( test1, (T)2 );
  ASSERT_EQ( val2, (T)1 );
  ASSERT_EQ( val3, (T)1 );

  // test dec ops
  T val4 = --test1;
  T val5 = test1--;
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( val4, (T)1 );
  ASSERT_EQ( val5, (T)1 );

  // test add/sub ops
  T val6 = (test1 += (T)23);
  ASSERT_EQ( test1, (T)23 );
  ASSERT_EQ( val6, (T)23 );
  T val7 = (test1 -= (T)22);
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( val7, (T)1 );

  // test add/sub methods
  T val8 = test1.fetch_add( (T)23 );
  ASSERT_EQ( test1, (T)24 );
  ASSERT_EQ( val8, (T)1 );
  T val9 = test1.fetch_sub( (T)23 );
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( val9, (T)24 );
}

REGISTER_TYPED_TEST_SUITE_P( AtomicRefBasicAddSubUnitTest,
                             BasicAddSubs
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( BasicAddSubUnitTest,
                                AtomicRefBasicAddSubUnitTest,
                                basic_types
                              );


// Pure CUDA test.
#if defined(RAJA_ENABLE_CUDA)
// CUDA Accessors

template <typename T>
class AtomicRefCUDAAddSubUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( AtomicRefCUDAAddSubUnitTest );

GPU_TYPED_TEST_P( AtomicRefCUDAAddSubUnitTest, CUDAAddSubs )
{
  using T = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  T * memaddr = nullptr;
  T * result1 = nullptr;
  T * result2 = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void **)&memaddr, sizeof(T));
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void **)&result1, sizeof(T));
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void **)&result2, sizeof(T));
  memaddr[0] = (T)0;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test inc ops
  forone<test_cuda>( [=] __device__ () {result1[0] = ++test1;} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( result1[0], (T)1 );
  forone<test_cuda>( [=] __device__ () {result2[0] = test1++;} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test1, (T)2 );
  ASSERT_EQ( result2[0], (T)1 );

  // test dec ops
  forone<test_cuda>( [=] __device__ () {result1[0] = --test1;} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( result1[0], (T)1 );
  forone<test_cuda>( [=] __device__ () {result2[0] = test1--;} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( result2[0], (T)1 );

  // test add/sub ops
  forone<test_cuda>( [=] __device__ () {result1[0] = (test1 += (T)23);} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test1, (T)23 );
  ASSERT_EQ( result1[0], (T)23 );
  forone<test_cuda>( [=] __device__ () {result2[0] = (test1 -= (T)22);} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( result2[0], (T)1 );

  // test add/sub methods
  forone<test_cuda>( [=] __device__ () {result1[0] = test1.fetch_add( (T)23 );} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test1, (T)24 );
  ASSERT_EQ( result1[0], (T)1 );
  forone<test_cuda>( [=] __device__ () {result2[0] = test1.fetch_sub( (T)23 );} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( result2[0], (T)24 );

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, memaddr);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, result1);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, result2);
}

REGISTER_TYPED_TEST_SUITE_P( AtomicRefCUDAAddSubUnitTest,
                             CUDAAddSubs
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( CUDAAddSubUnitTest,
                                AtomicRefCUDAAddSubUnitTest,
                                CUDA_types
                              );
#endif

