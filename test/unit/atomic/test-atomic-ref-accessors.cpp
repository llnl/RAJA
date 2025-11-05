//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic accessor methods
///

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA_unit-test-forone.hpp"
#endif

#include "test-atomic-ref.hpp"

// Basic Accessors

template <typename T>
class AtomicRefBasicAccessorUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( AtomicRefBasicAccessorUnitTest );

TYPED_TEST_P( AtomicRefBasicAccessorUnitTest, BasicAccessors )
{
  using T = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  // should also work with CUDA
  T theval = (T)0;
  T * memaddr = &theval;
  T result;

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test store method with op()
  test1.store( (T)19 );
  ASSERT_EQ( test1, (T)19 );

  // test assignment operator
  test1 = (T)23;
  ASSERT_EQ( test1, (T)23 );

  // test load method
  test1 = (T)29;
  ASSERT_EQ( test1.load(), (T)29 );

  // test ()
  result = (test1 = (T)31);
  ASSERT_EQ( test1, (T)31 );
  ASSERT_EQ( result, (T)31 );
}

REGISTER_TYPED_TEST_SUITE_P( AtomicRefBasicAccessorUnitTest,
                             BasicAccessors
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( BasicAccessUnitTest,
                                AtomicRefBasicAccessorUnitTest,
                                basic_types
                              );

// Pure CUDA test.
#if defined(RAJA_ENABLE_CUDA)
// CUDA Accessors

template <typename T>
class AtomicRefCUDAAccessorUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( AtomicRefCUDAAccessorUnitTest );

GPU_TYPED_TEST_P( AtomicRefCUDAAccessorUnitTest, CUDAAccessors )
{
  using T = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  T * memaddr = nullptr;
  T * result = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void **)&memaddr, sizeof(T));
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void **)&result, sizeof(T));
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test store method with op()
  forone<test_cuda>( [=] __device__ () {test1.store( (T)19 );} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test1, (T)19 );

  // test assignment operator
  forone<test_cuda>( [=] __device__ () {test1 = (T)23;} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test1, (T)23 );

  // test load method
  forone<test_cuda>( [=] __device__ () {test1 = (T)29; result[0] = test1.load();} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( result[0], (T)29 );
  ASSERT_EQ( test1, (T)29 );

  // test T()
  forone<test_cuda>( [=] __device__ () {test1 = (T)47; result[0] = test1;} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( result[0], (T)47 );
  ASSERT_EQ( test1, (T)47 );

  forone<test_cuda>( [=] __device__ () {result[0] = (test1 = (T)31);} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( result[0], (T)31 );
  ASSERT_EQ( test1, (T)31 );

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, memaddr);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, result);
}

REGISTER_TYPED_TEST_SUITE_P( AtomicRefCUDAAccessorUnitTest,
                             CUDAAccessors
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( CUDAAccessUnitTest,
                                AtomicRefCUDAAccessorUnitTest,
                                CUDA_types
                              );
#endif


