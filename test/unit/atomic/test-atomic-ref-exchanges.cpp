//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic exchange and swap methods
///

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA_unit-test-forone.hpp"
#endif

// Basic Exchange

template <typename T>
class AtomicRefBasicExchangeUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( AtomicRefBasicExchangeUnitTest );

TYPED_TEST_P( AtomicRefBasicExchangeUnitTest, BasicExchanges )
{
  using T = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  T swapper = (T)91;
  T theval = (T)0;
  T * memaddr = &theval;

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test exchange method
  swapper = test1.exchange( swapper );
  ASSERT_EQ( test1, (T)91 );
  ASSERT_EQ( swapper, (T)0 );

  // test CAS method
  swapper = test1.CAS( (T)91, swapper );
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( swapper, (T)91 );


  bool result = true;
  T testval = (T)19;
  T & valref = testval;

  // test strong exchange method
  result = test1.compare_exchange_strong( valref, testval );
  ASSERT_EQ( result, false );
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( swapper, (T)91 );
  ASSERT_EQ( testval, (T)0 );

  // test weak exchange method (same as strong exchange)
  result = test1.compare_exchange_weak( valref, swapper );
  ASSERT_EQ( result, true );
  ASSERT_EQ( test1, (T)91 );
  ASSERT_EQ( swapper, (T)91 );
  ASSERT_EQ( testval, (T)0 );
}

REGISTER_TYPED_TEST_SUITE_P( AtomicRefBasicExchangeUnitTest,
                             BasicExchanges
                           );

using basic_types = 
    ::testing::Types<
                      std::tuple<int, RAJA::builtin_atomic>,
                      std::tuple<int, RAJA::seq_atomic>,
                      std::tuple<unsigned int, RAJA::builtin_atomic>,
                      std::tuple<unsigned int, RAJA::seq_atomic>,
                      std::tuple<unsigned long long int, RAJA::builtin_atomic>,
                      std::tuple<unsigned long long int, RAJA::seq_atomic>,
                      std::tuple<float, RAJA::builtin_atomic>,
                      std::tuple<float, RAJA::seq_atomic>,
                      std::tuple<double, RAJA::builtin_atomic>,
                      std::tuple<double, RAJA::seq_atomic>
#if defined(RAJA_ENABLE_OPENMP)
                      ,
                      std::tuple<int, RAJA::omp_atomic>,
                      std::tuple<unsigned int, RAJA::omp_atomic>,
                      std::tuple<unsigned long long int, RAJA::omp_atomic>,
                      std::tuple<float, RAJA::omp_atomic>,
                      std::tuple<double, RAJA::omp_atomic>
#endif
#if defined(RAJA_ENABLE_CUDA)
                      ,
                      std::tuple<int, RAJA::auto_atomic>,
                      std::tuple<int, RAJA::cuda_atomic>,
                      std::tuple<unsigned int, RAJA::auto_atomic>,
                      std::tuple<unsigned int, RAJA::cuda_atomic>,
                      std::tuple<unsigned long long int, RAJA::auto_atomic>,
                      std::tuple<unsigned long long int, RAJA::cuda_atomic>,
                      std::tuple<float, RAJA::auto_atomic>,
                      std::tuple<float, RAJA::cuda_atomic>
#endif
                    >;

INSTANTIATE_TYPED_TEST_SUITE_P( BasicExchangeUnitTest,
                                AtomicRefBasicExchangeUnitTest,
                                basic_types
                              );


// Pure CUDA test.
#if defined(RAJA_ENABLE_CUDA)
// CUDA Accessors

template <typename T>
class AtomicRefCUDAExchangeUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( AtomicRefCUDAExchangeUnitTest );

GPU_TYPED_TEST_P( AtomicRefCUDAExchangeUnitTest, CUDAExchanges )
{
  using T = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  T * swapper = nullptr;
  T * memaddr = nullptr;
  T * testval = nullptr;
  bool * result = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, &swapper, sizeof(T));
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, &memaddr, sizeof(T));
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, &testval, sizeof(T));
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, &result, sizeof(bool));
  swapper[0] = (T)91;
  memaddr[0] = (T)0;
  testval[0] = (T)19;
  result[0] = true;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test exchange method
  forone<test_cuda>( [=] __device__ () {swapper[0] = test1.exchange( swapper[0] );} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test1, (T)91 );
  ASSERT_EQ( swapper[0], (T)0 );

  // test CAS method
  forone<test_cuda>( [=] __device__ () {swapper[0] = test1.CAS( (T)91, swapper[0] );} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( swapper[0], (T)91 );

  // test strong exchange method
  forone<test_cuda>( [=] __device__ () {result[0] = test1.compare_exchange_strong( testval[0], testval[0] );} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( result[0], false );
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( swapper[0], (T)91 );
  ASSERT_EQ( testval[0], (T)0 );

  // test weak exchange method (same as strong exchange)
  forone<test_cuda>( [=] __device__ () {result[0] = test1.compare_exchange_weak( testval[0], swapper[0] );} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( result[0], true );
  ASSERT_EQ( test1, (T)91 );
  ASSERT_EQ( swapper[0], (T)91 );
  ASSERT_EQ( testval[0], (T)0 );

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, swapper);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, memaddr);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, testval);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, result);
}

REGISTER_TYPED_TEST_SUITE_P( AtomicRefCUDAExchangeUnitTest,
                             CUDAExchanges
                           );

using CUDA_types = 
    ::testing::Types<
                      std::tuple<int, RAJA::auto_atomic>,
                      std::tuple<int, RAJA::cuda_atomic>,
                      std::tuple<unsigned int, RAJA::auto_atomic>,
                      std::tuple<unsigned int, RAJA::cuda_atomic>,
                      std::tuple<unsigned long long int, RAJA::auto_atomic>,
                      std::tuple<unsigned long long int, RAJA::cuda_atomic>,
                      std::tuple<float, RAJA::auto_atomic>,
                      std::tuple<float, RAJA::auto_atomic>
                    >;

INSTANTIATE_TYPED_TEST_SUITE_P( CUDAExchangeUnitTest,
                                AtomicRefCUDAExchangeUnitTest,
                                CUDA_types
                              );
#endif

