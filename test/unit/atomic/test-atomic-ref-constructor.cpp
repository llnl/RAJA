//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic ref constructors (and use of getPointer for verification)
///

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA_unit-test-forone.hpp"
#endif

#include "test-atomic-ref.hpp"

// Default constructors with basic types

template <typename T>
class AtomicRefDefaultConstructorUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P(AtomicRefDefaultConstructorUnitTest);

template <typename T>
void DefaultPolConstructors()
{
  T * memaddr = nullptr;

  // explicit constructor with memory address
  RAJA::AtomicRef<T> test1( memaddr );

  ASSERT_EQ( test1.getPointer(), nullptr );

  // ref constructor
  RAJA::AtomicRef<T> const & reft1 = test1;
  RAJA::AtomicRef<T> reftest1( reft1 );

  ASSERT_EQ( reftest1.getPointer(), nullptr );
}

TYPED_TEST_P( AtomicRefDefaultConstructorUnitTest, DefaultPolConstructors )
{
  DefaultPolConstructors<TypeParam>();
}

REGISTER_TYPED_TEST_SUITE_P( AtomicRefDefaultConstructorUnitTest,
                             DefaultPolConstructors
                           );

using default_types = ::testing::Types< int,
                                      float,
                                      double
                                    >;

INSTANTIATE_TYPED_TEST_SUITE_P( DefaultConstrUnitTest,
                                AtomicRefDefaultConstructorUnitTest,
                                default_types
                              );

// Basic Constructors with policies

template <typename T>
class AtomicRefBasicConstructorUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( AtomicRefBasicConstructorUnitTest );

TYPED_TEST_P( AtomicRefBasicConstructorUnitTest, BasicConstructors )
{
  using NumericType = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  NumericType * memaddr = nullptr;

  // explicit constructor with memory address
  RAJA::AtomicRef<NumericType, AtomicPolicy> test1( memaddr );

  ASSERT_EQ( test1.getPointer(), nullptr );

  // ref constructor
  RAJA::AtomicRef<NumericType, AtomicPolicy> const & reft1 = test1;
  RAJA::AtomicRef<NumericType, AtomicPolicy> reftest1( reft1 );

  ASSERT_EQ( reftest1.getPointer(), nullptr );
}

REGISTER_TYPED_TEST_SUITE_P( AtomicRefBasicConstructorUnitTest,
                             BasicConstructors
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( BasicConstrUnitTest,
                                AtomicRefBasicConstructorUnitTest,
                                basic_types
                              );

// Pure CUDA test.
#if defined(RAJA_ENABLE_CUDA)
// CUDA Constructors with policies

template <typename T>
class AtomicRefCUDAConstructorUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P(AtomicRefCUDAConstructorUnitTest);

GPU_TYPED_TEST_P( AtomicRefCUDAConstructorUnitTest, CUDAConstructors )
{
  using NumericType = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  NumericType * memaddr = nullptr;
  NumericType * proxy = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void **)&proxy, sizeof(NumericType));
  proxy = memaddr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  // explicit constructor with memory address
  RAJA::AtomicRef<NumericType, AtomicPolicy> test0( memaddr );
  RAJA::AtomicRef<NumericType, AtomicPolicy> test1( proxy );

  forone<test_cuda>( [=] __device__ () {test1.getPointer();} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  ASSERT_EQ( test0.getPointer(), nullptr );
  ASSERT_EQ( test1.getPointer(), nullptr );

  // ref constructor
  RAJA::AtomicRef<NumericType, AtomicPolicy> const & reft1 = test1;
  RAJA::AtomicRef<NumericType, AtomicPolicy> reftest1( reft1 );
  forone<test_cuda>( [=] __device__ () {reftest1.getPointer();} );
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  ASSERT_EQ( reftest1.getPointer(), nullptr );

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, proxy);
}

REGISTER_TYPED_TEST_SUITE_P( AtomicRefCUDAConstructorUnitTest,
                             CUDAConstructors
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( CUDAConstrUnitTest,
                                AtomicRefCUDAConstructorUnitTest,
                                CUDA_types
                              );
#endif

