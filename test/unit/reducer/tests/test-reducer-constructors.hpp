//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA reducer constructors and initialization.
///

#ifndef __TEST_REDUCER_CONSTRUCTOR__
#define __TEST_REDUCER_CONSTRUCTOR__

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "../test-reducer.hpp"

template <typename T>
class ReducerBasicConstructorUnitTest : public ::testing::Test
{
};

template <typename T>
class ReducerInitConstructorUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(ReducerBasicConstructorUnitTest);
TYPED_TEST_SUITE_P(ReducerInitConstructorUnitTest);

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
template <typename ReducePolicy,
          typename NumericType>
typename  std::enable_if<
#if defined(RAJA_ENABLE_CUDA) // CUDA policy does nothing.
            std::is_same<ReducePolicy, RAJA::cuda_reduce>::value
#elif defined(RAJA_ENABLE_HIP) // HIP policy does nothing.
            std::is_same<ReducePolicy, RAJA::hip_reduce>::value
#else
#error Please enable a supported GPU platform, e.g. CUDA or HIP.
#endif
          >::type
testReducerConstructor()
{
  // do nothing
}
#endif

// Basic constructor tests are only expected to be verified on the host.
// Should not run this on a GPU.
template <typename ReducePolicy,
          typename NumericType>
typename  std::enable_if< // CPU policy.
#if defined(RAJA_ENABLE_CUDA)
            !std::is_same<ReducePolicy, RAJA::cuda_reduce>::value
#elif defined(RAJA_ENABLE_HIP)
            !std::is_same<ReducePolicy, RAJA::hip_reduce>::value
#else
            true  // Always run for non-GPU policies.
#endif
          >::type
testReducerConstructor()
{
  RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum;
  RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min;
  RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max;
  RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc;
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc;

  RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup;
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup;

  ASSERT_EQ((NumericType)reduce_sum.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_min.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_max.get(), NumericType());

  ASSERT_EQ((NumericType)reduce_minloc.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_maxloc.get(), NumericType());
  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), RAJA::Index_type());

  ASSERT_EQ((NumericType)reduce_minloctup.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_maxloctup.get(), NumericType());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), RAJA::Index_type());
}

TYPED_TEST_P(ReducerBasicConstructorUnitTest, BasicReducerConstructor)
{
  using ReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;

  testReducerConstructor< ReducePolicy, NumericType >();
}

template  < typename ReducePolicy,
            typename NumericType,
            typename ForOnePol >
typename  std::enable_if< // Host policy does nothing.
            std::is_base_of<RunOnHost, ForOnePol>::value
          >::type
exec_dispatcher( NumericType * RAJA_UNUSED_ARG(workVal) )
{
  // Do nothing for host policies.
}

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
template  < typename ReducePolicy,
            typename NumericType,
            typename ForOnePol >
typename  std::enable_if< // GPU policy fiddles with value.
            std::is_base_of<RunOnDevice, ForOnePol>::value
          >::type
exec_dispatcher( NumericType * workVal )
{
  forone<ForOnePol>( [=] __device__ () {
                        workVal[0] += 1;
                        workVal[0] -= 1;
                 });
}
#endif

template <typename ReducePolicy,
          typename NumericType,
          typename WORKING_RES,
          typename ForOnePol>
void testInitReducerConstructor()
{
  camp::resources::Resource work_res{WORKING_RES::get_default()};
  camp::resources::Resource host_res{camp::resources::Host::get_default()};

  NumericType * theVal = nullptr;
  NumericType * workVal = nullptr;

  NumericType initVal = (NumericType)5;

  workVal = work_res.allocate<NumericType>(1);
  theVal = host_res.allocate<NumericType>(1);

  work_res.memcpy( workVal, &initVal, sizeof(initVal) );
  work_res.wait();
  theVal[0] = (NumericType)10;

  RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(initVal);
  RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(initVal);
  RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(initVal);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(initVal, 1);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(initVal, 1);

  RAJA::tuple<RAJA::Index_type, RAJA::Index_type> LocTup(1, 1);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup(initVal, LocTup);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup(initVal, LocTup);

  // move a value onto device and fiddle
  exec_dispatcher < ReducePolicy,
                    NumericType,
                    ForOnePol
                  >
                  ( workVal );

  work_res.memcpy( &initVal, workVal, sizeof(initVal) );
  work_res.wait();

  theVal[0] = initVal;

  ASSERT_EQ((NumericType)(theVal[0]), (NumericType)(initVal));

  ASSERT_EQ((NumericType)reduce_sum.get(), (NumericType)(initVal));
  ASSERT_EQ((NumericType)reduce_min.get(), (NumericType)(initVal));
  ASSERT_EQ((NumericType)reduce_max.get(), (NumericType)(initVal));

  ASSERT_EQ((NumericType)reduce_minloc.get(), (NumericType)(initVal));
  ASSERT_EQ((NumericType)reduce_maxloc.get(), (NumericType)(initVal));
  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), (RAJA::Index_type)1);

  ASSERT_EQ((NumericType)reduce_minloctup.get(), (NumericType)(initVal));
  ASSERT_EQ((NumericType)reduce_maxloctup.get(), (NumericType)(initVal));
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), (RAJA::Index_type)1);

  work_res.deallocate( workVal );
  host_res.deallocate( theVal );
}

TYPED_TEST_P(ReducerInitConstructorUnitTest, InitReducerConstructor)
{
  using ReduceType = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResourceType = typename camp::at<TypeParam, camp::num<2>>::type;
  using ForOneType = typename camp::at<TypeParam, camp::num<3>>::type;

  testInitReducerConstructor< ReduceType, NumericType, ResourceType, ForOneType >();
}


REGISTER_TYPED_TEST_SUITE_P(ReducerBasicConstructorUnitTest,
                            BasicReducerConstructor);

REGISTER_TYPED_TEST_SUITE_P(ReducerInitConstructorUnitTest,
                            InitReducerConstructor);

#endif  //__TEST_REDUCER_CONSTRUCTOR__
