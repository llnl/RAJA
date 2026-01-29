//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA reducer reset.
///

#ifndef __TEST_REDUCER_RESET__
#define __TEST_REDUCER_RESET__

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "../test-reducer.hpp"

template  < typename ReducePolicy,
            typename NumericType,
            typename Indexer,
            typename Tuple,
            typename ForOnePol
          >
typename  std::enable_if< // Empty function for non-device policy.
            std::is_base_of<RunOnHost, ForOnePol>::value
          >::type
exec_dispatcher(  RAJA::ReduceSum<ReducePolicy, NumericType> & RAJA_UNUSED_ARG(reduce_sum),
                  RAJA::ReduceMin<ReducePolicy, NumericType> & RAJA_UNUSED_ARG(reduce_min),
                  RAJA::ReduceMax<ReducePolicy, NumericType> & RAJA_UNUSED_ARG(reduce_max),
                  RAJA::ReduceMinLoc<ReducePolicy, NumericType, Indexer> & RAJA_UNUSED_ARG(reduce_minloc),
                  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, Indexer> & RAJA_UNUSED_ARG(reduce_maxloc),
                  RAJA::ReduceMinLoc<ReducePolicy, NumericType, Tuple> & RAJA_UNUSED_ARG(reduce_minloctup),
                  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, Tuple> & RAJA_UNUSED_ARG(reduce_maxloctup),
                  NumericType RAJA_UNUSED_ARG(initVal)
               )
{
  // Non-device policies should do nothing.
}

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
template  < typename ReducePolicy,
            typename NumericType,
            typename Indexer,
            typename Tuple,
            typename ForOnePol
          >
typename  std::enable_if< // GPU policy execution.
            std::is_base_of<RunOnDevice, ForOnePol>::value
          >::type
exec_dispatcher(  RAJA::ReduceSum<ReducePolicy, NumericType> & reduce_sum,
                  RAJA::ReduceMin<ReducePolicy, NumericType> & reduce_min,
                  RAJA::ReduceMax<ReducePolicy, NumericType> & reduce_max,
                  RAJA::ReduceMinLoc<ReducePolicy, NumericType, Indexer> & reduce_minloc,
                  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, Indexer> & reduce_maxloc,
                  RAJA::ReduceMinLoc<ReducePolicy, NumericType, Tuple> & reduce_minloctup,
                  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, Tuple> & reduce_maxloctup,
                  NumericType initVal
               )
{
  // Use device to activate any value for each reducer.
  forone<ForOnePol>( [=] __host__ __device__ () {
                    Tuple temploc(0,0);
                    reduce_sum += initVal;
                    reduce_min.min(0);
                    reduce_max.max(0);
                    reduce_minloc.minloc(0,0);
                    reduce_maxloc.maxloc(0,0);
                    reduce_minloctup.minloc(0,temploc);
                    reduce_maxloctup.maxloc(0,temploc);
                 });
  // Relying on implicit device synchronization in forone.
}
#endif

template <typename T>
class ReducerResetUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(ReducerResetUnitTest);

template <  typename ReducePolicy,
            typename NumericType,
            typename WORKING_RES,
            typename ForOnePol  >
void testReducerReset()
{
  camp::resources::Resource work_res{WORKING_RES::get_default()};
  camp::resources::Resource host_res{camp::resources::Host::get_default()};

  NumericType * resetVal = nullptr;
  NumericType * workVal = nullptr;

  NumericType initVal = (NumericType)5;

  workVal = work_res.allocate<NumericType>(1);
  resetVal = host_res.allocate<NumericType>(1);

  work_res.memcpy( workVal, &initVal, sizeof(initVal) );
  work_res.wait();
  resetVal[0] = (NumericType)10;

  RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(initVal);
  RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(initVal);
  RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(initVal);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(initVal, 1);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(initVal, 1);

  RAJA::tuple<RAJA::Index_type, RAJA::Index_type> LocTup(1, 1);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup(initVal, LocTup);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup(initVal, LocTup);

  // initiate some device computation if using device policy
  exec_dispatcher < ReducePolicy,
                    NumericType,
                    RAJA::Index_type,
                    RAJA::tuple<RAJA::Index_type, RAJA::Index_type>,
                    ForOnePol
                  >
                 (  reduce_sum,
                    reduce_min,
                    reduce_max,
                    reduce_minloc,
                    reduce_maxloc,
                    reduce_minloctup,
                    reduce_maxloctup,
                    initVal
                 );

  // perform real host resets
  reduce_sum.reset(resetVal[0]);
  reduce_min.reset(resetVal[0]);
  reduce_max.reset(resetVal[0]);
  reduce_minloc.reset(resetVal[0], -1);
  reduce_maxloc.reset(resetVal[0], -1);

  RAJA::tuple<RAJA::Index_type, RAJA::Index_type> resetLocTup(0, 0);
  reduce_maxloctup.reset(resetVal[0], resetLocTup);
  reduce_minloctup.reset(resetVal[0], resetLocTup);

  ASSERT_EQ((NumericType)reduce_sum.get(), (NumericType)(resetVal[0]));
  ASSERT_EQ((NumericType)reduce_min.get(), (NumericType)(resetVal[0]));
  ASSERT_EQ((NumericType)reduce_max.get(), (NumericType)(resetVal[0]));

  ASSERT_EQ((NumericType)reduce_minloc.get(), (NumericType)(resetVal[0]));
  ASSERT_EQ((NumericType)reduce_maxloc.get(), (NumericType)(resetVal[0]));
  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), (RAJA::Index_type)(-1));
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), (RAJA::Index_type)(-1));

  ASSERT_EQ((NumericType)reduce_minloctup.get(), (NumericType)(resetVal[0]));
  ASSERT_EQ((NumericType)reduce_maxloctup.get(), (NumericType)(resetVal[0]));

  // Reset of tuple loc defaults to 0
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), (RAJA::Index_type)0);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), (RAJA::Index_type)0);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), (RAJA::Index_type)0);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), (RAJA::Index_type)0);

  // reset locs to default of -1.
  reduce_minloc.reset(resetVal[0], -1);
  reduce_maxloc.reset(resetVal[0], -1);

  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), (RAJA::Index_type)(-1));
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), (RAJA::Index_type)(-1));

  work_res.deallocate( workVal );
  host_res.deallocate( resetVal );
}

TYPED_TEST_P(ReducerResetUnitTest, BasicReset)
{
  using ReduceType = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResourceType = typename camp::at<TypeParam, camp::num<2>>::type;
  using ForOneType = typename camp::at<TypeParam, camp::num<3>>::type;
  testReducerReset< ReduceType, NumericType, ResourceType, ForOneType >();
}

REGISTER_TYPED_TEST_SUITE_P(ReducerResetUnitTest,
                            BasicReset);

#endif  //__TEST_REDUCER_RESET__
