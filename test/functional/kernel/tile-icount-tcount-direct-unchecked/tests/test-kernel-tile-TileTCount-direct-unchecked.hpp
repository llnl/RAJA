//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_TILE_TILETCOUNT_DIRECT_UNCHECKED_HPP_
#define __TEST_KERNEL_TILE_TILETCOUNT_DIRECT_UNCHECKED_HPP_

//
// Value struct for manipulating tile sizes in parameterized tests.
//
template<int VALUE>
struct Value {
  static constexpr int value = VALUE;
};


template<typename IDX_TYPE, typename EXEC_POLICY, typename REDUCE_POLICY, typename USE_REDUCER_PARAM>
std::enable_if_t<USE_REDUCER_PARAM::value>
CallKernel(IDX_TYPE& trip_count,
            IDX_TYPE& tile_count,
            IDX_TYPE t,
            IDX_TYPE N,
            IDX_TYPE tsize) {
  RAJA::kernel_param<EXEC_POLICY>(
    RAJA::make_tuple( RAJA::TypedRangeSegment<IDX_TYPE>(0, N) ),
    RAJA::make_tuple( static_cast<IDX_TYPE>(0),
      RAJA::expt::Reduce<RAJA::operators::plus>(&trip_count),
      RAJA::expt::Reduce<RAJA::operators::plus>(&tile_count)
    ),

    [=] RAJA_HOST_DEVICE(IDX_TYPE i, IDX_TYPE ti,
                         RAJA::expt::ValOp<IDX_TYPE, RAJA::operators::plus>& _trip_count,
                         RAJA::expt::ValOp<IDX_TYPE, RAJA::operators::plus>& _tile_count) {
      _trip_count += 1;
      if ( i / tsize == t && ti == t ) {
        _tile_count += 1;
      }
    }
  );

}

template<typename IDX_TYPE, typename EXEC_POLICY, typename REDUCE_POLICY, typename USE_REDUCER_PARAM>
std::enable_if_t<!USE_REDUCER_PARAM::value>
CallKernel(IDX_TYPE& _trip_count,
            IDX_TYPE& _tile_count,
            IDX_TYPE t,
            IDX_TYPE N,
            IDX_TYPE tsize) {
  RAJA::ReduceSum<REDUCE_POLICY, IDX_TYPE> tile_count(_tile_count);
  RAJA::ReduceSum<REDUCE_POLICY, IDX_TYPE> trip_count(_trip_count);
  RAJA::kernel_param<EXEC_POLICY>(
    RAJA::make_tuple( RAJA::TypedRangeSegment<IDX_TYPE>(0, N) ),
    RAJA::make_tuple( static_cast<IDX_TYPE>(0) ),

    [=] RAJA_HOST_DEVICE(IDX_TYPE i, IDX_TYPE ti) {
      trip_count += 1;
      if ( i / tsize == t && ti == t ) {
        tile_count += 1;
      }
    }
  );

  _trip_count = trip_count.get();
  _tile_count = tile_count.get();
}

template <typename IDX_TYPE, typename EXEC_POLICY, typename REDUCE_POLICY, typename USE_REDUCER_PARAM>
void KernelTileTileTCountDirectUncheckedTestImpl(IDX_TYPE N, IDX_TYPE tsize)
{

  IDX_TYPE NT = (N + tsize - 1) / tsize;
  IDX_TYPE trip_count(0);

  for (IDX_TYPE t = 0; t < NT; ++t) {
    IDX_TYPE tile_count = 0;
    CallKernel<IDX_TYPE, EXEC_POLICY, REDUCE_POLICY, USE_REDUCER_PARAM>(trip_count, tile_count, t, N, tsize);

    IDX_TYPE tile_expect = tsize;
    if ( (t + 1) * tsize > N ) {
      tile_expect = N - t * tsize;
    }
    ASSERT_EQ(tile_count, tile_expect);
    ASSERT_EQ(trip_count, (t+1) * N);
  }
}


TYPED_TEST_SUITE_P(KernelTileTileTCountDirectUncheckedTest);
template <typename T>
class KernelTileTileTCountDirectUncheckedTest : public ::testing::Test
{
};


TYPED_TEST_P(KernelTileTileTCountDirectUncheckedTest, TileTCountTileKernel)
{
  using IDX_TYPE    = typename camp::at<TypeParam, camp::num<0>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<1>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;
  using USE_REDUCER_PARAM = typename camp::at<TypeParam, camp::num<3>>::type;

  IDX_TYPE tsize = camp::at_v<TypeParam, 4>::value;

  KernelTileTileTCountDirectUncheckedTestImpl<IDX_TYPE, EXEC_POLICY, REDUCE_POLICY, USE_REDUCER_PARAM>(
    IDX_TYPE(0), tsize);
  KernelTileTileTCountDirectUncheckedTestImpl<IDX_TYPE, EXEC_POLICY, REDUCE_POLICY, USE_REDUCER_PARAM>(
    IDX_TYPE(tsize), tsize);
  KernelTileTileTCountDirectUncheckedTestImpl<IDX_TYPE, EXEC_POLICY, REDUCE_POLICY, USE_REDUCER_PARAM>(
    IDX_TYPE(13*tsize), tsize);

}

REGISTER_TYPED_TEST_SUITE_P(KernelTileTileTCountDirectUncheckedTest,
                            TileTCountTileKernel);

#endif  // __TEST_KERNEL_TILE_TILETCOUNT_DIRECT_UNCHECKED_HPP_
