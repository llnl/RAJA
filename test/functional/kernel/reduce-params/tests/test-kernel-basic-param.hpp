//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_REDUCELOC_MAX2D_HPP__
#define __TEST_KERNEL_REDUCELOC_MAX2D_HPP__

namespace {
template <typename T>
struct val_t_impl {
  using type = T;
};

template <RAJA::concepts::IndexValued T>
struct val_t_impl<T> {
  using type = typename T::value_type;
};

template <typename T>
using VAL_T = typename val_t_impl<T>::type;

template<RAJA::concepts::IndexValued T>
typename T::value_type get_val(T index_val) { return *index_val; }

template<typename T>
requires (!RAJA::concepts::IndexValued<T>)
auto get_val(T index_val) { return index_val; }

}

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename FORALL_POLICY, typename EXEC_POLICY, typename REDUCE_POLICY>
void KernelParamReduceTestImpl(const INDEX_TYPE xdim, const INDEX_TYPE ydim)
{
  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE* work_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  // square 2D array, xdim x ydim
  INDEX_TYPE array_length = xdim * ydim;

  allocateForallTestData<DATA_TYPE> ( array_length,
                                      work_res,
                                      &work_array,
                                      &check_array,
                                      &test_array
                                    );

  // using index_setup_type = VAL_T<INDEX_TYPE>;
  // set rows to point to check and work _arrays
  RAJA::TypedRangeSegment<INDEX_TYPE> seg(0, ydim);
  using LayoutType = RAJA::TypedLayout<INDEX_TYPE, camp::tuple<INDEX_TYPE, INDEX_TYPE>>;
  using ViewType = RAJA::View<DATA_TYPE, LayoutType>;
  ViewType TestView (test_array, xdim, ydim );
  ViewType WorkView (work_array, xdim, ydim );
  ViewType CheckView (check_array, xdim, ydim );
  // initializing  values
  RAJA::forall<RAJA::seq_exec>(seg, [=] RAJA_HOST_DEVICE (INDEX_TYPE zz)
  {
    for ( INDEX_TYPE xx (0); xx < xdim; ++xx )
    {
      CheckView(zz, xx) = get_val(zz * xdim + xx ) % 100 + 1;
    }
    // Make a unique min
    CheckView(ydim - 1, xdim - 1) = 0;
    // Make a unique max
    CheckView(ydim / 2, xdim / 2) = 101;
  });

  work_res.memcpy(work_array, check_array, sizeof(DATA_TYPE) * get_val(array_length));

  RAJA::TypedRangeSegment<INDEX_TYPE> colrange(0, xdim);
  RAJA::TypedRangeSegment<INDEX_TYPE> rowrange(0, ydim);

  using VALLOC_DATA_TYPE = RAJA::expt::ValLoc<DATA_TYPE, Index2D<INDEX_TYPE>>;
  using VALOP_DATA_TYPE_SUM = RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::plus>;
  using VALOP_DATA_TYPE_MIN = RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::minimum>;
  using VALOP_DATA_TYPE_MAX = RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::maximum>;
  using VALOPLOC_DATA_TYPE_MIN = RAJA::expt::ValLocOp<DATA_TYPE, Index2D<INDEX_TYPE>, RAJA::operators::minimum>;
  using VALOPLOC_DATA_TYPE_MAX = RAJA::expt::ValLocOp<DATA_TYPE, Index2D<INDEX_TYPE>, RAJA::operators::maximum>;

  VALLOC_DATA_TYPE seq_minloc(std::numeric_limits<DATA_TYPE>::max(), Index2D<INDEX_TYPE>(-1,-1));
  VALLOC_DATA_TYPE seq_maxloc(std::numeric_limits<DATA_TYPE>::min(), Index2D<INDEX_TYPE>(-1,-1));
  Index2D<INDEX_TYPE> seq_minloc2(-1, -1);
  Index2D<INDEX_TYPE> seq_maxloc2(-1, -1);
  DATA_TYPE seq_sum = 0;
  DATA_TYPE seq_min = std::numeric_limits<DATA_TYPE>::max();
  DATA_TYPE seq_max = std::numeric_limits<DATA_TYPE>::min();
  DATA_TYPE seq_min2 = std::numeric_limits<DATA_TYPE>::max();
  DATA_TYPE seq_max2 = std::numeric_limits<DATA_TYPE>::min();

  VALLOC_DATA_TYPE minloc(std::numeric_limits<DATA_TYPE>::max(), Index2D<INDEX_TYPE>(-1,-1));
  VALLOC_DATA_TYPE maxloc(std::numeric_limits<DATA_TYPE>::min(), Index2D<INDEX_TYPE>(-1,-1));
  Index2D<INDEX_TYPE> minloc2(-1, -1);
  Index2D<INDEX_TYPE> maxloc2(-1, -1);
  DATA_TYPE sum = 0;
  DATA_TYPE min2 = std::numeric_limits<DATA_TYPE>::max();
  DATA_TYPE max2 = std::numeric_limits<DATA_TYPE>::min();
  DATA_TYPE min = std::numeric_limits<DATA_TYPE>::max();
  DATA_TYPE max = std::numeric_limits<DATA_TYPE>::min();

  RAJA::kernel_param<EXEC_POLICY>(
    // segs
    RAJA::make_tuple(colrange, rowrange),
    // params
    RAJA::make_tuple(
      RAJA::expt::Reduce<RAJA::operators::plus>(&sum),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&min),
      RAJA::expt::Reduce<RAJA::operators::maximum>(&max),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&minloc),
      RAJA::expt::Reduce<RAJA::operators::maximum>(&maxloc),
      RAJA::expt::ReduceLoc<RAJA::operators::minimum>(&min2, &minloc2),
      RAJA::expt::ReduceLoc<RAJA::operators::maximum>(&max2, &maxloc2)
    ),
      [=] RAJA_HOST_DEVICE (INDEX_TYPE c,
                            INDEX_TYPE r,
                            VALOP_DATA_TYPE_SUM &_sum,
                            VALOP_DATA_TYPE_MIN &_min,
                            VALOP_DATA_TYPE_MAX &_max,
                            VALOPLOC_DATA_TYPE_MIN &_minloc,
                            VALOPLOC_DATA_TYPE_MAX &_maxloc,
                            VALOPLOC_DATA_TYPE_MIN &_minloc2,
                            VALOPLOC_DATA_TYPE_MAX &_maxloc2) {
        _sum += WorkView(r,c);
        _min.min(WorkView(r,c));
        _max.max(WorkView(r,c));

        // loc
        _minloc.minloc(WorkView(r,c), Index2D<INDEX_TYPE>(c, r));
        _maxloc.maxloc(WorkView(r,c), Index2D<INDEX_TYPE>(c, r));
        _minloc2.minloc(WorkView(r,c), Index2D<INDEX_TYPE>(c, r));
        _maxloc2.maxloc(WorkView(r,c), Index2D<INDEX_TYPE>(c, r));
      });

  // CPU answer

  RAJA::forall<RAJA::seq_exec>(rowrange,
    RAJA::expt::Reduce<RAJA::operators::plus>(&seq_sum),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&seq_min),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&seq_max),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&seq_minloc),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&seq_maxloc),
    RAJA::expt::ReduceLoc<RAJA::operators::minimum>(&seq_min2, &seq_minloc2),
    RAJA::expt::ReduceLoc<RAJA::operators::maximum>(&seq_max2, &seq_maxloc2),
    [=] (INDEX_TYPE r,
         VALOP_DATA_TYPE_SUM &_sum,
         VALOP_DATA_TYPE_MIN &_min,
         VALOP_DATA_TYPE_MAX &_max,
         VALOPLOC_DATA_TYPE_MIN &_minloc,
         VALOPLOC_DATA_TYPE_MAX &_maxloc,
         VALOPLOC_DATA_TYPE_MIN &_minloc2,
         VALOPLOC_DATA_TYPE_MAX &_maxloc2
    ) {
    for (INDEX_TYPE c(0); c < xdim; ++c)
    {
      _sum += CheckView(r,c);
      _min = _min.min(CheckView(r,c));
      _max = _max.max(CheckView(r,c));

      // loc
      _minloc.minloc(CheckView(r,c), Index2D<INDEX_TYPE>(c, r));
      _maxloc.maxloc(CheckView(r,c), Index2D<INDEX_TYPE>(c, r));
      _minloc2.minloc(CheckView(r,c), Index2D<INDEX_TYPE>(c, r));
      _maxloc2.maxloc(CheckView(r,c), Index2D<INDEX_TYPE>(c, r));
    }
  });

  DATA_TYPE DEBUG_SUM = 0;
  for (INDEX_TYPE r (0) ; r < ydim; ++r) {
    for (INDEX_TYPE c (0); c < xdim; ++c) {
      DEBUG_SUM += CheckView(r,c);
    }
  }

  ASSERT_DOUBLE_EQ(seq_sum, sum);
  ASSERT_FLOAT_EQ(DEBUG_SUM, sum);
  ASSERT_DOUBLE_EQ(seq_min2, min2);
  ASSERT_DOUBLE_EQ(seq_max2, max2);
  ASSERT_DOUBLE_EQ(seq_min, min);
  ASSERT_DOUBLE_EQ(seq_max, max);

  ASSERT_EQ(seq_maxloc.getLoc().idx, maxloc.getLoc().idx);
  ASSERT_EQ(seq_maxloc.getLoc().idy, maxloc.getLoc().idy);

  ASSERT_EQ(seq_maxloc2.idx, maxloc2.idx);
  ASSERT_EQ(seq_maxloc2.idy, maxloc2.idy);

  ASSERT_EQ(seq_minloc.getLoc().idx, minloc.getLoc().idx);
  ASSERT_EQ(seq_minloc.getLoc().idy, minloc.getLoc().idy);

  ASSERT_EQ(seq_minloc2.idx, minloc2.idx);
  ASSERT_EQ(seq_minloc2.idy, minloc2.idy);


  deallocateForallTestData<DATA_TYPE> ( work_res,
                                        work_array,
                                        check_array,
                                        test_array
                                      );

}


TYPED_TEST_SUITE_P(KernelReduceParamsTest);
template <typename T>
class KernelReduceParamsTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelReduceParamsTest, ParamReduceKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using FORALL_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<5>>::type;

  KernelParamReduceTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(10, 10);
  KernelParamReduceTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(100, 100);
  KernelParamReduceTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(151, 151);
  KernelParamReduceTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(362, 362);
}

REGISTER_TYPED_TEST_SUITE_P(KernelReduceParamsTest,
                            ParamReduceKernel);

#endif  // __TEST_KERNEL_REDUCELOC_MAX2D_HPP__
