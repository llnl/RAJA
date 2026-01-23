//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTEDLOOP_OFFSETVIEW2D_HPP__
#define __TEST_KERNEL_NESTEDLOOP_OFFSETVIEW2D_HPP__


template <typename IDX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void KernelOffsetView2DTestImpl(std::array<RAJA::idx_t, 2> dim,
                                std::array<RAJA::idx_t, 2> offset_lo,
                                std::array<RAJA::idx_t, 2> offset_hi)
{
  camp::resources::Resource working_res{WORKING_RES::get_default()};
  IDX_TYPE* working_array;
  IDX_TYPE* check_array;
  IDX_TYPE* test_array;

  RAJA::idx_t N = dim.at(0) * dim.at(1);

  RAJA::idx_t off_dim0 = offset_hi.at(0) - offset_lo.at(0);
  RAJA::idx_t off_dim1 = offset_hi.at(1) - offset_lo.at(1);
  EXPECT_LT( off_dim0, dim.at(0) );
  EXPECT_LT( off_dim1, dim.at(1) );

  allocateForallTestData<IDX_TYPE>(N,
                                   working_res,
                                   &working_array,
                                   &check_array,
                                   &test_array);

  memset(static_cast<void*>(test_array), 0, sizeof(IDX_TYPE) * N);

  working_res.memcpy(working_array, test_array, sizeof(IDX_TYPE) * N);

  for (RAJA::idx_t i = 0; i < off_dim0; ++i) {
    for (RAJA::idx_t j = 0; j < off_dim1; ++j) {
      test_array[j + dim.at(1) * i] = static_cast<IDX_TYPE>(1);
    }
  }


  RAJA::OffsetLayout<2> layout =
    RAJA::make_offset_layout<2>( {{offset_lo.at(0), offset_lo.at(1)}},
                                 {{offset_lo.at(0) + dim.at(0),
                                   offset_lo.at(1) + dim.at(1)}} );
  RAJA::View< IDX_TYPE, RAJA::OffsetLayout<2> > view(working_array, layout);

  RAJA::TypedRangeSegment<IDX_TYPE> iseg( offset_lo.at(0), offset_hi.at(0));
  RAJA::TypedRangeSegment<IDX_TYPE> jseg( offset_lo.at(1), offset_hi.at(1));

  RAJA::kernel<EXEC_POLICY>(
    RAJA::make_tuple( iseg, jseg ),
    [=] RAJA_HOST_DEVICE(IDX_TYPE i, IDX_TYPE j) {
      view(i, j) = static_cast<IDX_TYPE>(1);
    }
  );

  working_res.memcpy(check_array, working_array, sizeof(IDX_TYPE) * N);

  for (RAJA::idx_t ii = 0; ii < N; ++ii) {
    ASSERT_EQ(test_array[ii], check_array[ii]);
  }

  deallocateForallTestData<IDX_TYPE>(working_res,
                                     working_array,
                                     check_array,
                                     test_array);
}


TYPED_TEST_SUITE_P(KernelNestedLoopOffsetView2DTest);
template <typename T>
class KernelNestedLoopOffsetView2DTest : public ::testing::Test
{
};


TYPED_TEST_P(KernelNestedLoopOffsetView2DTest, OffsetView2DKernelTest)
{
  using IDX_TYPE    = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;


  RAJA::idx_t dim0 = 21;
  RAJA::idx_t dim1 = 23;
  std::array<RAJA::idx_t, 2> dim {{dim0, dim1}};

  //
  // Square views
  //
  std::array<RAJA::idx_t, 2> offset_lo {{0, 2}};
  std::array<RAJA::idx_t, 2> offset_hi {{dim0-3, dim1-4}};
  KernelOffsetView2DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim,
                                                                 offset_lo,
                                                                 offset_hi);

  offset_lo = std::array<RAJA::idx_t, 2> {{-1, -2}};
  offset_hi = std::array<RAJA::idx_t, 2> {{dim0-3, dim1-6}};
  KernelOffsetView2DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim,
                                                                 offset_lo,
                                                                 offset_hi);

  //
  // Non-square views
  //
  offset_lo = std::array<RAJA::idx_t, 2> {{0, 1}};
  offset_hi = std::array<RAJA::idx_t, 2> {{dim0-3, dim1-1}};
  KernelOffsetView2DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim,
                                                                 offset_lo,
                                                                 offset_hi);

  offset_lo = std::array<RAJA::idx_t, 2> {{-1, -1}};
  offset_hi = std::array<RAJA::idx_t, 2> {{dim0-3, dim1-4}};
  KernelOffsetView2DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim,
                                                                 offset_lo,
                                                                 offset_hi);
}

REGISTER_TYPED_TEST_SUITE_P(KernelNestedLoopOffsetView2DTest,
                            OffsetView2DKernelTest);

#endif  // __TEST_KERNEL_NESTEDLOOP_OFFSETVIEW2D_HPP__
