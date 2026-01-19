//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTEDLOOP_OFFSETVIEW3D_HPP__
#define __TEST_KERNEL_NESTEDLOOP_OFFSETVIEW3D_HPP__


template <typename IDX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void KernelOffsetView3DTestImpl(std::array<RAJA::idx_t, 3> dim,
                                std::array<RAJA::idx_t, 3> offset_lo,
                                std::array<RAJA::idx_t, 3> offset_hi)
{
  camp::resources::Resource working_res{WORKING_RES::get_default()};
  IDX_TYPE* working_array;
  IDX_TYPE* check_array;
  IDX_TYPE* test_array;

  RAJA::idx_t N = dim.at(0) * dim.at(1) * dim.at(2);

  RAJA::idx_t off_dim0 = offset_hi.at(0) - offset_lo.at(0);
  RAJA::idx_t off_dim1 = offset_hi.at(1) - offset_lo.at(1);
  RAJA::idx_t off_dim2 = offset_hi.at(2) - offset_lo.at(2);
  EXPECT_LT( off_dim0, dim.at(0) );
  EXPECT_LT( off_dim1, dim.at(1) );
  EXPECT_LT( off_dim2, dim.at(2) );

  allocateForallTestData<IDX_TYPE>(N,
                                   working_res,
                                   &working_array,
                                   &check_array,
                                   &test_array);

  memset(static_cast<void*>(test_array), 0, sizeof(IDX_TYPE) * N);

  working_res.memcpy(working_array, test_array, sizeof(IDX_TYPE) * N);

  for (RAJA::idx_t i = 0; i < off_dim0; ++i) {
    for (RAJA::idx_t j = 0; j < off_dim1; ++j) {
      for (RAJA::idx_t k = 0; k < off_dim2; ++k) {
        test_array[k + dim.at(2) * j + dim.at(1) * dim.at(2) * i] =
          static_cast<IDX_TYPE>(1);
      }
    }
  }


  RAJA::OffsetLayout<3> layout =
    RAJA::make_offset_layout<3>( {{offset_lo.at(0),
                                   offset_lo.at(1),
                                   offset_lo.at(2)}},
                                 {{offset_lo.at(0) + dim.at(0),
                                   offset_lo.at(1) + dim.at(1),
                                   offset_lo.at(2) + dim.at(2)}} );

  RAJA::View< IDX_TYPE, RAJA::OffsetLayout<3> > view(working_array, layout);

  RAJA::TypedRangeSegment<IDX_TYPE> iseg( offset_lo.at(0), offset_hi.at(0));
  RAJA::TypedRangeSegment<IDX_TYPE> jseg( offset_lo.at(1), offset_hi.at(1));
  RAJA::TypedRangeSegment<IDX_TYPE> kseg( offset_lo.at(2), offset_hi.at(2));

  RAJA::kernel<EXEC_POLICY>(
    RAJA::make_tuple( iseg, jseg, kseg ),
    [=] RAJA_HOST_DEVICE(IDX_TYPE i, IDX_TYPE j, IDX_TYPE k) {
      view(i, j, k) = static_cast<IDX_TYPE>(1);
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


TYPED_TEST_SUITE_P(KernelNestedLoopOffsetView3DTest);
template <typename T>
class KernelNestedLoopOffsetView3DTest : public ::testing::Test
{
};


TYPED_TEST_P(KernelNestedLoopOffsetView3DTest, OffsetView3DKernelTest)
{
  using IDX_TYPE    = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;


  RAJA::idx_t dim0 = 13;
  RAJA::idx_t dim1 = 19;
  RAJA::idx_t dim2 = 16;
  std::array<RAJA::idx_t, 3> dim {{dim0, dim1, dim2}};

  //
  // Square views
  //
  std::array<RAJA::idx_t, 3> offset_lo {{0, 2, 1}};
  std::array<RAJA::idx_t, 3> offset_hi {{dim0-2, dim1-6, dim2-4}};
  KernelOffsetView3DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim,
                                                                 offset_lo,
                                                                 offset_hi);

  offset_lo = std::array<RAJA::idx_t, 3> {{-1, -2, -3}};
  offset_hi = std::array<RAJA::idx_t, 3> {{dim0-3, dim1-10, dim2-8}};
  KernelOffsetView3DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim,
                                                                 offset_lo,
                                                                 offset_hi);

  //
  // Non-square views
  //
  offset_lo = std::array<RAJA::idx_t, 3> {{0, 1, 2}};
  offset_hi = std::array<RAJA::idx_t, 3> {{dim0-3, dim1-2, dim2-2}};
  KernelOffsetView3DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim,
                                                                 offset_lo,
                                                                 offset_hi);

  offset_lo = std::array<RAJA::idx_t, 3> {{-1, -1, 0}};
  offset_hi = std::array<RAJA::idx_t, 3> {{dim0-3, dim1-4, dim2-2}};
  KernelOffsetView3DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim,
                                                                 offset_lo,
                                                                 offset_hi);
}

REGISTER_TYPED_TEST_SUITE_P(KernelNestedLoopOffsetView3DTest,
                            OffsetView3DKernelTest);

#endif  // __TEST_KERNEL_NESTEDLOOP_OFFSETVIEW3D_HPP__
