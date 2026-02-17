//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTEDLOOP_PERMUTEDOFFSETVIEW3D_HPP__
#define __TEST_KERNEL_NESTEDLOOP_PERMUTEDOFFSETVIEW3D_HPP__

template <typename IDX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void KernelPermutedOffsetView3DTestImpl(std::array<RAJA::idx_t, 3> dim,
                                        std::array<RAJA::idx_t, 3> perm)
{
  camp::resources::Resource working_res{WORKING_RES::get_default()};
  IDX_TYPE* A_work_array;
  IDX_TYPE* A_check_array;
  IDX_TYPE* A_test_array;
  IDX_TYPE* B_work_array;
  IDX_TYPE* B_check_array;
  IDX_TYPE* B_test_array;

  //
  // These are used for RAJA Layout, Segment definitions in the test.
  //
  // Note that we assume a finite difference stencil width of one.
  //
  std::array<RAJA::idx_t, 3> Nint_len {{dim.at(0), dim.at(1), dim.at(2)}};
  std::array<RAJA::idx_t, 3> Ntot_len {{dim.at(0) + 2 * 1,
                                        dim.at(1) + 2 * 1,
                                        dim.at(2) + 2 * 1}};

  //
  // These are used in data initialization and setting reference solution.
  // We set loop bounds baed on permutation, so inner loop is always stride-1,
  // etc.
  //
  // Also, we assume a finite difference stencil width of one.
  //
  RAJA::idx_t Nint_outer  = dim.at( perm.at(0) );
  RAJA::idx_t Nint_middle = dim.at( perm.at(1) );
  RAJA::idx_t Nint_inner  = dim.at( perm.at(2) );

  RAJA::idx_t Ntot_outer  = Nint_outer + 2 * 1;
  RAJA::idx_t Ntot_middle = Nint_middle + 2 * 1;
  RAJA::idx_t Ntot_inner  = Nint_inner + 2 * 1;

  RAJA::idx_t Nint = Nint_outer * Nint_middle * Nint_inner;
  RAJA::idx_t Ntot = Ntot_outer * Ntot_middle * Ntot_inner;


  allocateForallTestData<IDX_TYPE>(Ntot,
                                   working_res,
                                   &B_work_array,
                                   &B_check_array,
                                   &B_test_array);

  memset(static_cast<void*>(B_test_array), 0, sizeof(IDX_TYPE) * Ntot);

  for (RAJA::idx_t i = 1; i <= Nint_outer; ++i) {
    for (RAJA::idx_t j = 1; j <= Nint_middle; ++j) {
      for (RAJA::idx_t k = 1; k <= Nint_inner; ++k) {
        B_test_array[k + j * Ntot_inner + i * Ntot_inner * Ntot_middle] =
          static_cast<IDX_TYPE>(1);
      }
    }
  }


  working_res.memcpy(B_work_array, B_test_array, sizeof(IDX_TYPE) * Ntot);


  allocateForallTestData<IDX_TYPE>(Nint,
                                   working_res,
                                   &A_work_array,
                                   &A_check_array,
                                   &A_test_array);

  memset(static_cast<void*>(A_test_array), 0, sizeof(IDX_TYPE) * Nint);

  working_res.memcpy(A_work_array, A_test_array, sizeof(IDX_TYPE) * Nint);

  for (RAJA::idx_t i = 0; i < Nint_outer; ++i) {
    for (RAJA::idx_t j = 0; j < Nint_middle; ++j) {
      for (RAJA::idx_t k = 0; k < Nint_inner; ++k) {

        int A_idx = k + j * Nint_inner + i * Nint_inner * Nint_middle;
        int B_idx =
          (k + 1) + (j + 1) * Ntot_inner + (i + 1) * Ntot_inner * Ntot_middle;

        A_test_array[A_idx] =
          B_test_array[B_idx] +                              // C
          B_test_array[B_idx - 1] +                          // W
          B_test_array[B_idx + 1] +                          // E
          B_test_array[B_idx - Ntot_inner] +                 // S
          B_test_array[B_idx + Ntot_inner] +                 // N
          B_test_array[B_idx - (Ntot_inner*Ntot_middle)] +   // B
          B_test_array[B_idx + (Ntot_inner*Ntot_middle)];    // T

      }
    }
  }


  RAJA::OffsetLayout<3> B_layout =
    RAJA::make_permuted_offset_layout<3>( {{-1, -1, -1}},
                                          {{Ntot_len.at(0)-1,
                                            Ntot_len.at(1)-1,
                                            Ntot_len.at(2)-1}},
                                          perm );
  RAJA::Layout<3> A_layout =
    RAJA::make_permuted_layout( {{Nint_len.at(0),
                                  Nint_len.at(1),
                                  Nint_len.at(2)}}, perm );

  RAJA::View< IDX_TYPE, RAJA::OffsetLayout<3> > B_view(B_work_array, B_layout);
  RAJA::View< IDX_TYPE, RAJA::Layout<3> >  A_view(A_work_array, A_layout);

  RAJA::TypedRangeSegment<IDX_TYPE> iseg( 0, Nint_len.at(0) );
  RAJA::TypedRangeSegment<IDX_TYPE> jseg( 0, Nint_len.at(1) );
  RAJA::TypedRangeSegment<IDX_TYPE> kseg( 0, Nint_len.at(2) );

  RAJA::kernel<EXEC_POLICY>(
    RAJA::make_tuple( iseg, jseg, kseg ),
    [=] RAJA_HOST_DEVICE(IDX_TYPE i, IDX_TYPE j, IDX_TYPE k) {
      A_view(i, j, k) = B_view(i, j, k) +
                        B_view(i - 1, j, k) + B_view(i + 1, j, k) +
                        B_view(i, j - 1, k) + B_view(i, j + 1, k) +
                        B_view(i, j, k - 1) + B_view(i, j, k + 1);
    }
  );

  working_res.memcpy(A_check_array, A_work_array, sizeof(IDX_TYPE) * Nint);

  for (RAJA::idx_t ii = 0; ii < Nint; ++ii) {
    ASSERT_EQ(A_test_array[ii], A_check_array[ii]);
  }

  deallocateForallTestData<IDX_TYPE>(working_res,
                                     A_work_array,
                                     A_check_array,
                                     A_test_array);

  deallocateForallTestData<IDX_TYPE>(working_res,
                                     B_work_array,
                                     B_check_array,
                                     B_test_array);
}


TYPED_TEST_SUITE_P(KernelNestedLoopPermutedOffsetView3DTest);
template <typename T>
class KernelNestedLoopPermutedOffsetView3DTest : public ::testing::Test
{
};


TYPED_TEST_P(KernelNestedLoopPermutedOffsetView3DTest, PermutedOffsetView3DKernelTest)
{
  using IDX_TYPE    = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;


#if 0
  RAJA::idx_t dim0 = 23;
  RAJA::idx_t dim1 = 37;
  RAJA::idx_t dim2 = 15;
#else
  RAJA::idx_t dim0 = 3;
  RAJA::idx_t dim1 = 4;
  RAJA::idx_t dim2 = 5;
#endif
  std::array<RAJA::idx_t, 3> dim {{dim0, dim1, dim2}};

  std::array<RAJA::idx_t, 3> perm {{0, 1, 2}};
  KernelPermutedOffsetView3DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim,
                                                                         perm);
  perm = std::array<RAJA::idx_t, 3> {{1, 2, 0}};
  KernelPermutedOffsetView3DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim,
                                                                         perm);

  perm = std::array<RAJA::idx_t, 3> {{2, 0, 1}};
  KernelPermutedOffsetView3DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim,
                                                                         perm);
}

REGISTER_TYPED_TEST_SUITE_P(KernelNestedLoopPermutedOffsetView3DTest,
                            PermutedOffsetView3DKernelTest);

#endif  // __TEST_KERNEL_NESTEDLOOP_PERMUTEDOFFSETVIEW3D_HPP__
