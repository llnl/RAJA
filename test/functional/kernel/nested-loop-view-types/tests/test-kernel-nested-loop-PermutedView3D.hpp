//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTEDLOOP_PERMUTEDVIEW3D_HPP_
#define __TEST_KERNEL_NESTEDLOOP_PERMUTEDVIEW3D_HPP_


template <typename IDX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void KernelPermutedView3DTestImpl(std::array<IDX_TYPE, 3> dim,
                                  std::array<RAJA::idx_t, 3> perm)
{
  camp::resources::Resource working_res{WORKING_RES::get_default()};
  IDX_TYPE* working_array;
  IDX_TYPE* check_array;
  IDX_TYPE* test_array;

  std::array<RAJA::idx_t, 3>
    dim_strip {{ static_cast<RAJA::idx_t>( RAJA::stripIndexType(dim.at(0)) ),
                 static_cast<RAJA::idx_t>( RAJA::stripIndexType(dim.at(1)) ),
                 static_cast<RAJA::idx_t>( RAJA::stripIndexType(dim.at(2)) ) }};
  RAJA::idx_t N = dim_strip.at(0) * dim_strip.at(1) * dim_strip.at(2);

  allocateForallTestData<IDX_TYPE>(N,
                                   working_res,
                                   &working_array,
                                   &check_array,
                                   &test_array);

  memset(static_cast<void*>(test_array), 0, sizeof(IDX_TYPE) * N);

  working_res.memcpy(working_array, test_array, sizeof(IDX_TYPE) * N);

  int mod_val = dim.at( perm.at(1) ) * dim.at( perm.at(2) );
  for (RAJA::idx_t ii = 0; ii < N; ++ii) {
    test_array[ii] = static_cast<IDX_TYPE>(ii % mod_val);
  }

  RAJA::Layout<3> layout = RAJA::make_permuted_layout(dim_strip, perm);
  RAJA::View< IDX_TYPE, RAJA::Layout<3, int> > view(working_array, layout);

  RAJA::kernel<EXEC_POLICY>(
    RAJA::make_tuple( RAJA::TypedRangeSegment<IDX_TYPE>(0, dim_strip.at(0)),
                      RAJA::TypedRangeSegment<IDX_TYPE>(0, dim_strip.at(1)),
                      RAJA::TypedRangeSegment<IDX_TYPE>(0, dim_strip.at(2)) ),
    [=] RAJA_HOST_DEVICE(IDX_TYPE i, IDX_TYPE j, IDX_TYPE k) {
      int val = RAJA::stripIndexType(layout(i, j, k)) % mod_val;
      view(i, j, k) = static_cast<IDX_TYPE>(val);
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


TYPED_TEST_SUITE_P(KernelNestedLoopPermutedView3DTest);
template <typename T>
class KernelNestedLoopPermutedView3DTest : public ::testing::Test
{
};


TYPED_TEST_P(KernelNestedLoopPermutedView3DTest, PermutedView3DKernelTest)
{
  using IDX_TYPE    = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;


  std::array<RAJA::idx_t, 3> perm {{0, 1, 2}};
  //
  // Square view
  //
  std::array<IDX_TYPE, 3> dim_s  {{static_cast<IDX_TYPE>(21),
                                   static_cast<IDX_TYPE>(21),
                                   static_cast<IDX_TYPE>(21)}};
  KernelPermutedView3DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim_s, perm);

  perm = std::array<RAJA::idx_t, 3> {{1, 2, 0}};
  KernelPermutedView3DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim_s, perm);

  perm = std::array<RAJA::idx_t, 3> {{2, 0, 1}};
  KernelPermutedView3DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim_s, perm);

  perm = std::array<RAJA::idx_t, 3> {{0, 1, 2}};
  //
  // Non-square view
  //
  std::array<IDX_TYPE, 3> dim_ns  {{static_cast<IDX_TYPE>(15),
                                    static_cast<IDX_TYPE>(24),
                                    static_cast<IDX_TYPE>(17)}};
  KernelPermutedView3DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim_ns, perm);

  perm = std::array<RAJA::idx_t, 3> {{1, 2, 0}};
  KernelPermutedView3DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim_ns, perm);

  perm = std::array<RAJA::idx_t, 3> {{2, 0, 1}};
  KernelPermutedView3DTestImpl<IDX_TYPE, WORKING_RES, EXEC_POLICY>(dim_ns, perm);
}

REGISTER_TYPED_TEST_SUITE_P(KernelNestedLoopPermutedView3DTest,
                            PermutedView3DKernelTest);

#endif  // __TEST_KERNEL_NESTEDLOOP_PERMUTEDVIEW3D_HPP_
