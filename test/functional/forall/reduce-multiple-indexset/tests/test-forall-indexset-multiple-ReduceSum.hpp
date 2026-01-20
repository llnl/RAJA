//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_INDEXSET_MULTIPLE_REDUCESUM_HPP__
#define __TEST_FORALL_INDEXSET_MULTIPLE_REDUCESUM_HPP__

#include <cstdlib>
#include <numeric>

//
// Test runs 4 reductions (2 int, 2 double) over disjoint chunks
// of an array using an indexset with four range segments
// not aligned with warp boundaries, for example, to check that reduction
// mechanics don't depend on any sort of special indexing.
//
template <typename IDX_TYPE, typename WORKING_RES, 
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallIndexSetReduceSumMultipleTestImpl()
{
  using RangeSegType = RAJA::TypedRangeSegment<IDX_TYPE>;
  using IdxSetType = RAJA::TypedIndexSet<RangeSegType>;

  RAJA::TypedRangeSegment<IDX_TYPE> r1(1, 1037);
  RAJA::TypedRangeSegment<IDX_TYPE> r2(1043, 2036);
  RAJA::TypedRangeSegment<IDX_TYPE> r3(4098, 6103);
  RAJA::TypedRangeSegment<IDX_TYPE> r4(10243, 15286);

  IdxSetType iset;
  iset.push_back(r1); 
  iset.push_back(r2); 
  iset.push_back(r3); 
  iset.push_back(r4); 

  const IDX_TYPE alen = 15286;

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  double* dworking_array;
  double* dcheck_array;
  double* dtest_array;

  allocateForallTestData<double>(alen,
                                 working_res,
                                 &dworking_array,
                                 &dcheck_array,
                                 &dtest_array);

  int* iworking_array;
  int* icheck_array;
  int* itest_array;

  allocateForallTestData<int>(alen,
                              working_res,
                              &iworking_array,
                              &icheck_array,
                              &itest_array);

  const double dinit_val = 0.1;
  const int iinit_val = 1;

  for (IDX_TYPE i = 0; i < alen; ++i) {
    dtest_array[i] = dinit_val;
    itest_array[i] = iinit_val;
  }
  
  working_res.memcpy(dworking_array, dtest_array, sizeof(double) * alen);
  working_res.memcpy(iworking_array, itest_array, sizeof(int) * alen);

  const double drinit = 5.0;
  const int irinit = 4;
  const int test_repeat = 4;

  RAJA::ReduceSum<REDUCE_POLICY, double> dsum0(drinit * 1.0);
  RAJA::ReduceSum<REDUCE_POLICY, int> isum1(irinit * 2);
  RAJA::ReduceSum<REDUCE_POLICY, double> dsum2(drinit * 3.0);
  RAJA::ReduceSum<REDUCE_POLICY, int> isum3(irinit * 4);

  for (int tcount = 1; tcount <= test_repeat; ++tcount) {
 
    RAJA::forall<EXEC_POLICY>(iset, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
      dsum0 += 1.0 * dworking_array[idx];
      isum1 += 2 * iworking_array[idx];
      dsum2 += 3.0 * dworking_array[idx];
      isum3 += 4 * iworking_array[idx];
    });

    double dchk_val = dinit_val * static_cast<double>(iset.getLength());
    int ichk_val = iinit_val * static_cast<int>(iset.getLength());

    ASSERT_FLOAT_EQ(static_cast<double>(dsum0.get()), 
                               tcount * (1 * dchk_val) + (drinit * 1.0) );
    ASSERT_EQ(static_cast<int>(isum1.get()), 
                               tcount * (2 * ichk_val) + (irinit * 2) );
    ASSERT_FLOAT_EQ(static_cast<double>(dsum2.get()),
                               tcount * (3 * dchk_val) + (drinit * 3.0) );
    ASSERT_EQ(static_cast<int>(isum3.get()), 
                               tcount * (4 * ichk_val) + (irinit * 4) );

  }

  deallocateForallTestData<double>(working_res,
                                   dworking_array,
                                   dcheck_array,
                                   dtest_array);

  deallocateForallTestData<int>(working_res,
                                iworking_array,
                                icheck_array,
                                itest_array);
}

TYPED_TEST_SUITE_P(ForallIndexSetReduceSumMultipleTest);
template <typename T>
class ForallIndexSetReduceSumMultipleTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallIndexSetReduceSumMultipleTest, 
             ReduceSumMultipleForallIndexSet)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<2>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  ForallIndexSetReduceSumMultipleTestImpl<IDX_TYPE, WORKING_RES,
                                          EXEC_POLICY, REDUCE_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ForallIndexSetReduceSumMultipleTest,
                            ReduceSumMultipleForallIndexSet);

#endif  // __TEST_FORALL_INDEXSET_MULTIPLE_REDUCESUM_HPP__
