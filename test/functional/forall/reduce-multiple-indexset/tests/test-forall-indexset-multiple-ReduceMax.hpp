//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_INDEXSET_MULTIPLE_REDUCEMAX_HPP__
#define __TEST_FORALL_INDEXSET_MULTIPLE_REDUCEMAX_HPP__

#include <cfloat>
#include <cstdlib>
#include <iostream>
#include <random>

//
// Test runs 2 reductions (double) over disjoint chunks
// of an array using an indexset with four range segments
// not aligned with warp boundaries, for example, to check that reduction
// mechanics don't depend on any sort of special indexing.
//
template <typename IDX_TYPE, typename WORKING_RES,
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallIndexSetReduceMaxMultipleTestImpl()
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

  double* working_array;
  double* check_array;
  double* test_array;

  allocateForallTestData<double>(alen,
                                 working_res,
                                 &working_array,
                                 &check_array,
                                 &test_array);

  const double default_val = -DBL_MAX;

  for (IDX_TYPE i = 0; i < alen; ++i) {
    test_array[i] = default_val;
  }

  // for setting random values in arrays
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(-10, 10);

  double current_max = default_val;
  const int test_repeat = 4;

  RAJA::ReduceMax<REDUCE_POLICY, double> dmax0(default_val);
  RAJA::ReduceMax<REDUCE_POLICY, double> dmax1(default_val);

  for (int tcount = 1; tcount <= test_repeat; ++tcount) {

     // pick an index in one of the segments
     int index = 5127;  // seg 3
     if (tcount == 2) index = 1938; // seg2
     if (tcount == 3) index = 13333; // seg4
     if (tcount == 4) index = 52; // seg1

     double droll = dist(mt);
     if (test_array[index] > droll) {
       test_array[index] = droll;
       current_max = RAJA_MAX(current_max, droll);
     }

     working_res.memcpy(working_array, test_array, sizeof(double) * alen);

     RAJA::forall<EXEC_POLICY>(iset, [=] RAJA_HOST_DEVICE(IDX_TYPE i) {
       dmax0.max(working_array[i]);
       dmax1.max(2 * working_array[i]);
     });

     ASSERT_FLOAT_EQ(static_cast<double>(dmax0.get()), current_max);
     ASSERT_FLOAT_EQ(static_cast<double>(dmax1.get()), 2 * current_max);

  }

  deallocateForallTestData<double>(working_res,
                                   working_array,
                                   check_array,
                                   test_array);
}

TYPED_TEST_SUITE_P(ForallIndexSetReduceMaxMultipleTest);
template <typename T>
class ForallIndexSetReduceMaxMultipleTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallIndexSetReduceMaxMultipleTest,
             ReduceMaxMultipleForallIndexSet)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<2>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  ForallIndexSetReduceMaxMultipleTestImpl<IDX_TYPE, WORKING_RES,
                                          EXEC_POLICY, REDUCE_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ForallIndexSetReduceMaxMultipleTest,
                            ReduceMaxMultipleForallIndexSet);

#endif  // __TEST_FORALL_INDEXSET_MULTIPLE_REDUCEMAX_HPP__
