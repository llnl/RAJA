//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_INDEXSET_MULTIPLE_REDUCEMIN_HPP__
#define __TEST_FORALL_INDEXSET_MULTIPLE_REDUCEMIN_HPP__

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
void ForallIndexSetReduceMinMultipleTestImpl()
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

  const double default_val = DBL_MAX;

  for (IDX_TYPE i = 0; i < alen; ++i) {
    test_array[i] = default_val;
  }
  
  // for setting random values in arrays
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(-10, 10);

  double current_min = default_val;
  const int test_repeat = 4;

  RAJA::ReduceMin<REDUCE_POLICY, double> dmin0(default_val);
  RAJA::ReduceMin<REDUCE_POLICY, double> dmin1(default_val);

  for (int tcount = 1; tcount <= test_repeat; ++tcount) {

     // pick an index in one of the segments
     int index = 5127;  // seg 3
     if (tcount == 2) index = 1938; // seg2
     if (tcount == 3) index = 13333; // seg4
     if (tcount == 4) index = 52; // seg1

     double droll = dist(mt);
     if (test_array[index] > droll) {
       test_array[index] = droll;
       current_min = RAJA_MIN(current_min, droll);
     }
 
     working_res.memcpy(working_array, test_array, sizeof(double) * alen);

     RAJA::forall<EXEC_POLICY>(iset, [=] RAJA_HOST_DEVICE(IDX_TYPE i) {
       dmin0.min(working_array[i]);
       dmin1.min(2 * working_array[i]);
     });

     ASSERT_FLOAT_EQ(static_cast<double>(dmin0.get()), current_min);
     ASSERT_FLOAT_EQ(static_cast<double>(dmin1.get()), 2 * current_min);

  }

  deallocateForallTestData<double>(working_res,
                                   working_array,
                                   check_array,
                                   test_array);
}

TYPED_TEST_SUITE_P(ForallIndexSetReduceMinMultipleTest);
template <typename T>
class ForallIndexSetReduceMinMultipleTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallIndexSetReduceMinMultipleTest, 
             ReduceMinMultipleForallIndexSet)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<2>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  ForallIndexSetReduceMinMultipleTestImpl<IDX_TYPE, WORKING_RES,
                                          EXEC_POLICY, REDUCE_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ForallIndexSetReduceMinMultipleTest,
                            ReduceMinMultipleForallIndexSet);

#endif  // __TEST_FORALL_INDEXSET_MULTIPLE_REDUCEMIN_HPP__
