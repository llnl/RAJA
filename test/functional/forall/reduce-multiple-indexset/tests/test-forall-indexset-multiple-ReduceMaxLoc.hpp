//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_INDEXSET_MULTIPLE_REDUCEMAXLOC_HPP__
#define __TEST_FORALL_INDEXSET_MULTIPLE_REDUCEMAXLOC_HPP__

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
void ForallIndexSetReduceMaxLocMultipleTestImpl()
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

  double current_max = -DBL_MAX;
  IDX_TYPE current_loc = -1;

  for (IDX_TYPE i = 0; i < alen; ++i) {
    test_array[i] = current_max;
  }
  
  const int test_repeat = 4;

  RAJA::ReduceMaxLoc<REDUCE_POLICY, double, IDX_TYPE> dmax0(current_max, current_loc);
  RAJA::ReduceMaxLoc<REDUCE_POLICY, double, IDX_TYPE> dmax1(current_max, current_loc);

  for (int tcount = 1; tcount <= test_repeat; ++tcount) {

     // set max val 
     current_max = 100.0 + tcount * 10.0;

     // pick an index in one of the segments
     current_loc = 5127;  // seg 3
     if (tcount == 2) current_loc = 1938; // seg2
     if (tcount == 3) current_loc = 13333; // seg4
     if (tcount == 4) current_loc = 52; // seg1

     test_array[current_loc] = current_max;
 
     working_res.memcpy(working_array, test_array, sizeof(double) * alen);

     RAJA::forall<EXEC_POLICY>(iset, [=] RAJA_HOST_DEVICE(IDX_TYPE i) {
       dmax0.maxloc(working_array[i], i);
       dmax1.maxloc(2 * working_array[i], i);
     });

     ASSERT_FLOAT_EQ(static_cast<double>(dmax0.get()), current_max);
     ASSERT_EQ(static_cast<IDX_TYPE>(dmax0.getLoc()), current_loc);
     ASSERT_FLOAT_EQ(static_cast<double>(dmax1.get()), 2 * current_max);
     ASSERT_EQ(static_cast<IDX_TYPE>(dmax1.getLoc()), current_loc);

  }

  deallocateForallTestData<double>(working_res,
                                   working_array,
                                   check_array,
                                   test_array);
}

TYPED_TEST_SUITE_P(ForallIndexSetReduceMaxLocMultipleTest);
template <typename T>
class ForallIndexSetReduceMaxLocMultipleTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallIndexSetReduceMaxLocMultipleTest, 
             ReduceMaxLocMultipleForallIndexSet)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<2>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  ForallIndexSetReduceMaxLocMultipleTestImpl<IDX_TYPE, WORKING_RES,
                                             EXEC_POLICY, REDUCE_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ForallIndexSetReduceMaxLocMultipleTest,
                            ReduceMaxLocMultipleForallIndexSet);

#endif  // __TEST_FORALL_INDEXSET_MULTIPLE_REDUCEMAXLOC_HPP__
