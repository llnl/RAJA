//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_CombiningAdapter_3D_HPP__
#define __TEST_FORALL_CombiningAdapter_3D_HPP__

#include <numeric>
#include <cstring>

#include "RAJA/util/CombiningAdapter.hpp"

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallCombiningAdapter3DTestImpl(INDEX_TYPE first0, INDEX_TYPE last0,
                                  INDEX_TYPE first1, INDEX_TYPE last1,
                                  INDEX_TYPE first2, INDEX_TYPE last2)
{
  RAJA::TypedRangeSegment<INDEX_TYPE> r0(RAJA::stripIndexType(first0), RAJA::stripIndexType(last0));
  RAJA::TypedRangeSegment<INDEX_TYPE> r1(RAJA::stripIndexType(first1), RAJA::stripIndexType(last1));
  RAJA::TypedRangeSegment<INDEX_TYPE> r2(RAJA::stripIndexType(first2), RAJA::stripIndexType(last2));
  INDEX_TYPE N0 = static_cast<INDEX_TYPE>(r0.end() - r0.begin());
  INDEX_TYPE N1 = static_cast<INDEX_TYPE>(r1.end() - r1.begin());
  INDEX_TYPE N2 = static_cast<INDEX_TYPE>(r2.end() - r2.begin());
  INDEX_TYPE N = N0 * N1 * N2;

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  size_t data_len = RAJA::stripIndexType(N)+1;

  allocateForallTestData<INDEX_TYPE>(data_len,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  {

    for (INDEX_TYPE i0 = INDEX_TYPE(0); i0 < N0; i0++) {
      for (INDEX_TYPE i1 = INDEX_TYPE(0); i1 < N1; i1++) {
        for (INDEX_TYPE i2 = INDEX_TYPE(0); i2 < N2; i2++) {
          test_array[i0 * N1*N2 +
                     i1 * N2 +
                     i2] = i0 * N1 * N2 +
                           i1 * N2 +
                           i2;
        }
      }
    }
    test_array[RAJA::stripIndexType(N)] = INDEX_TYPE(0);

    working_res.memset(working_array, 0, sizeof(INDEX_TYPE) * data_len);

    auto adapter = RAJA::make_CombiningAdapter([=] RAJA_HOST_DEVICE(INDEX_TYPE idx0, INDEX_TYPE idx1, INDEX_TYPE idx2) {
      if (idx0 >= first0 && idx0 < last0 &&
          idx1 >= first1 && idx1 < last1 &&
          idx2 >= first2 && idx2 < last2) {
        // in bounds
        working_array[RAJA::stripIndexType((idx0 - first0) * N1 * N2 +
                                           (idx1 - first1) * N2 +
                                           (idx2 - first2))] += (idx0 - first0) * N1 * N2 +
                                                                (idx1 - first1) * N2 +
                                                                (idx2 - first2);
      } else {
        // out of bounds
        working_array[RAJA::stripIndexType(N)]++;
      }
    }, r0, r1, r2);

    RAJA::forall<EXEC_POLICY>(adapter.getRange(), adapter);

  }

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * data_len);

  for (INDEX_TYPE i = INDEX_TYPE(0); i <= N; i++) {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}


TYPED_TEST_SUITE_P(ForallCombiningAdapter3DTest);
template <typename T>
class ForallCombiningAdapter3DTest : public ::testing::Test
{
};

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_unsigned<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeTests()
{
}

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeTests()
{
  // test zero-length range segment
  ForallCombiningAdapter3DTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(-5),
                                                                     INDEX_TYPE(-3), INDEX_TYPE(-3),
                                                                     INDEX_TYPE(-1), INDEX_TYPE(-1));

  ForallCombiningAdapter3DTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(0),
                                                                     INDEX_TYPE(-3), INDEX_TYPE(0),
                                                                     INDEX_TYPE(-4), INDEX_TYPE(0));
  ForallCombiningAdapter3DTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(5),
                                                                     INDEX_TYPE(-3), INDEX_TYPE(2),
                                                                     INDEX_TYPE(-7), INDEX_TYPE(-2));
}


TYPED_TEST_P(ForallCombiningAdapter3DTest, Forall3D)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  // test zero-length range segment
  ForallCombiningAdapter3DTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(3), INDEX_TYPE(3),
                                                                     INDEX_TYPE(5), INDEX_TYPE(5),
                                                                     INDEX_TYPE(7), INDEX_TYPE(7));
  ForallCombiningAdapter3DTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(3), INDEX_TYPE(3),
                                                                     INDEX_TYPE(5), INDEX_TYPE(6),
                                                                     INDEX_TYPE(7), INDEX_TYPE(8));
  ForallCombiningAdapter3DTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(3), INDEX_TYPE(4),
                                                                     INDEX_TYPE(5), INDEX_TYPE(5),
                                                                     INDEX_TYPE(7), INDEX_TYPE(8));
  ForallCombiningAdapter3DTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(3), INDEX_TYPE(4),
                                                                     INDEX_TYPE(5), INDEX_TYPE(6),
                                                                     INDEX_TYPE(7), INDEX_TYPE(7));

  ForallCombiningAdapter3DTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(0), INDEX_TYPE(7),
                                                                     INDEX_TYPE(0), INDEX_TYPE(6),
                                                                     INDEX_TYPE(0), INDEX_TYPE(3));
  ForallCombiningAdapter3DTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(1), INDEX_TYPE(13),
                                                                     INDEX_TYPE(4), INDEX_TYPE(17),
                                                                     INDEX_TYPE(6), INDEX_TYPE(11));
  ForallCombiningAdapter3DTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(13), INDEX_TYPE(46),
                                                                     INDEX_TYPE(17), INDEX_TYPE(51),
                                                                     INDEX_TYPE(4), INDEX_TYPE(31));

  runNegativeTests<INDEX_TYPE, WORKING_RES, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ForallCombiningAdapter3DTest,
                            Forall3D);

#endif  // __TEST_FORALL_CombiningAdapter_3D_HPP__
