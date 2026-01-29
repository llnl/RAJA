//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_LISTSEGMENTVIEW_HPP__
#define __TEST_FORALL_LISTSEGMENTVIEW_HPP__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <type_traits>

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallListSegmentViewTestImpl(INDEX_TYPE N)
{

  // Create and initialize indices in idx_array used to create list segment
  std::vector<INDEX_TYPE> idx_array;

  srand ( time(NULL) );

  for (INDEX_TYPE i = 0; i < N; ++i) {
    INDEX_TYPE randval = rand() % N;
    if ( i < randval ) {
      idx_array.push_back(i);
    }     
  }

  size_t idxlen = idx_array.size();

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  RAJA::TypedListSegment<INDEX_TYPE> lseg(&idx_array[0], idxlen, 
                                          working_res);

  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  memset( test_array, 0, sizeof(INDEX_TYPE) * N );  

  working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * N);

  for (size_t i = 0; i < idxlen; ++i) {
    test_array[ idx_array[i] ] = idx_array[i];
  }

  using layout_type = RAJA::Layout<1, INDEX_TYPE, 0>;
  using view_type = RAJA::View< INDEX_TYPE, layout_type >;
#if (!(defined(_GLIBCXX_RELEASE) || defined(RAJA_COMPILER_INTEL) || defined(RAJA_COMPILER_MSVC)))\
    || _GLIBCXX_RELEASE >= 20150716
  #if (__GNUG__ && __GNUC__ < 5)
  #define IS_TRIVIALLY_COPYABLE(T) __has_trivial_copy(T)
  #else
  #define IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
  #endif
  static_assert(IS_TRIVIALLY_COPYABLE(layout_type),
                "These layouts should always be triviallly copyable");

  // AJK: see ViewBase Ctor notes in RAJA/Util/TypedViewBase.hpp
#if (!defined(RAJA_ENABLE_CUDA) && !defined(RAJA_ENABLE_CLANG_CUDA))
  static_assert(IS_TRIVIALLY_COPYABLE(view_type),
                "These views should always be triviallly copyable");
#endif


#endif
  
  RAJA::Layout<1> layout(N);
  view_type work_view(working_array, layout);

  RAJA::forall<EXEC_POLICY>(lseg, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    work_view( idx ) = idx;
  });

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * N);

  for (INDEX_TYPE i = 0; i < N; i++) {
    ASSERT_EQ(test_array[i], check_array[i]);
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallListSegmentOffsetViewTestImpl(INDEX_TYPE N, INDEX_TYPE offset)
{

  // Create and initialize indices in idx_array used to create list segment
  std::vector<INDEX_TYPE> idx_array;

  srand ( time(NULL) );

  for (INDEX_TYPE i = 0; i < N; ++i) {
    INDEX_TYPE randval = rand() % N;
    if ( i < randval ) {
      idx_array.push_back(i+offset);
    }     
  }

  size_t idxlen = idx_array.size();

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  RAJA::TypedListSegment<INDEX_TYPE> lseg(&idx_array[0], idxlen, 
                                          working_res);

  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  memset( test_array, 0, sizeof(INDEX_TYPE) * N );  

  working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * N);

  for (size_t i = 0; i < idxlen; ++i) {
    test_array[ idx_array[i]-offset ] = idx_array[i];
  }

  using layout_type = RAJA::OffsetLayout<1, INDEX_TYPE>;
  using view_type = RAJA::View< INDEX_TYPE, layout_type >;

  INDEX_TYPE N_offset = N + offset;
  view_type work_view(working_array, 
                      RAJA::make_offset_layout<1, INDEX_TYPE>( {{offset}}, 
                                                               {{N_offset}} ));

  RAJA::forall<EXEC_POLICY>(lseg, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    work_view( idx ) = idx;
  });

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * N);

  for (INDEX_TYPE i = 0; i < N; i++) {
    ASSERT_EQ(test_array[i], check_array[i]);
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}

TYPED_TEST_SUITE_P(ForallListSegmentViewTest);
template <typename T>
class ForallListSegmentViewTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallListSegmentViewTest, ListSegmentForallView)
{
  using INDEX_TYPE       = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY      = typename camp::at<TypeParam, camp::num<2>>::type;

  ForallListSegmentViewTestImpl<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(13);
  ForallListSegmentViewTestImpl<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(2047);
  ForallListSegmentViewTestImpl<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(32000);

  ForallListSegmentOffsetViewTestImpl<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(13, 1);
  ForallListSegmentOffsetViewTestImpl<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(2047, 2);
  ForallListSegmentOffsetViewTestImpl<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(32000, 3);
}

REGISTER_TYPED_TEST_SUITE_P(ForallListSegmentViewTest,
                            ListSegmentForallView);

#endif  // __TEST_FORALL_LISTSEGMENTVIEW_HPP__
