//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for ListSegment
///

#include "RAJA_test-base.hpp"

#include "RAJA_unit-test-types.hpp"

#include "camp/resource.hpp"

#include <vector>

template<typename T>
class ListSegmentUnitTest : public ::testing::Test {};

TYPED_TEST_SUITE(ListSegmentUnitTest, UnitIndexTypes);

//
// Resource object used to construct list segment objects with indices
// living in host (CPU) memory. Used in all tests in this file. 
//
camp::resources::Resource host_res{camp::resources::Host::get_default()};


TYPED_TEST(ListSegmentUnitTest, Constructors)
{
  std::vector<TypeParam> idx;
  for (TypeParam i = 0; i < 5; ++i){
    idx.push_back(i);
  }

  RAJA::TypedListSegment<TypeParam> list1( &idx[0], idx.size(), host_res);
  ASSERT_EQ(list1.size(), idx.size());
  ASSERT_EQ(list1.getIndexOwnership(), RAJA::Owned);

  RAJA::TypedListSegment<TypeParam> copied(list1);
  ASSERT_EQ(list1, copied);
  ASSERT_EQ(copied.getIndexOwnership(), RAJA::Unowned);

  RAJA::TypedListSegment<TypeParam> moved(std::move(list1));
  ASSERT_EQ(list1.size(), 0);
  ASSERT_EQ(moved, copied);

  RAJA::TypedListSegment<TypeParam> container(idx, host_res);
  ASSERT_EQ(container.getIndexOwnership(), RAJA::Owned);
  ASSERT_EQ(moved, container); 
}

TYPED_TEST(ListSegmentUnitTest, Swaps)
{
  std::vector<TypeParam> idx1;
  std::vector<TypeParam> idx2;
  for (TypeParam i = 0; i < 5; ++i){
    idx1.push_back(i);
    idx2.push_back(i+5);
  }

  RAJA::TypedListSegment<TypeParam> list1( idx1, host_res );
  RAJA::TypedListSegment<TypeParam> list2( idx2, host_res );
  auto list3 = RAJA::TypedListSegment<TypeParam>(list1);
  auto list4 = RAJA::TypedListSegment<TypeParam>(list2);

  list1.swap(list2);

  ASSERT_EQ(list2, list3);
  ASSERT_EQ(list1, list4);

  std::swap(list1, list2);

  ASSERT_EQ(list1, list3);
  ASSERT_EQ(list2, list4);
}

TYPED_TEST(ListSegmentUnitTest, Equality)
{
  std::vector<TypeParam> idx1{5,3,1,2};
  RAJA::TypedListSegment<TypeParam> list( idx1, host_res );

  std::vector<TypeParam> idx2{2,1,3,5};
  
  ASSERT_EQ(list.indicesEqual( &idx2.begin()[0], idx2.size() ), false);

  std::reverse( idx2.begin(), idx2.end() );

  ASSERT_EQ(list.indicesEqual( &idx2.begin()[0], idx2.size() ), true);
}

TYPED_TEST(ListSegmentUnitTest, Iterators)
{
  std::vector<TypeParam> idx1{5,3,1,2};
  RAJA::TypedListSegment<TypeParam> list( idx1, host_res );

  ASSERT_EQ(TypeParam(5), *list.begin());
  ASSERT_EQ(TypeParam(2), *(list.end()-1));

  ASSERT_EQ(4, list.size());
}

