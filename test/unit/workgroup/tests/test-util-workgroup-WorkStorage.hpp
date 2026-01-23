//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup constructors.
///

#ifndef __TEST_UTIL_WORKGROUP_WORKSTORAGE__
#define __TEST_UTIL_WORKGROUP_WORKSTORAGE__

#include "RAJA_test-workgroup.hpp"

#include <random>
#include <array>
#include <cstddef>


template < typename T >
struct TestCallable
{
  TestCallable(T _val)
    : val(_val)
  { }

  TestCallable(TestCallable const&) = delete;
  TestCallable& operator=(TestCallable const&) = delete;

  TestCallable(TestCallable&& o)
    : val(o.val)
    , move_constructed(true)
  {
    o.moved_from = true;
  }

  TestCallable& operator=(TestCallable&& o)
  {
    val = o.val;
    o.moved_from = true;
    return *this;
  }

  RAJA_HOST_DEVICE void operator()(
      void* val_ptr, bool* move_constructed_ptr, bool* moved_from_ptr) const
  {
    *static_cast<T*>(val_ptr) = val;
    *move_constructed_ptr = move_constructed;
    *moved_from_ptr = moved_from;
  }

private:
  T val;
public:
  bool move_constructed = false;
  bool moved_from = false;
};


// work around inconsistent std::array support over stl versions
template < typename T, size_t N >
struct TestArray
{
  T a[N]{};
  T& operator[](size_t i) { return a[i]; }
  T const& operator[](size_t i) const { return a[i]; }
  friend inline bool operator==(TestArray const& lhs, TestArray const& rhs)
  {
    for (size_t i = 0; i < N; ++i) {
      if (lhs[i] == rhs[i]) continue;
      else return false;
    }
    return true;
  }
  friend inline bool operator!=(TestArray const& lhs, TestArray const& rhs)
  {
    return !(lhs == rhs);
  }
};

#endif  //__TEST_UTIL_WORKGROUP_WORKSTORAGE__
