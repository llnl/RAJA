//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Fundamental includes and structs used throughout RAJA tests.
//

#ifndef __RAJA_test_base_HPP__
#define __RAJA_test_base_HPP__

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"

//
// Unroll types for gtest testing::Types
//
template <class T>
struct Test;

template <class... T>
struct Test<camp::list<T...>> {
  using Types = ::testing::Types<T...>;
};


#endif // __RAJA_test_base_HPP__
