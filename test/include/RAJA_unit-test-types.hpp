//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Types used in RAJA unit tests (template params to GTest)
//

#ifndef __RAJA_unit_test_types_HPP__
#define __RAJA_unit_test_types_HPP__

#include "RAJA_test-base.hpp"

//
// List of integral types used in RAJA index unit tests
//
using UnitIntegralTypes = ::testing::Types<char,
                                           unsigned char,
                                           short,
                                           unsigned short,
                                           int,
                                           unsigned int,
                                           long,
                                           unsigned long,
                                           long int,
                                           unsigned long int,
                                           long long,
                                           unsigned long long>;

//
// Expanded integral types used in RAJA index unit tests
//
#ifndef RAJA_UNIT_EXPANDED_INTEGRAL_TYPES
  #define RAJA_UNIT_EXPANDED_INTEGRAL_TYPES \
    RAJA::Index_type,    \
    char,                \
    unsigned char,       \
    short,               \
    unsigned short,      \
    int,                 \
    unsigned int,        \
    long,                \
    unsigned long,       \
    long int,            \
    unsigned long int,   \
    long long,           \
    unsigned long long
#endif // RAJA_UNIT_EXPANDED_INTEGRAL_TYPES

#ifndef RAJA_UNIT_FLOAT_TYPES
#ifndef __clang__
  #define RAJA_UNIT_FLOAT_TYPES \
    float,               \
    double,              \
    long double
#else
  #define RAJA_UNIT_FLOAT_TYPES \
    float,               \
    double
#endif // __clang__
#endif // FLOATING_TYPES

using UnitExpandedIntegralTypes = 
  ::testing::Types<RAJA_UNIT_EXPANDED_INTEGRAL_TYPES>;

using UnitFloatTypes = ::testing::Types<RAJA_UNIT_FLOAT_TYPES>;

using UnitIntFloatTypes = 
  ::testing::Types<RAJA_UNIT_EXPANDED_INTEGRAL_TYPES,RAJA_UNIT_FLOAT_TYPES>;

//
// Standard list of index types used in RAJA index unit tests
//
using UnitIndexTypes = ::testing::Types<RAJA::Index_type,
                                        int,
#if defined(RAJA_TEST_EXHAUSTIVE)
                                        unsigned int,
                                        char,
                                        unsigned char,
                                        short,
                                        unsigned short,
                                        long,
                                        unsigned long,
                                        long int,
                                        unsigned long int,
                                        long long,
#endif
                                        unsigned long long>;

#endif  // __RAJA_unit_test_types_HPP__
