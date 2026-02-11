//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __RAJA_test_abs_HPP__
#define __RAJA_test_abs_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/camp.hpp"

#include <cmath>

namespace RAJA {

  template<typename T>
  camp::concepts::enable_if_t<T, std::is_floating_point<T> >
  test_abs(T&& val) {
    return std::fabs(val);
  } 

  template<typename T>
  camp::concepts::enable_if_t<T, std::is_integral<T> >
  test_abs(T&& val) {
    return std::abs(val);
  }

} // namespace RAJA

#endif // __RAJA_test_abs_HPP__
