//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Custom index type used for loc reductions.
//

#ifndef __RAJA_test_reduceloc_types_HPP__
#define __RAJA_test_reduceloc_types_HPP__

#include "RAJA/RAJA.hpp"
#include "RAJA/util/types.hpp"
#include "camp/list.hpp"

struct Index2D {
   RAJA::Index_type idx, idy;
   constexpr Index2D() : idx(-1), idy(-1) {}
   constexpr Index2D(RAJA::Index_type init) : idx(init), idy(init) {}
   constexpr Index2D(RAJA::Index_type ix, RAJA::Index_type iy) : idx(ix), idy(iy) {}
   template<typename T>
   RAJA_HOST_DEVICE void operator=(T rhs) { idx = rhs; idx = rhs; }
};

#endif // __RAJA_test_reduceloc_types_HPP__
