/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA wrapper header file for a C++ style allocator with a resource.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_RESOURCE_ALLOCATOR_HPP
#define RAJA_RESOURCE_ALLOCATOR_HPP

#include <cstddef>

#include "RAJA/config.hpp"
#include "RAJA/util/resource.hpp"
#include "RAJA/util/macros.hpp"

namespace RAJA
{
template<typename T, typename Resource>
using ResourceAllocator = typename RAJA::resources::ResourceAllocator<
    Resource>::template allocator<T>;
}  // namespace RAJA

#endif  // RAJA_RESOURCE_ALLOCATOR_HPP
