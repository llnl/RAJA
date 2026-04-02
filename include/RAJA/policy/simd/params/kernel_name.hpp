//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef SIMD_KERNELNAME_HPP
#define SIMD_KERNELNAME_HPP

#include "RAJA/pattern/params/kernel_name.hpp"

namespace RAJA
{
namespace expt
{
namespace detail
{

// Init
template<typename EXEC_POL>
camp::concepts::enable_if<std::is_same<EXEC_POL, RAJA::simd_exec>> param_init(
    EXEC_POL const&,
    RAJA::detail::Name&)
{
  // TODO: Define kernel naming
}

// Combine
template<typename EXEC_POL, typename T>
RAJA_HOST_DEVICE camp::concepts::enable_if<
    std::is_same<EXEC_POL, RAJA::simd_exec>>
param_combine(EXEC_POL const&, RAJA::detail::Name&, T)
{}

// Resolve
template<typename EXEC_POL>
camp::concepts::enable_if<std::is_same<EXEC_POL, RAJA::simd_exec>>
param_resolve(EXEC_POL const&, RAJA::detail::Name&)
{
  // TODO: Define kernel naming
}

}  //  namespace detail
}  //  namespace expt
}  //  namespace RAJA


#endif  //  NEW_REDUCE_SIMD_REDUCE_HPP
