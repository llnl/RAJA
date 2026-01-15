//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef SYCL_KERNELNAME_HPP
#define SYCL_KERNELNAME_HPP

#include "RAJA/pattern/params/kernel_name.hpp"

namespace RAJA
{
namespace expt
{
namespace detail
{

#if defined(RAJA_ENABLE_SYCL)

// Init
template<typename EXEC_POL>
camp::concepts::enable_if<RAJA::type_traits::is_sycl_policy<EXEC_POL>>
param_init(EXEC_POL const&, RAJA::detail::Name&)
{
  // TODO: Define kernel naming
}

// Combine
template<typename EXEC_POL, typename T>
camp::concepts::enable_if<RAJA::type_traits::is_sycl_policy<EXEC_POL>>
    SYCL_EXTERNAL param_combine(EXEC_POL const&, RAJA::detail::Name&, T)
{}

// Resolve
template<typename EXEC_POL>
camp::concepts::enable_if<RAJA::type_traits::is_sycl_policy<EXEC_POL>>
param_resolve(EXEC_POL const&, RAJA::detail::Name&)
{
  // TODO: Define kernel naming
}

#endif

}  //  namespace detail
}  //  namespace expt
}  //  namespace RAJA


#endif  //  NEW_REDUCE_SYCL_REDUCE_HPP
