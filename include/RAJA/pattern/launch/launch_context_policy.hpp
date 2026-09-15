/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing a helper to
 *           determine the launch context type
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

#ifndef RAJA_pattern_context_policy_HPP
#define RAJA_pattern_context_policy_HPP

#include <type_traits>

#include "RAJA/util/FunctionTypeTraits.hpp"

namespace RAJA
{

template<typename LaunchContextPolicy>
class LaunchContextT;

class LaunchContextHostPolicy;

namespace detail
{

template<typename T, typename = void>
struct launch_context_type
{
  using type = LaunchContextT<LaunchContextHostPolicy>;
};

template<typename T>
struct launch_context_type<
    T,
    std::void_t<internal::func_arg_t<0, decltype(&camp::decay<T>::operator())>>>
{
  using type = camp::decay<
      internal::func_arg_t<0, decltype(&camp::decay<T>::operator())>>;
};


}  // namespace detail

}  // namespace RAJA
#endif
