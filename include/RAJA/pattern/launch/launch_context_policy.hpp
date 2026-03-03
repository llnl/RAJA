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

namespace RAJA
{

template<typename LaunchContextPolicy>
class LaunchContextT;

class LaunchContextDefaultPolicy;

namespace detail
{

template<typename T>
struct first_argument;

template<typename R, typename Arg0, typename... Args>
struct first_argument<R(Arg0, Args...)>
{
  using type = Arg0;
};

template<typename C, typename R, typename Arg0, typename... Args>
struct first_argument<R (C::*)(Arg0, Args...)>
    : first_argument<R(Arg0, Args...)>
{};

template<typename C, typename R, typename Arg0, typename... Args>
struct first_argument<R (C::*)(Arg0, Args...) const>
    : first_argument<R(Arg0, Args...)>
{};

template<typename C, typename R, typename Arg0, typename... Args>
struct first_argument<R (C::*)(Arg0, Args...) noexcept>
    : first_argument<R(Arg0, Args...)>
{};

template<typename C, typename R, typename Arg0, typename... Args>
struct first_argument<R (C::*)(Arg0, Args...) const noexcept>
    : first_argument<R(Arg0, Args...)>
{};

template<typename T, typename = void>
struct callable_signature
{
  using type = camp::decay<T>;
};

template<typename T>
struct callable_signature<T, std::void_t<decltype(&camp::decay<T>::operator())>>
{
  using type = decltype(&camp::decay<T>::operator());
};

template<typename T, typename = void>
struct launch_context_type
{
  using type = LaunchContextT<LaunchContextDefaultPolicy>;
};

template<typename T>
struct launch_context_type<T,
                           std::void_t<typename first_argument<camp::decay<
                               typename callable_signature<T>::type>>::type>>
{
  using type = camp::decay<
      typename first_argument<typename callable_signature<T>::type>::type>;
};


}  // namespace detail

}  // namespace RAJA
#endif
