/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for utility metaprogramming related to functions.
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

#ifndef RAJA_FUNCTION_SIGNATURE_UTILS_HPP
#define RAJA_FUNCTION_SIGNATURE_UTILS_HPP

#include "camp/tuple.hpp"

namespace RAJA
{
namespace internal
{

template<typename Fn>
struct signature_impl;

template<typename C, typename R, typename... Args>
struct signature_impl<R (C::*) (Args...)>
{
  using type   = R(Args...);
  using args_t = camp::tuple<Args...>;
};

template<typename C, typename R, typename... Args>
struct signature_impl<R (C::*) (Args...) const> : public signature_impl<R(C::*)(Args...)>
{
};

template<typename C, typename R, typename... Args>
struct signature_impl<R(C::*)(Args...) noexcept> : public signature_impl<R(C::*)(Args...)>
{
};

template<typename C, typename R, typename... Args>
struct signature_impl<R(C::*)(Args...) const noexcept> : public signature_impl<R(C::*)(Args...)>
{
};

template <typename T>
using signature_t = typename signature_impl<T>::type;

template <std::size_t ArgNum, typename T>
using func_arg_t = camp::tuple_element_t<ArgNum, typename signature_impl<T>::args_t>;

}  // namespace internal
}  // namespace RAJA
#endif // RAJA_FUNCTION_SIGNATURE_UTILS_HPP
