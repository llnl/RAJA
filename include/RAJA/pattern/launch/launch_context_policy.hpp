/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing template types of RAJA::LaunchContextT
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_context_policy_HPP
#define RAJA_pattern_context_policy_HPP

namespace RAJA
{

template<typename LaunchContextPolicy>
class LaunchContextT;

class LaunchContextDefaultPolicy;

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
class LaunchContextDim3Policy;
#endif

namespace detail
{


template < typename T, typename = void >
struct has_single_call_operator : std::false_type {};

template < typename T >
struct has_single_call_operator<T, std::enable_if_t<!std::is_same_v<
    decltype(&std::decay_t<T>::operator()), void>>> : std::true_type {};


template <typename T>
struct function_traits{};

template <typename R, typename... Args>
struct function_traits<R(Args...)> {
  using result_type = R;
  static constexpr std::size_t arity = sizeof...(Args);

  template <std::size_t N>
  struct arg {
    static_assert(N < arity, "argument index out of range");
    using type = typename std::tuple_element<N, std::tuple<Args...>>::type;
  };
};

template <typename R, typename... Args>
struct function_traits<R(*)(Args...)> : function_traits<R(Args...)> {};

template <typename R, typename... Args>
struct function_traits<R(&)(Args...)> : function_traits<R(Args...)> {};

template <typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...) const>
    : function_traits<R(Args...)> {
  using functional_type = C;
};

template <typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...)>
    : function_traits<R(Args...)> {
  using functional_type = C;
};


template <typename T, bool hasCallOp = has_single_call_operator<std::decay_t<T>>::value>
struct functional_traits : function_traits<std::decay_t<T>> {};

template <typename T>
struct functional_traits<T, true> : function_traits<decltype(&std::decay_t<T>::operator())> {};


template <typename T, typename = void>
struct has_arg0 : std::false_type {};

template <typename T>
struct has_arg0<
  T,
  typename std::enable_if_t<
    !std::is_same_v<typename functional_traits<T>::template arg<0>::type, void>
  >
> : std::true_type {};


template <typename T, bool HasArg0 = has_arg0<T>::value>
struct launch_context_type {
  using type = LaunchContextT<LaunchContextDefaultPolicy>;
};

template <typename T>
struct launch_context_type<T, true> {
  using type = typename functional_traits<T>::template arg<0>::type;
};


} // namespace detail

}  // namespace RAJA
#endif
