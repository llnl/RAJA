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

class LaunchContextDefaultPolicy;

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
class LaunchContextDim3Policy;
#endif

namespace detail
{

// Primary template
template <typename T>
struct function_traits;

// Specialization for plain function pointers
template <typename R, typename... Args>
struct function_traits<R(*)(Args...)> {
    using result_type = R;
    static constexpr std::size_t arity = sizeof...(Args);

    template <std::size_t N>
    struct arg {
        static_assert(N < arity, "argument index out of range");
        using type = typename std::tuple_element<N, std::tuple<Args...>>::type;
    };
};

// Specialization for const member function pointers,
// which is what a non-mutable lambda's operator() usually is.
template <typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...) const>
    : function_traits<R(*)(Args...)> {};

// Optional: handle mutable lambdas (non-const operator())
template <typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...)>
    : function_traits<R(*)(Args...)> {};

// Convenience alias for lambdas and other callable objects
template <typename Lambda>
using lambda_traits = function_traits<decltype(&Lambda::operator())>;

}

}  // namespace RAJA
#endif
