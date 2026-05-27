/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining core C++ concepts for use in development of
 *RAJA
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

#ifndef RAJA_type_concepts_HPP
#define RAJA_type_concepts_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/detail/TypeTraits.hpp"
#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/policy/MultiPolicy.hpp"
#include "RAJA/util/resource.hpp"
#include "RAJA/index/IndexSet.hpp"

namespace RAJA
{

namespace concepts
{

/// A RAJA ExecutionPolicy is a backend-specific directive supplied by the user
/// like hip_exec or seq_exec that instructs RAJA how to configure a parallel
/// kernel
template<typename Pol>
concept ExecutionPolicy =
    RAJA::type_traits::is_same_decay_v<decltype(Pol::policy), ::RAJA::Policy> &&
    RAJA::type_traits::is_same_decay_v<decltype(Pol::pattern),
                                       ::RAJA::Pattern> &&
    RAJA::type_traits::is_same_decay_v<decltype(Pol::launch), ::RAJA::Launch> &&
    RAJA::type_traits::is_same_decay_v<decltype(Pol::platform),
                                       ::RAJA::Platform>;

template<typename T>
concept IndexSetType =
    static_cast<bool>(RAJA::type_traits::is_index_set<std::decay_t<T>>::value);

template<typename T>
concept IndexSetPolicy =
    static_cast<bool>(type_traits::is_indexset_policy<std::decay_t<T>>::value);

template<typename T>
concept Resource = RAJA::type_traits::is_resource<std::decay_t<T>>::value;

template<typename T>
concept ForallParams =
    expt::type_traits::is_ForallParamPack<std::decay_t<T>>::value;

template<typename T>
concept MultiPolicyConcept =
    static_cast<bool>(RAJA::type_traits::is_multi_policy<T>::value);

template<class Function, class Return, class Arg1 = Return, class Arg2 = Arg1>
concept BinaryFunction = std::is_invocable_r_v<Return, Function&, Arg1, Arg2>;

template<class Function, class Return, class Arg1 = Return>
concept UnaryFunction = std::is_invocable_r_v<Return, Function&, Arg1>;

}  // namespace concepts

namespace type_traits
{
DefineTypeTraitFromConcept(is_execution_policy,
                           RAJA::concepts::ExecutionPolicy);

template<class Function, class Return, class Arg1 = Return, class Arg2 = Arg1>
struct is_binary_function
    : std::bool_constant<
          RAJA::concepts::BinaryFunction<Function, Return, Arg1, Arg2>>
{};
template<class Function, class Return, class Arg1 = Return, class Arg2 = Arg1>
inline constexpr bool is_binary_function_v =
    is_binary_function<Function, Return, Arg1, Arg2>::value;

template<class Function, class Return, class Arg = Return>
struct is_unary_function
    : std::bool_constant<RAJA::concepts::UnaryFunction<Function, Return, Arg>>
{};
template<class Function, class Return, class Arg1 = Return>
inline constexpr bool is_unary_function_v =
    is_unary_function<Function, Return, Arg1>::value;

}  // namespace type_traits


}  // namespace RAJA

#endif