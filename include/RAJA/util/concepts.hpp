/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA concept definitions.
 *
 *          Definitions in this file will propagate to all RAJA header files.
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

#ifndef RAJA_concepts_HPP
#define RAJA_concepts_HPP

#include <iterator>
#include "RAJA/util/types.hpp"
#include <type_traits>

#include "RAJA/index/IndexValue.hpp"
#include "camp/concepts.hpp"

namespace RAJA
{

namespace concepts
{
template<class Function, class Return, class Arg1 = Return, class Arg2 = Arg1>
concept BinaryFunction = std::is_invocable_r_v<Return, Function&, Arg1, Arg2>;

template<class Function, class Return, class Arg1 = Return>
concept UnaryFunction = std::is_invocable_r_v<Return, Function&, Arg1>;

template<typename T, typename U>
concept RangeConstructible =
    (IndexValued<T> && IndexValued<U> &&
     std::is_same_v<std::remove_cvref_t<T>, std::remove_cvref_t<U>>) ||
    (!IndexValued<T> && !IndexValued<U> && requires {
      typename std::common_type_t<std::remove_cvref_t<T>,
                                  std::remove_cvref_t<U>>;
    });

template<typename StrongT, typename T>
concept StrongOrSignedIntegralStrideArg =
    ((IndexValued<T> &&
      std::is_same_v<std::remove_cvref_t<T>, std::remove_cvref_t<StrongT>>) ||
     (!IndexValued<T> &&
      std::is_integral_v<strip_index_type_t<std::remove_cvref_t<T>>>)) &&
    std::is_signed_v<strip_index_type_t<std::remove_cvref_t<T>>>;

template<typename StrongT, typename T>
concept StrongRangeBound =
    IndexValued<T> &&
    std::is_same_v<std::remove_cvref_t<T>, std::remove_cvref_t<StrongT>>;

template<typename BeginT, typename EndT, typename StrideT>
concept RangeStrideConstructible =
    (!IndexValued<BeginT> && !IndexValued<EndT> && !IndexValued<StrideT> &&
     std::is_signed_v<strip_index_type_t<std::remove_cvref_t<StrideT>>> &&
     requires {
       typename std::common_type_t<std::remove_cvref_t<BeginT>,
                                   std::remove_cvref_t<EndT>,
                                   make_signed_t<strip_index_type_t<
                                       std::remove_cvref_t<StrideT>>>>;
     }) ||
    (IndexValued<BeginT> && StrongRangeBound<BeginT, EndT> &&
     StrongOrSignedIntegralStrideArg<BeginT, StrideT>);

using namespace camp::concepts;
}  // namespace concepts

namespace type_traits
{

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

using namespace camp::type_traits;
}  // namespace type_traits

}  // end namespace RAJA

#endif
