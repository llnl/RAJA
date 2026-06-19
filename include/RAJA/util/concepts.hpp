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

#include "camp/concepts.hpp"

namespace RAJA
{

namespace concepts
{
template<class Function, class Return, class Arg1 = Return, class Arg2 = Arg1>
concept BinaryFunction = std::is_invocable_r_v<Return, Function&, Arg1, Arg2>;

template<class Function, class Return, class Arg1 = Return>
concept UnaryFunction = std::is_invocable_r_v<Return, Function&, Arg1>;

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
