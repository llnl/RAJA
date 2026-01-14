/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header containing helper type traits for work with Reducers
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


#ifndef RAJA_TYPETRAITS_HPP
#define RAJA_TYPETRAITS_HPP

#include <type_traits>
#include <camp/camp.hpp>

namespace RAJA
{
namespace expt
{
//===========================================================================
//
//
// Forward declarations for types used by type trait helpers
//
//

// Forward declaration of ForallParamPack
template<typename... Params>
struct ForallParamPack;

// Forward declaration of Reducer
namespace detail
{
template<typename Op, typename T, typename VOp>
struct Reducer;
}

//===========================================================================
//
//
// Type traits for SFINAE work.
//
//
namespace type_traits
{
template<typename T>
struct is_ForallParamPack : std::false_type
{};

template<typename... Args>
struct is_ForallParamPack<ForallParamPack<Args...>> : std::true_type
{};

template<typename T>
struct is_ForallParamPack_empty : std::true_type
{};

template<typename First, typename... Rest>
struct is_ForallParamPack_empty<ForallParamPack<First, Rest...>>
    : std::false_type
{};

template<>
struct is_ForallParamPack_empty<ForallParamPack<>> : std::true_type
{};
}  // namespace type_traits

template<typename T>
struct is_instance_of_Reducer : std::false_type
{};

template<typename Op, typename T, typename VOp>
struct is_instance_of_Reducer<detail::Reducer<Op, T, VOp>> : std::true_type
{};

template<typename T>
struct tuple_contains_Reducers : std::false_type
{};

template<typename... Params>
struct tuple_contains_Reducers<camp::tuple<Params...>>
    : std::integral_constant<
          bool,
          camp::concepts::any_of<is_instance_of_Reducer<Params>...>::value>
{};

}  // namespace expt
}  // namespace RAJA

#endif  //  RAJA_TYPETRAITS_HPP
