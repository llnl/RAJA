/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for type helpers used in tests.
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

#ifndef _TYPE_HELPER_HPP_
#define _TYPE_HELPER_HPP_

#include "gtest/gtest.h"

#include <tuple>

namespace types
{


template <typename S, typename T>
struct type_cat;

template <typename... Ss, typename... Ts>
struct type_cat<std::tuple<Ss...>, std::tuple<Ts...>> {
  using type = std::tuple<Ss..., Ts...>;
};


template <typename S, typename T>
struct product;

template <typename S, typename... Ss, typename... Ts>
struct product<std::tuple<S, Ss...>, std::tuple<Ts...>> {
  // the cartesian product of {S} and {Ts...}
  // is a list of pairs -- here: a std::tuple of 2-element std::tuples
  using S_cross_Ts = std::tuple<std::tuple<S, Ts>...>;

  // the cartesian product of {Ss...} and {Ts...} (computed recursively)
  using Ss_cross_Ts = typename product<std::tuple<Ss...>, std::tuple<Ts...>>::type;

  // concatenate both products
  using type = typename type_cat<S_cross_Ts, Ss_cross_Ts>::type;
};

template <typename... Ss, typename... Ts, typename... Smembers>
struct product<std::tuple<std::tuple<Smembers...>, Ss...>, std::tuple<Ts...>> {
  // the cartesian product of {S} and {Ts...}
  // is a list of pairs -- here: a std::tuple of 2-element std::tuples
  using S_cross_Ts = std::tuple<std::tuple<Smembers..., Ts>...>;

  // the cartesian product of {Ss...} and {Ts...} (computed recursively)
  using Ss_cross_Ts = typename product<std::tuple<Ss...>, std::tuple<Ts...>>::type;

  // concatenate both products
  using type = typename type_cat<S_cross_Ts, Ss_cross_Ts>::type;
};

// end the recursion
template <typename... Ts>
struct product<std::tuple<>, std::tuple<Ts...>> {
  using type = std::tuple<>;
};
}  // namespace types


namespace tt
{
template <typename...>
struct concat;

template <template <class...> class T, typename U>
struct concat<T<U>> {
  using type = U;
};

template <typename T>
struct concat<T> {
  using type = T;
};

template <template <class...> class T,
          class... Front,
          class... Next,
          class... Rest>
struct concat<T<Front...>, T<Next...>, Rest...> {
  using type = typename concat<T<Front..., Next...>, Rest...>::type;
};

template <typename... Ts>
using concat_t = typename concat<Ts...>::type;

template <class T>
struct collapse {
  using type = T;
};

template <template <class...> class T, class... U>
struct collapse<T<T<U...>>> {
  using type = typename collapse<T<U...>>::type;
};

template <typename T>
using collapse_t = typename collapse<T>::type;

template <template <class> class, class>
struct apply;

template <template <class...> class L, template <class> class Fn, class... Ts>
struct apply<Fn, L<Ts...>> {
  using type = collapse_t<L<concat_t<Fn<Ts>...>>>;
};

template <template <class> class Outer, class T>
using apply_t = typename apply<Outer, T>::type;

}  // namespace tt


namespace detail
{
template <typename T>
struct ForTesting;

template <template <class...> class T, typename... Ts>
struct ForTesting<T<Ts...>> {
  using type = ::testing::Types<Ts...>;
};
}  // namespace detail

template <typename T>
using ForTesting = typename ::detail::ForTesting<T>::type;

#endif
