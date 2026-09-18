/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA for_each templates.
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

#ifndef RAJA_util_for_each_HPP
#define RAJA_util_for_each_HPP

#include "RAJA/config.hpp"

#include <iterator>
#include <type_traits>

#include "camp/concepts.hpp"
#include "camp/list.hpp"
#include "camp/number.hpp"
#include "camp/tuple.hpp"

#include "RAJA/pattern/detail/algorithm.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

namespace detail
{

// compile-time expansion applying func to each index in order
RAJA_SUPPRESS_HD_WARN
template<typename UnaryFunc, camp::idx_t... Is>
constexpr RAJA_HOST_DEVICE RAJA_INLINE UnaryFunc
for_each_index(camp::idx_seq<Is...>, UnaryFunc func)
{
  // braced init lists are evaluated in order
  // create integral_constant type to allow UnaryFunc arguments to be used in
  // compile-time context
  int seq_unused_array[] = {
      0, (func(std::integral_constant<std::size_t, Is> {}), 0)...};
  RAJA_UNUSED_VAR(seq_unused_array);

  return func;
}

// runtime loop applying func to each element in the range in order
RAJA_SUPPRESS_HD_WARN
template<typename Iter, typename UnaryFunc>
constexpr RAJA_HOST_DEVICE RAJA_INLINE UnaryFunc for_each(Iter begin,
                                                          Iter end,
                                                          UnaryFunc func)
{
  for (; begin != end; ++begin)
  {
    func(*begin);
  }

  return func;
}

// compile time expansion applying func to a each type in the list in order
RAJA_SUPPRESS_HD_WARN
template<typename UnaryFunc, typename... Ts>
constexpr RAJA_HOST_DEVICE RAJA_INLINE UnaryFunc
for_each_type(camp::list<Ts...> const&, UnaryFunc func)
{
  // braced init lists are evaluated in order
  int seq_unused_array[] = {0, (func(Ts {}), 0)...};
  RAJA_UNUSED_VAR(seq_unused_array);

  return func;
}

// compile time expansion applying func to a each type in the tuple in order
RAJA_SUPPRESS_HD_WARN
template<typename Tuple, typename UnaryFunc, camp::idx_t... Is>
constexpr RAJA_HOST_DEVICE RAJA_INLINE UnaryFunc
for_each_tuple(Tuple&& t, UnaryFunc func, camp::idx_seq<Is...>)
{
  using camp::get;
  // braced init lists are evaluated in order
  int seq_unused_array[] = {0, (func(get<Is>(std::forward<Tuple>(t))), 0)...};
  RAJA_UNUSED_VAR(seq_unused_array);

  return func;
}

// compile-time expansion applying func to each tuple object and index in order
RAJA_SUPPRESS_HD_WARN
template<typename Tuple, typename BinaryFunc, camp::idx_t... Is>
constexpr RAJA_HOST_DEVICE RAJA_INLINE BinaryFunc
for_each_tuple_index(Tuple&& t, BinaryFunc func, camp::idx_seq<Is...>)
{
  using camp::get;
  // braced init lists are evaluated in order
  int seq_unused_array[] = {0,
                            (func(get<Is>(std::forward<Tuple>(t)),
                                  std::integral_constant<std::size_t, Is> {}),
                             0)...};
  RAJA_UNUSED_VAR(seq_unused_array);

  return func;
}

}  // namespace detail

/*!
  \brief Apply func to each index in [0, N) in order using a compile-time
  expansion in O(N) operations and O(1) extra memory
*/
RAJA_SUPPRESS_HD_WARN
template<size_t N, typename UnaryFunc>
constexpr RAJA_HOST_DEVICE RAJA_INLINE UnaryFunc for_each_index(UnaryFunc func)
{
  return detail::for_each_index(camp::make_idx_seq_t<N>(), std::move(func));
}

/*!
  \brief Apply func to all the elements in the given range in order
  using a sequential for loop in O(N) operations and O(1) extra memory
    see https://en.cppreference.com/w/cpp/algorithm/for_each
*/
RAJA_SUPPRESS_HD_WARN
template<typename Container, typename UnaryFunc>
constexpr RAJA_HOST_DEVICE RAJA_INLINE
    camp::concepts::enable_if_t<UnaryFunc,
                                camp::type_traits::is_range<Container>>
    for_each(Container&& c, UnaryFunc func)
{
  using std::begin;
  using std::end;

  return detail::for_each(begin(c), end(c), std::move(func));
}

/*!
  \brief Apply func to each type in the given list in order
  using a compile-time expansion in O(N) operations and O(1) extra memory
*/
RAJA_SUPPRESS_HD_WARN
template<typename UnaryFunc, typename... Ts>
constexpr RAJA_HOST_DEVICE RAJA_INLINE UnaryFunc
for_each_type(camp::list<Ts...> const& c, UnaryFunc func)
{
  return detail::for_each_type(c, std::move(func));
}

/*!
  \brief Apply func to each object in the given tuple or tuple like type in
  order using a compile-time expansion in O(N) operations and O(1) extra memory
*/
RAJA_SUPPRESS_HD_WARN
template<typename Tuple, typename UnaryFunc>
constexpr RAJA_HOST_DEVICE RAJA_INLINE UnaryFunc for_each_tuple(Tuple&& t,
                                                                UnaryFunc func)
{
  return detail::for_each_tuple(
      std::forward<Tuple>(t), std::move(func),
      camp::make_idx_seq_t<std::tuple_size<camp::decay<Tuple>>::value> {});
}

/*!
  \brief Apply func to each object in the given tuple or tuple like type as well
  as the index of the tuple element in order using a compile-time expansion in
  O(N) operations and O(1) extra memory
*/
RAJA_SUPPRESS_HD_WARN
template<typename Tuple, typename BinaryFunc>
constexpr RAJA_HOST_DEVICE RAJA_INLINE BinaryFunc
for_each_tuple_index(Tuple&& t, BinaryFunc func)
{
  return detail::for_each_tuple_index(
      std::forward<Tuple>(t), std::move(func),
      camp::make_idx_seq_t<std::tuple_size<camp::decay<Tuple>>::value> {});
}

}  // namespace RAJA

#endif
