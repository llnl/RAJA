/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for multi-iterator Zip Views.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_util_zip_ref_HPP
#define RAJA_util_zip_ref_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/detail/algorithm.hpp"
#include "RAJA/util/camp_aliases.hpp"
#include "RAJA/util/concepts.hpp"

namespace RAJA
{

template<bool is_val, typename... Ts>
struct zip_tuple;

template<camp::idx_t I, typename ZT>
struct zip_tuple_element;

template<camp::idx_t I, bool is_val, typename... Ts>
struct zip_tuple_element<I, zip_tuple<is_val, Ts...>>
    : camp::tuple_element<I, typename zip_tuple<is_val, Ts...>::value_type>
{};

template<camp::idx_t I, typename ZT>
using zip_tuple_element_t = typename zip_tuple_element<I, ZT>::type;

// get function declarations for zip_tuple
// the reference type returned by get depends on the reference type
// of the zip_tuple that get is called on
template<camp::idx_t I, bool is_val, typename... Ts>
RAJA_HOST_DEVICE RAJA_INLINE constexpr RAJA::
    zip_tuple_element_t<I, zip_tuple<is_val, Ts...>>&
    get(zip_tuple<is_val, Ts...>& z) noexcept
{
  return z.template get<I>();
}

template<camp::idx_t I, bool is_val, typename... Ts>
RAJA_HOST_DEVICE RAJA_INLINE constexpr RAJA::
    zip_tuple_element_t<I, zip_tuple<is_val, Ts...>> const&
    get(zip_tuple<is_val, Ts...> const& z) noexcept
{
  return z.template get<I>();
}

template<camp::idx_t I, bool is_val, typename... Ts>
RAJA_HOST_DEVICE RAJA_INLINE constexpr std::remove_reference_t<
    RAJA::zip_tuple_element_t<I, zip_tuple<is_val, Ts...>>>&&
get(zip_tuple<is_val, Ts...>&& z) noexcept
{
  return std::move(z).template get<I>();
}

template<camp::idx_t I, bool is_val, typename... Ts>
RAJA_HOST_DEVICE RAJA_INLINE constexpr std::remove_reference_t<
    RAJA::zip_tuple_element_t<I, zip_tuple<is_val, Ts...>>> const&&
get(zip_tuple<is_val, Ts...> const&& z) noexcept
{
  return std::move(z).template get<I>();
}

namespace detail
{

struct PassThrough
{
  template<typename T>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr auto operator()(T&& t) const
      -> decltype(std::forward<T>(t))
  {
    return std::forward<T>(t);
  }
};

struct Move
{
  template<typename T>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr auto operator()(T&& t) const
      -> decltype(std::move(t))
  {
    return std::move(t);
  }
};

struct PreInc
{
  template<typename Iter>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr auto operator()(Iter&& iter) const
      -> decltype(++std::forward<Iter>(iter))
  {
    return ++std::forward<Iter>(iter);
  }
};

struct PreDec
{
  template<typename Iter>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr auto operator()(Iter&& iter) const
      -> decltype(--std::forward<Iter>(iter))
  {
    return --std::forward<Iter>(iter);
  }
};

template<typename difference_type>
struct PlusEq
{
  const difference_type& rhs;

  template<typename Iter>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr auto operator()(Iter&& iter) const
      -> decltype(std::forward<Iter>(iter) += rhs)
  {
    return std::forward<Iter>(iter) += rhs;
  }
};

template<typename difference_type>
struct MinusEq
{
  const difference_type& rhs;

  template<typename Iter>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr auto operator()(Iter&& iter) const
      -> decltype(std::forward<Iter>(iter) -= rhs)
  {
    return std::forward<Iter>(iter) -= rhs;
  }
};

struct DeRef
{
  template<typename Iter>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr auto operator()(Iter&& iter) const
      -> decltype(*std::forward<Iter>(iter))
  {
    return *std::forward<Iter>(iter);
  }
};

struct Swap
{
  template<typename T0, typename T1>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr int operator()(T0&& t0, T1&& t1) const
  {
    using camp::safe_swap;
    safe_swap(std::forward<T0>(t0), std::forward<T1>(t1));
    return 1;
  }
};

struct IterSwap
{
  template<typename T0, typename T1>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr int operator()(T0&& t0, T1&& t1) const
  {
    using RAJA::safe_iter_swap;
    safe_iter_swap(std::forward<T0>(t0), std::forward<T1>(t1));
    return 1;
  }
};

/*!
    \brief Call f on each member of t (f(t)...).
*/
template<typename Tuple, typename F, camp::idx_t... Is>
RAJA_HOST_DEVICE RAJA_INLINE constexpr void zip_for_each_impl(
    Tuple&& t,
    F&& f,
    camp::idx_seq<Is...>)
{
  camp::sink(std::forward<F>(f)(RAJA::get<Is>(std::forward<Tuple>(t)))...);
}

/*!
    \brief Call f on each member of t0 and t1 (f(t0, t1)...).
*/
template<typename Tuple0, typename Tuple1, typename F, camp::idx_t... Is>
RAJA_HOST_DEVICE RAJA_INLINE constexpr void zip_for_each_impl(
    Tuple0&& t0,
    Tuple1&& t1,
    F&& f,
    camp::idx_seq<Is...>)
{
  camp::sink(std::forward<F>(f)(RAJA::get<Is>(std::forward<Tuple0>(t0)),
                                RAJA::get<Is>(std::forward<Tuple1>(t1)))...);
}

/*!
    \brief Call f on each member of t (f(t)...).
*/
template<typename Tuple, typename F>
RAJA_HOST_DEVICE RAJA_INLINE constexpr void zip_for_each(Tuple&& t, F&& f)
{
  zip_for_each_impl(std::forward<Tuple>(t), std::forward<F>(f),
                    typename camp::decay<Tuple>::IdxSeq {});
}

/*!
    \brief Call f on each member of t0 and t1 (f(t0, t1)...).
*/
template<typename Tuple0, typename Tuple1, typename F>
RAJA_HOST_DEVICE RAJA_INLINE constexpr void zip_for_each(Tuple0&& t0,
                                                         Tuple1&& t1,
                                                         F&& f)
{
  static_assert(std::is_same<typename camp::decay<Tuple0>::IdxSeq,
                             typename camp::decay<Tuple1>::IdxSeq>::value,
                "Tuple0 and Tuple1 must have the same size");
  zip_for_each_impl(std::forward<Tuple0>(t0), std::forward<Tuple1>(t1),
                    std::forward<F>(f),
                    typename camp::decay<Tuple0>::IdxSeq {});
}

}  // end namespace detail

/*!
    \brief Tuple used by ZipIterator for storing multiple references and values.
    Acts like a reference to its members allowing copy/move
   construction/assignment based on the reference type of the zip_tuple.
*/
template<bool is_val, typename... Ts>
struct zip_tuple
{
  using value_type = RAJA::tuple<Ts...>;

  template<typename T>
  using opp_type =
      typename std::conditional<is_val,
                                typename std::add_lvalue_reference<T>::type,
                                typename std::remove_reference<T>::type>::type;

  // zip_tuple type with opposite is_val
  using opp_tuple = zip_tuple<!is_val, opp_type<Ts>...>;

  // camp::idx_seq for this type, also used by zip_for_each
  using IdxSeq = camp::make_idx_seq_t<sizeof...(Ts)>;

  // constructor from types convertible to Ts
  template<
      typename... Os,
      typename = concepts::enable_if<type_traits::convertible_to<Os&&, Ts>...>>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple(Os&&... os)
      : m_tuple(std::forward<Os>(os)...)
  {}

  // assignment from types convertible to Ts
  template<typename... Os,
           typename = concepts::enable_if<type_traits::convertible_to<
               Os&&,
               typename std::remove_reference<Ts>::type>...>>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& assign(Os&&... os)
  {
    return assign_helper(IdxSeq {}, std::forward<Os>(os)...);
  }

  // copy and move constructors
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple(zip_tuple& o)
      : zip_tuple(o, IdxSeq {})
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple(zip_tuple const& o)
      : zip_tuple(o, IdxSeq {})
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple(zip_tuple&& o)
      : zip_tuple(std::move(o), IdxSeq {})
  {}  // move if is_val, pass-through otherwise

  // copy and move assignment operators
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& operator=(zip_tuple& o)
  {
    return assign_helper(o, IdxSeq {});
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& operator=(
      zip_tuple const& o)
  {
    return assign_helper(o, IdxSeq {});
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& operator=(zip_tuple&& o)
  {
    return assign_helper(std::move(o), IdxSeq {});
  }

  // copy and move constructors from opp_tuple type zip_tuples
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple(opp_tuple& o)
      : zip_tuple(o, IdxSeq {})
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple(opp_tuple const& o)
      : zip_tuple(o, IdxSeq {})
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple(opp_tuple&& o)
      : zip_tuple(std::move(o), IdxSeq {})
  {}  // move if is_val, pass-through otherwise

  // copy and move assignment operators from opp_tuple type zip_tuples
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& operator=(opp_tuple& o)
  {
    return assign_helper(o, IdxSeq {});
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& operator=(
      opp_tuple const& o)
  {
    return assign_helper(o, IdxSeq {});
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& operator=(opp_tuple&& o)
  {
    return assign_helper(std::move(o), IdxSeq {});
  }

  // get member functions for zip_tuples
  // the reference type returned by get depends on the reference type
  // of the zip_tuple that get is called on
  template<camp::idx_t I>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr RAJA::tuple_element_t<I, value_type>&
  get() & noexcept
  {
    return RAJA::get<I>(m_tuple);
  }

  template<camp::idx_t I>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr RAJA::
      tuple_element_t<I, value_type> const&
      get() const& noexcept
  {
    return RAJA::get<I>(m_tuple);
  }

  template<camp::idx_t I>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr std::remove_reference_t<
      RAJA::tuple_element_t<I, value_type>>&&
  get() && noexcept
  {
    return std::move(RAJA::get<I>(m_tuple));
  }

  template<camp::idx_t I>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr std::remove_reference_t<
      RAJA::tuple_element_t<I, value_type>> const&&
  get() const&& noexcept
  {
    return std::move(RAJA::get<I>(m_tuple));
  }

  // safe_swap that calls swap on each pair in the tuple
  RAJA_HOST_DEVICE RAJA_INLINE constexpr friend void safe_swap(zip_tuple& lhs,
                                                               zip_tuple& rhs)
  {
    detail::zip_for_each(lhs, rhs, detail::Swap {});
  }

  // safe_swap for swapping zip_tuples with opposite is_val
  // calls swap on each pair in the tuple
  RAJA_HOST_DEVICE RAJA_INLINE constexpr friend void safe_swap(zip_tuple& lhs,
                                                               opp_tuple& rhs)
  {
    detail::zip_for_each(lhs, rhs, detail::Swap {});
  }

  // allow printing of zip_tuples by printing value_type
  friend inline std::ostream& operator<<(std::ostream& o, zip_tuple const& v)
  {
    return o << v.m_tuple;
  }

private:
  // move if is_val is true, otherwise copy in move constructor
  // this allows values to be moved, and references to stay lvalue references
  using IsValMover = typename std::
      conditional<is_val, detail::Move, detail::PassThrough>::type;

  value_type m_tuple;

  // assignment helper from types convertible to Ts
  template<typename... Os, camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& assign_helper(
      camp::idx_seq<Is...>,
      Os&&... os)
  {
    camp::sink(get<Is>() = std::forward<Os>(os)...);
    return *this;
  }

  // copy and move constructor helpers
  template<camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple(zip_tuple& o,
                                                   camp::idx_seq<Is...>)
      : zip_tuple(RAJA::get<Is>(o)...)
  {}

  template<camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple(zip_tuple const& o,
                                                   camp::idx_seq<Is...>)
      : zip_tuple(RAJA::get<Is>(o)...)
  {}

  template<camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple(zip_tuple&& o,
                                                   camp::idx_seq<Is...>)
      : zip_tuple(RAJA::get<Is>(IsValMover {}(o))...)
  {}  // move if is_val, pass-through otherwise

  // copy and move assignment operator helpers
  template<camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& assign_helper(
      zip_tuple& o,
      camp::idx_seq<Is...>)
  {
    if (this != &o)
    {
      camp::sink(get<Is>() = RAJA::get<Is>(o)...);
    }
    return *this;
  }

  template<camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& assign_helper(
      zip_tuple const& o,
      camp::idx_seq<Is...>)
  {
    if (this != &o)
    {
      camp::sink(get<Is>() = RAJA::get<Is>(o)...);
    }
    return *this;
  }

  template<camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& assign_helper(
      zip_tuple&& o,
      camp::idx_seq<Is...>)
  {
    if (this != &o)
    {
      camp::sink(get<Is>() = RAJA::get<Is>(std::move(o))...);
    }
    return *this;
  }

  // copy and move constructor helpers from opp_tuple type zip_tuples
  template<camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple(opp_tuple& o,
                                                   camp::idx_seq<Is...>)
      : zip_tuple(RAJA::get<Is>(o)...)
  {}

  template<camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple(opp_tuple const& o,
                                                   camp::idx_seq<Is...>)
      : zip_tuple(RAJA::get<Is>(o)...)
  {}

  template<camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple(opp_tuple&& o,
                                                   camp::idx_seq<Is...>)
      : zip_tuple(RAJA::get<Is>(IsValMover {}(o))...)
  {}  // move if is_val, pass-through otherwise

  // copy and move assignment operator helpers from opp_tuple type zip_tuples
  template<camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& assign_helper(
      opp_tuple& o,
      camp::idx_seq<Is...>)
  {
    camp::sink(get<Is>() = RAJA::get<Is>(o)...);
    return *this;
  }

  template<camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& assign_helper(
      opp_tuple const& o,
      camp::idx_seq<Is...>)
  {
    camp::sink(get<Is>() = RAJA::get<Is>(o)...);
    return *this;
  }

  template<camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr zip_tuple& assign_helper(
      opp_tuple&& o,
      camp::idx_seq<Is...>)
  {
    camp::sink(get<Is>() = RAJA::get<Is>(std::move(o))...);
    return *this;
  }
};

// alias zip_ref to zip_tuple capable of storing references (!is_val)
template<typename... Ts>
using zip_ref = zip_tuple<false, Ts...>;

// alias zip_val to zip_tuple suitable for storing values (is_val)
template<typename... Ts>
using zip_val = zip_tuple<true, Ts...>;

}  // end namespace RAJA

#endif
