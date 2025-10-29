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


#ifndef RAJA_util_zip_HPP
#define RAJA_util_zip_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/detail/algorithm.hpp"
#include "RAJA/util/camp_aliases.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/util/zip_tuple.hpp"
#include "RAJA/util/Span.hpp"

namespace RAJA
{

/*!
    \brief ZipIterator class for simultaneously iterating over
    multiple iterators. This is not a standards compliant iterator.
*/
template<typename... Iters>
struct ZipIterator
{
  static_assert(
      concepts::all_of<type_traits::is_random_access_iterator<Iters>...>::value,
      "ZipIterator can only contain random access iterators");
  static_assert(sizeof...(Iters) > 1,
                "ZipIterator must contain one or more iterators");

  using value_type =
      zip_val<typename std::iterator_traits<Iters>::value_type...>;
  using difference_type = std::ptrdiff_t;
  using pointer         = void;
  using reference = zip_ref<typename std::iterator_traits<Iters>::reference...>;
  using creference =
      zip_ref<const typename std::iterator_traits<Iters>::reference...>;
  using iterator_category = std::random_access_iterator_tag;

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator() : m_iterators() {}

  template<typename... Args,
           typename = concepts::enable_if<
               type_traits::convertible_to<Args&&, Iters>...>>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator(Args&&... args)
      : m_iterators(std::forward<Args>(args)...)
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator(const ZipIterator& rhs)
      : m_iterators(rhs.m_iterators)
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator(ZipIterator&& rhs)
      : m_iterators(std::move(rhs.m_iterators))
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator& operator=(
      const ZipIterator& rhs)
  {
    m_iterators = rhs.m_iterators;
    return *this;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator& operator=(
      ZipIterator&& rhs)
  {
    m_iterators = std::move(rhs.m_iterators);
    return *this;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr difference_type get_stride() const
  {
    return 1;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr bool operator==(
      const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) == RAJA::get<0>(rhs.m_iterators);
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr bool operator!=(
      const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) != RAJA::get<0>(rhs.m_iterators);
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr bool operator>(
      const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) > RAJA::get<0>(rhs.m_iterators);
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr bool operator<(
      const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) < RAJA::get<0>(rhs.m_iterators);
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr bool operator>=(
      const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) >= RAJA::get<0>(rhs.m_iterators);
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr bool operator<=(
      const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) <= RAJA::get<0>(rhs.m_iterators);
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator& operator++()
  {
    detail::zip_for_each(m_iterators, detail::PreInc {});
    return *this;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator& operator--()
  {
    detail::zip_for_each(m_iterators, detail::PreDec {});
    return *this;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator operator++(int)
  {
    ZipIterator tmp(*this);
    ++(*this);
    return tmp;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator operator--(int)
  {
    ZipIterator tmp(*this);
    --(*this);
    return tmp;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator& operator+=(
      const difference_type& rhs)
  {
    detail::zip_for_each(m_iterators, detail::PlusEq<difference_type> {rhs});
    return *this;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator& operator-=(
      const difference_type& rhs)
  {
    detail::zip_for_each(m_iterators, detail::MinusEq<difference_type> {rhs});
    return *this;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr difference_type operator-(
      const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) - RAJA::get<0>(rhs.m_iterators);
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator operator+(
      const difference_type& rhs) const
  {
    ZipIterator tmp(*this);
    tmp += rhs;
    return tmp;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ZipIterator operator-(
      const difference_type& rhs) const
  {
    ZipIterator tmp(*this);
    tmp -= rhs;
    return tmp;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr friend ZipIterator operator+(
      difference_type lhs,
      const ZipIterator& rhs)
  {
    ZipIterator tmp(rhs);
    tmp += lhs;
    return tmp;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr reference operator*() const
  {
    return deref_helper(camp::make_idx_seq_t<sizeof...(Iters)> {});
  }

  // TODO:: figure out what to do with this
  // RAJA_HOST_DEVICE RAJA_INLINE constexpr reference operator->() const
  // {
  //   return *(*this);
  // }
  RAJA_HOST_DEVICE RAJA_INLINE constexpr reference operator[](
      difference_type rhs) const
  {
    return *((*this) + rhs);
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr friend void safe_iter_swap(
      ZipIterator lhs,
      ZipIterator rhs)
  {
    detail::zip_for_each(lhs.m_iterators, rhs.m_iterators, detail::IterSwap {});
  }

private:
  zip_val<camp::decay<Iters>...> m_iterators;

  template<camp::idx_t... Is>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr reference deref_helper(
      camp::idx_seq<Is...>) const
  {
    return reference(*RAJA::get<Is>(m_iterators)...);
  }
};

/*!
    \brief Zip multiple iterators together to iterate them simultaneously with
    a single ZipIterator object.
*/
template<typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto zip(Args&&... args)
    -> ZipIterator<camp::decay<Args>...>
{
  return {std::forward<Args>(args)...};
}

/*!
    \brief Zip multiple containers together to iterate them simultaneously with
    ZipIterator objects.
*/
template<typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto zip_span(Args&&... args)
    -> Span<ZipIterator<detail::ContainerIter<camp::decay<Args>>...>,
            typename ZipIterator<
                detail::ContainerIter<camp::decay<Args>>...>::difference_type>
{
  using std::begin;
  using std::end;
  return Span<ZipIterator<detail::ContainerIter<camp::decay<Args>>...>,
              typename ZipIterator<detail::ContainerIter<
                  camp::decay<Args>>...>::difference_type>(
      zip(begin(std::forward<Args>(args))...),
      zip(end(std::forward<Args>(args))...));
}

/*!
    \brief Comparator object that compares the first member
    of tuple like objects.
*/
template<typename T, typename Compare>
struct CompareFirst
{
  RAJA_HOST_DEVICE RAJA_INLINE constexpr CompareFirst(Compare comp_)
      : comp(comp_)
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr bool operator()(T const& lhs,
                                                         T const& rhs) const
  {
    return comp(RAJA::get<0>(lhs), RAJA::get<0>(rhs));
  }

private:
  Compare comp;
};

/*!
    \brief Make a comparator to compare first member of tuple
    like objects of type T.
*/
template<typename T, typename Compare>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto compare_first(Compare comp)
    -> CompareFirst<T, Compare>
{
  return {comp};
}

}  // end namespace RAJA

#endif
