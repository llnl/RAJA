/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA span constructs.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_SPAN_HPP
#define RAJA_SPAN_HPP

#include <type_traits>

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/macros.hpp"

namespace RAJA
{

/*!
 * @brief A view to a sequence of objects.
 *
 * Creates a view or container object given a random access iterator and length.
 * Allows use of container interface functions using iterators.
 * Indices are of the type of the second template parameter.
 *
 * For example:
 *
 *     // Create a span object for an array of ints
 *     Span<int*, int> int_span(int_ptr, int_len);
 *
 *     // Use with RAJA sort
 *     RAJA::sort<policy>(int_span);
 *
 *     // Create a span object another way
 *     auto double_span = make_span(double_ptr, double_len);
 *
 *     // Use with RAJA scan
 *     RAJA::inclusive_scan_inplace<policy>(double_span);
 *
 * Based on the std::span template.
 * Differs in that it supports:
 *   random access instead of contiguous iterators
 *   different index types
 * and does not support:
 *   compile time extents
 *
 */
template<typename IterType, typename IndexType>
struct Span
{
  using element_type    = typename std::iterator_traits<IterType>::value_type;
  using value_type      = camp::decay<element_type>;
  using size_type       = IndexType;
  using difference_type = std::ptrdiff_t;
  using reference       = element_type&;
  using const_reference = const element_type&;
  using iterator        = IterType;
  using const_iterator  = IterType;

  static_assert(type_traits::is_integral<IndexType>::value,
                "IndexType must model Integral");
  static_assert(type_traits::is_random_access_iterator<IterType>::value,
                "IterType must model RandomAccessIterator");

  constexpr RAJA_HOST_DEVICE Span(iterator begin, iterator end)
      : m_begin {begin},
        m_end {end}
  {}

  constexpr RAJA_HOST_DEVICE Span(iterator begin, size_type size)
      : m_begin {begin},
        m_end {begin + size}
  {}

  constexpr RAJA_HOST_DEVICE RAJA_INLINE iterator begin() { return m_begin; }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE iterator end() { return m_end; }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE const_iterator begin() const
  {
    return m_begin;
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE const_iterator end() const
  {
    return m_end;
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE const_iterator cbegin() const
  {
    return m_begin;
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE const_iterator cend() const
  {
    return m_end;
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE friend iterator begin(Span& s)
  {
    return s.begin();
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE friend iterator end(Span& s)
  {
    return s.end();
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE friend const_iterator begin(
      const Span& s)
  {
    return s.begin();
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE friend const_iterator end(
      const Span& s)
  {
    return s.end();
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE friend const_iterator cbegin(
      const Span& s)
  {
    return s.cbegin();
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE friend const_iterator cend(
      const Span& s)
  {
    return s.cend();
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE reference front() const
  {
    return *begin();
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE reference back() const
  {
    return *(end() - 1);
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE reference operator[](size_type i) const
  {
    return data()[i];
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE iterator data() const
  {
    return m_begin;
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE size_type size() const
  {
    return static_cast<size_type>(m_end - m_begin);
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE bool empty() const
  {
    return size() == static_cast<size_type>(0);
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE Span first(size_type count) const
  {
    return slice(0, count);
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE Span last(size_type count) const
  {
    return slice(size() - count, count);
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE Span subspan(size_type begin,
                                                      size_type length) const
  {
    return slice(begin, length);
  }

  constexpr RAJA_HOST_DEVICE RAJA_INLINE Span slice(size_type begin,
                                                    size_type length) const
  {
    auto start = m_begin + begin;
    auto end   = start + length > m_end ? m_end : start + length;
    return Span(start, end);
  }

private:
  iterator m_begin;
  iterator m_end;
};

/*!
 * @brief Creates a span from a random access iterator and length.
 * @param begin beginning of the sequence being spanned
 * @param size length of the sequence being spanned
 * @return Returns a Span representing the given sequence
 *
 * Creates a span object given a random access iterator and length.
 *
 * For example:
 *
 *     // the span type will have IndexType size_t
 *     size_t len = ...;
 *
 *     // Create a span object
 *     auto my_span = make_span(begin, len);
 *
 *     // Use with RAJA scan
 *     RAJA::inclusive_scan_inplace<policy>(my_span);
 *
 */
template<typename IterType, typename IndexType>
constexpr RAJA_HOST_DEVICE RAJA_INLINE Span<IterType, IndexType> make_span(
    IterType begin,
    IndexType size)
{
  return Span<IterType, IndexType>(begin, size);
}

template<typename Iter>
constexpr RAJA_HOST_DEVICE RAJA_INLINE auto make_span(Iter& iterable)
{
  using std::begin;
  using std::distance;
  using std::end;
  return Span<typename Iter::iterator,
              decltype(distance(begin(iterable), end(iterable)))>(
      begin(iterable), end(iterable));
}

}  // end namespace RAJA

#endif /* RAJA_SPAN_HPP */
