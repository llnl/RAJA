/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA RepeatView constructs.
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

#ifndef RAJA_REPEATVIEW_HPP
#define RAJA_REPEATVIEW_HPP

#include <cstddef>
#include <utility>
#include <type_traits>

#include "RAJA/util/macros.hpp"

namespace RAJA
{

/*!
 * @brief A view of a single object repeated a certain number of times.
 *
 * Creates a view or container object given an object and length.
 * Allows use of container interface functions if you want to repeat a
 * single object.
 *
 * For example:
 *
 *     // Create a repeat view object for the int 2 repeated int_len times
 *     RepeatView<int> int_repeated(2, int_len);
 *
 *     // Use with RAJA for_each
 *     RAJA::for_each(int_repeated, [&](int val) {
 *       sum += val;
 *     });
 *
 * Based on the std::ranges::repeat_view template.
 * Differs in that it does not support:
 *   compile time extents
 *   unbounded extents
 *
 */
template<typename T>
struct RepeatView
{
  struct iterator
  {
    using difference_type = std::ptrdiff_t;
    using value_type      = T;
    using reference       = value_type const&;

    iterator() = default;

    constexpr iterator(const T* base, size_t index)
        : m_value(base),
          m_index(index)
    {}

    constexpr reference operator*() const noexcept { return *m_value; }

    constexpr reference operator[](difference_type index) const noexcept
    {
      return *(*this + index);
    }

    constexpr iterator& operator++()
    {
      ++m_index;
      return *this;
    }

    constexpr iterator operator++(int)
    {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }

    constexpr iterator& operator--()
    {
      --m_index;
      return *this;
    }

    constexpr iterator operator--(int)
    {
      auto tmp = *this;
      --(*this);
      return tmp;
    }

    constexpr iterator& operator+=(difference_type rhs)
    {
      m_index += rhs;
      return *this;
    }

    constexpr iterator& operator-=(difference_type rhs)
    {
      m_index -= rhs;
      return *this;
    }

    friend constexpr iterator operator+(iterator lhs, difference_type rhs)
    {
      lhs += rhs;
      return lhs;
    }

    friend constexpr iterator operator+(difference_type lhs, iterator rhs)
    {
      rhs += lhs;
      return rhs;
    }

    friend constexpr iterator operator-(iterator lhs, difference_type rhs)
    {
      lhs -= rhs;
      return lhs;
    }

    friend constexpr difference_type operator-(iterator const& lhs,
                                               iterator const& rhs)
    {
      return static_cast<difference_type>(lhs.m_index) -
             static_cast<difference_type>(rhs.m_index);
    }

    friend constexpr bool operator==(iterator const& lhs, iterator const& rhs)
    {
      return lhs.m_index == rhs.m_index;
    }

    friend constexpr bool operator!=(iterator const& lhs, iterator const& rhs)
    {
      return !(lhs == rhs);
    }

    friend constexpr bool operator<(iterator const& lhs, iterator const& rhs)
    {
      return lhs.m_index < rhs.m_index;
    }

    friend constexpr bool operator<=(iterator const& lhs, iterator const& rhs)
    {
      return !(rhs < lhs);
    }

    friend constexpr bool operator>(iterator const& lhs, iterator const& rhs)
    {
      return rhs < lhs;
    }

    friend constexpr bool operator>=(iterator const& lhs, iterator const& rhs)
    {
      return !(lhs < rhs);
    }

  private:
    const T* m_value = nullptr;
    size_t m_index   = 0;
  };

  RepeatView() = delete;

  constexpr RepeatView(T const& value, size_t bound)
      : m_bound(bound),
        m_value(value)
  {}

  constexpr RepeatView(T&& value, size_t bound)
      : m_bound(bound),
        m_value(std::move(value))
  {}

  constexpr T const& front() const { return m_value; }

  constexpr T const& back() const { return m_value; }

  constexpr T const& operator[](size_t RAJA_UNUSED_ARG(index)) const
  {
    return m_value;
  }

  constexpr iterator begin() const { return iterator(&m_value, 0); }

  constexpr iterator cbegin() const { return iterator(&m_value, 0); }

  constexpr iterator end() const { return iterator(&m_value, m_bound); }

  constexpr iterator cend() const { return iterator(&m_value, m_bound); }

  constexpr explicit operator bool() const { return m_bound != 0; }

  constexpr bool empty() const { return m_bound == 0; }

  constexpr size_t size() const { return m_bound; }

private:
  size_t m_bound = 0;
  T m_value;
};

}  // end namespace RAJA

#endif /* RAJA_REPEATVIEW_HPP */
