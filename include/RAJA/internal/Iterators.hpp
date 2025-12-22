/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA iterator constructs.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_ITERATORS_HPP
#define RAJA_ITERATORS_HPP

#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "RAJA/config.hpp"
#include "RAJA/index/IndexValue.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{
namespace Iterators
{

// Containers

#if defined(RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG)
template<typename LType, typename RType>
std::string overflow_msg(LType lhs, RType rhs)
{
  return "Iterator Overflow detected between operation of :\n\ttype : " +
         (std::string) typeid(lhs).name() + " val : " + std::to_string(lhs) +
         "\n\ttype : " + typeid(rhs).name() + " val : " + std::to_string(rhs) +
         "\n";
}

template<typename Type, typename DifferenceType>
RAJA_HOST_DEVICE bool is_addition_overflow(Type lhs, DifferenceType rhs)
{
  if (std::is_unsigned<Type>::value)
  {
    if ((rhs > 0) && (lhs > std::numeric_limits<Type>::max() - rhs))
      return true;
    if ((rhs < 0) && (lhs < std::numeric_limits<Type>::min() - rhs))
      return true;
  }
  return false;
}

template<typename Type, typename DifferenceType>
RAJA_HOST_DEVICE bool is_subtraction_overflow(Type lhs,
                                              DifferenceType rhs,
                                              bool iterator_on_left = true)
{
  if (iterator_on_left)
  {

    if (std::is_unsigned<Type>::value)
    {
      if ((rhs > 0) && (lhs < std::numeric_limits<Type>::min() + rhs))
        return true;
      if ((rhs < 0) && (lhs > std::numeric_limits<Type>::max() + rhs))
        return true;
    }
  }
  else
  {  // Special case where operation is : value(lhs) - iterator(rhs).

    if (std::is_unsigned<DifferenceType>::value)
    {
      if ((lhs > 0) && (rhs < std::numeric_limits<DifferenceType>::min() + lhs))
        return true;
      if ((lhs < 0)) return true;
    }
  }
  return false;
}

template<typename Type, typename DifferenceType>
RAJA_HOST_DEVICE void check_is_addition_overflow(Type lhs, DifferenceType rhs)
{
  if (is_addition_overflow(lhs, rhs))
    throw std::runtime_error(overflow_msg(lhs, rhs));
}

template<typename Type, typename DifferenceType>
RAJA_HOST_DEVICE void check_is_subtraction_overflow(Type lhs,
                                                    DifferenceType rhs)
{
  if (is_subtraction_overflow(lhs, rhs))
    throw std::runtime_error(overflow_msg(lhs, rhs));
}
#endif

template<typename Type           = Index_type,
         typename DifferenceType = Type,
         typename PointerType    = Type*>
class numeric_iterator
{
public:
  using value_type          = Type;
  using stripped_value_type = strip_index_type_t<Type>;
  using difference_type     = DifferenceType;
  using pointer             = PointerType;
  using reference           = value_type&;
  using iterator_category   = std::random_access_iterator_tag;

  constexpr numeric_iterator() noexcept                         = default;
  constexpr numeric_iterator(const numeric_iterator&) noexcept  = default;
  constexpr numeric_iterator(numeric_iterator&&) noexcept       = default;
  numeric_iterator& operator=(const numeric_iterator&) noexcept = default;
  numeric_iterator& operator=(numeric_iterator&&) noexcept      = default;

  RAJA_HOST_DEVICE constexpr numeric_iterator(const stripped_value_type& rhs)
      : val(rhs)
  {}

  RAJA_HOST_DEVICE inline DifferenceType get_stride() const { return 1; }

  RAJA_HOST_DEVICE inline bool operator==(const numeric_iterator& rhs) const
  {
    return val == rhs.val;
  }

  RAJA_HOST_DEVICE inline bool operator!=(const numeric_iterator& rhs) const
  {
    return val != rhs.val;
  }

  RAJA_HOST_DEVICE inline bool operator>(const numeric_iterator& rhs) const
  {
    return val > rhs.val;
  }

  RAJA_HOST_DEVICE inline bool operator<(const numeric_iterator& rhs) const
  {
    return val < rhs.val;
  }

  RAJA_HOST_DEVICE inline bool operator>=(const numeric_iterator& rhs) const
  {
    return val >= rhs.val;
  }

  RAJA_HOST_DEVICE inline bool operator<=(const numeric_iterator& rhs) const
  {
    return val <= rhs.val;
  }

  RAJA_HOST_DEVICE inline numeric_iterator& operator++()
  {
    ++val;
    return *this;
  }

  RAJA_HOST_DEVICE inline numeric_iterator& operator--()
  {
    --val;
    return *this;
  }

  RAJA_HOST_DEVICE inline numeric_iterator operator++(int)
  {
    numeric_iterator tmp(*this);
    ++val;
    return tmp;
  }

  RAJA_HOST_DEVICE inline numeric_iterator operator--(int)
  {
    numeric_iterator tmp(*this);
    --val;
    return tmp;
  }

  RAJA_HOST_DEVICE inline numeric_iterator& operator+=(
      const difference_type& rhs)
  {
#if defined(RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG)
    check_is_addition_overflow(val, rhs);
#endif
    val += rhs;
    return *this;
  }

  RAJA_HOST_DEVICE inline numeric_iterator& operator-=(
      const difference_type& rhs)
  {
#if defined(RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG)
    check_is_subtraction_overflow(val, rhs);
#endif
    val -= rhs;
    return *this;
  }

  RAJA_HOST_DEVICE inline numeric_iterator& operator+=(
      const numeric_iterator& rhs)
  {
    val += rhs.val;
    return *this;
  }

  RAJA_HOST_DEVICE inline numeric_iterator& operator-=(
      const numeric_iterator& rhs)
  {
    val -= rhs.val;
    return *this;
  }

  RAJA_HOST_DEVICE inline stripped_value_type operator+(
      const numeric_iterator& rhs) const
  {
    return val + rhs.val;
  }

  RAJA_HOST_DEVICE inline stripped_value_type operator-(
      const numeric_iterator& rhs) const
  {
    return val - rhs.val;
  }

  RAJA_HOST_DEVICE inline numeric_iterator operator+(
      const difference_type& rhs) const
  {
#if defined(RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG)
    check_is_addition_overflow(val, rhs);
#endif
    return numeric_iterator(val + rhs);
  }

  RAJA_HOST_DEVICE inline numeric_iterator operator-(
      const difference_type& rhs) const
  {
#if defined(RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG)
    check_is_subtraction_overflow(val, rhs);
#endif
    return numeric_iterator(val - rhs);
  }

  RAJA_HOST_DEVICE friend constexpr numeric_iterator operator+(
      difference_type lhs,
      const numeric_iterator& rhs)
  {
#if defined(RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG)
    return is_addition_overflow(rhs.val, lhs)
               ? throw std::runtime_error(overflow_msg(lhs, rhs.val))
               : numeric_iterator(lhs + rhs.val);
#else
    return numeric_iterator(lhs + rhs.val);
#endif
  }

  RAJA_HOST_DEVICE friend constexpr numeric_iterator operator-(
      difference_type lhs,
      const numeric_iterator& rhs)
  {
#if defined(RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG)
    return is_subtraction_overflow(rhs.val, lhs, false)
               ? throw std::runtime_error(overflow_msg(lhs, rhs.val))
               : numeric_iterator(lhs - rhs.val);
#else
    return numeric_iterator(lhs - rhs.val);
#endif
  }

  RAJA_HOST_DEVICE inline value_type operator*() const
  {
    return value_type(val);
  }

  RAJA_HOST_DEVICE inline value_type operator->() const
  {
    return value_type(val);
  }

  RAJA_HOST_DEVICE constexpr value_type operator[](difference_type rhs) const
  {
    return value_type(val + rhs);
  }

private:
  stripped_value_type val = 0;
};

template<typename Type           = Index_type,
         typename DifferenceType = Type,
         typename PointerType    = Type*>
class strided_numeric_iterator
{
public:
  using value_type          = Type;
  using stripped_value_type = strip_index_type_t<Type>;
  using difference_type     = DifferenceType;
  using pointer             = DifferenceType*;
  using reference           = DifferenceType&;
  using iterator_category   = std::random_access_iterator_tag;

  constexpr strided_numeric_iterator() noexcept = default;
  constexpr strided_numeric_iterator(const strided_numeric_iterator&) noexcept =
      default;
  constexpr strided_numeric_iterator(strided_numeric_iterator&&) noexcept =
      default;
  strided_numeric_iterator& operator=(
      const strided_numeric_iterator&) noexcept = default;
  strided_numeric_iterator& operator=(strided_numeric_iterator&&) noexcept =
      default;

  RAJA_HOST_DEVICE constexpr strided_numeric_iterator(
      stripped_value_type rhs,
      DifferenceType stride_ = DifferenceType(1))
      : val(rhs),
        stride(stride_)
  {}

  RAJA_HOST_DEVICE inline DifferenceType get_stride() const { return stride; }

  RAJA_HOST_DEVICE inline strided_numeric_iterator& operator++()
  {
    val += stride;
    return *this;
  }

  RAJA_HOST_DEVICE inline strided_numeric_iterator& operator--()
  {
    val -= stride;
    return *this;
  }

  RAJA_HOST_DEVICE inline strided_numeric_iterator& operator+=(
      const difference_type& rhs)
  {
#if defined(RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG)
    check_is_addition_overflow(val, rhs * stride);
#endif
    val += rhs * stride;
    return *this;
  }

  RAJA_HOST_DEVICE inline strided_numeric_iterator& operator-=(
      const difference_type& rhs)
  {
#if defined(RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG)
    check_is_subtraction_overflow(val, rhs * stride);
#endif
    val -= rhs * stride;
    return *this;
  }

  RAJA_HOST_DEVICE inline difference_type operator+(
      const strided_numeric_iterator& rhs) const
  {
    return (static_cast<difference_type>(val) +
            (static_cast<difference_type>(rhs.val))) /
           stride;
  }

  RAJA_HOST_DEVICE inline difference_type operator-(
      const strided_numeric_iterator& rhs) const
  {
    difference_type diff = (static_cast<difference_type>(val) -
                            (static_cast<difference_type>(rhs.val)));

    return (diff % stride != difference_type {0})
               ? (difference_type {1} + diff / stride)
               : diff / stride;
  }

  RAJA_HOST_DEVICE inline strided_numeric_iterator operator+(
      const difference_type& rhs) const
  {
#if defined(RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG)
    check_is_addition_overflow(val, rhs * stride);
#endif
    return strided_numeric_iterator(val + rhs * stride, stride);
  }

  RAJA_HOST_DEVICE inline strided_numeric_iterator operator-(
      const difference_type& rhs) const
  {
#if defined(RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG)
    check_is_subtraction_overflow(val, rhs * stride);
#endif
    return strided_numeric_iterator(val - rhs * stride, stride);
  }

  // Specialized comparison to allow normal iteration to work on off-stride
  // multiples by adjusting rhs to the nearest *higher* multiple of stride
  RAJA_HOST_DEVICE inline bool operator!=(
      const strided_numeric_iterator& rhs) const
  {
    return (val - rhs.val) / stride;
  }

  RAJA_HOST_DEVICE inline bool operator==(
      const strided_numeric_iterator& rhs) const
  {
    return !((val - rhs.val) / stride);
  }

  RAJA_HOST_DEVICE inline bool operator>(
      const strided_numeric_iterator& rhs) const
  {
    return val * stride > rhs.val * stride;
  }

  RAJA_HOST_DEVICE inline bool operator<(
      const strided_numeric_iterator& rhs) const
  {
    return val * stride < rhs.val * stride;
  }

  RAJA_HOST_DEVICE inline bool operator>=(
      const strided_numeric_iterator& rhs) const
  {
    return val * stride >= rhs.val * stride;
  }

  RAJA_HOST_DEVICE inline bool operator<=(
      const strided_numeric_iterator& rhs) const
  {
    return val * stride <= rhs.val * stride;
  }

  RAJA_HOST_DEVICE inline value_type operator*() const
  {
    return value_type(val);
  }

  RAJA_HOST_DEVICE inline value_type operator->() const
  {
    return value_type(val);
  }

  RAJA_HOST_DEVICE constexpr value_type operator[](difference_type rhs) const
  {
    return value_type(val + rhs * stride);
  }

private:
  stripped_value_type val = 0;
  DifferenceType stride   = 1;
};


}  // namespace Iterators

}  // namespace RAJA

#endif /* RAJA_ITERATORS_HPP */
