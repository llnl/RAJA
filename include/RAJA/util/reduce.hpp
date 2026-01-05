/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA sort templates.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_util_reduce_HPP
#define RAJA_util_reduce_HPP

#include "RAJA/config.hpp"

#include <climits>
#include <iterator>
#include <new>
#include <type_traits>

#include "RAJA/pattern/detail/algorithm.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/util/math.hpp"
#include "RAJA/util/Operators.hpp"

namespace RAJA
{

/*!
    \brief Reduce class that does a reduction with a left fold.
*/
template<typename T, typename BinaryOp>
struct LeftFoldReduce
{
  RAJA_HOST_DEVICE RAJA_INLINE constexpr explicit LeftFoldReduce(
      T init      = BinaryOp::identity(),
      BinaryOp op = BinaryOp {}) noexcept
      : m_op(std::move(op)),
        m_accumulated_value(std::move(init))
  {}

  LeftFoldReduce(LeftFoldReduce const&)            = delete;
  LeftFoldReduce& operator=(LeftFoldReduce const&) = delete;
  LeftFoldReduce(LeftFoldReduce&&)                 = delete;
  LeftFoldReduce& operator=(LeftFoldReduce&&)      = delete;

  ~LeftFoldReduce() = default;

  /*!
      \brief reset the combined value of the reducer to the identity
  */
  RAJA_HOST_DEVICE RAJA_INLINE void clear() noexcept
  {
    m_accumulated_value = BinaryOp::identity();
  }

  /*!
      \brief return the combined value and clear the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE T get_and_clear()
  {
    T accumulated_value = std::move(m_accumulated_value);

    clear();

    return accumulated_value;
  }

  /*!
      \brief return the combined value
  */
  RAJA_HOST_DEVICE RAJA_INLINE T get() { return m_accumulated_value; }

  /*!
      \brief combine a value into the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE void combine(T val)
  {
    m_accumulated_value = m_op(std::move(m_accumulated_value), std::move(val));
  }

  /*!
      \brief combine a value into the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE void operator+=(T val)
  {
    combine(std::move(val));
  }

private:
  BinaryOp m_op;
  T m_accumulated_value;
};

/*!
    \brief Reduce class that does a reduction with a binary tree.
*/
template<typename T,
         typename BinaryOp,
         typename SizeType     = size_t,
         SizeType t_num_levels = CHAR_BIT * sizeof(SizeType)>
struct BinaryTreeReduce
{
  static_assert(std::is_unsigned<SizeType>::value, "SizeType must be unsigned");
  static_assert(
      t_num_levels <= CHAR_BIT * sizeof(SizeType),
      "SizeType must be large enough to act at a bitset for num_levels");

  static constexpr SizeType num_levels = t_num_levels;

  RAJA_HOST_DEVICE RAJA_INLINE constexpr explicit BinaryTreeReduce(
      T init      = BinaryOp::identity(),
      BinaryOp op = BinaryOp {}) noexcept
      : m_op(std::move(op))
  {
    combine(std::move(init));
  }

  BinaryTreeReduce(BinaryTreeReduce const&)            = delete;
  BinaryTreeReduce& operator=(BinaryTreeReduce const&) = delete;
  BinaryTreeReduce(BinaryTreeReduce&&)                 = delete;
  BinaryTreeReduce& operator=(BinaryTreeReduce&&)      = delete;

  RAJA_HOST_DEVICE RAJA_INLINE ~BinaryTreeReduce() { clear(); }

  /*!
      \brief reset the combined value of the reducer to the identity
  */
  RAJA_HOST_DEVICE RAJA_INLINE void clear() noexcept
  {
    // destroy all values on the tree stack and reset count to 0
    for (SizeType level = 0, mask = 1; m_count; ++level, mask <<= 1)
    {

      if (m_count & mask)
      {

        get_value(level)->~T();

        m_count ^= mask;
      }
    }
  }

  /*!
      \brief return the combined value and clear the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE T get_and_clear()
  {
    // accumulate all values
    T value = BinaryOp::identity();

    for (SizeType level = 0, mask = 1; m_count; ++level, mask <<= 1)
    {

      if (m_count & mask)
      {

        value = m_op(std::move(value), std::move(*get_value(level)));
        get_value(level)->~T();

        m_count ^= mask;
      }
    }

    return value;
  }

  /*!
      \brief return the combined value
  */
  RAJA_HOST_DEVICE RAJA_INLINE T get()
  {
    // accumulate all values
    T value = BinaryOp::identity();

    for (SizeType count = m_count, level = 0, mask = 1; count;
         ++level, mask <<= 1)
    {

      if (count & mask)
      {

        value = m_op(std::move(value), *get_value(level));

        count ^= mask;
      }
    }

    return value;
  }

  /*!
      \brief combine a value into the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE void combine(T value)
  {
    // accumulate values and store in the first unused level found
    // clear values from used levels along the way
    SizeType level = 0;
    for (SizeType mask = 1; m_count & mask; ++level, mask <<= 1)
    {

      value = m_op(std::move(*get_value(level)), std::move(value));
      get_value(level)->~T();
    }

    new (get_storage(level)) T(std::move(value));

    ++m_count;
  }

  /*!
      \brief combine a value into the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE void operator+=(T val)
  {
    combine(std::move(val));
  }

private:
  BinaryOp m_op;

  // A counter of the number of inputs combined.
  // The bits of count indicate which levels of tree stack have a value
  SizeType m_count = 0;

  // Each level in tree stack has a value that holds the accumulation of 2^level
  // values or is unused and has no value.
  std::aligned_storage_t<sizeof(T), alignof(T)> m_tree_stack[num_levels];

  RAJA_HOST_DEVICE RAJA_INLINE void* get_storage(SizeType level)
  {
    return &m_tree_stack[level];
  }

  RAJA_HOST_DEVICE RAJA_INLINE T* get_value(SizeType level)
  {
#if __cplusplus >= 201703L && !defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
    // TODO: check that launder is supported in device code
    return std::launder(reinterpret_cast<T*>(&m_tree_stack[level]));
#else
    return reinterpret_cast<T*>(&m_tree_stack[level]);
#endif
  }
};

/*!
    \brief Reduce class that does a reduction with a left fold.

    \note KahanSum does not take an binary operation as the only valid operation
          is plus.
*/
template<typename T>
struct KahanSum
{
  static_assert(std::is_floating_point_v<T>, "T must be a floating point type");

  RAJA_HOST_DEVICE RAJA_INLINE constexpr explicit KahanSum(
      T init = T()) noexcept
      : m_accumulated_value(std::move(init)),
        m_accumulated_carry(T())
  {}

  KahanSum(KahanSum const&)            = delete;
  KahanSum& operator=(KahanSum const&) = delete;
  KahanSum(KahanSum&&)                 = delete;
  KahanSum& operator=(KahanSum&&)      = delete;

  ~KahanSum() = default;

  /*!
      \brief reset the combined value of the reducer to the identity
  */
  RAJA_HOST_DEVICE RAJA_INLINE void clear() noexcept
  {
    m_accumulated_value = T();
    m_accumulated_carry = T();
  }

  /*!
      \brief return the combined value and clear the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE T get_and_clear()
  {
    T accumulated_value = std::move(m_accumulated_value);

    clear();

    return accumulated_value;
  }

  /*!
      \brief return the combined value
  */
  RAJA_HOST_DEVICE RAJA_INLINE T get() { return m_accumulated_value; }

  /*!
      \brief combine a value into the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE void combine(T val)
  {
    // volatile used to prevent compiler optimizations that assume
    // floating-point operations are associative
    T y                 = val - m_accumulated_carry;
    volatile T t        = m_accumulated_value + y;
    volatile T z        = t - m_accumulated_value;
    m_accumulated_carry = z - y;
    m_accumulated_value = t;
  }

  /*!
      \brief combine a value into the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE void operator+=(T val)
  {
    combine(std::move(val));
  }

private:
  T m_accumulated_value;
  T m_accumulated_carry;
};

template<typename T, typename BinaryOp>
using HighAccuracyReduce =
    std::conditional_t<RAJA::operators::is_fp_associative<T>::value,
                       BinaryTreeReduce<T, BinaryOp>,
                       LeftFoldReduce<T, BinaryOp>>;

/*!
  \brief Accumulate given range to a single value
  using a left fold algorithm in O(N) operations and O(1) extra memory
    see https://en.cppreference.com/w/cpp/algorithm/accumulate
*/
template<typename Container,
         typename T        = detail::ContainerVal<Container>,
         typename BinaryOp = operators::plus<T>>
RAJA_HOST_DEVICE RAJA_INLINE
    concepts::enable_if_t<T, type_traits::is_range<Container>>
    left_fold_reduce(Container&& c,
                     T init      = BinaryOp::identity(),
                     BinaryOp op = BinaryOp {})
{
  using std::begin;
  using std::end;
  static_assert(type_traits::is_binary_function<BinaryOp, T, T, T>::value,
                "BinaryOp must model BinaryFunction");

  auto begin_it = begin(c);
  auto end_it   = end(c);

  LeftFoldReduce<T, BinaryOp> reducer(std::move(init), std::move(op));

  for (; begin_it != end_it; ++begin_it)
  {
    reducer.combine(*begin_it);
  }

  return reducer.get_and_clear();
}

/*!
  \brief Reduce given range to a single value
  using a binary tree algorithm in O(N) operations and O(lg(N)) extra memory
    see https://en.cppreference.com/w/cpp/algorithm/reduce
*/
template<typename Container,
         typename T        = detail::ContainerVal<Container>,
         typename BinaryOp = operators::plus<T>>
RAJA_HOST_DEVICE RAJA_INLINE
    concepts::enable_if_t<T, type_traits::is_range<Container>>
    binary_tree_reduce(Container&& c,
                       T init      = BinaryOp::identity(),
                       BinaryOp op = BinaryOp {})
{
  using std::begin;
  using std::distance;
  using std::end;
  static_assert(type_traits::is_binary_function<BinaryOp, T, T, T>::value,
                "BinaryOp must model BinaryFunction");

  auto begin_it  = begin(c);
  auto end_it    = end(c);
  using SizeType = std::make_unsigned_t<decltype(distance(begin_it, end_it))>;

  BinaryTreeReduce<T, BinaryOp, SizeType> reducer(std::move(init),
                                                  std::move(op));

  for (; begin_it != end_it; ++begin_it)
  {
    reducer.combine(*begin_it);
  }

  return reducer.get_and_clear();
}

/*!
  \brief Accumulate given range to a single value
  using a left fold algorithm in O(N) operations and O(1) extra memory
    see https://en.cppreference.com/w/cpp/algorithm/accumulate
*/
template<typename Container, typename T = detail::ContainerVal<Container>>
RAJA_HOST_DEVICE RAJA_INLINE concepts::
    enable_if_t<T, type_traits::is_range<Container>, std::is_floating_point<T>>
    kahan_sum(Container&& c, T init = T())
{
  using std::begin;
  using std::end;

  auto begin_it = begin(c);
  auto end_it   = end(c);

  KahanSum<T> reducer(std::move(init));

  for (; begin_it != end_it; ++begin_it)
  {
    reducer.combine(*begin_it);
  }

  return reducer.get_and_clear();
}

/*!
  \brief Reduce given range to a single value
  using an algorithm with high accuracy when floating point round off is a
  concern
    see https://en.cppreference.com/w/cpp/algorithm/reduce
*/
template<typename Container,
         typename T        = detail::ContainerVal<Container>,
         typename BinaryOp = operators::plus<T>>
RAJA_HOST_DEVICE RAJA_INLINE
    concepts::enable_if_t<T, type_traits::is_range<Container>>
    high_accuracy_reduce(Container&& c,
                         T init      = BinaryOp::identity(),
                         BinaryOp op = BinaryOp {})
{
  using std::begin;
  using std::end;
  static_assert(type_traits::is_binary_function<BinaryOp, T, T, T>::value,
                "BinaryOp must model BinaryFunction");

  auto begin_it = begin(c);
  auto end_it   = end(c);

  HighAccuracyReduce<T, BinaryOp> reducer(std::move(init), std::move(op));

  for (; begin_it != end_it; ++begin_it)
  {
    reducer.combine(*begin_it);
  }

  return reducer.get_and_clear();
}

/*!
  \brief Accumulate given range to a single value
  using a left fold algorithm in O(N) operations and O(1) extra memory
    see https://en.cppreference.com/w/cpp/algorithm/accumulate
*/
template<typename Container,
         typename T        = detail::ContainerVal<Container>,
         typename BinaryOp = operators::plus<T>>
RAJA_HOST_DEVICE RAJA_INLINE
    concepts::enable_if_t<T, type_traits::is_range<Container>>
    accumulate(Container&& c,
               T init      = BinaryOp::identity(),
               BinaryOp op = BinaryOp {})
{
  using std::begin;
  using std::end;
  static_assert(type_traits::is_binary_function<BinaryOp, T, T, T>::value,
                "BinaryOp must model BinaryFunction");

  auto begin_it = begin(c);
  auto end_it   = end(c);

  LeftFoldReduce<T, BinaryOp> reducer(std::move(init), std::move(op));

  for (; begin_it != end_it; ++begin_it)
  {
    reducer.combine(*begin_it);
  }

  return reducer.get_and_clear();
}

}  // namespace RAJA

#endif
