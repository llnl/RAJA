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
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
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
      : m_storage(std::move(op), std::move(init))
  {}

  /*!
      \brief reset the combined value of the reducer to the identity
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr void reset(
      T init = BinaryOp::identity()) noexcept
  {
    m_storage.m_accumulated_value = std::move(init);
  }

  /*!
      \brief return the combined value and reset the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr T get_and_reset(
      T init = BinaryOp::identity())
  {
    T value = get();

    reset(std::move(init));

    return value;
  }

  /*!
      \brief return the combined value
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr T get() const
  {
    return m_storage.m_accumulated_value;
  }

  /*!
      \brief combine a value into the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr void combine(T value)
  {
    m_storage.m_accumulated_value = m_storage.get_op()(
        std::move(m_storage.m_accumulated_value), std::move(value));
  }

  /*!
      \brief combine a value into the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr void operator+=(T value)
  {
    combine(std::move(value));
  }

private:
  // use a struct derived from BinaryOp to avoid extra storage when BinaryOp
  // is an empty class
  struct Storage : BinaryOp
  {
    T m_accumulated_value;

    RAJA_HOST_DEVICE RAJA_INLINE constexpr Storage(BinaryOp op, T init)
        : BinaryOp(std::move(op)),
          m_accumulated_value(std::move(init))
    {}

    RAJA_HOST_DEVICE RAJA_INLINE constexpr BinaryOp& get_op() noexcept
    {
      return *this;
    }

    RAJA_HOST_DEVICE RAJA_INLINE constexpr BinaryOp const& get_op()
        const noexcept
    {
      return *this;
    }
  };

  Storage m_storage;
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
      : m_storage(std::move(op))
  {
    combine(std::move(init));
  }

  /*!
      \brief reset the combined value of the reducer to the identity
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr void reset(
      T init = BinaryOp::identity()) noexcept
  {
    m_storage.m_count = 0;

    combine(std::move(init));
  }

  /*!
      \brief return the combined value and reset the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr T get_and_reset(
      T init = BinaryOp::identity())
  {
    T value = get();

    reset(std::move(init));

    return value;
  }

  /*!
      \brief return the combined value
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr T get() const
  {
    // accumulate all values
    T value = BinaryOp::identity();

    for (SizeType count = m_storage.m_count, level = 0, mask = 1; count;
         ++level, mask <<= 1)
    {
      if (count & mask)
      {
        value =
            m_storage.get_op()(std::move(value), m_storage.m_tree_stack[level]);

        count ^= mask;
      }
    }

    return value;
  }

  /*!
      \brief combine a value into the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr void combine(T value)
  {
    // accumulate values and store in the first unused level found
    // reset values from used levels along the way
    SizeType level = 0;
    for (SizeType mask = 1; m_storage.m_count & mask; ++level, mask <<= 1)
    {
      value = m_storage.get_op()(std::move(m_storage.m_tree_stack[level]),
                                 std::move(value));
    }

    m_storage.m_tree_stack[level] = std::move(value);

    ++m_storage.m_count;
  }

  /*!
      \brief combine a value into the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr void operator+=(T value)
  {
    combine(std::move(value));
  }

private:
  // use a struct derived from BinaryOp to avoid extra storage when BinaryOp
  // is an empty class
  struct Storage : BinaryOp
  {
    // A counter of the number of inputs combined.
    // The bits of count indicate which levels of tree stack have a value
    SizeType m_count = 0;

    // Each level in tree stack has a value that holds the accumulation of
    // 2^level values or is unused and has no value.
    T m_tree_stack[num_levels];

    RAJA_HOST_DEVICE RAJA_INLINE constexpr Storage(BinaryOp op)
        : BinaryOp(std::move(op))
    {}

    RAJA_HOST_DEVICE RAJA_INLINE constexpr BinaryOp& get_op() noexcept
    {
      return *this;
    }

    RAJA_HOST_DEVICE RAJA_INLINE constexpr BinaryOp const& get_op()
        const noexcept
    {
      return *this;
    }
  };

  Storage m_storage;
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

  /*!
      \brief reset the combined value of the reducer to the identity
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr void reset(T init = T()) noexcept
  {
    m_accumulated_value = std::move(init);
    m_accumulated_carry = T();
  }

  /*!
      \brief return the combined value and reset the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr T get_and_reset(T init = T())
  {
    T value = get();

    reset(std::move(init));

    return value;
  }

  /*!
      \brief return the combined value
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr T get() const
  {
    return m_accumulated_value;
  }

  /*!
      \brief combine a value into the reducer
  */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr void combine(T val)
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
  RAJA_HOST_DEVICE RAJA_INLINE constexpr void operator+=(T val)
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
RAJA_HOST_DEVICE RAJA_INLINE constexpr concepts::
    enable_if_t<T, type_traits::is_range<Container>>
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

  return reducer.get_and_reset();
}

/*!
  \brief Reduce given range to a single value
  using a binary tree algorithm in O(N) operations and O(lg(N)) extra memory
    see https://en.cppreference.com/w/cpp/algorithm/reduce
*/
template<typename Container,
         typename T        = detail::ContainerVal<Container>,
         typename BinaryOp = operators::plus<T>>
RAJA_HOST_DEVICE RAJA_INLINE constexpr concepts::
    enable_if_t<T, type_traits::is_range<Container>>
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

  return reducer.get_and_reset();
}

/*!
  \brief Accumulate given range to a single value
  using a left fold algorithm in O(N) operations and O(1) extra memory
    see https://en.cppreference.com/w/cpp/algorithm/accumulate
*/
template<typename Container, typename T = detail::ContainerVal<Container>>
RAJA_HOST_DEVICE RAJA_INLINE constexpr concepts::
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

  return reducer.get_and_reset();
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
RAJA_HOST_DEVICE RAJA_INLINE constexpr concepts::
    enable_if_t<T, type_traits::is_range<Container>>
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

  return reducer.get_and_reset();
}

/*!
  \brief Accumulate given range to a single value
  using a left fold algorithm in O(N) operations and O(1) extra memory
    see https://en.cppreference.com/w/cpp/algorithm/accumulate
*/
template<typename Container,
         typename T        = detail::ContainerVal<Container>,
         typename BinaryOp = operators::plus<T>>
RAJA_HOST_DEVICE RAJA_INLINE constexpr concepts::
    enable_if_t<T, type_traits::is_range<Container>>
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

  return reducer.get_and_reset();
}

}  // namespace RAJA

#endif
