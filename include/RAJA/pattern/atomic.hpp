/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining atomic operations.
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

#ifndef RAJA_pattern_atomic_HPP
#define RAJA_pattern_atomic_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/atomic_auto.hpp"
#include "RAJA/policy/atomic_builtin.hpp"

#include "RAJA/util/macros.hpp"

namespace RAJA
{


/*!
 * \file
 * Atomic operation functions in the namespace RAJA::atomic
 *
 * The dispatch of all of these is:
 *
 * T atomicAdd<Policy>(T *acc, T value)      -- User facing API
 *
 * calls
 *
 * T atomicAdd(Policy{}, T *acc, T value)    -- Policy specific implementation
 *
 *
 * With the exception of the auto_atomic policy which then calls the
 * "appropriate" policy implementation.
 *
 *
 * Current supported policies include:
 *
 *   auto_atomic       -- Attempts to do "the right thing"
 *
 *   cuda_atomic       -- Only atomic supported in CUDA device functions
 *
 *   omp_atomic        -- Available (and default) when OpenMP is enabled
 *                        these are safe inside and outside of OMP parallel
 *                        regions
 *
 *   builtin_atomic    -- Use the (nonstandard) __sync_fetch_and_XXX functions
 *
 *   seq_atomic        -- Non-atomic, does an unprotected (raw) operation
 *
 *
 * Current supported data types include:
 *
 *   32-bit and 64-bit integral types:
 *      -Native atomic support for "unsigned", "long", "unsigned long long" and
 *      "long long"
 *
 *      -General support, via CAS algorithm, for any 32-bit or 64-bit datatype
 *
 *   32-bit and 64-bit floating point types:  float and double
 *
 *
 * The implementation code lives in:
 * RAJA/policy/atomic_auto.hpp     -- for auto_atomic
 * RAJA/policy/atomic_builtin.hpp  -- for builtin_atomic
 * RAJA/policy/XXX/atomic.hpp      -- for omp_atomic, cuda_atomic, etc.
 *
 */


/*!
 * @brief Atomic load
 * @param acc Pointer to location of value
 * @return Value at acc
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicLoad(T* acc)
{
  return RAJA::atomicLoad(Policy {}, acc);
}

/*!
 * @brief Atomic store
 * @param acc Pointer to location of value
 * @param value Value to store at *acc
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE void atomicStore(T* acc, T value)
{
  RAJA::atomicStore(Policy {}, acc, value);
}

/*!
 * @brief Atomic add
 * @param acc Pointer to location of result value
 * @param value Value to add to *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicAdd(T* acc, T value)
{
  return RAJA::atomicAdd(Policy {}, acc, value);
}

/*!
 * @brief Atomic subtract
 * @param acc Pointer to location of result value
 * @param value Value to subtract from *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicSub(T* acc, T value)
{
  return RAJA::atomicSub(Policy {}, acc, value);
}

/*!
 * @brief Atomic minimum equivalent to (*acc) = std::min(*acc, value)
 * @param acc Pointer to location of result value
 * @param value Value to compare to *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicMin(T* acc, T value)
{
  return RAJA::atomicMin(Policy {}, acc, value);
}

/*!
 * @brief Atomic maximum equivalent to (*acc) = std::max(*acc, value)
 * @param acc Pointer to location of result value
 * @param value Value to compare to *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicMax(T* acc, T value)
{
  return RAJA::atomicMax(Policy {}, acc, value);
}

/*!
 * @brief Atomic increment
 * @param acc Pointer to location of value to increment
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicInc(T* acc)
{
  return RAJA::atomicInc(Policy {}, acc);
}

/*!
 * @brief Atomic increment with bound
 * Equivalent to *acc = ((*acc >= compare) ? 0 : ((*acc)+1))
 * This is for compatability with the CUDA atomicInc.
 * @param acc Pointer to location of value to increment
 * @param compare Bound value
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicInc(T* acc, T compare)
{
  return RAJA::atomicInc(Policy {}, acc, compare);
}

/*!
 * @brief Atomic decrement
 * @param acc Pointer to location of value to decrement
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicDec(T* acc)
{
  return RAJA::atomicDec(Policy {}, acc);
}

/*!
 * @brief Atomic decrement with bound
 * Equivalent to *acc = (((*acc==0)|(*acc>compare))?compare:((*acc)-1))
 * This is for compatability with the CUDA atomicDec.
 * @param acc Pointer to location of value to decrement
 * @param compare Bound value
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicDec(T* acc, T compare)
{
  return RAJA::atomicDec(Policy {}, acc, compare);
}

/*!
 * @brief Atomic bitwise AND equivalent to (*acc) = (*acc) & value
 * This only works with integral data types
 * @param acc Pointer to location of result value
 * @param value Value to bitwise AND with *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicAnd(T* acc, T value)
{
  static_assert(std::is_integral<T>::value,
                "atomicAnd can only be used on integral types");
  return RAJA::atomicAnd(Policy {}, acc, value);
}

/*!
 * @brief Atomic bitwise OR equivalent to (*acc) = (*acc) | value
 * This only works with integral data types
 * @param acc Pointer to location of result value
 * @param value Value to bitwise OR with *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicOr(T* acc, T value)
{
  static_assert(std::is_integral<T>::value,
                "atomicOr can only be used on integral types");
  return RAJA::atomicOr(Policy {}, acc, value);
}

/*!
 * @brief Atomic bitwise XOR equivalent to (*acc) = (*acc) ^ value
 * This only works with integral data types
 * @param acc Pointer to location of result value
 * @param value Value to bitwise XOR with *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicXor(T* acc, T value)
{
  static_assert(std::is_integral<T>::value,
                "atomicXor can only be used on integral types");
  return RAJA::atomicXor(Policy {}, acc, value);
}

/*!
 * @brief Atomic value exchange
 * @param acc Pointer to location to store value
 * @param value Value to exchange with *acc
 * @return Returns value at *acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicExchange(T* acc, T value)
{
  return RAJA::atomicExchange(Policy {}, acc, value);
}

/*!
 * @brief Atomic compare and swap
 * @param acc Pointer to location to store value
 * @param value Value to exchange with *acc
 * @param compare Value to compare with *acc
 * @return Returns value at *acc immediately before this operation completed
 */

RAJA_SUPPRESS_HD_WARN
template<typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicCAS(T* acc, T compare, T value)
{
  return RAJA::atomicCAS(Policy {}, acc, compare, value);
}

/*!
 * \brief Atomic wrapper object
 *
 * Provides an interface akin to that provided by std::atomic, but for an
 * arbitrary memory location.
 *
 * This object provides an OO interface to the global function calls provided
 * as RAJA::atomicXXX
 */
template<typename T, typename Policy = auto_atomic>
class AtomicRef
{
public:
  using value_type = T;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr explicit AtomicRef(value_type* value_ptr) : m_value_ptr(value_ptr)
  {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr AtomicRef(AtomicRef const& c) : m_value_ptr(c.m_value_ptr) {}

  AtomicRef& operator=(AtomicRef const&) = delete;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type* getPointer() const { return m_value_ptr; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  void store(value_type rhs) const
  {
    RAJA::atomicStore<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type operator=(value_type rhs) const
  {
    RAJA::atomicStore<Policy>(m_value_ptr, rhs);
    return rhs;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type load() const { return RAJA::atomicLoad<Policy>(m_value_ptr); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  operator value_type() const { return RAJA::atomicLoad<Policy>(m_value_ptr); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type exchange(value_type rhs) const
  {
    return RAJA::atomicExchange<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type CAS(value_type compare, value_type rhs) const
  {
    return RAJA::atomicCAS<Policy>(m_value_ptr, compare, rhs);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  bool compare_exchange_strong(value_type& expect, value_type rhs) const
  {
    value_type compare = expect;
    value_type old     = RAJA::atomicCAS<Policy>(m_value_ptr, compare, rhs);
    if (compare == old)
    {
      return true;
    }
    else
    {
      expect = old;
      return false;
    }
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  bool compare_exchange_weak(value_type& expect, value_type rhs) const
  {
    return this->compare_exchange_strong(expect, rhs);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type operator++() const
  {
    return RAJA::atomicInc<Policy>(m_value_ptr) + 1;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type operator++(int) const
  {
    return RAJA::atomicInc<Policy>(m_value_ptr);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type operator--() const
  {
    return RAJA::atomicDec<Policy>(m_value_ptr) - 1;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type operator--(int) const
  {
    return RAJA::atomicDec<Policy>(m_value_ptr);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type fetch_add(value_type rhs) const
  {
    return RAJA::atomicAdd<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type operator+=(value_type rhs) const
  {
    return RAJA::atomicAdd<Policy>(m_value_ptr, rhs) + rhs;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type fetch_sub(value_type rhs) const
  {
    return RAJA::atomicSub<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type operator-=(value_type rhs) const
  {
    return RAJA::atomicSub<Policy>(m_value_ptr, rhs) - rhs;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type fetch_min(value_type rhs) const
  {
    return RAJA::atomicMin<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type min(value_type rhs) const
  {
    value_type old = RAJA::atomicMin<Policy>(m_value_ptr, rhs);
    return old < rhs ? old : rhs;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type fetch_max(value_type rhs) const
  {
    return RAJA::atomicMax<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type max(value_type rhs) const
  {
    value_type old = RAJA::atomicMax<Policy>(m_value_ptr, rhs);
    return old > rhs ? old : rhs;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type fetch_and(value_type rhs) const
  {
    return RAJA::atomicAnd<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type operator&=(value_type rhs) const
  {
    return RAJA::atomicAnd<Policy>(m_value_ptr, rhs) & rhs;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type fetch_or(value_type rhs) const
  {
    return RAJA::atomicOr<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type operator|=(value_type rhs) const
  {
    return RAJA::atomicOr<Policy>(m_value_ptr, rhs) | rhs;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type fetch_xor(value_type rhs) const
  {
    return RAJA::atomicXor<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  value_type operator^=(value_type rhs) const
  {
    return RAJA::atomicXor<Policy>(m_value_ptr, rhs) ^ rhs;
  }

private:
  value_type* m_value_ptr;
};


}  // namespace RAJA

#endif
