/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing functionality similar to std mutex header.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_util_mutex_HPP
#define RAJA_util_mutex_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)
#include <omp.h>
#endif

namespace RAJA
{

#if defined(RAJA_ENABLE_OPENMP)
namespace omp
{

//! class wrapping omp_lock_t with std::mutex interface
class mutex
{
public:
  using native_handle_type = omp_lock_t;

  mutex() { omp_init_lock(&m_lock); }

  mutex(const mutex&)            = delete;
  mutex(mutex&&)                 = delete;
  mutex& operator=(const mutex&) = delete;
  mutex& operator=(mutex&&)      = delete;

  void lock() { omp_set_lock(&m_lock); }

  bool try_lock() { return omp_test_lock(&m_lock) != 0; }

  void unlock() { omp_unset_lock(&m_lock); }

  native_handle_type& native_handle() { return m_lock; }

  ~mutex() { omp_destroy_lock(&m_lock); }

private:
  native_handle_type m_lock;
};

}  // namespace omp
#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP)

//! class providing functionality of std::lock_guard
template<typename mutex_type>
class lock_guard
{
public:
  explicit lock_guard(mutex_type& m) : m_mutex(m) { m_mutex.lock(); }

  lock_guard(const lock_guard&)            = delete;
  lock_guard(lock_guard&&)                 = delete;
  lock_guard& operator=(const lock_guard&) = delete;
  lock_guard& operator=(lock_guard&&)      = delete;

  ~lock_guard() { m_mutex.unlock(); }

private:
  mutex_type& m_mutex;
};

}  // namespace RAJA

#endif  // closing endif for header file include guard
