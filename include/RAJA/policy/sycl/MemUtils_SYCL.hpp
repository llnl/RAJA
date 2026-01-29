/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for SYCL reductions and other operations.
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

#ifndef RAJA_MemUtils_SYCL_HPP
#define RAJA_MemUtils_SYCL_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "RAJA/util/sycl_compat.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <type_traits>
#include <unordered_map>


#include "RAJA/util/basic_mempool.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/sycl/policy.hpp"

namespace RAJA
{

namespace sycl
{

namespace detail
{

//! struct containing data necessary to coordinate kernel launches with reducers
struct syclInfo
{
  sycl_dim_t gridDim {0};
  sycl_dim_t blockDim {0};
  ::sycl::queue qu    = ::sycl::queue();
  bool setup_reducers = false;
};

extern syclInfo g_status;

thread_local extern syclInfo tl_status;

extern std::unordered_map<::sycl::queue, bool> g_queue_info_map;

}  // namespace detail

//! Allocator for pinned memory for use in basic_mempool
struct PinnedAllocator
{

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    ::sycl::queue* q = ::camp::resources::Sycl::get_default().get_queue();
    ptr              = ::sycl::malloc_host(nbytes, *q);
    return ptr;
  }

  // returns true on success
  // Will throw if ptr is not in q's context
  bool free(void* ptr)
  {
    ::sycl::queue* q = ::camp::resources::Sycl::get_default().get_queue();
    ::sycl::free(ptr, *q);
    return true;
  }
};

//! Allocator for device memory for use in basic_mempool
struct DeviceAllocator
{

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    ::sycl::queue* q = ::camp::resources::Sycl::get_default().get_queue();
    ptr              = ::sycl::malloc_device(nbytes, *q);
    return ptr;
  }

  // returns true on success
  // Will throw if ptr is not in q's context
  bool free(void* ptr)
  {
    ::sycl::queue* q = ::camp::resources::Sycl::get_default().get_queue();
    ::sycl::free(ptr, *q);
    return true;
  }
};

//! Allocator for pre-zeroed device memory for use in basic_mempool
//  Note: Memory must be zero when returned to mempool
struct DeviceZeroedAllocator
{

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    ::sycl::queue* q = ::camp::resources::Sycl::get_default().get_queue();
    ptr              = ::sycl::malloc_device(nbytes, *q);
    q->memset(ptr, 0, nbytes);
    return ptr;
  }

  // Returns true on success
  // Will throw if ptr is not in q's context
  bool free(void* ptr)
  {
    ::sycl::queue* q = ::camp::resources::Sycl::get_default().get_queue();
    ::sycl::free(ptr, *q);
    return true;
  }
};

using device_mempool_type = basic_mempool::MemPool<DeviceAllocator>;
using device_zeroed_mempool_type =
    basic_mempool::MemPool<DeviceZeroedAllocator>;
using pinned_mempool_type = basic_mempool::MemPool<PinnedAllocator>;

}  // namespace sycl

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_SYCL

#endif  // closing endif for header file include guard
