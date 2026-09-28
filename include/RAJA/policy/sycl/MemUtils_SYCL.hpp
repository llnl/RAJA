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

#include <cstddef>

#include "RAJA/util/basic_mempool.hpp"

#include "camp/resource/sycl.hpp"

namespace RAJA
{

namespace sycl
{

//! Allocator for pinned memory for use in basic_mempool
struct PinnedAllocator
{

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    auto resource    = ::camp::resources::Sycl::get_default();
    ::sycl::queue& q = resource.get_queue();
    ptr              = ::sycl::malloc_host(nbytes, q);
    return ptr;
  }

  // returns true on success
  // Will throw if ptr is not in q's context
  bool free(void* ptr)
  {
    auto resource    = ::camp::resources::Sycl::get_default();
    ::sycl::queue& q = resource.get_queue();
    ::sycl::free(ptr, q);
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
    auto resource    = ::camp::resources::Sycl::get_default();
    ::sycl::queue& q = resource.get_queue();
    ptr              = ::sycl::malloc_device(nbytes, q);
    return ptr;
  }

  // returns true on success
  // Will throw if ptr is not in q's context
  bool free(void* ptr)
  {
    auto resource    = ::camp::resources::Sycl::get_default();
    ::sycl::queue& q = resource.get_queue();
    ::sycl::free(ptr, q);
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
    auto resource    = ::camp::resources::Sycl::get_default();
    ::sycl::queue& q = resource.get_queue();
    ptr              = ::sycl::malloc_device(nbytes, q);
    q.memset(ptr, 0, nbytes);
    return ptr;
  }

  // Returns true on success
  // Will throw if ptr is not in q's context
  bool free(void* ptr)
  {
    auto resource    = ::camp::resources::Sycl::get_default();
    ::sycl::queue& q = resource.get_queue();
    ::sycl::free(ptr, q);
    return true;
  }
};

using device_mempool_type = basic_mempool::MemPool<DeviceAllocator>;
using device_zeroed_mempool_type =
    basic_mempool::MemPool<DeviceZeroedAllocator>;
using pinned_mempool_type = basic_mempool::MemPool<PinnedAllocator>;

inline size_t release_unused_internal_memory()
{
  size_t released = 0;
  released += device_mempool_type::getInstance().release_unused();
  released += device_zeroed_mempool_type::getInstance().release_unused();
  released += pinned_mempool_type::getInstance().release_unused();
  return released;
}

}  // namespace sycl

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_SYCL

#endif  // closing endif for header file include guard
