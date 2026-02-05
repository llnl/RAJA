/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Internal memory functions and classes to manage memory for
 *          CPU reductions and other operations.
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

#ifndef RAJA_MemUtils_CPU_HPP
#define RAJA_MemUtils_CPU_HPP

#include "RAJA/config.hpp"

#include <cstddef>
#include <cstdlib>
#include <memory>

#include "RAJA/util/types.hpp"

#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) ||                \
    defined(__MINGW32__) || defined(__BORLANDC__)
#define RAJA_PLATFORM_WINDOWS
#include <malloc.h>
#endif

namespace RAJA
{

/*!
 * Typed aligned memory allocation
 *
 * Convenience function to allocate aligned memory using the
 * std::aligned_alloc function but returns a pointer of the allocated
 * type rather than void*
 *
 * Memory should be deallocated using standard C++ free.
 */
template<typename T>
inline T* allocate_aligned_type(size_t alignment, size_t size)
{
  return reinterpret_cast<T*>(std::aligned_alloc(alignment, size));
}

/*!
 * Deleter function object for memory allocated with std allocation
 * methods.
 *
 * Can be used with the RAJA::allocate_aligned_type function.
 */
struct FreeAligned
{
  void operator()(void* ptr) { free(ptr); }
};

/*!
 * Deleter function object for memory allocated with
 * allocate_aligned_type that calls the destructor for the first `size`
 * objects in the storage.
 */
template<typename T, typename index_type>
struct FreeAlignedType : FreeAligned
{
  index_type size = 0;

  void operator()(T* ptr)
  {
    for (index_type i = size; i > 0; --i)
    {
      ptr[i - 1].~T();
    }
    FreeAligned::operator()(ptr);
  }
};

}  // namespace RAJA

#endif  // closing endif for header file include guard
