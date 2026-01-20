/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for CPU reductions and other operations.
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

///
/// Portable aligned memory allocation
///
inline void* allocate_aligned(size_t alignment, size_t size)
{
#if defined(RAJA_HAVE_POSIX_MEMALIGN)
  // posix_memalign available
  void* ret = nullptr;
  int err   = posix_memalign(&ret, alignment, size);
  return err ? nullptr : ret;
#elif defined(RAJA_HAVE_ALIGNED_ALLOC)
  return std::aligned_alloc(alignment, size);
#elif defined(RAJA_HAVE_MM_MALLOC)
  return _mm_malloc(size, alignment);
#elif defined(RAJA_PLATFORM_WINDOWS)
  return _aligned_malloc(size, alignment);
#else
  char* mem = (char*)malloc(size + alignment + sizeof(void*));
  if (nullptr == mem) return nullptr;
  void** ptr = (void**)((std::uintptr_t)(mem + alignment + sizeof(void*)) &
                        ~(alignment - 1));
  // Store the original address one position behind what we give the user.
  ptr[-1] = mem;
  return ptr;
#endif
}

///
/// Portable aligned memory allocation
///
template<typename T>
inline T* allocate_aligned_type(size_t alignment, size_t size)
{
  return reinterpret_cast<T*>(allocate_aligned(alignment, size));
}

///
/// Portable aligned memory free - required for Windows
///
inline void free_aligned(void* ptr)
{
#if defined(RAJA_HAVE_POSIX_MEMALIGN) || defined(RAJA_HAVE_ALIGNED_ALLOC)
  free(ptr);
#elif defined(RAJA_HAVE_MM_MALLOC)
  _mm_free(ptr);
#elif defined(RAJA_PLATFORM_WINDOWS)
  _aligned_free(ptr);
#else
  // Free the address stored one position behind the user data in ptr.
  // This is valid for pointers allocated with allocate_aligned
  free(((void**)ptr)[-1]);
#endif
}

///
/// Deleter function object for memory allocated with allocate_aligned
///
struct FreeAligned
{
  void operator()(void* ptr) { free_aligned(ptr); }
};

///
/// Deleter function object for memory allocated with allocate_aligned_type
/// that calls the destructor for the fist size objects in the storage.
///
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
