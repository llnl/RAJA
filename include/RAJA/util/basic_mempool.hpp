/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing an implementation of a memory pool.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_BASIC_MEMPOOL_HPP
#define RAJA_BASIC_MEMPOOL_HPP

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>

#include "RAJA/util/align.hpp"
#include "RAJA/util/mutex.hpp"

namespace RAJA
{

namespace basic_mempool
{

namespace detail
{


/*! \class MemoryArena
 ******************************************************************************
 *
 * \brief  MemoryArena is a map based subclass for class MemPool
 * provides book-keeping to divy a large chunk of pre-allocated memory to avoid
 * the overhead of  malloc/free or cudaMalloc/cudaFree, etc
 *
 * get/give are the primary calls used by class MemPool to get aligned memory
 * from the pool or give it back
 *
 *
 ******************************************************************************
 */
class MemoryArena
{
public:
  using free_type       = std::map<void*, void*>;
  using free_value_type = typename free_type::value_type;
  using used_type       = std::map<void*, void*>;
  using used_value_type = typename used_type::value_type;

  MemoryArena(void* ptr, size_t size)
      : m_allocation {ptr, static_cast<char*>(ptr) + size},
        m_free_space(),
        m_used_space()
  {
    m_free_space[ptr] = static_cast<char*>(ptr) + size;
    if (m_allocation.begin == nullptr)
    {
      fprintf(stderr, "Attempt to create MemoryArena with no memory");
      std::abort();
    }
  }

  MemoryArena(MemoryArena const&)            = delete;
  MemoryArena& operator=(MemoryArena const&) = delete;

  MemoryArena(MemoryArena&&)            = default;
  MemoryArena& operator=(MemoryArena&&) = default;

  size_t capacity()
  {
    return static_cast<char*>(m_allocation.end) -
           static_cast<char*>(m_allocation.begin);
  }

  bool unused() { return m_used_space.empty(); }

  void* get_allocation() { return m_allocation.begin; }

  void* get(size_t nbytes, size_t alignment)
  {
    void* ptr_out = nullptr;
    if (capacity() >= nbytes)
    {
      free_type::iterator end = m_free_space.end();
      for (free_type::iterator iter = m_free_space.begin(); iter != end; ++iter)
      {

        void* adj_ptr = iter->first;
        size_t cap =
            static_cast<char*>(iter->second) - static_cast<char*>(adj_ptr);

        if (::RAJA::align(alignment, nbytes, adj_ptr, cap))
        {

          ptr_out = adj_ptr;

          remove_free_chunk(iter, adj_ptr,
                            static_cast<char*>(adj_ptr) + nbytes);

          add_used_chunk(adj_ptr, static_cast<char*>(adj_ptr) + nbytes);

          break;
        }
      }
    }
    return ptr_out;
  }

  bool give(void* ptr)
  {
    if (m_allocation.begin <= ptr && ptr < m_allocation.end)
    {

      used_type::iterator found = m_used_space.find(ptr);

      if (found != m_used_space.end())
      {

        add_free_chunk(found->first, found->second);

        m_used_space.erase(found);
      }
      else
      {
        fprintf(stderr, "Invalid free %p", ptr);
        std::abort();
      }

      return true;
    }
    else
    {
      return false;
    }
  }

private:
  struct memory_chunk
  {
    void* begin;
    void* end;
  };

  void add_free_chunk(void* begin, void* end)
  {
    // integrates a chunk of memory into free_space
    free_type::iterator invl = m_free_space.end();
    free_type::iterator next = m_free_space.lower_bound(begin);

    // check if prev exists
    if (next != m_free_space.begin())
    {
      // check if prev can cover [begin, end)
      free_type::iterator prev = next;
      --prev;
      if (prev->second == begin)
      {
        // extend prev to cover [begin, end)
        prev->second = end;

        // check if prev can cover next too
        if (next != invl)
        {
          assert(next->first != begin);

          if (next->first == end)
          {
            // extend prev to cover next too
            prev->second = next->second;

            // remove redundant next
            m_free_space.erase(next);
          }
        }
        return;
      }
    }

    if (next != invl)
    {
      assert(next->first != begin);

      if (next->first == end)
      {
        // extend next to cover [begin, end)
        m_free_space.insert(next, free_value_type {begin, next->second});
        m_free_space.erase(next);

        return;
      }
    }

    // no free space adjacent to this chunk, add seperate free chunk [begin,
    // end)
    m_free_space.insert(next, free_value_type {begin, end});
  }

  void remove_free_chunk(free_type::iterator iter, void* begin, void* end)
  {

    void* ptr     = iter->first;
    void* ptr_end = iter->second;

    // fixup m_free_space, shrinking and adding chunks as needed
    if (ptr != begin)
    {

      // shrink end of current free region to [ptr, begin)
      iter->second = begin;

      if (end != ptr_end)
      {

        // insert free region [end, ptr_end) after current free region
        free_type::iterator next = iter;
        ++next;
        m_free_space.insert(next, free_value_type {end, ptr_end});
      }
    }
    else if (end != ptr_end)
    {

      // shrink beginning of current free region to [end, ptr_end)
      free_type::iterator next = iter;
      ++next;
      m_free_space.insert(next, free_value_type {end, ptr_end});
      m_free_space.erase(iter);
    }
    else
    {

      // can not reuse current region, erase
      m_free_space.erase(iter);
    }
  }

  void add_used_chunk(void* begin, void* end)
  {
    // simply inserts a chunk of memory into used_space
    m_used_space.insert(used_value_type {begin, end});
  }

  memory_chunk m_allocation;
  free_type m_free_space;
  used_type m_used_space;
};

} /* end namespace detail */

/*! \class MemPool
 ******************************************************************************
 *
 * \brief  MemPool pre-allocates a large chunk of memory and provides generic
 * malloc/free for the user to allocate aligned data within the pool
 *
 * MemPool uses MemoryArena to do the heavy lifting of maintaining access to
 * the used/free space.
 *
 * MemPool provides an example generic_allocator which can guide more
 *specialized
 * allocators. The following are some examples
 *
 * using device_mempool_type = basic_mempool::MemPool<cuda::DeviceAllocator>;
 * using device_zeroed_mempool_type =
 *basic_mempool::MemPool<cuda::DeviceZeroedAllocator>;
 * using pinned_mempool_type = basic_mempool::MemPool<cuda::PinnedAllocator>;
 *
 * The user provides the specialized allocator, for example :
 * struct DeviceAllocator {
 *
 *  // returns a valid pointer on success, nullptr on failure
 *  void* malloc(size_t nbytes)
 *  {
 *    void* ptr;
 *    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMalloc, &ptr, nbytes);
 *    return ptr;
 *  }
 *
 *  // returns true on success, false on failure
 *  bool free(void* ptr)
 *  {
 *    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, ptr);
 *    return true;
 *  }
 * };
 *
 *
 ******************************************************************************
 */
template<typename allocator_t>
class MemPool
{
public:
  using allocator_type = allocator_t;

  static inline MemPool<allocator_t>& getInstance()
  {
    static MemPool<allocator_t> pool {};
    return pool;
  }

  static const size_t default_default_arena_size = 32ull * 1024ull * 1024ull;

  MemPool()
      : m_arenas(),
        m_default_arena_size(default_default_arena_size),
        m_alloc()
  {}

  ~MemPool()
  {
    // With static objects like MemPool, cudaErrorCudartUnloading is a possible
    // error with cudaFree
    // So no more cuda calls here
  }

  /// Free all backing allocations, even if they are currently in use
  void free_chunks()
  {
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(m_mutex);
#endif

    while (!m_arenas.empty())
    {
      void* allocation_ptr = m_arenas.front().get_allocation();
      m_alloc.free(allocation_ptr);
      m_arenas.pop_front();
    }
  }

  size_t arena_size()
  {
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(m_mutex);
#endif

    return m_default_arena_size;
  }

  size_t arena_size(size_t new_size)
  {
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(m_mutex);
#endif

    size_t prev_size     = m_default_arena_size;
    m_default_arena_size = new_size;
    return prev_size;
  }

  template<typename T>
  T* malloc(size_t nTs, size_t alignment = alignof(T))
  {
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(m_mutex);
#endif

    const size_t size                  = nTs * sizeof(T);
    void* ptr                          = nullptr;
    arena_container_type::iterator end = m_arenas.end();
    for (arena_container_type::iterator iter = m_arenas.begin(); iter != end;
         ++iter)
    {
      ptr = iter->get(size, alignment);
      if (ptr != nullptr)
      {
        break;
      }
    }

    if (ptr == nullptr)
    {
      const size_t alloc_size =
          std::max(size + alignment, m_default_arena_size);
      void* arena_ptr = m_alloc.malloc(alloc_size);
      if (arena_ptr != nullptr)
      {
        m_arenas.emplace_front(arena_ptr, alloc_size);
        ptr = m_arenas.front().get(size, alignment);
      }
    }

    return static_cast<T*>(ptr);
  }

  void free(const void* cptr)
  {
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(m_mutex);
#endif

    void* ptr                          = const_cast<void*>(cptr);
    arena_container_type::iterator end = m_arenas.end();
    for (arena_container_type::iterator iter = m_arenas.begin(); iter != end;
         ++iter)
    {
      if (iter->give(ptr))
      {
        ptr = nullptr;
        break;
      }
    }
    if (ptr != nullptr)
    {
      fprintf(stderr, "Unknown pointer %p", ptr);
    }
  }

private:
  using arena_container_type = std::list<detail::MemoryArena>;

#if defined(RAJA_ENABLE_OPENMP)
  omp::mutex m_mutex;
#endif

  arena_container_type m_arenas;
  size_t m_default_arena_size;
  allocator_t m_alloc;
};

//! example allocator for basic_mempool using malloc/free
struct generic_allocator
{

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes) { return std::malloc(nbytes); }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    std::free(ptr);
    return true;
  }
};

} /* end namespace basic_mempool */

} /* end namespace RAJA */


#endif /* RAJA_BASIC_MEMPOOL_HPP */
