/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for a C++ style allocator with a resource.
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

#ifndef RAJA_RESOURCE_ALLOCATOR_HPP
#define RAJA_RESOURCE_ALLOCATOR_HPP

#include <cstddef>

#include "RAJA/config.hpp"
#include "RAJA/util/resource.hpp"
#include "RAJA/util/macros.hpp"

namespace RAJA
{

namespace detail
{
template<typename Resource>
struct ResourceAllocator
{
  template<typename T>
  struct allocator
  {
    using value_type = T;

    allocator()
        : allocator(Resource::get_default(),
                    RAJA::resources::MemoryAccess::Pinned)
    {}

    allocator(Resource res,
              RAJA::resources::MemoryAccess mem_type =
                  RAJA::resources::MemoryAccess::Pinned)
        : m_res {res},
          m_mem_type {mem_type}
    {}

    allocator(allocator const&) = default;
    allocator(allocator&&)      = default;

    allocator& operator=(allocator const&) = default;
    allocator& operator=(allocator&&)      = default;

    template<typename U>
    allocator(allocator<U> const& other) noexcept
        : m_res(other.get_resource()),
          m_mem_type {other.get_mem_access()}
    {}

    /*[[nodiscard]]*/
    value_type* allocate(std::size_t num)
    {
      if (num > std::numeric_limits<std::size_t>::max() / sizeof(value_type))
      {
        throw std::bad_alloc();
      }

      value_type* ptr = m_res.template allocate<value_type>(num, m_mem_type);

      if (!ptr)
      {
        throw std::bad_alloc();
      }

      return ptr;
    }

    void deallocate(value_type* ptr, std::size_t) noexcept
    {
      m_res.deallocate(ptr, m_mem_type);
    }

    Resource const& get_resource() const { return m_res; }

    Resource get_resource() { return m_res; }

    RAJA::resources::MemoryAccess get_mem_access() const { return m_mem_type; }

    RAJA::resources::MemoryAccess get_mem_access() { return m_mem_type; }

    template<typename U>
    friend inline bool operator==(allocator const& lhs, allocator<U> const& rhs)
    {
      return lhs.get_resource() == rhs.get_resource();
    }

  private:
    Resource m_res;
    RAJA::resources::MemoryAccess m_mem_type;
  };
};
}  // namespace detail

template<typename T, typename Resource>
using ResourceAllocator =
    typename detail::ResourceAllocator<Resource>::template allocator<T>;

}  // namespace RAJA

#endif  // RAJA_RESOURCE_ALLOCATOR_HPP
