//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_WORKGROUP_UTILS_HPP__
#define __TEST_WORKGROUP_UTILS_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"

#include "RAJA_unit-test-forone.hpp"

#include <cstddef>
#include <limits>
#include <new>
#include <unordered_map>

namespace detail {

struct indirect_function_call_dispatch_typer {
  template < typename ... >
  using type = ::RAJA::indirect_function_call_dispatch;
};

struct indirect_virtual_function_dispatch_typer {
  template < typename ... >
  using type = ::RAJA::indirect_virtual_function_dispatch;
};

struct direct_dispatch_typer {
  template < typename ... Ts >
  using type = ::RAJA::direct_dispatch<Ts...>;
};


template < typename Resource >
struct ResourceAllocator
{
  template < typename T >
  struct std_allocator
  {
    using value_type = T;

    std_allocator() = default;

    std_allocator(std_allocator const&) = default;
    std_allocator(std_allocator &&) = default;

    std_allocator& operator=(std_allocator const&) = default;
    std_allocator& operator=(std_allocator &&) = default;

    template < typename U >
    std_allocator(std_allocator<U> const& other) noexcept
      : m_res(other.get_resource())
    { }

    /*[[nodiscard]]*/
    value_type* allocate(size_t num)
    {
      if (num > std::numeric_limits<size_t>::max() / sizeof(value_type)) {
        throw std::bad_alloc();
      }

      value_type* ptr = m_res.template allocate<value_type>(num, camp::resources::MemoryAccess::Pinned);

      if (!ptr) {
        throw std::bad_alloc();
      }

      return ptr;
    }

    void deallocate(value_type* ptr, size_t) noexcept
    {
      m_res.deallocate(ptr, camp::resources::MemoryAccess::Pinned);
    }

    Resource const& get_resource() const
    {
      return m_res;
    }

    template <typename U>
    friend inline bool operator==(std_allocator const& /*lhs*/, std_allocator<U> const& /*rhs*/)
    {
      return true; // lhs.get_resource() == rhs.get_resource(); // TODO not equality comparable yet
    }

    template <typename U>
    friend inline bool operator!=(std_allocator const& lhs, std_allocator<U> const& rhs)
    {
      return !(lhs == rhs);
    }

  private:
    Resource m_res;
  };
};

struct NeverEqualAllocator
{
  using propagate_on_container_copy_assignment = std::false_type;
  using propagate_on_container_move_assignment = std::false_type;
  using propagate_on_container_swap = std::true_type;

  NeverEqualAllocator() = default;

  NeverEqualAllocator(NeverEqualAllocator const&) = default;
  NeverEqualAllocator(NeverEqualAllocator &&) = default;

  NeverEqualAllocator& operator=(NeverEqualAllocator const&) = default;
  NeverEqualAllocator& operator=(NeverEqualAllocator &&) = default;

  NeverEqualAllocator select_on_container_copy_construction()
  {
    return NeverEqualAllocator{};
  }

  ~NeverEqualAllocator()
  {
    if (!m_allocations.empty()) {
      RAJA_ABORT_OR_THROW("allocation map not empty at destruction");
    }
  }

  /*[[nodiscard]]*/
  void* allocate(size_t size)
  {
    void* ptr = malloc(size);
    auto iter_b = m_allocations.emplace(ptr, size);
    if (!iter_b.second) {
      RAJA_ABORT_OR_THROW("failed to add allocation to map");
    }
    return ptr;
  }

  void deallocate(void* ptr, size_t size) noexcept
  {
    auto iter = m_allocations.find(ptr);
    if (iter == m_allocations.end()) {
      RAJA_ABORT_OR_THROW("failed to find allocation in map");
    }
    if (iter->second != size) {
      RAJA_ABORT_OR_THROW("allocation size does not match known in map");
    }
    m_allocations.erase(iter);
    free(ptr);
  }

  bool operator==(NeverEqualAllocator const&) const
  {
    return false;
  }

private:
  std::unordered_map<void*, size_t> m_allocations;
};

struct AlwaysEqualAllocator
{
  using propagate_on_container_copy_assignment = std::false_type;
  using propagate_on_container_move_assignment = std::false_type;
  using propagate_on_container_swap = std::false_type;

  AlwaysEqualAllocator() = default;

  AlwaysEqualAllocator(AlwaysEqualAllocator const&) = default;
  AlwaysEqualAllocator(AlwaysEqualAllocator &&) = default;

  AlwaysEqualAllocator& operator=(AlwaysEqualAllocator const&) = default;
  AlwaysEqualAllocator& operator=(AlwaysEqualAllocator &&) = default;

  AlwaysEqualAllocator select_on_container_copy_construction()
  {
    return *this;
  }

  /*[[nodiscard]]*/
  void* allocate(size_t size)
  {
    return get_allocator().allocate(size);
  }

  void deallocate(void* ptr, size_t size) noexcept
  {
    get_allocator().deallocate(ptr, size);
  }

  bool operator==(AlwaysEqualAllocator const&) const
  {
    return true;
  }

private:
  static inline NeverEqualAllocator& get_allocator()
  {
    static NeverEqualAllocator s_allocator;
    return s_allocator;
  }
};

struct PropogatingAllocator : NeverEqualAllocator
{
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

  PropogatingAllocator() = default;

  PropogatingAllocator(PropogatingAllocator const&) = default;
  PropogatingAllocator(PropogatingAllocator &&) = default;

  PropogatingAllocator& operator=(PropogatingAllocator const&) = default;
  PropogatingAllocator& operator=(PropogatingAllocator &&) = default;

  PropogatingAllocator select_on_container_copy_construction()
  {
    return PropogatingAllocator(NeverEqualAllocator::select_on_container_copy_construction());
  }

private:
  PropogatingAllocator(NeverEqualAllocator&& nea)
    : NeverEqualAllocator(std::move(nea))
  { }
};

template < typename AllocatorImpl >
struct WorkStorageTestAllocator
{
  template < typename T >
  struct std_allocator
  {
    using value_type = T;
    using propagate_on_container_copy_assignment = typename AllocatorImpl::propagate_on_container_copy_assignment;
    using propagate_on_container_move_assignment = typename AllocatorImpl::propagate_on_container_move_assignment;
    using propagate_on_container_swap = typename AllocatorImpl::propagate_on_container_swap;

    std_allocator() = default;

    std_allocator(std_allocator const&) = default;
    std_allocator(std_allocator &&) = default;

    std_allocator& operator=(std_allocator const&) = default;
    std_allocator& operator=(std_allocator &&) = default;

    template < typename U >
    std_allocator(std_allocator<U> const& other) noexcept
      : m_impl(other.get_impl())
    { }

    std_allocator select_on_container_copy_construction()
    {
      return std_allocator(m_impl.select_on_container_copy_construction());
    }

    /*[[nodiscard]]*/
    value_type* allocate(size_t num)
    {
      if (num > std::numeric_limits<size_t>::max() / sizeof(value_type)) {
        throw std::bad_alloc();
      }

      value_type* ptr = static_cast<value_type*>(m_impl.allocate(num*sizeof(value_type)));

      if (!ptr) {
        throw std::bad_alloc();
      }

      return ptr;
    }

    void deallocate(value_type* ptr, size_t num) noexcept
    {
      m_impl.deallocate(static_cast<void*>(ptr), num*sizeof(value_type));
    }

    AllocatorImpl const& get_impl() const
    {
      return m_impl;
    }

    template <typename U>
    friend inline bool operator==(std_allocator const& lhs, std_allocator<U> const& rhs)
    {
      return lhs.get_impl() == rhs.get_impl();
    }

    template <typename U>
    friend inline bool operator!=(std_allocator const& lhs, std_allocator<U> const& rhs)
    {
      return !(lhs == rhs);
    }

  private:
    std_allocator(AllocatorImpl&& impl)
      : m_impl(std::move(impl))
    { }

    AllocatorImpl m_impl;
  };
};

} // namespace detail


//
// Data types
//
using IndexTypeTypeList = camp::list<
                                 int,
                                 long,
                                 RAJA::Index_type
                               >;

using XargsTypeList = camp::list<
                                 RAJA::xargs<>,
                                 RAJA::xargs<int*>,
                                 RAJA::xargs<int, int*>
                               >;

using SequentialExecPolicyList =
    camp::list<
                RAJA::seq_work
              >;
using SequentialOrderedPolicyList =
    camp::list<
                RAJA::ordered,
                RAJA::reverse_ordered
              >;
using SequentialOrderPolicyList =
    camp::list<
                RAJA::ordered,
                RAJA::reverse_ordered
              >;
using SequentialStoragePolicyList =
    camp::list<
                RAJA::array_of_pointers,
                RAJA::ragged_array_of_objects,
                RAJA::constant_stride_array_of_objects
              >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPExecPolicyList =
    camp::list<
                RAJA::omp_work
              >;
using OpenMPOrderedPolicyList = SequentialOrderedPolicyList;
using OpenMPOrderPolicyList   = SequentialOrderPolicyList;
using OpenMPStoragePolicyList = SequentialStoragePolicyList;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetExecPolicyList =
    camp::list<
                RAJA::omp_target_work
              >;
using OpenMPTargetOrderedPolicyList = SequentialOrderedPolicyList;
using OpenMPTargetOrderPolicyList   = SequentialOrderPolicyList;
using OpenMPTargetStoragePolicyList = SequentialStoragePolicyList;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaExecPolicyList =
    camp::list<
                #if defined(RAJA_TEST_EXHAUSTIVE)
                // avoid compilation error:
                // tpl/camp/include/camp/camp.hpp(104): error #456: excessive recursion at instantiation of class
                RAJA::cuda_work<256>,
                #endif
                RAJA::cuda_work<1024>,
                RAJA::cuda_work_explicit<256, 2>
              >;
using CudaOrderedPolicyList = SequentialOrderedPolicyList;
using CudaOrderPolicyList   =
    camp::list<
                RAJA::ordered,
                RAJA::reverse_ordered,
                RAJA::unordered_cuda_loop_y_block_iter_x_threadblock_average
              >;
using CudaStoragePolicyList = SequentialStoragePolicyList;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipExecPolicyList =
    camp::list<
                #if defined(RAJA_TEST_EXHAUSTIVE)
                RAJA::hip_work<256>,
                #endif
                RAJA::hip_work<1024>
              >;
using HipOrderedPolicyList = SequentialOrderedPolicyList;
using HipOrderPolicyList   =
    camp::list<
                RAJA::ordered,
                RAJA::reverse_ordered
              , RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average
              >;
using HipStoragePolicyList = SequentialStoragePolicyList;
#endif


//
// Dispatch policy type lists, broken up for compile time reasons
//
using IndirectFunctionDispatchTyperList = camp::list<detail::indirect_function_call_dispatch_typer>;
using IndirectVirtualDispatchTyperList = camp::list<detail::indirect_virtual_function_dispatch_typer>;
using DirectDispatchTyperList = camp::list<detail::direct_dispatch_typer>;


//
// Memory resource Allocator types
//
using HostAllocatorList = camp::list<typename detail::ResourceAllocator<camp::resources::Host>::template std_allocator<char>>;

using SequentialAllocatorList = HostAllocatorList;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPAllocatorList = HostAllocatorList;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaAllocatorList = camp::list<typename detail::ResourceAllocator<camp::resources::Cuda>::template std_allocator<char>>;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipAllocatorList = camp::list<typename detail::ResourceAllocator<camp::resources::Hip>::template std_allocator<char>>;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetAllocatorList = camp::list<typename detail::ResourceAllocator<camp::resources::Omp>::template std_allocator<char>>;
#endif


//
// Memory resource types for testing different std allocator requirements
//
using WorkStorageAllocatorList = camp::list<typename detail::WorkStorageTestAllocator<detail::AlwaysEqualAllocator>::template std_allocator<char>,
                                            typename detail::WorkStorageTestAllocator<detail::NeverEqualAllocator>::template std_allocator<char>,
                                            typename detail::WorkStorageTestAllocator<detail::PropogatingAllocator>::template std_allocator<char>>;

#endif  // __TEST_WORKGROUP_UTILS_HPP__
