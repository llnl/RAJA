/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA WorkStorage.
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

#ifndef RAJA_PATTERN_WORKGROUP_WorkStorage_HPP
#define RAJA_PATTERN_WORKGROUP_WorkStorage_HPP

#include "RAJA/config.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <type_traits>

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/macros.hpp"

#include "RAJA/internal/RAJAVec.hpp"

#include "RAJA/pattern/WorkGroup/WorkStruct.hpp"

namespace RAJA
{

namespace detail
{

// iterator class that implements the random access iterator interface
// in terms of a in terms of a few basic operations
//   operator *  (                      )
//   operator += ( difference_type      )
//   operator -  ( iterator_base const& )
//   operator == ( iterator_base const& )
//   operator <  ( iterator_base const& )
template<typename iterator_base>
struct random_access_iterator : iterator_base
{
  using base              = iterator_base;
  using value_type        = const typename base::value_type;
  using pointer           = typename base::pointer;
  using reference         = typename base::reference;
  using difference_type   = typename base::difference_type;
  using iterator_category = std::random_access_iterator_tag;

  using base::base;

  random_access_iterator(random_access_iterator const&) = default;
  random_access_iterator(random_access_iterator&&)      = default;

  random_access_iterator& operator=(random_access_iterator const&) = default;
  random_access_iterator& operator=(random_access_iterator&&)      = default;

  RAJA_HOST_DEVICE reference operator*() const
  {
    return *static_cast<base const&>(*this);
  }

  RAJA_HOST_DEVICE pointer operator->() const { return &(*(*this)); }

  RAJA_HOST_DEVICE reference operator[](difference_type i) const
  {
    random_access_iterator copy = *this;
    copy += i;
    return *copy;
  }

  RAJA_HOST_DEVICE random_access_iterator& operator++()
  {
    (*this) += 1;
    return *this;
  }

  RAJA_HOST_DEVICE random_access_iterator operator++(int)
  {
    random_access_iterator copy = *this;
    ++(*this);
    return copy;
  }

  RAJA_HOST_DEVICE random_access_iterator& operator--()
  {
    (*this) -= 1;
    return *this;
  }

  RAJA_HOST_DEVICE random_access_iterator operator--(int)
  {
    random_access_iterator copy = *this;
    --(*this);
    return copy;
  }

  RAJA_HOST_DEVICE random_access_iterator& operator+=(difference_type rhs)
  {
    static_cast<base&>(*this) += rhs;
    return *this;
  }

  RAJA_HOST_DEVICE random_access_iterator& operator-=(difference_type rhs)
  {
    (*this) += -rhs;
    return *this;
  }

  RAJA_HOST_DEVICE friend inline random_access_iterator operator+(
      random_access_iterator const& lhs,
      difference_type rhs)
  {
    random_access_iterator copy = lhs;
    copy += rhs;
    return copy;
  }

  RAJA_HOST_DEVICE friend inline random_access_iterator operator+(
      difference_type lhs,
      random_access_iterator const& rhs)
  {
    random_access_iterator copy = rhs;
    copy += lhs;
    return copy;
  }

  RAJA_HOST_DEVICE friend inline random_access_iterator operator-(
      random_access_iterator const& lhs,
      difference_type rhs)
  {
    random_access_iterator copy = lhs;
    copy -= rhs;
    return copy;
  }

  RAJA_HOST_DEVICE friend inline difference_type operator-(
      random_access_iterator const& lhs,
      random_access_iterator const& rhs)
  {
    return static_cast<base const&>(lhs) - static_cast<base const&>(rhs);
  }

  RAJA_HOST_DEVICE friend inline bool operator==(
      random_access_iterator const& lhs,
      random_access_iterator const& rhs)
  {
    return static_cast<base const&>(lhs) == static_cast<base const&>(rhs);
  }

  RAJA_HOST_DEVICE friend inline bool operator!=(
      random_access_iterator const& lhs,
      random_access_iterator const& rhs)
  {
    return !(lhs == rhs);
  }

  RAJA_HOST_DEVICE friend inline bool operator<(
      random_access_iterator const& lhs,
      random_access_iterator const& rhs)
  {
    return static_cast<base const&>(lhs) < static_cast<base const&>(rhs);
  }

  RAJA_HOST_DEVICE friend inline bool operator<=(
      random_access_iterator const& lhs,
      random_access_iterator const& rhs)
  {
    return !(rhs < lhs);
  }

  RAJA_HOST_DEVICE friend inline bool operator>(
      random_access_iterator const& lhs,
      random_access_iterator const& rhs)
  {
    return rhs < lhs;
  }

  RAJA_HOST_DEVICE friend inline bool operator>=(
      random_access_iterator const& lhs,
      random_access_iterator const& rhs)
  {
    return !(lhs < rhs);
  }
};


/*!
 * A storage container for work groups
 */
template<typename STORAGE_POLICY_T, typename ALLOCATOR_T, typename Dispatcher_T>
class WorkStorage;

template<typename ALLOCATOR_T, typename Dispatcher_T>
class WorkStorage<RAJA::array_of_pointers, ALLOCATOR_T, Dispatcher_T>
{
  using allocator_traits_type = std::allocator_traits<ALLOCATOR_T>;
  using propagate_on_container_copy_assignment =
      typename allocator_traits_type::propagate_on_container_copy_assignment;
  using propagate_on_container_move_assignment =
      typename allocator_traits_type::propagate_on_container_move_assignment;
  using propagate_on_container_swap =
      typename allocator_traits_type::propagate_on_container_swap;
  static_assert(
      std::is_same<typename allocator_traits_type::value_type, char>::value,
      "WorkStorage expects an allocator for 'char's.");

public:
  using storage_policy  = RAJA::array_of_pointers;
  using dispatcher_type = Dispatcher_T;

  template<typename holder>
  using true_value_type = WorkStruct<sizeof(holder), dispatcher_type>;

  using value_type      = GenericWorkStruct<dispatcher_type>;
  using allocator_type  = ALLOCATOR_T;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference       = value_type&;
  using const_reference = const value_type&;
  using pointer         = value_type*;
  using const_pointer   = const value_type*;

private:
  // struct used in storage vector to retain pointer and allocation size
  struct pointer_and_size
  {
    pointer ptr;
    size_type size;
  };

public:
  // iterator base class for accessing stored WorkStructs outside of the
  // container
  struct const_iterator_base
  {
    using value_type        = const typename WorkStorage::value_type;
    using pointer           = typename WorkStorage::const_pointer;
    using reference         = typename WorkStorage::const_reference;
    using difference_type   = typename WorkStorage::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    const_iterator_base(const pointer_and_size* ptrptr) : m_ptrptr(ptrptr) {}

    RAJA_HOST_DEVICE reference operator*() const { return *(m_ptrptr->ptr); }

    RAJA_HOST_DEVICE const_iterator_base& operator+=(difference_type n)
    {
      m_ptrptr += n;
      return *this;
    }

    RAJA_HOST_DEVICE friend inline difference_type operator-(
        const_iterator_base const& lhs_iter,
        const_iterator_base const& rhs_iter)
    {
      return lhs_iter.m_ptrptr - rhs_iter.m_ptrptr;
    }

    RAJA_HOST_DEVICE friend inline bool operator==(
        const_iterator_base const& lhs_iter,
        const_iterator_base const& rhs_iter)
    {
      return lhs_iter.m_ptrptr == rhs_iter.m_ptrptr;
    }

    RAJA_HOST_DEVICE friend inline bool operator<(
        const_iterator_base const& lhs_iter,
        const_iterator_base const& rhs_iter)
    {
      return lhs_iter.m_ptrptr < rhs_iter.m_ptrptr;
    }

  private:
    const pointer_and_size* m_ptrptr;
  };

  using const_iterator = random_access_iterator<const_iterator_base>;

  explicit WorkStorage(allocator_type const& aloc)
      : m_vec(0, aloc),
        m_aloc(aloc)
  {}

  WorkStorage(WorkStorage const&)            = delete;
  WorkStorage& operator=(WorkStorage const&) = delete;

  WorkStorage(WorkStorage&& rhs)
      : m_vec(std::move(rhs.m_vec)),
        m_aloc(std::move(rhs.m_aloc))
  {}

  WorkStorage& operator=(WorkStorage&& rhs)
  {
    if (this != &rhs)
    {
      move_assign_private(std::move(rhs),
                          propagate_on_container_move_assignment {});
    }
    return *this;
  }

  // reserve may be used to allocate enough memory to store num_loops
  // and loop_storage_size is ignored in this version because each
  // object has its own allocation
  void reserve(size_type num_loops, size_type loop_storage_size)
  {
    RAJA_UNUSED_VAR(loop_storage_size);
    m_vec.reserve(num_loops);
  }

  // number of loops stored
  size_type size() const { return m_vec.size(); }

  const_iterator begin() const { return const_iterator(m_vec.begin()); }

  const_iterator end() const { return const_iterator(m_vec.end()); }

  // number of bytes used for storage of loops
  size_type storage_size() const
  {
    size_type storage_size_nbytes = 0;
    for (size_t i = 0; i < m_vec.size(); ++i)
    {
      storage_size_nbytes += m_vec[i].size;
    }
    return storage_size_nbytes;
  }

  template<typename holder, typename... holder_ctor_args>
  void emplace(const dispatcher_type* dispatcher,
               holder_ctor_args&&... ctor_args)
  {
    m_vec.emplace_back(create_value<holder>(
        dispatcher, std::forward<holder_ctor_args>(ctor_args)...));
  }

  // destroy all stored loops, deallocates all storage
  void clear()
  {
    while (!m_vec.empty())
    {
      destroy_value(m_vec.back());
      m_vec.pop_back();
    }
    m_vec.shrink_to_fit();
  }

  ~WorkStorage() { clear(); }

private:
  RAJAVec<
      pointer_and_size,
      typename allocator_traits_type::template rebind_alloc<pointer_and_size>>
      m_vec;
  allocator_type m_aloc;

  // move assignment if allocator propagates on move assignment
  void move_assign_private(WorkStorage&& rhs, std::true_type)
  {
    clear();
    m_vec  = std::move(rhs.m_vec);
    m_aloc = std::move(rhs.m_aloc);
  }

  // move assignment if allocator does not propagate on move assignment
  void move_assign_private(WorkStorage&& rhs, std::false_type)
  {
    clear();
    if (m_aloc == rhs.m_aloc)
    {
      // take storage if allocators compare equal
      m_vec = std::move(rhs.m_vec);
    }
    else
    {
      // allocate new storage if allocators do not compare equal
      for (size_type i = 0; i < rhs.m_vec.size(); ++i)
      {
        m_vec.emplace_back(move_destroy_value(std::move(rhs), rhs.m_vec[i]));
      }
      rhs.m_vec.clear();
      rhs.clear();
    }
  }

  // allocate and construct value in storage
  template<typename holder, typename... holder_ctor_args>
  pointer_and_size create_value(const dispatcher_type* dispatcher,
                                holder_ctor_args&&... ctor_args)
  {
    const size_type value_size = sizeof(true_value_type<holder>);

    pointer value_ptr = reinterpret_cast<pointer>(
        allocator_traits_type::allocate(m_aloc, value_size));

    value_type::template construct<holder>(
        value_ptr, dispatcher, std::forward<holder_ctor_args>(ctor_args)...);

    return pointer_and_size {value_ptr, value_size};
  }

  // allocate and move construct object as copy of other value and
  // destroy and deallocate other value
  pointer_and_size move_destroy_value(WorkStorage&& rhs,
                                      pointer_and_size other_value_and_size)
  {
    pointer value_ptr = reinterpret_cast<pointer>(
        allocator_traits_type::allocate(m_aloc, other_value_and_size.size));

    value_type::move_destroy(value_ptr, other_value_and_size.ptr);

    allocator_traits_type::deallocate(
        rhs.m_aloc, reinterpret_cast<char*>(other_value_and_size.ptr),
        other_value_and_size.size);

    return pointer_and_size {value_ptr, other_value_and_size.size};
  }

  // destroy and deallocate value
  void destroy_value(pointer_and_size value_and_size_ptr)
  {
    value_type::destroy(value_and_size_ptr.ptr);
    allocator_traits_type::deallocate(
        m_aloc, reinterpret_cast<char*>(value_and_size_ptr.ptr),
        value_and_size_ptr.size);
  }
};

template<typename ALLOCATOR_T, typename Dispatcher_T>
class WorkStorage<RAJA::ragged_array_of_objects, ALLOCATOR_T, Dispatcher_T>
{
  using allocator_traits_type = std::allocator_traits<ALLOCATOR_T>;
  using propagate_on_container_copy_assignment =
      typename allocator_traits_type::propagate_on_container_copy_assignment;
  using propagate_on_container_move_assignment =
      typename allocator_traits_type::propagate_on_container_move_assignment;
  using propagate_on_container_swap =
      typename allocator_traits_type::propagate_on_container_swap;
  static_assert(
      std::is_same<typename allocator_traits_type::value_type, char>::value,
      "WorkStorage expects an allocator for 'char's.");

public:
  using storage_policy  = RAJA::ragged_array_of_objects;
  using dispatcher_type = Dispatcher_T;

  template<typename holder>
  using true_value_type = WorkStruct<sizeof(holder), dispatcher_type>;

  using value_type      = GenericWorkStruct<dispatcher_type>;
  using allocator_type  = ALLOCATOR_T;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference       = value_type&;
  using const_reference = const value_type&;
  using pointer         = value_type*;
  using const_pointer   = const value_type*;

  // iterator base class for accessing stored WorkStructs outside of the
  // container
  struct const_iterator_base
  {
    using value_type        = const typename WorkStorage::value_type;
    using pointer           = typename WorkStorage::const_pointer;
    using reference         = typename WorkStorage::const_reference;
    using difference_type   = typename WorkStorage::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    const_iterator_base(const char* array_begin, const size_type* offset_iter)
        : m_array_begin(array_begin),
          m_offset_iter(offset_iter)
    {}

    RAJA_HOST_DEVICE reference operator*() const
    {
      return *reinterpret_cast<pointer>(m_array_begin + *m_offset_iter);
    }

    RAJA_HOST_DEVICE const_iterator_base& operator+=(difference_type n)
    {
      m_offset_iter += n;
      return *this;
    }

    RAJA_HOST_DEVICE friend inline difference_type operator-(
        const_iterator_base const& lhs_iter,
        const_iterator_base const& rhs_iter)
    {
      return lhs_iter.m_offset_iter - rhs_iter.m_offset_iter;
    }

    RAJA_HOST_DEVICE friend inline bool operator==(
        const_iterator_base const& lhs_iter,
        const_iterator_base const& rhs_iter)
    {
      return lhs_iter.m_offset_iter == rhs_iter.m_offset_iter;
    }

    RAJA_HOST_DEVICE friend inline bool operator<(
        const_iterator_base const& lhs_iter,
        const_iterator_base const& rhs_iter)
    {
      return lhs_iter.m_offset_iter < rhs_iter.m_offset_iter;
    }

  private:
    const char* m_array_begin;
    const size_type* m_offset_iter;
  };

  using const_iterator = random_access_iterator<const_iterator_base>;

  explicit WorkStorage(allocator_type const& aloc)
      : m_offsets(0, aloc),
        m_aloc(aloc)
  {}

  WorkStorage(WorkStorage const&)            = delete;
  WorkStorage& operator=(WorkStorage const&) = delete;

  WorkStorage(WorkStorage&& rhs)
      : m_offsets(std::move(rhs.m_offsets)),
        m_array_begin(rhs.m_array_begin),
        m_array_end(rhs.m_array_end),
        m_array_cap(rhs.m_array_cap),
        m_aloc(std::move(rhs.m_aloc))
  {
    rhs.m_array_begin = nullptr;
    rhs.m_array_end   = nullptr;
    rhs.m_array_cap   = nullptr;
  }

  WorkStorage& operator=(WorkStorage&& rhs)
  {
    if (this != &rhs)
    {
      move_assign_private(std::move(rhs),
                          propagate_on_container_move_assignment {});
    }
    return *this;
  }

  // reserve space for num_loops in the array of offsets
  // and space for loop_storage_size bytes of loop storage
  void reserve(size_type num_loops, size_type loop_storage_size)
  {
    m_offsets.reserve(num_loops);
    array_reserve(loop_storage_size);
  }

  // number of loops stored
  size_type size() const { return m_offsets.size(); }

  const_iterator begin() const
  {
    return const_iterator(m_array_begin, m_offsets.begin());
  }

  const_iterator end() const
  {
    return const_iterator(m_array_begin, m_offsets.end());
  }

  // number of bytes used for storage of loops
  size_type storage_size() const { return m_array_end - m_array_begin; }

  template<typename holder, typename... holder_ctor_args>
  void emplace(const dispatcher_type* dispatcher,
               holder_ctor_args&&... ctor_args)
  {
    size_type value_offset = storage_size();
    size_type value_size   = create_value<holder>(
        value_offset, dispatcher, std::forward<holder_ctor_args>(ctor_args)...);
    m_offsets.emplace_back(value_offset);
    m_array_end += value_size;
  }

  // destroy loops and deallocate all storage
  void clear()
  {
    array_clear();
    if (m_array_begin != nullptr)
    {
      allocator_traits_type::deallocate(m_aloc, m_array_begin,
                                        storage_capacity());
      m_array_begin = nullptr;
      m_array_end   = nullptr;
      m_array_cap   = nullptr;
    }
  }

  ~WorkStorage() { clear(); }

private:
  RAJAVec<size_type,
          typename allocator_traits_type::template rebind_alloc<size_type>>
      m_offsets;
  char* m_array_begin = nullptr;
  char* m_array_end   = nullptr;
  char* m_array_cap   = nullptr;
  allocator_type m_aloc;

  // move assignment if allocator propagates on move assignment
  void move_assign_private(WorkStorage&& rhs, std::true_type)
  {
    clear();

    m_offsets     = std::move(rhs.m_offsets);
    m_array_begin = rhs.m_array_begin;
    m_array_end   = rhs.m_array_end;
    m_array_cap   = rhs.m_array_cap;
    m_aloc        = std::move(rhs.m_aloc);

    rhs.m_array_begin = nullptr;
    rhs.m_array_end   = nullptr;
    rhs.m_array_cap   = nullptr;
  }

  // move assignment if allocator does not propagate on move assignment
  void move_assign_private(WorkStorage&& rhs, std::false_type)
  {
    clear();
    if (m_aloc == rhs.m_aloc)
    {

      m_offsets     = std::move(rhs.m_offsets);
      m_array_begin = rhs.m_array_begin;
      m_array_end   = rhs.m_array_end;
      m_array_cap   = rhs.m_array_cap;

      rhs.m_array_begin = nullptr;
      rhs.m_array_end   = nullptr;
      rhs.m_array_cap   = nullptr;
    }
    else
    {
      array_reserve(rhs.storage_size());

      for (size_type i = 0; i < rhs.size(); ++i)
      {
        m_array_end = m_array_begin + rhs.m_offsets[i];
        move_destroy_value(m_array_end, rhs.m_array_begin + rhs.m_offsets[i]);
        m_offsets.emplace_back(rhs.m_offsets[i]);
      }
      m_array_end     = m_array_begin + rhs.storage_size();
      rhs.m_array_end = rhs.m_array_begin;
      rhs.m_offsets.clear();
      rhs.clear();
    }
  }

  // get loop storage capacity, used and unused in bytes
  size_type storage_capacity() const { return m_array_cap - m_array_begin; }

  // get unused loop storage capacity in bytes
  size_type storage_unused() const { return m_array_cap - m_array_end; }

  // reserve space for loop_storage_size bytes of loop storage
  void array_reserve(size_type loop_storage_size)
  {
    if (loop_storage_size > storage_capacity())
    {

      char* new_array_begin =
          allocator_traits_type::allocate(m_aloc, loop_storage_size);
      char* new_array_end = new_array_begin + storage_size();
      char* new_array_cap = new_array_begin + loop_storage_size;

      for (size_type i = 0; i < size(); ++i)
      {
        move_destroy_value(new_array_begin + m_offsets[i],
                           m_array_begin + m_offsets[i]);
      }

      if (m_array_begin != nullptr)
      {
        allocator_traits_type::deallocate(m_aloc, m_array_begin,
                                          storage_capacity());
      }

      m_array_begin = new_array_begin;
      m_array_end   = new_array_end;
      m_array_cap   = new_array_cap;
    }
  }

  // destroy loop objects (does not deallocate array storage)
  void array_clear()
  {
    while (!m_offsets.empty())
    {
      destroy_value(m_offsets.back());
      m_array_end = m_array_begin + m_offsets.back();
      m_offsets.pop_back();
    }
    m_offsets.shrink_to_fit();
  }

  // ensure there is enough storage to hold the next loop body at value offset
  // and store the loop body
  template<typename holder, typename... holder_ctor_args>
  size_type create_value(size_type value_offset,
                         const dispatcher_type* dispatcher,
                         holder_ctor_args&&... ctor_args)
  {
    const size_type value_size = sizeof(true_value_type<holder>);

    if (value_size > storage_unused())
    {
      array_reserve(
          std::max(storage_size() + value_size, 2 * storage_capacity()));
    }

    pointer value_ptr = reinterpret_cast<pointer>(m_array_begin + value_offset);

    value_type::template construct<holder>(
        value_ptr, dispatcher, std::forward<holder_ctor_args>(ctor_args)...);

    return value_size;
  }

  // move construct the loop body into value from other, and destroy the
  // loop body in other
  void move_destroy_value(char* value_ptr, char* other_value_ptr)
  {
    value_type::move_destroy(reinterpret_cast<pointer>(value_ptr),
                             reinterpret_cast<pointer>(other_value_ptr));
  }

  // destroy the loop body at value offset
  void destroy_value(size_type value_offset)
  {
    pointer value_ptr = reinterpret_cast<pointer>(m_array_begin + value_offset);
    value_type::destroy(value_ptr);
  }
};

template<typename ALLOCATOR_T, typename Dispatcher_T>
class WorkStorage<RAJA::constant_stride_array_of_objects,
                  ALLOCATOR_T,
                  Dispatcher_T>
{
  using allocator_traits_type = std::allocator_traits<ALLOCATOR_T>;
  using propagate_on_container_copy_assignment =
      typename allocator_traits_type::propagate_on_container_copy_assignment;
  using propagate_on_container_move_assignment =
      typename allocator_traits_type::propagate_on_container_move_assignment;
  using propagate_on_container_swap =
      typename allocator_traits_type::propagate_on_container_swap;
  static_assert(
      std::is_same<typename allocator_traits_type::value_type, char>::value,
      "WorkStorage expects an allocator for 'char's.");

public:
  using storage_policy  = RAJA::constant_stride_array_of_objects;
  using dispatcher_type = Dispatcher_T;

  template<typename holder>
  using true_value_type = WorkStruct<sizeof(holder), dispatcher_type>;

  using value_type      = GenericWorkStruct<dispatcher_type>;
  using allocator_type  = ALLOCATOR_T;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference       = value_type&;
  using const_reference = const value_type&;
  using pointer         = value_type*;
  using const_pointer   = const value_type*;

  // iterator base class for accessing stored WorkStructs outside of the
  // container
  struct const_iterator_base
  {
    using value_type        = const typename WorkStorage::value_type;
    using pointer           = typename WorkStorage::const_pointer;
    using reference         = typename WorkStorage::const_reference;
    using difference_type   = typename WorkStorage::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    const_iterator_base(const char* array_pos, size_type stride)
        : m_array_pos(array_pos),
          m_stride(stride)
    {}

    RAJA_HOST_DEVICE reference operator*() const
    {
      return *reinterpret_cast<const value_type*>(m_array_pos);
    }

    RAJA_HOST_DEVICE const_iterator_base& operator+=(difference_type n)
    {
      m_array_pos += n * m_stride;
      return *this;
    }

    RAJA_HOST_DEVICE friend inline difference_type operator-(
        const_iterator_base const& lhs_iter,
        const_iterator_base const& rhs_iter)
    {
      return (lhs_iter.m_array_pos - rhs_iter.m_array_pos) / lhs_iter.m_stride;
    }

    RAJA_HOST_DEVICE friend inline bool operator==(
        const_iterator_base const& lhs_iter,
        const_iterator_base const& rhs_iter)
    {
      return lhs_iter.m_array_pos == rhs_iter.m_array_pos;
    }

    RAJA_HOST_DEVICE friend inline bool operator<(
        const_iterator_base const& lhs_iter,
        const_iterator_base const& rhs_iter)
    {
      return lhs_iter.m_array_pos < rhs_iter.m_array_pos;
    }

  private:
    const char* m_array_pos;
    size_type m_stride;
  };

  using const_iterator = random_access_iterator<const_iterator_base>;

  explicit WorkStorage(allocator_type const& aloc) : m_aloc(aloc) {}

  WorkStorage(WorkStorage const&)            = delete;
  WorkStorage& operator=(WorkStorage const&) = delete;

  WorkStorage(WorkStorage&& rhs)
      : m_aloc(std::move(rhs.m_aloc)),
        m_stride(rhs.m_stride),
        m_array_begin(rhs.m_array_begin),
        m_array_end(rhs.m_array_end),
        m_array_cap(rhs.m_array_cap)
  {
    // do not reset stride, leave it for reuse
    rhs.m_array_begin = nullptr;
    rhs.m_array_end   = nullptr;
    rhs.m_array_cap   = nullptr;
  }

  WorkStorage& operator=(WorkStorage&& rhs)
  {
    if (this != &rhs)
    {
      move_assign_private(std::move(rhs),
                          propagate_on_container_move_assignment {});
    }
    return *this;
  }

  // reserve space for at least loop_storage_size bytes of loop storage
  // and num_loops at current stride
  void reserve(size_type num_loops, size_type loop_storage_size)
  {
    size_type num_storage_loops =
        std::max(num_loops, (loop_storage_size + m_stride - 1) / m_stride);
    array_reserve(num_storage_loops * m_stride, m_stride);
  }

  // number of loops stored
  size_type size() const { return storage_size() / m_stride; }

  const_iterator begin() const
  {
    return const_iterator(m_array_begin, m_stride);
  }

  const_iterator end() const { return const_iterator(m_array_end, m_stride); }

  // amount of storage in bytes used to store loops
  size_type storage_size() const { return m_array_end - m_array_begin; }

  template<typename holder, typename... holder_ctor_args>
  void emplace(const dispatcher_type* dispatcher,
               holder_ctor_args&&... ctor_args)
  {
    create_value<holder>(dispatcher,
                         std::forward<holder_ctor_args>(ctor_args)...);
    m_array_end += m_stride;
  }

  // destroy stored loop bodies and deallocates all storage
  void clear()
  {
    array_clear();
    if (m_array_begin != nullptr)
    {
      allocator_traits_type::deallocate(m_aloc, m_array_begin,
                                        storage_capacity());
      m_array_begin = nullptr;
      m_array_end   = nullptr;
      m_array_cap   = nullptr;
    }
  }

  ~WorkStorage() { clear(); }

private:
  allocator_type m_aloc;
  size_type m_stride  = 1;  // can't be 0 because size divides stride
  char* m_array_begin = nullptr;
  char* m_array_end   = nullptr;
  char* m_array_cap   = nullptr;

  // move assignment if allocator propagates on move assignment
  void move_assign_private(WorkStorage&& rhs, std::true_type)
  {
    clear();

    m_aloc        = std::move(rhs.m_aloc);
    m_stride      = rhs.m_stride;
    m_array_begin = rhs.m_array_begin;
    m_array_end   = rhs.m_array_end;
    m_array_cap   = rhs.m_array_cap;

    // do not reset stride, leave it for reuse
    rhs.m_array_begin = nullptr;
    rhs.m_array_end   = nullptr;
    rhs.m_array_cap   = nullptr;
  }

  // move assignment if allocator does not propagate on move assignment
  void move_assign_private(WorkStorage&& rhs, std::false_type)
  {
    clear();
    if (m_aloc == rhs.m_aloc)
    {

      m_stride      = rhs.m_stride;
      m_array_begin = rhs.m_array_begin;
      m_array_end   = rhs.m_array_end;
      m_array_cap   = rhs.m_array_cap;

      // do not reset stride, leave it for reuse
      rhs.m_array_begin = nullptr;
      rhs.m_array_end   = nullptr;
      rhs.m_array_cap   = nullptr;
    }
    else
    {

      m_stride = rhs.m_stride;
      array_reserve(rhs.storage_size(), rhs.m_stride);

      for (size_type i = 0; i < rhs.size(); ++i)
      {
        move_destroy_value(m_array_end, rhs.m_array_begin + i * rhs.m_stride);
        m_array_end += m_stride;
      }
      rhs.m_array_end = rhs.m_array_begin;
      rhs.clear();
    }
  }

  // storage capacity, used and unused, in bytes
  size_type storage_capacity() const { return m_array_cap - m_array_begin; }

  // unused storage capacity in bytes
  size_type storage_unused() const { return m_array_cap - m_array_end; }

  // allocate enough storage for loop_storage_size bytes with
  // each loop body separated by new_stride bytes
  // note that this can reallocate storage with or without changing
  // the storage stride
  // Note that loop_storage_size must be a multiple of new_stride
  void array_reserve(size_type loop_storage_size, size_type new_stride)
  {
    if (loop_storage_size > storage_capacity() || new_stride > m_stride)
    {

      char* new_array_begin =
          allocator_traits_type::allocate(m_aloc, loop_storage_size);
      char* new_array_end = new_array_begin + size() * new_stride;
      char* new_array_cap = new_array_begin + loop_storage_size;

      for (size_type i = 0; i < size(); ++i)
      {
        move_destroy_value(new_array_begin + i * new_stride,
                           m_array_begin + i * m_stride);
      }

      if (m_array_begin != nullptr)
      {
        allocator_traits_type::deallocate(m_aloc, m_array_begin,
                                          storage_capacity());
      }

      m_stride      = new_stride;
      m_array_begin = new_array_begin;
      m_array_end   = new_array_end;
      m_array_cap   = new_array_cap;
    }
  }

  // destroy the loops in storage (does not deallocate loop storage)
  void array_clear()
  {
    for (size_type value_offset = storage_size(); value_offset > 0;
         value_offset -= m_stride)
    {
      destroy_value(value_offset - m_stride);
      m_array_end -= m_stride;
    }
  }

  // ensure there is enough storage to store the loop body
  // and construct the body in storage.
  template<typename holder, typename... holder_ctor_args>
  void create_value(const dispatcher_type* dispatcher,
                    holder_ctor_args&&... ctor_args)
  {
    const size_type value_size = sizeof(true_value_type<holder>);

    if (value_size > storage_unused() && value_size <= m_stride)
    {
      array_reserve(std::max(storage_size() + m_stride, 2 * storage_capacity()),
                    m_stride);
    }
    else if (value_size > m_stride)
    {
      array_reserve((size() + 1) * value_size, value_size);
    }

    size_type value_offset = storage_size();
    pointer value_ptr = reinterpret_cast<pointer>(m_array_begin + value_offset);

    value_type::template construct<holder>(
        value_ptr, dispatcher, std::forward<holder_ctor_args>(ctor_args)...);
  }

  // move construct the loop body in value from other and
  // destroy the loop body in other
  void move_destroy_value(char* value_ptr, char* other_value_ptr)
  {
    value_type::move_destroy(reinterpret_cast<pointer>(value_ptr),
                             reinterpret_cast<pointer>(other_value_ptr));
  }

  // destroy the loop body at value offset
  void destroy_value(size_type value_offset)
  {
    pointer value_ptr = reinterpret_cast<pointer>(m_array_begin + value_offset);
    value_type::destroy(value_ptr);
  }
};

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
