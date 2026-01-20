/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for simple vector template class that enables
 *          RAJA to be used with or without the C++ STL.
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

#ifndef RAJAVec_HPP
#define RAJAVec_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

#include "RAJA/internal/MemUtils_CPU.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Class template that provides a simple vector implementation
 *         sufficient to insulate RAJA entities from the STL.
 *
 *         Note: This class has limited functionality sufficient to
 *               support its usage for RAJA TypedIndexSet operations. However,
 *               it does provide a push_front method that is not found
 *               in the STL vector container.
 *
 *               Template type should support standard semantics for
 *               copy, swap, etc.
 *
 *               Note that this class has no exception safety guarantees.
 *
 ******************************************************************************
 */
template<typename T, typename Allocator = std::allocator<T>>
class RAJAVec
{
  using allocator_traits_type = std::allocator_traits<Allocator>;
  using propagate_on_container_copy_assignment =
      typename allocator_traits_type::propagate_on_container_copy_assignment;
  using propagate_on_container_move_assignment =
      typename allocator_traits_type::propagate_on_container_move_assignment;
  using propagate_on_container_swap =
      typename allocator_traits_type::propagate_on_container_swap;

public:
  using value_type      = T;
  using allocator_type  = Allocator;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference       = value_type&;
  using const_reference = const value_type&;
  using pointer         = typename allocator_traits_type::pointer;
  using const_pointer   = typename allocator_traits_type::const_pointer;
  using iterator        = value_type*;
  using const_iterator  = const value_type*;

  ///
  /// Construct empty vector with given capacity.
  ///
  explicit RAJAVec(size_type init_cap      = 0,
                   const allocator_type& a = allocator_type())
      : m_data(nullptr),
        m_allocator(a),
        m_capacity(0),
        m_size(0)
  {
    reserve(init_cap);
  }

  ///
  /// Copy ctor for vector.
  ///
  RAJAVec(const RAJAVec& other)
      : m_data(nullptr),
        m_allocator(
            allocator_traits_type::select_on_container_copy_construction(
                other.m_allocator)),
        m_capacity(0),
        m_size(0)
  {
    reserve(other.size());
    copy_construct_items_back(other.size(), other.data());
  }

  ///
  /// Move ctor for vector.
  ///
  RAJAVec(RAJAVec&& other)
      : m_data(other.m_data),
        m_allocator(std::move(other.m_allocator)),
        m_capacity(other.m_capacity),
        m_size(other.m_size)
  {
    other.m_data     = nullptr;
    other.m_capacity = 0;
    other.m_size     = 0;
  }

  ///
  /// Copy-assignment operator for vector.
  ///
  RAJAVec& operator=(const RAJAVec& rhs)
  {
    if (&rhs != this)
    {
      copy_assign_private(rhs, propagate_on_container_copy_assignment {});
    }
    return *this;
  }

  ///
  /// Move-assignment operator for vector.
  ///
  RAJAVec& operator=(RAJAVec&& rhs)
  {
    if (&rhs != this)
    {
      move_assign_private(std::move(rhs),
                          propagate_on_container_move_assignment {});
    }
    return *this;
  }

  ///
  /// Destroy vector and its data.
  ///
  ~RAJAVec()
  {
    clear();
    shrink_to_fit();
  }

  ///
  /// Swap function for copy-and-swap idiom.
  ///
  void swap(RAJAVec& other)
  {
    swap_private(other, propagate_on_container_swap {});
  }

  ///
  /// Get a pointer to the beginning of the contiguous vector
  ///
  pointer data() { return m_data; }

  ///
  const_pointer data() const { return m_data; }

  ///
  /// Get an iterator to the end.
  ///
  iterator end() { return m_data + m_size; }

  ///
  const_iterator end() const { return m_data + m_size; }

  ///
  const_iterator cend() const { return m_data + m_size; }

  ///
  /// Get an iterator to the beginning.
  ///
  iterator begin() { return m_data; }

  ///
  const_iterator begin() const { return m_data; }

  ///
  const_iterator cbegin() const { return m_data; }

  ///
  /// Return true if vector has size zero; false otherwise.
  ///
  bool empty() const { return (m_size == 0); }

  ///
  /// Return current size of vector.
  ///
  size_type size() const { return m_size; }

  ///
  /// Return current capacity of vector.
  ///
  size_type capacity() const { return m_capacity; }

  ///
  /// Get the allocator used by the container.
  ///
  allocator_type get_allocator() const { return m_allocator; }

  ///
  /// Grow the capacity of the vector.
  ///
  void reserve(size_type target_capacity) { grow_cap(target_capacity); }

  ///
  /// Shrink the capacity of the vector to the current size.
  ///
  void shrink_to_fit() { shrink_cap(m_size); }

  ///
  /// Empty vector of all data.
  ///
  void clear() { destroy_items_after(0); }

  ///
  /// Change the size of the vector,
  /// default initializing any new items,
  /// destroying any extra items.
  ///
  RAJA_INLINE
  void resize(size_type new_size)
  {
    if (new_size >= size())
    {
      reserve(new_size);
      construct_items_back(new_size);
    }
    else
    {
      destroy_items_after(new_size);
    }
  }

  ///
  /// Change the size of the vector,
  /// initializing any new items with new_value,
  /// destroying any extra items.
  ///
  RAJA_INLINE
  void resize(size_type new_size, const_reference new_value)
  {
    if (new_size >= size())
    {
      reserve(new_size);
      construct_items_back(new_size, new_value);
    }
    else
    {
      destroy_items_after(new_size);
    }
  }

  ///
  /// Bracket operator accessor.
  ///
  reference operator[](difference_type i) { return m_data[i]; }

  ///
  const_reference operator[](difference_type i) const { return m_data[i]; }

  ///
  /// Access the last item of the vector.
  ///
  reference front() { return m_data[0]; }

  ///
  const_reference front() const { return m_data[0]; }

  ///
  /// Access the last item of the vector.
  ///
  reference back() { return m_data[m_size - 1]; }

  ///
  const_reference back() const { return m_data[m_size - 1]; }

  ///
  /// Add item to front end of vector. Note that this operation is unique to
  /// this class; it is not part of the C++ standard library vector interface.
  ///
  void push_front(const_reference item) { emplace_front_private(item); }

  ///
  void push_front(value_type&& item) { emplace_front_private(std::move(item)); }

  ///
  template<typename... Os>
  void emplace_front(Os&&... os)
  {
    emplace_front_private(std::forward<Os>(os)...);
  }

  ///
  /// Add item to back end of vector.
  ///
  void push_back(const_reference item) { emplace_back_private(item); }

  ///
  void push_back(value_type&& item) { emplace_back_private(std::move(item)); }

  ///
  template<typename... Os>
  void emplace_back(Os&&... os)
  {
    emplace_back_private(std::forward<Os>(os)...);
  }

  ///
  /// Remove the last item of the vector.
  ///
  void pop_back() { destroy_items_after(m_size - 1); }

private:
  pointer m_data;
  allocator_type m_allocator;
  size_type m_capacity;
  size_type m_size;

  ///
  /// Copy assignment implementation
  /// when propagate on container copy assignment is true.
  ///
  void copy_assign_private(RAJAVec const& rhs, std::true_type)
  {
    if (m_allocator != rhs.m_allocator)
    {
      clear();
      shrink_to_fit();
      m_allocator = rhs.m_allocator;
    }

    copy_assign_private(rhs, std::false_type {});
  }

  ///
  /// Copy assignment implementation
  /// when propagate on container copy assignment is false.
  ///
  void copy_assign_private(RAJAVec const& rhs, std::false_type)
  {
    reserve(rhs.size());
    if (size() < rhs.size())
    {
      copy_assign_items(0, size(), rhs.data());
      copy_construct_items_back(rhs.size(), rhs.data());
    }
    else
    {
      copy_assign_items(0, rhs.size(), rhs.data());
      destroy_items_after(size());
    }
  }

  ///
  /// Move assignment implementation
  /// when propagate on container copy assignment is true.
  ///
  void move_assign_private(RAJAVec&& rhs, std::true_type)
  {
    clear();
    shrink_to_fit();

    m_data      = rhs.m_data;
    m_allocator = std::move(rhs.m_allocator);
    m_capacity  = rhs.m_capacity;
    m_size      = rhs.m_size;

    rhs.m_data     = nullptr;
    rhs.m_capacity = 0;
    rhs.m_size     = 0;
  }

  ///
  /// Move assignment implementation
  /// when propagate on container copy assignment is false.
  ///
  void move_assign_private(RAJAVec&& rhs, std::false_type)
  {
    if (m_allocator == rhs.m_allocator)
    {
      clear();
      shrink_to_fit();

      m_data     = rhs.m_data;
      m_capacity = rhs.m_capacity;
      m_size     = rhs.m_size;

      rhs.m_data     = nullptr;
      rhs.m_capacity = 0;
      rhs.m_size     = 0;
    }
    else
    {
      reserve(rhs.size());
      if (size() < rhs.size())
      {
        move_assign_items(0, size(), rhs.data());
        move_construct_items_back(rhs.size(), rhs.data());
      }
      else
      {
        move_assign_items(0, rhs.size(), rhs.data());
        destroy_items_after(size());
      }
    }
  }

  ///
  /// Swap implementation when propagate on swap is true.
  ///
  void swap_private(RAJAVec& other, std::true_type)
  {
    using std::swap;
    swap(m_data, other.m_data);
    swap(m_allocator, other.m_allocator);
    swap(m_capacity, other.m_capacity);
    swap(m_size, other.m_size);
  }

  ///
  /// Swap implementation when propagate on swap is false.
  ///
  void swap_private(RAJAVec& other, std::false_type)
  {
    using std::swap;
    swap(m_data, other.m_data);
    swap(m_capacity, other.m_capacity);
    swap(m_size, other.m_size);
  }

  //
  // Copy items [first, last) from o_data.
  //
  void copy_assign_items(size_type first, size_type last, const_pointer o_data)
  {
    for (size_type i = first; i < last; ++i)
    {
      m_data[i] = o_data[i];
    }
  }

  //
  // Move items [first, last) from o_data.
  //
  void move_assign_items(size_type first, size_type last, pointer o_data)
  {
    for (size_type i = first; i < last; ++i)
    {
      m_data[i] = std::move(o_data[i]);
    }
  }

  //
  // Construct items [m_size, new_size) from args.
  //
  template<typename... Os>
  void construct_items_back(size_type new_size, Os&&... os)
  {
    for (; m_size < new_size; ++m_size)
    {
      allocator_traits_type::construct(m_allocator, m_data + m_size,
                                       std::forward<Os>(os)...);
    }
  }

  //
  // Copy construct items [m_size, new_size) from o_data.
  //
  void copy_construct_items_back(size_type new_size, const_pointer o_data)
  {
    for (; m_size < new_size; ++m_size)
    {
      allocator_traits_type::construct(m_allocator, m_data + m_size,
                                       o_data[m_size]);
    }
  }

  //
  // Move construct items [m_size, new_size) from o_data.
  //
  void move_construct_items_back(size_type new_size, pointer o_data)
  {
    for (; m_size < new_size; ++m_size)
    {
      allocator_traits_type::construct(m_allocator, m_data + m_size,
                                       std::move(o_data[m_size]));
    }
  }

  //
  // Destroy items [new_end, m_size).
  //
  void destroy_items_after(size_type new_end)
  {
    for (; m_size > new_end; --m_size)
    {
      allocator_traits_type::destroy(m_allocator, m_data + m_size - 1);
    }
  }

  //
  // Add an item to the front, shifting all existing items back one.
  //
  template<typename... Os>
  void emplace_front_private(Os&&... os)
  {
    reserve(m_size + 1);

    if (m_size > 0)
    {
      size_type i = m_size;
      allocator_traits_type::construct(m_allocator, m_data + i,
                                       std::move(m_data[i - 1]));
      for (--i; i > 0; --i)
      {
        m_data[i] = std::move(m_data[i - 1]);
      }
      allocator_traits_type::destroy(m_allocator, m_data);
    }
    allocator_traits_type::construct(m_allocator, m_data,
                                     std::forward<Os>(os)...);
    m_size++;
  }

  //
  // Add an item to the back.
  //
  template<typename... Os>
  void emplace_back_private(Os&&... os)
  {
    reserve(m_size + 1);
    allocator_traits_type::construct(m_allocator, m_data + m_size,
                                     std::forward<Os>(os)...);
    m_size++;
  }

  //
  // The following private members and methods provide a quick and dirty
  // memory allocation scheme to mimick std::vector behavior without
  // relying on STL directly.
  //
  static constexpr const size_type s_init_cap = 8;
  static constexpr const double s_grow_fac    = 1.5;

  //
  // Get the next value for capacity given a target and minimum.
  //
  size_type get_next_cap(size_type target_size)
  {
    size_type next_cap = s_init_cap;
    if (m_capacity != 0)
    {
      next_cap = static_cast<size_type>(m_capacity * s_grow_fac);
    }
    return std::max(target_size, next_cap);
  }

  //
  // Increase capacity to at least target_size.
  //
  void grow_cap(size_type target_size)
  {
    if (m_capacity < target_size)
    {
      change_cap(get_next_cap(target_size));
    }
  }

  //
  // Decrease capacity to at most target_size or size if size is greater.
  //
  void shrink_cap(size_type target_size)
  {
    if (m_capacity > target_size)
    {
      change_cap(std::max(m_size, target_size));
    }
  }

  //
  // Reallocate to change capacity to next_cap.
  // NOTE: assumes next_cap >= size()
  //
  void change_cap(size_type next_cap)
  {
    pointer tdata = nullptr;
    if (next_cap != 0)
    {
      tdata = allocator_traits_type::allocate(m_allocator, next_cap);
    }

    if (m_data)
    {
      for (size_type i = 0; i < m_size; ++i)
      {
        allocator_traits_type::construct(m_allocator, tdata + i,
                                         std::move(m_data[i]));
        allocator_traits_type::destroy(m_allocator, m_data + i);
      }
      allocator_traits_type::deallocate(m_allocator, m_data, m_capacity);
    }

    m_data     = tdata;
    m_capacity = next_cap;
  }
};

}  // namespace RAJA

#endif  // closing endif for header file include guard
