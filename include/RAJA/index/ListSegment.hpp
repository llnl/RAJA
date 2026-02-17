/*!
 ******************************************************************************
 *
 * \file ListSegment.hpp
 *
 * \brief  Header file containing definition of RAJA list segment class.
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

#ifndef RAJA_ListSegment_HPP
#define RAJA_ListSegment_HPP

#include "RAJA/config.hpp"

#include <memory>
#include <type_traits>
#include <utility>

#include "camp/resource.hpp"

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/Span.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \class ListSegment
 *
 * \brief  Segment class representing an arbitrary collection of indices.
 *
 * \tparam StorageT underlying data type for the segment indices (required)
 *
 * A TypedRangeSegment models an Iterable interface:
 *
 *  begin() -- returns a StorageT*
 *  end() -- returns a StorageT*
 *  size() -- returns size of the Segment iteration space (RAJA::Index_type)
 *
 * NOTE: TypedListSegment supports the option for the segment to own the
 *       its index data or simply use the index array passed to the constructor.
 *       Owning the index data is the default; an array is created in the
 *       memory space specified by the camp resource object and the values are
 *       copied from the input array to that. Ownership of the indices is
 *       determined by an optional ownership enum value passed to the
 *       constructor.
 *
 * Usage:
 *
 * A common C-style loop traversal pattern using an indirection array would be:
 *
 * \verbatim
 * const T* indices = ...;
 * for (T i = begin; i < end; ++i) {
 *   // loop body -- use indices[i] as index value
 * }
 * \endverbatim
 *
 * A TypedListSegment would be used with a RAJA forall execution template as:
 *
 * \verbatim
 * camp::resources::Resource resource{ camp resource type };
 * TypedListSegment<T> listseg(indices, length, resource);
 *
 * forall<exec_pol>(listseg, [=] (T i) {
 *   // loop body -- use i as index value
 * });
 * \endverbatim
 *
 ******************************************************************************
 */
template<typename StorageT>
class TypedListSegment
{
public:
  //@{
  //!   @name Types used in implementation based on template parameter.

  //! The underlying value type for index storage
  using value_type = StorageT;

  //! The underlying terator type
  using iterator = StorageT*;

  //! Expose underlying index type for consistency with other segment types
  using IndexType = StorageT;

  //@}

  //@{
  //!   @name Constructors and destructor.

  /*!
   * \brief Construct a list segment from given array with specified length
   *        and use given camp resource to allocate list segment index data
   *        if owned by this list segment.
   *
   * \param values array of indices defining iteration space of segment
   * \param length number of indices
   * \param resource camp resource defining memory space where index data live
   * \param owned optional enum value indicating whether segment owns indices
   * (Owned or Unowned). Default is Owned.
   *
   * If 'Unowned' is passed as last argument, the segment will not own its
   * index data. In this case, caller must manage array lifetime properly.
   */
  TypedListSegment(const value_type* values,
                   Index_type length,
                   camp::resources::Resource resource,
                   IndexOwnership owned = Owned)
      : m_resource(nullptr),
        m_owned(Unowned),
        m_data(nullptr),
        m_size(0)
  {
    initIndexData(values, length, resource, owned);
  }

  /*!
   * \brief Construct a list segment from given container of indices.
   *
   * \param container container of indices for segment
   * \param resource camp resource defining memory space where index data live
   *
   * The given container must provide methods begin(), end(), and size(). The
   * segment constructor will make a copy of the container's index data in
   * the memory space defined by the resource argument.
   *
   * Constructor assumes container data lives in host memory space.
   */
  template<typename Container>
  TypedListSegment(const Container& container,
                   camp::resources::Resource resource)
      : m_resource(nullptr),
        m_owned(Unowned),
        m_data(nullptr),
        m_size(container.size())
  {
    if (m_size > 0)
    {

      camp::resources::Resource host_res {camp::resources::Host::get_default()};

      value_type* tmp = host_res.allocate<value_type>(m_size);

      auto dest      = tmp;
      auto src       = container.begin();
      auto const end = container.end();
      while (src != end)
      {
        *dest = *src;
        ++dest;
        ++src;
      }

      m_resource = new camp::resources::Resource(resource);
      m_data     = m_resource->allocate<value_type>(m_size);
      m_resource->memcpy(m_data, tmp, sizeof(value_type) * m_size);
      m_resource->wait();
      m_owned = Owned;

      host_res.deallocate(tmp);
    }
  }

  //! Disable compiler generated constructor
  TypedListSegment() = delete;

  //! Copy constructor for list segment
  //  As this may be called from a lambda in a
  //  RAJA method we perform a shallow copy
  RAJA_HOST_DEVICE TypedListSegment(const TypedListSegment& other)
      : m_resource(nullptr),
        m_owned(Unowned),
        m_data(other.m_data),
        m_size(other.m_size)
  {}

  //! Copy assignment for list segment
  //  As this may be called from a lambda in a
  //  RAJA method we perform a shallow copy
  RAJA_HOST_DEVICE TypedListSegment& operator=(const TypedListSegment& other)
  {
    clear();
    m_resource = nullptr;
    m_owned    = Unowned;
    m_data     = other.m_data;
    m_size     = other.m_size;
  }

  //! move assignment for list segment
  //  As this may be called from a lambda in a
  //  RAJA method we perform a shallow copy
  RAJA_HOST_DEVICE TypedListSegment& operator=(TypedListSegment&& rhs)
  {
    clear();
    m_resource = rhs.m_resource;
    m_owned    = rhs.m_owned;
    m_data     = rhs.m_data;
    m_size     = rhs.m_size;

    rhs.m_resource = nullptr;
    rhs.m_owned    = Unowned;
    rhs.m_data     = nullptr;
    rhs.m_size     = 0;
  }

  //! Move constructor for list segment
  RAJA_HOST_DEVICE TypedListSegment(TypedListSegment&& rhs)
      : m_resource(rhs.m_resource),
        m_owned(rhs.m_owned),
        m_data(rhs.m_data),
        m_size(rhs.m_size)
  {
    rhs.m_owned    = Unowned;
    rhs.m_resource = nullptr;
    rhs.m_size     = 0;
    rhs.m_data     = nullptr;
  }

  //! List segment destructor
  RAJA_HOST_DEVICE ~TypedListSegment() { clear(); }

  //! Clear method to be called
  RAJA_HOST_DEVICE void clear()
  {

#if !defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
    if (m_data != nullptr && m_owned == Owned)
    {
      m_resource->deallocate(m_data);
      delete m_resource;
    }
#endif
    m_data     = nullptr;
    m_resource = nullptr;
    m_owned    = Unowned;
    m_size     = 0;
  }

  //@}

  //@{
  //!   @name Accessor methods

  /*!
   * \brief Get iterator to the beginning of this segment
   */
  RAJA_HOST_DEVICE iterator begin() const { return m_data; }

  /*!
   * \brief Get iterator to the end of this segment
   */
  RAJA_HOST_DEVICE iterator end() const { return m_data + m_size; }

  /*!
   * \brief Get size of this segment (number of indices)
   */
  RAJA_HOST_DEVICE Index_type size() const { return m_size; }

  /*!
   * \brief Get ownership of index data (Owned/Unowned)
   */
  RAJA_HOST_DEVICE IndexOwnership getIndexOwnership() const { return m_owned; }

  //@}

  //@{
  //!   @name Segment comparison methods

  /*!
   * \brief Compare this segment's indices to an array of values
   *
   * \param container pointer to array of values
   * \param len number of values to compare
   *
   * \return true if segment size is same as given length value and values in
   *         given array match segment index values, else false
   *
   * Method assumes values in given array and segment indices both live in host
   * memory space.
   */
  RAJA_HOST_DEVICE bool indicesEqual(const value_type* container,
                                     Index_type len) const
  {
    if (container == m_data) return len == m_size;
    if (len != m_size || container == nullptr || m_data == nullptr)
      return false;
    for (Index_type i = 0; i < m_size; ++i)
      if (m_data[i] != container[i]) return false;
    return true;
  }

  /*!
   * \brief Compare this segment to another for equality
   *
   * \return true if both segments are the same size and indices match,
   *         else false
   *
   * Method assumes indices in both segments live in host memory space.
   */
  RAJA_HOST_DEVICE bool operator==(const TypedListSegment& other) const
  {
    return (indicesEqual(other.m_data, other.m_size));
  }

  /*!
   * \brief Compare this segment to another for inequality
   *
   * \return true if segments are not the same size or indices do not match,
   *         else false
   *
   * Method assumes indices in both segments live in host memory space.
   */
  RAJA_HOST_DEVICE bool operator!=(const TypedListSegment& other) const
  {
    return (!(*this == other));
  }

  //@}

  /*!
   * \brief Swap this segment with another
   */
  RAJA_HOST_DEVICE void swap(TypedListSegment& other)
  {
    camp::safe_swap(m_resource, other.m_resource);
    camp::safe_swap(m_data, other.m_data);
    camp::safe_swap(m_size, other.m_size);
    camp::safe_swap(m_owned, other.m_owned);
  }

private:
  //
  // Initialize segment data based on whether object owns the index data.
  //
  void initIndexData(const value_type* container,
                     Index_type len,
                     camp::resources::Resource resource_,
                     IndexOwnership container_own)
  {

    // empty list segment
    if (len <= 0 || container == nullptr)
    {
      m_data  = nullptr;
      m_size  = 0;
      m_owned = Unowned;
      return;
    }

    // some non-zero size -- initialize accordingly
    m_size  = len;
    m_owned = container_own;
    if (m_owned == Owned)
    {

      m_resource = new camp::resources::Resource(resource_);

      camp::resources::Resource host_res {camp::resources::Host::get_default()};

      value_type* tmp = host_res.allocate<value_type>(m_size);

      for (Index_type i = 0; i < m_size; ++i)
      {
        tmp[i] = container[i];
      }

      m_data = m_resource->allocate<value_type>(m_size);
      m_resource->memcpy(m_data, tmp, sizeof(value_type) * m_size);
      m_resource->wait();

      host_res.deallocate(tmp);

      return;
    }

    // list segment accesses container data directly.
    // Uh-oh. Using evil const_cast....
    m_data = const_cast<value_type*>(container);
  }

  // Copy of camp resource passed to ctor
  camp::resources::Resource* m_resource;

  // Ownership flag to guide data copying/management
  IndexOwnership m_owned;

  // Buffer storage for segment index data
  value_type* RAJA_RESTRICT m_data;

  // Size of list segment
  Index_type m_size;
};

//! Alias for A TypedListSegment<Index_type>
using ListSegment = TypedListSegment<Index_type>;

}  // namespace RAJA

namespace std
{

//! Specialization of std::swap for TypedListSegment
template<typename StorageT>
RAJA_INLINE void swap(RAJA::TypedListSegment<StorageT>& a,
                      RAJA::TypedListSegment<StorageT>& b)
{
  a.swap(b);
}
}  // namespace std

#endif  // closing endif for header file include guard
