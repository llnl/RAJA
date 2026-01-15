/*!
 ******************************************************************************
 *
 * \file RangeSegment.hpp
 *
 * \brief  Header file containing definitions of RAJA range segment classes.
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

#ifndef RAJA_RangeSegment_HPP
#define RAJA_RangeSegment_HPP

#include "RAJA/config.hpp"

#include <iostream>

#include "RAJA/internal/Iterators.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/index/IndexValue.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \class TypedRangeSegment
 *
 * \brief  Segment class representing a contiguous range of typed indices
 *
 * \tparam StorageT underlying data type for the segment indices (required)
 * \tparam DiffT underlying data type for the difference between two segment
 *         indices (optional)
 *
 * A TypedRangeSegment models an Iterable interface:
 *
 *  begin() -- returns an iterator (TypedRangeSegment::iterator)
 *  end() -- returns an iterator (TypedRangeSegment::iterator)
 *  size() -- returns the total size of the Segment iteration space (DiffT)
 *
 * NOTE: TypedRangeSegment::iterator is a RandomAccessIterator
 *
 * NOTE: TypedRangeSegment supports negative indices; e.g., an interval of
 *       indices [-5, 3).
 *
 * NOTE: Proper handling of indices strides requires that StorageT is a
 *       signed type.
 *
 * Usage:
 *
 * A common C-style loop traversal pattern would be:
 *
 * \verbatim
 * for (T i = begin; i < end; ++i) {
 *   // loop body -- use i as index value
 * }
 * \endverbatim
 *
 * Using a TypedRangeSegment, this becomes:
 *
 * \verbatim
 * TypedRangeSegment<T> seg (begin, end);
 * for (auto i = seg.begin(); i != seg.end(); ++i) {
 *   // loop body -- use (*i) as index value
 * }
 * \endverbatim
 *
 * This can also be used in a C++11 style range-based for:
 *
 * \verbatim
 * for (auto i : TypedRangeSegment<T>(begin, end)) {
 *   // loop body -- use i as index value
 * }
 * \endverbatim
 *
 * Or, as it would be commonly used with a RAJA forall execution template:
 * \verbatim
 * forall<exec_pol>(TypedRangeSegment<T>(beg, end), [=] (T i) {
 *   // loop body -- use i as index value
 * });
 * \endverbatim
 *
 ******************************************************************************
 */
template<typename StorageT,
         typename DiffT = make_signed_t<strip_index_type_t<StorageT>>>
struct TypedRangeSegment
{

  //
  // Static asserts to provide some useful error messages during compilation
  // for incorrect usage.
  //
  static_assert(std::is_signed<DiffT>::value,
                "TypedRangeSegment DiffT requires signed type.");
  static_assert(!std::is_floating_point<StorageT>::value,
                "TypedRangeSegment Type must be non floating point.");

  //@{
  //!   @name Types used in implementation based on template parameters.

  //! The underlying iterator type
  using iterator = Iterators::numeric_iterator<StorageT, DiffT>;

  //! The underlying value type
  using value_type = StorageT;

  //! The underlying type for a difference in index values
  using IndexType = DiffT;

  //@}

  //@{
  //!   @name Constructors, destructor, and copy assignment.

  /*!
   * \brief Construct a range segment repreenting the interval [begin, end)
   *
   * \param begin start value (inclusive) for the range
   * \param end end value (exclusive) for the range
   */
  using StripStorageT = strip_index_type_t<StorageT>;

  RAJA_HOST_DEVICE constexpr TypedRangeSegment(StripStorageT begin,
                                               StripStorageT end)
      : m_begin(iterator(begin)),
        m_end(begin > end ? m_begin : iterator(end))
  {}

  //! Disable compiler generated constructor
  RAJA_HOST_DEVICE TypedRangeSegment() = delete;

  //! Defaulted move constructor
  constexpr TypedRangeSegment(TypedRangeSegment&&) = default;

  //! Defaulted copy constructor
  constexpr TypedRangeSegment(TypedRangeSegment const&) = default;

  //! Defaulted copy assignment operator
  RAJA_INLINE TypedRangeSegment& operator=(TypedRangeSegment const&) = default;

  //! Defaulted destructor
  RAJA_INLINE ~TypedRangeSegment() = default;

  //@}

  //@{
  //!   @name Accessor methods

  /*!
   * \brief Get iterator to the beginning of this segment
   */
  RAJA_HOST_DEVICE RAJA_INLINE iterator begin() const { return m_begin; }

  /*!
   * \brief Get iterator to the end of this segment
   */
  RAJA_HOST_DEVICE RAJA_INLINE iterator end() const { return m_end; }

  /*!
   * \brief Get size of this segment (end - begin)
   */
  RAJA_HOST_DEVICE RAJA_INLINE DiffT size() const { return m_end - m_begin; }

  //@}

  //@{
  //!   @name Segment comparison methods

  /*!
   * \brief Compare this segment to another for equality
   *
   * \return true if begin and end match, else false
   */
  RAJA_HOST_DEVICE RAJA_INLINE bool operator==(TypedRangeSegment const& o) const
  {
    // someday this shall be replaced with a compiler-generated operator==
    return m_begin == o.m_begin && m_end == o.m_end;
  }

  /*!
   * \brief Compare this segment to another for inequality
   *
   * \return true if begin or end does not match, else false
   */
  RAJA_HOST_DEVICE RAJA_INLINE bool operator!=(TypedRangeSegment const& o) const
  {
    return !(operator==(o));
  }

  //@}

  /*!
   * \brief Get a new TypedRangeSegment instance representing a slice of
   *        existing segment
   *
   * \param begin start iterate of new range
   * \param length maximum length of new range
   * \return TypedRangeSegment representing the interval
   *         [ *begin() + begin, min( *begin() + begin + length, *end() ) )
   *
   * Here's an example of a slice operation on a range segment with negative
   * indices:
   *
   *   \verbatim
   *
   *     // r represents the index interval [-4, 4)
   *     auto r = RAJA::TypedRangeSegment<int>(-4, 4);
   *
   *     // s repreents the subinterval  [-3, 2)
   *     auto s = r.slice(1, 5);
   *
   *   \endverbatim
   */
  RAJA_HOST_DEVICE RAJA_INLINE TypedRangeSegment slice(StorageT begin,
                                                       DiffT length) const
  {
    StorageT start = m_begin[0] + begin;
    StorageT end   = start + length > m_end[0] ? m_end[0] : start + length;

    return TypedRangeSegment {stripIndexType(start), stripIndexType(end)};
  }

  /*!
   * \brief Swap this segment with another
   */
  RAJA_HOST_DEVICE RAJA_INLINE void swap(TypedRangeSegment& other)
  {
    camp::safe_swap(m_begin, other.m_begin);
    camp::safe_swap(m_end, other.m_end);
  }

private:
  // Member variable for begin iterator
  iterator m_begin;

  // Member variable for end iterator
  iterator m_end;
};

/*!
 ******************************************************************************
 *
 * \class TypedRangeStrideSegment
 *
 * \brief  Segment class representing a strided range of typed indices
 *
 * \tparam StorageT underlying data type for the segment indices (required)
 * \tparam DiffT underlying data type for the difference between two segment
 *         indices (optional)
 *
 * A TypedRangeStrideSegment models an Iterable interface:
 *
 *  begin() -- returns an iterator (TypedRangeStrideSegment::iterator)
 *  end() -- returns an iterator (TypedRangeStrideSegment::iterator)
 *  size() -- returns total size of the Segment iteration space, which is
 *            not the distance between begin() and end() for non-unit strides
 *
 * NOTE: TypedRangeStrideSegment::iterator is a RandomAccessIterator
 *
 * NOTE: TypedRangeStrideSegment allows for positive or negative strides and
 *       indices. This allows for forward (stride > 0) or backward (stride < 0)
 *       traversal of the iteration space. A stride of zero is undefined and
 *       will cause divide-by-zero errors.
 *
 * As with RangeSegment, the iteration space is inclusive of begin() and
 * exclusive of end()
 *
 * For positive strides, begin() > end() implies size()==0
 * For negative strides, begin() < end() implies size()==0
 *
 * NOTE: Proper handling of negative strides and indices requires that
 *       StorageT is a signed type.
 *
 * Usage:
 *
 * A common C-style loop traversal pattern would be:
 *
 * \verbatim
 * for (T i = begin; i < end; i += stride) {
 *   // loop body -- use i as index value
 * }
 * \endverbatim
 *
 * Or, equivalently for a negative stride (stride < 0):
 *
 * \verbatim
 * for (T i = begin; i > end; i += stride) {
 *   // loop body -- use i as index value
 * }
 * \endverbatim
 *
 * \verbatim
 * Using a TypedRangeStrideSegment, this becomes:
 * TypedRangeStrideSegment<T> seg (begin, end, stride);
 * for (auto i = seg.begin(); i != seg.end(); ++i) {
 *   // loop body -- use (*i) as index value
 * }
 * \endverbatim
 *
 * This can also be used in a C++11 style range-based for:
 *
 * \verbatim
 * for (auto i : TypedRangeStrideSegment<T>(begin, end, stride)) {
 *   // loop body -- use i as index value
 * }
 * \endverbatim
 *
 * Or, as it would be commonly used with a RAJA forall execution template:
 * \verbatim
 * forall<exec_pol>(TypedRangeStrideSegment<T>(beg, end, stride), [=] (T i) {
 *   // loop body -- use i as index value
 * });
 * \endverbatim
 *
 ******************************************************************************
 */
template<typename StorageT,
         typename DiffT = make_signed_t<strip_index_type_t<StorageT>>>
struct TypedRangeStrideSegment
{

  //
  // Static asserts to provide some useful error messages during compilation
  // for incorrect usage.
  //
  static_assert(std::is_signed<DiffT>::value,
                "TypedRangeStrideSegment DiffT requires signed type.");
  static_assert(!std::is_floating_point<StorageT>::value,
                "TypedRangeStrideSegment Type must be non floating point.");

  //@{
  //!   @name Types used in implementation based on template parameters.

  //! The underlying iterator type
  using iterator = Iterators::strided_numeric_iterator<StorageT, DiffT>;

  //! The underlying value type
  using value_type = StorageT;

  //! The underlying type for a difference in index values
  using IndexType = DiffT;

  //@}

  //@{
  //!   @name Constructors, destructor, and copy assignment.

  /*!
   * \brief Construct a range segment for the interval [begin, end) with
   *        given stride
   *
   * \param begin start value (inclusive) for the range
   * \param end end value (exclusive) for the range
   * \param stride stride value when iterating over the range
   */
  using StripStorageT = strip_index_type_t<StorageT>;

  RAJA_HOST_DEVICE TypedRangeStrideSegment(StripStorageT begin,
                                           StripStorageT end,
                                           DiffT stride)
      : m_begin(iterator(begin, stride)),
        m_end(iterator(end, stride)),
        // essentially a ceil((end-begin)/stride) but using integer math,
        // and allowing for negative strides
        m_size((end - begin + stride - (stride > 0 ? 1 : -1)) / stride)
  {
    // clamp range when end is unreachable from begin without wrapping
    if (stride < 0 && end > begin)
    {
      m_end = m_begin;
    }
    else if (stride > 0 && end < begin)
    {
      m_end = m_begin;
    }
    // m_size initialized as negative indicates a zero iteration space
    m_size = m_size < DiffT {0} ? DiffT {0} : m_size;
  }

  //! Disable compiler generated constructor
  TypedRangeStrideSegment() = delete;

  //! Defaulted move constructor
  TypedRangeStrideSegment(TypedRangeStrideSegment&&) = default;

  //! Defaulted copy constructor
  TypedRangeStrideSegment(TypedRangeStrideSegment const&) = default;

  //! Defaulted copy assignment operator
  TypedRangeStrideSegment& operator=(TypedRangeStrideSegment const&) = default;

  //! Defaulted destructore
  ~TypedRangeStrideSegment() = default;

  //@}

  //@{
  //!   @name Accessor methods

  /*!
   * \brief Get iterator to the beginning of this segment
   */
  RAJA_HOST_DEVICE iterator begin() const { return m_begin; }

  /*!
   * \brief Get iterator to the end of this segment
   */
  RAJA_HOST_DEVICE iterator end() const { return m_end; }

  /*!
   * \brief Get size of this segment
   *
   * The size is the number of iterates in the
   * interval [begin, end) when striding over it
   */
  RAJA_HOST_DEVICE DiffT size() const { return m_size; }

  //@}

  //@{
  //!   @name Segment comparison methods

  /*!
   * \brief Compare this segment to another for equality
   *
   * \return true if begin, end, and size match, else false
   */
  RAJA_HOST_DEVICE bool operator==(TypedRangeStrideSegment const& o) const
  {
    // someday this shall be replaced with a compiler-generated operator==
    return m_begin == o.m_begin && m_end == o.m_end && m_size == o.m_size;
  }

  /*!
   * \brief Compare this segment to another for inequality
   *
   * \return true if begin, end, or size does not match, else false
   */
  RAJA_HOST_DEVICE RAJA_INLINE bool operator!=(
      TypedRangeStrideSegment const& o) const
  {
    return !(operator==(o));
  }

  //@}

  /*!
   * \brief Get a new TypedRangeStrideSegment instance representing a slice of
   *        existing segment
   *
   * \param begin start iterate of new range
   * \param length maximum length of new range
   *
   * \return TypedRangeStrideSegment representing the interval
   *         [ *begin() + begin * stride,
   *           min( *begin() + (begin + length) * stride, *end() )
   *
   * Here's an example of a slice operation on a range segment with a negative
   * stride:
   *
   *   \verbatim
   *
   *     // r represents the set of indices {10, 9, ..., 1, 0}
   *     auto r = RAJA::TypedRangeSegment<int>(10, -1, -1);
   *
   *     // s repreents the subset of indices {4, 3, 2, 1, 0}
   *     // Note: the length of s is 5, not 6, because there are only
   *     //       5 indices in r starting at the 6th entry
   *     auto s = r.slice(6, 6);
   *
   *   \endverbatim
   */
  RAJA_HOST_DEVICE TypedRangeStrideSegment slice(StorageT begin,
                                                 DiffT length) const
  {
    StorageT stride = m_begin.get_stride();
    StorageT start  = m_begin[0] + begin * stride;
    StorageT end    = start + stride * length;

    if (stride > 0)
    {
      end = end > m_end[0] ? m_end[0] : end;
    }
    else
    {
      end = end < m_end[0] ? m_end[0] : end;
    }

    return TypedRangeStrideSegment {stripIndexType(start), stripIndexType(end),
                                    m_begin.get_stride()};
  }

  /*!
   * \brief Swap this segment with another
   */
  RAJA_HOST_DEVICE void swap(TypedRangeStrideSegment& other)
  {
    camp::safe_swap(m_begin, other.m_begin);
    camp::safe_swap(m_end, other.m_end);
    camp::safe_swap(m_size, other.m_size);
  }

private:
  // Member variable for begin iterator
  iterator m_begin;

  // Member variable for end iterator
  iterator m_end;

  // Member variable for size of segment
  DiffT m_size;
};

//! Alias for TypedRangeSegment<Index_type>
using RangeSegment = TypedRangeSegment<Index_type>;

//! Alias for TypedRangeStrideSegment<Index_type>
using RangeStrideSegment = TypedRangeStrideSegment<Index_type>;

namespace detail
{

template<typename T, typename... Rest>
struct common_type
    : std::common_type<T, typename std::common_type<Rest...>::type>
{};

template<typename T>
struct common_type<T>
{
  using type = T;
};

template<typename... Ts>
using common_type_t = typename common_type<Ts...>::type;

}  // namespace detail

/*!
 * \brief Function to make a TypedRangeSegment for the interval [begin, end)
 *
 *  \return a newly constructed TypedRangeSegment where the
 *          value_type is equivilent to the common type of
 *          @begin and @end. If there is no common type, then
 *          a compiler error will be produced.
 */
template<typename BeginT,
         typename EndT,
         typename Common = detail::common_type_t<BeginT, EndT>>
RAJA_HOST_DEVICE TypedRangeSegment<Common> make_range(BeginT&& begin,
                                                      EndT&& end)
{
  return {begin, end};
}

/*!
 * \brief Function to make a TypedRangeStride Segment for the interval
 *        [begin, end) with given stride
 *
 *  \return a newly constructed TypedRangeStrideSegment where
 *          the value_type is equivilent to the common type of
 *          @begin, @end, and @stride. If there is no common
 *          type, then a compiler error will be produced.
 */
template<typename BeginT,
         typename EndT,
         typename StrideT,
         typename Common = detail::common_type_t<BeginT, EndT>>
RAJA_HOST_DEVICE TypedRangeStrideSegment<Common> make_strided_range(
    BeginT&& begin,
    EndT&& end,
    StrideT&& stride)
{
  static_assert(std::is_signed<StrideT>::value,
                "make_strided_segment : stride must be signed.");
  static_assert(
      std::is_same<make_signed_t<EndT>, StrideT>::value,
      "make_stride_segment : stride and end must be of similar types.");
  return {begin, end, stride};
}

namespace concepts
{

template<typename T, typename U>
struct RangeConstructible
    : DefineConcept(camp::val<RAJA::detail::common_type_t<T, U>>())
{};

template<typename T, typename U, typename V>
struct RangeStrideConstructible
    : DefineConcept(camp::val<RAJA::detail::common_type_t<T, U, V>>())
{};

}  // namespace concepts

namespace type_traits
{

DefineTypeTraitFromConcept(is_range_constructible,
                           RAJA::concepts::RangeConstructible);

DefineTypeTraitFromConcept(is_range_stride_constructible,
                           RAJA::concepts::RangeStrideConstructible);

}  // namespace type_traits

}  // namespace RAJA

namespace std
{

//! Specialization of std::swap for TypedRangeSegment
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE void swap(RAJA::TypedRangeSegment<T>& a,
                                       RAJA::TypedRangeSegment<T>& b)
{
  a.swap(b);
}

//! Specialization of std::swap for TypedRangeStrideSegment
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE void swap(RAJA::TypedRangeStrideSegment<T>& a,
                                       RAJA::TypedRangeStrideSegment<T>& b)
{
  a.swap(b);
}

}  // namespace std

#endif  // closing endif for header file include guard
