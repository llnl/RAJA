/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for strongly-typed integer class.
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

#ifndef RAJA_INDEXVALUE_HPP
#define RAJA_INDEXVALUE_HPP

#include "RAJA/config.hpp"

#include <string>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

struct IndexValueBase
{};

/*!
 * \brief Strongly typed "integer" class.
 *
 * Allows integers to be associated with a type, and disallows automatic
 * conversion.
 *
 * Useful for maintaining correctness in multidimensional loops and arrays.
 *
 * Use the RAJA_INDEX_VALUE(NAME) macro to define new indices.
 *
 * Yes, this uses the curiously-recurring template pattern.
 */
template<typename TYPE, typename VALUE = RAJA::Index_type>
struct IndexValue : public IndexValueBase
{

  using value_type = VALUE;

  //! Default constructor initializes value to 0.
  RAJA_INLINE constexpr IndexValue()                   = default;
  constexpr RAJA_INLINE IndexValue(IndexValue const&)  = default;
  constexpr RAJA_INLINE IndexValue(IndexValue&&)       = default;
  RAJA_INLINE IndexValue& operator=(IndexValue const&) = default;
  RAJA_INLINE IndexValue& operator=(IndexValue&&)      = default;

  /*!
   * \brief Explicit constructor.
   * \param v   Initial value
   */
  RAJA_HOST_DEVICE RAJA_INLINE constexpr explicit IndexValue(value_type v)
      : value(v)
  {}

  //! Dereference provides cast-to-integer.
  RAJA_HOST_DEVICE RAJA_INLINE value_type& operator*() { return value; }

  //! Dereference provides cast-to-integer.
  RAJA_HOST_DEVICE RAJA_INLINE const value_type& operator*() const
  {
    return value;
  }

  //! postincrement -- returns a copy
  RAJA_HOST_DEVICE RAJA_INLINE TYPE operator++(int)
  {
    TYPE self(value);
    value++;
    return self;
  }

  //! preincrement stored index
  RAJA_HOST_DEVICE RAJA_INLINE TYPE& operator++()
  {
    value++;
    return static_cast<TYPE&>(*this);
  }

  //! postdecrement -- returns a copy
  RAJA_HOST_DEVICE RAJA_INLINE TYPE operator--(int)
  {
    TYPE self(value);
    value--;
    return self;
  }

  //! preincrement stored index
  RAJA_HOST_DEVICE RAJA_INLINE TYPE& operator--()
  {
    value--;
    return static_cast<TYPE&>(*this);
  }

  //! addition to underlying index from an Index_type
  RAJA_HOST_DEVICE RAJA_INLINE TYPE operator+(value_type a) const
  {
    return TYPE(value + a);
  }

  //! addition to underlying index from another strong type
  RAJA_HOST_DEVICE RAJA_INLINE TYPE operator+(TYPE a) const
  {
    return TYPE(value + a.value);
  }

  //! subtraction to underlying index from an Index_type
  RAJA_HOST_DEVICE RAJA_INLINE TYPE operator-(value_type a) const
  {
    return TYPE(value - a);
  }

  //! subtraction to underlying index from another strong type
  RAJA_HOST_DEVICE RAJA_INLINE TYPE operator-(TYPE a) const
  {
    return TYPE(value - a.value);
  }

  //! multiplication to underlying index from an Index_type
  RAJA_HOST_DEVICE RAJA_INLINE TYPE operator*(value_type a) const
  {
    return TYPE(value * a);
  }

  //! multiplication to underlying index from another strong type
  RAJA_HOST_DEVICE RAJA_INLINE TYPE operator*(TYPE a) const
  {
    return TYPE(value * a.value);
  }

  //! division to underlying index from an Index_type
  RAJA_HOST_DEVICE RAJA_INLINE TYPE operator/(value_type a) const
  {
    return TYPE(value / a);
  }

  //! division to underlying index from another strong type
  RAJA_HOST_DEVICE RAJA_INLINE TYPE operator/(TYPE a) const
  {
    return TYPE(value / a.value);
  }

  //! modulus to underlying index from an Index_type
  RAJA_HOST_DEVICE RAJA_INLINE TYPE operator%(value_type a) const
  {
    return TYPE(value % a);
  }

  //! modulus to underlying index from another strong type
  RAJA_HOST_DEVICE RAJA_INLINE TYPE operator%(TYPE a) const
  {
    return TYPE(value % a.value);
  }

  RAJA_HOST_DEVICE RAJA_INLINE TYPE& operator+=(value_type x)
  {
    value += x;
    return static_cast<TYPE&>(*this);
  }

  RAJA_HOST_DEVICE RAJA_INLINE TYPE& operator+=(TYPE x)
  {
    value += x.value;
    return static_cast<TYPE&>(*this);
  }

  RAJA_HOST_DEVICE RAJA_INLINE TYPE& operator-=(value_type x)
  {
    value -= x;
    return static_cast<TYPE&>(*this);
  }

  RAJA_HOST_DEVICE RAJA_INLINE TYPE& operator-=(TYPE x)
  {
    value -= x.value;
    return static_cast<TYPE&>(*this);
  }

  RAJA_HOST_DEVICE RAJA_INLINE TYPE& operator*=(value_type x)
  {
    value *= x;
    return static_cast<TYPE&>(*this);
  }

  RAJA_HOST_DEVICE RAJA_INLINE TYPE& operator*=(TYPE x)
  {
    value *= x.value;
    return static_cast<TYPE&>(*this);
  }

  RAJA_HOST_DEVICE RAJA_INLINE TYPE& operator/=(value_type x)
  {
    value /= x;
    return static_cast<TYPE&>(*this);
  }

  RAJA_HOST_DEVICE RAJA_INLINE TYPE& operator/=(TYPE x)
  {
    value /= x.value;
    return static_cast<TYPE&>(*this);
  }

  RAJA_HOST_DEVICE RAJA_INLINE bool operator<(value_type x) const
  {
    return (value < x);
  }

  RAJA_HOST_DEVICE RAJA_INLINE bool operator<(TYPE x) const
  {
    return (value < x.value);
  }

  RAJA_HOST_DEVICE RAJA_INLINE bool operator<=(value_type x) const
  {
    return (value <= x);
  }

  RAJA_HOST_DEVICE RAJA_INLINE bool operator<=(TYPE x) const
  {
    return (value <= x.value);
  }

  RAJA_HOST_DEVICE RAJA_INLINE bool operator>(value_type x) const
  {
    return (value > x);
  }

  RAJA_HOST_DEVICE RAJA_INLINE bool operator>(TYPE x) const
  {
    return (value > x.value);
  }

  RAJA_HOST_DEVICE RAJA_INLINE bool operator>=(value_type x) const
  {
    return (value >= x);
  }

  RAJA_HOST_DEVICE RAJA_INLINE bool operator>=(TYPE x) const
  {
    return (value >= x.value);
  }

  RAJA_HOST_DEVICE RAJA_INLINE bool operator==(value_type x) const
  {
    return (value == x);
  }

  RAJA_HOST_DEVICE RAJA_INLINE bool operator==(TYPE x) const
  {
    return (value == x.value);
  }

  RAJA_HOST_DEVICE RAJA_INLINE bool operator!=(value_type x) const
  {
    return (value != x);
  }

  RAJA_HOST_DEVICE RAJA_INLINE bool operator!=(TYPE x) const
  {
    return (value != x.value);
  }

  // This is not implemented... but should be by the derived type
  // this is done by the macro
  static std::string getName();

protected:
  value_type value = 0;
};

namespace internal
{

template<typename TO, typename FROM>
constexpr RAJA_HOST_DEVICE RAJA_INLINE TO convertIndex_helper(FROM const val)
{
  return TO(val);
}

template<typename TO, typename FROM>
constexpr RAJA_HOST_DEVICE RAJA_INLINE TO
convertIndex_helper(typename FROM::IndexValueType const val)
{
  return static_cast<TO>(*val);
}


}  // namespace internal

/*!
 * \brief Function provides a way to take either an int or any Index<> type, and
 * convert it to another type, possibly another Index or an int.
 *
 */
template<typename TO, typename FROM>
constexpr RAJA_HOST_DEVICE RAJA_INLINE TO convertIndex(FROM const val)
{
  return internal::convertIndex_helper<TO, FROM>(val);
}

/*!
 * \brief Function that strips the strongly typed Index<> and returns its
 * underlying value_type value.
 */
// This version is enabled if FROM is a strongly typed class
template<typename FROM>
constexpr RAJA_HOST_DEVICE RAJA_INLINE
    typename std::enable_if<std::is_base_of<IndexValueBase, FROM>::value,
                            typename FROM::value_type>::type
    stripIndexType(FROM const val)
{
  return *val;
}

/*
 * enabled if FROM is not a strongly typed class
 */
template<typename FROM>
constexpr RAJA_HOST_DEVICE RAJA_INLINE
    typename std::enable_if<!std::is_base_of<IndexValueBase, FROM>::value,
                            FROM>::type
    stripIndexType(FROM const val)
{
  return val;
}

namespace internal
{
template<typename FROM, typename Enable = void>
struct StripIndexTypeT
{
  using type = FROM;
};

template<typename FROM>
struct StripIndexTypeT<
    FROM,
    typename std::enable_if<std::is_base_of<IndexValueBase, FROM>::value>::type>
{
  using type = typename FROM::value_type;
};
}  // namespace internal

/*!
 * \brief Strips a strongly typed index to its underlying type
 * In the case of a non-strongly typed index, the original type is returned.
 *
 * \param FROM the original type
 */
template<typename FROM>
using strip_index_type_t = typename internal::StripIndexTypeT<FROM>::type;

/*!
 * \brief Converts a type into a signed type. Also handles floating point
 * types as std::make_signed only supports integral types.
 *
 * \param FROM the original type
 */
template<typename FROM>
using make_signed_t =
    typename std::conditional<std::is_floating_point<FROM>::value,
                              std::common_type<FROM>,
                              std::make_signed<FROM>>::type::type;

}  // namespace RAJA

/*!
 * \brief Helper Macro to create new Index types.
 * \param TYPE the name of the type
 * \param NAME a string literal to identify this index type
 */
#define RAJA_INDEX_VALUE(TYPE, NAME)                                           \
  class TYPE : public ::RAJA::IndexValue<TYPE>                                 \
  {                                                                            \
    using parent = ::RAJA::IndexValue<TYPE>;                                   \
                                                                               \
  public:                                                                      \
    using IndexValueType = TYPE;                                               \
    RAJA_HOST_DEVICE RAJA_INLINE TYPE() : parent::IndexValue() {}              \
    RAJA_HOST_DEVICE RAJA_INLINE explicit TYPE(::RAJA::Index_type v)           \
        : parent::IndexValue(v)                                                \
    {}                                                                         \
    static inline std::string getName() { return NAME; }                       \
  };

/*!
 * \brief Helper Macro to create new Index types.
 * \param TYPE the name of the type
 * \param IDXT the index types value type
 * \param NAME a string literal to identify this index type
 */
#define RAJA_INDEX_VALUE_T(TYPE, IDXT, NAME)                                   \
  class TYPE : public ::RAJA::IndexValue<TYPE, IDXT>                           \
  {                                                                            \
  public:                                                                      \
    RAJA_HOST_DEVICE RAJA_INLINE TYPE()                                        \
        : RAJA::IndexValue<TYPE, IDXT>::IndexValue()                           \
    {}                                                                         \
    RAJA_HOST_DEVICE RAJA_INLINE explicit TYPE(IDXT v)                         \
        : RAJA::IndexValue<TYPE, IDXT>::IndexValue(v)                          \
    {}                                                                         \
    static inline std::string getName() { return NAME; }                       \
  };

#endif
