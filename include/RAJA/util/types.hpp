/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA type definitions.
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_Types_HPP
#define RAJA_Types_HPP

#include "RAJA/config.hpp"

#include <cstddef>

#if defined(RAJA_USE_COMPLEX)
#include <complex>
#endif

#include "camp/helpers.hpp"

#include "RAJA/util/macros.hpp"

namespace RAJA
{

///
/// Enum for named values with special usage.
///
enum named_usage : int
{
  ignored     = -1,
  unspecified = 0
};

///
/// Enum for named dimensions.
///
enum struct named_dim : int
{
  x = 0,
  y = 1,
  z = 2
};

///
/// Enum for synchronization requirements in some kernel constructs.
///
enum struct kernel_sync_requirement : int
{
  none = 0,
  sync = 1
};

///
/// Classes used to indicate how to map iterations in a loop to indices.
///
namespace iteration_mapping
{

struct DirectUncheckedBase
{};

struct DirectBase
{};

struct LoopBase
{};

struct ContiguousLoopBase : LoopBase
{};

struct StridedLoopBase : LoopBase
{};

struct UnsizedLoopBase
{};

struct SizedLoopBase
{};

template<size_t t_max_iterations>
struct SizedLoopSpecifyingBase : SizedLoopBase
{
  static constexpr size_t max_iterations = t_max_iterations;
};

///
/// DirectUnchecked assumes the loop has the same number of iterations and
/// indices and maps directly without bounds checking from an iteration to an
/// index.
///
/// For example a loop with 4 iterations mapping indices from a range of size 4.
///   int iterations = 4;
///   int range_size = 4;
///   for (int i = 0; i < iterations; ++i) {
///     int index = i;
///     printf("%i -> {%i}", i, index);
///   }
///   // 0 -> {0}
///   // 1 -> {1}
///   // 2 -> {2}
///   // 3 -> {3}
///
struct DirectUnchecked : DirectUncheckedBase
{};

///
/// Direct assumes the loop has enough iterations for all of the indices and
/// maps directly from an iteration to an index.
///
/// For example a loop with 5 iterations mapping indices from a range of size 4.
///   int iterations = 5;
///   int range_size = 4;
///   for (int i = 0; i < iterations; ++i) {
///     if (i < range_size) {
///       int index = i;
///       printf("%i -> {%i}", i, index);
///     } else {
///       printf("%i -> {safely-ignored}", i);
///     }
///   }
///   // 0 -> {0}
///   // 1 -> {1}
///   // 2 -> {2}
///   // 3 -> {3}
///   // 4 -> {safely-ignored}
///
struct Direct : DirectBase
{};

///
/// Contiguousloop assumes the loop has fewer iterations than indices and
/// maps a single iteration to a range of contiguous indices.
///
/// For example a loop with 3 iterations mapping indices from a range of size 8.
///   int iterations = 3;
///   int range_size = 8;
///   int indices_per_iteration = (range_size + iterations-1) / iterations;
///   for (int i = 0; i < iterations; ++i) {
///     printf("%i -> {", i);
///     int index = indices_per_iteration*i;
///     if (index < range_size) {
///       printf("%i", i);
///       for (++index;
///            index < indices_per_iteration*(i+1) && index < range_size;
///            ++index) {
///         printf(", %i", i);
///       }
///     }
///     printf("}");
///   }
///   // 0 -> {0, 1, 2}
///   // 1 -> {3, 4, 5}
///   // 2 -> {6, 7}
///
template<size_t max_iterations>
struct Contiguousloop
    : ContiguousLoopBase,
      std::conditional_t<(max_iterations != named_usage::unspecified),
                         SizedLoopSpecifyingBase<max_iterations>,
                         UnsizedLoopBase>
{};

///
/// StridedLoop assumes the loop has fewer iterations than indices and
/// maps a single iteration to a range of indices strided by the number of
/// iterations in the loop.
///
/// For example a loop with 3 iterations mapping indices from a range of size 8.
///   int iterations = 3;
///   int range_size = 8;
///   for (int i = 0; i < iterations; ++i) {
///     printf("%i -> {", i);
///     int index = i;
///     if (index < range_size) {
///       printf("%i", i);
///       for (index += iterations;
///            index < range_size;
///            index += iterations) {
///         printf(", %i", i);
///       }
///     }
///     printf("}");
///   }
///   // 0 -> {0, 3, 6}
///   // 1 -> {1, 4, 7}
///   // 2 -> {2, 5}
///
template<size_t max_iterations>
struct StridedLoop
    : StridedLoopBase,
      std::conditional_t<(max_iterations != named_usage::unspecified),
                         SizedLoopSpecifyingBase<max_iterations>,
                         UnsizedLoopBase>
{};

}  // namespace iteration_mapping

///
/// Enumeration used to indicate whether ListSegment object owns data
/// representing its indices.
///
enum IndexOwnership
{
  Unowned,
  Owned
};

///
/// Type use for all loop indexing in RAJA constructs.
///
using Index_type = std::ptrdiff_t;

///
/// Integer value for undefined indices and other integer values.
/// Although this is a magic value, it avoids sprinkling them throughout code.
///
const int UndefinedValue = -9999999;

///
/// Template list of sizes
///
template<Index_type... Sizes>
struct SizeList
{};

///
/// Compile time fraction for use with integral types
///
template<typename int_t, int_t numerator, int_t denominator>
struct Fraction
{
  static_assert(denominator != int_t(0), "denominator must not be zero");

  using inverse = Fraction<int_t, denominator, numerator>;

  template<typename new_int_t>
  using rebind =
      Fraction<new_int_t, new_int_t(numerator), new_int_t(denominator)>;

  static constexpr int_t multiply(int_t val) noexcept
  {
    return (val / denominator) * numerator +
           (val % denominator) * numerator / denominator;
  }
};

/*!
 ******************************************************************************
 *
 * \brief RAJA scalar type definitions.
 *
 ******************************************************************************
 */

#if defined(RAJA_USE_DOUBLE)
///
using Real_type = double;

#elif defined(RAJA_USE_FLOAT)
///
using Real_type         = float;

#else
#error RAJA Real_type is undefined!

#endif

#if defined(RAJA_USE_COMPLEX)
///
using Complex_type = std::complex<Real_type>;
#endif

/*
 ******************************************************************************
 *
 * The following items include some setup items for definitions that follow.
 *
 ******************************************************************************
 */

#if defined(RAJA_COMPILER_ICC)
//
// alignment attribute supported for versions > 12
//
#if __ICC >= 1300
using TDRAReal_ptr =
    Real_type* RAJA_RESTRICT __attribute__((align_value(RAJA::DATA_ALIGN)));

using const_TDRAReal_ptr = const TDRAReal_ptr;
#endif

#elif defined(RAJA_COMPILER_GNU)

#elif defined(RAJA_COMPILER_CLANG)
using TDRAReal_ptr =
    Real_type* RAJA_RESTRICT __attribute__((aligned(RAJA::DATA_ALIGN)));

using const_TDRAReal_ptr = const TDRAReal_ptr;

#else

using TDRAReal_ptr = Real_type* RAJA_RESTRICT;

using const_TDRAReal_ptr = const TDRAReal_ptr;

#endif

#if defined(RAJA_USE_PTR_CLASS)
/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted Real_type const pointer.
 *
 ******************************************************************************
 */
class ConstRestrictRealPtr
{
public:
  ///
  /// Ctors and assignment op.
  ///

  constexpr ConstRestrictRealPtr() : dptr(0) { ; }

  constexpr ConstRestrictRealPtr(const Real_type* d) : dptr(d) { ; }

  constexpr ConstRestrictRealPtr& operator=(const Real_type* d)
  {
    ConstRestrictRealPtr copy(d);
    std::swap(dptr, copy.dptr);
    return *this;
  }

  ///
  /// NOTE: Using compiler-generated copy ctor, dtor, and copy assignment op.
  ///

  ///
  ///  Implicit conversion operator to bare const pointer.
  ///
  constexpr operator const Real_type*() { return dptr; }

  ///
  ///  "Explicit conversion operator" to bare const pointer,
  ///  consistent with boost shared ptr.
  ///
  constexpr const Real_type* get() const { return dptr; }

  ///
  /// Bracket operator.
  ///
  constexpr const Real_type& operator[](Index_type i) const
  {
    return ((const Real_type* RAJA_RESTRICT)dptr)[i];
  }

  ///
  /// + operator for pointer arithmetic.
  ///
  constexpr const Real_type* operator+(Index_type i) const { return dptr + i; }

private:
  const Real_type* dptr;
};

/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted Real_type (non-const) pointer.
 *
 ******************************************************************************
 */
class RestrictRealPtr
{
public:
  ///
  /// Ctors and assignment op.
  ///

  constexpr RestrictRealPtr() : dptr(0) { ; }

  constexpr RestrictRealPtr(Real_type* d) : dptr(d) { ; }

  constexpr RestrictRealPtr& operator=(Real_type* d)
  {
    RestrictRealPtr copy(d);
    std::swap(dptr, copy.dptr);
    return *this;
  }

  ///
  /// NOTE: Using compiler-generated copy ctor, dtor, and copy assignment op.
  ///

  ///
  ///  Implicit conversion operator to (non-const) bare pointer.
  ///
  constexpr operator Real_type*() { return dptr; }

  ///
  ///  Implicit conversion operator to const bare pointer.
  ///
  constexpr operator const Real_type*() const { return dptr; }

  ///
  ///  "Explicit conversion operator" to (non-const) bare pointer,
  ///  consistent with boost shared ptr.
  ///
  constexpr Real_type* get() { return dptr; }

  ///
  ///  "Explicit conversion operator" to const bare pointer,
  ///  consistent with boost shared ptr.
  ///
  constexpr const Real_type* get() const { return dptr; }

  ///
  ///  Operator that enables implicit conversion from RestrictRealPtr to
  ///  RestrictRealConstPtr.
  ///
  constexpr operator ConstRestrictRealPtr()
  {
    return ConstRestrictRealPtr(dptr);
  }

  ///
  /// Bracket operator.
  ///
  constexpr Real_type& operator[](Index_type i)
  {
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
  }

  ///
  /// + operator for (non-const) pointer arithmetic.
  ///
  constexpr Real_type* operator+(Index_type i) { return dptr + i; }

  ///
  /// + operator for const pointer arithmetic.
  ///
  constexpr const Real_type* operator+(Index_type i) const { return dptr + i; }

private:
  Real_type* dptr;
};

/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted aligned Real_type const pointer.
 *
 ******************************************************************************
 */
class ConstRestrictAlignedRealPtr
{
public:
  ///
  /// Ctors and assignment op.
  ///

  constexpr ConstRestrictAlignedRealPtr() : dptr(0) { ; }

  constexpr ConstRestrictAlignedRealPtr(const Real_type* d) : dptr(d) { ; }

  constexpr ConstRestrictAlignedRealPtr& operator=(const Real_type* d)
  {
    ConstRestrictAlignedRealPtr copy(d);
    std::swap(dptr, copy.dptr);
    return *this;
  }

  ///
  /// NOTE: Using compiler-generated copy ctor, dtor, and copy assignment op.
  ///

  ///
  ///  Implicit conversion operator to bare const pointer.
  ///
  constexpr operator const Real_type*() { return dptr; }

  ///
  ///  "Explicit conversion operator" to bare const pointer,
  ///  consistent with boost shared ptr.
  ///
  constexpr const Real_type* get() const { return dptr; }

  ///
  /// Compiler-specific bracket operators.
  ///

#if defined(RAJA_COMPILER_ICC)
  ///
  constexpr const Real_type& operator[](Index_type i) const
  {
#if __ICC < 1300  // use alignment intrinsic
    RAJA_ALIGN_DATA(dptr);
    return ((const Real_type* RAJA_RESTRICT)dptr)[i];
#else  // use alignment attribute
    return ((const_TDRAReal_ptr)dptr)[i];
#endif
  }

#elif defined(RAJA_COMPILER_GNU)
  ///
  constexpr const Real_type& operator[](Index_type i) const
  {
#if 1  // NOTE: alignment instrinsic not available for older GNU compilers
    return ((const Real_type* RAJA_RESTRICT)RAJA_ALIGN_DATA(dptr))[i];
#else
    return ((const Real_type* RAJA_RESTRICT)dptr)[i];
#endif
  }

#elif defined(RAJA_COMPILER_XLC)
  constexpr const Real_type& operator[](Index_type i) const
  {
    RAJA_ALIGN_DATA(dptr);
    return ((const Real_type* RAJA_RESTRICT)dptr)[i];
  }

#elif defined(RAJA_COMPILER_CLANG)
  constexpr const Real_type& operator[](Index_type i) const
  {
    return ((const_TDRAReal_ptr)dptr)[i];
  }

#else
#error RAJA compiler macro is undefined!

#endif

  ///
  /// + operator for pointer arithmetic.
  ///
  constexpr const Real_type* operator+(Index_type i) const { return dptr + i; }

private:
  const Real_type* dptr;
};

/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted aligned Real_type (non-const) pointer.
 *
 ******************************************************************************
 */
class RestrictAlignedRealPtr
{
public:
  ///
  /// Ctors and assignment op.
  ///

  constexpr RestrictAlignedRealPtr() : dptr(0) { ; }

  constexpr RestrictAlignedRealPtr(Real_type* d) : dptr(d) { ; }

  constexpr RestrictAlignedRealPtr& operator=(Real_type* d)
  {
    RestrictAlignedRealPtr copy(d);
    std::swap(dptr, copy.dptr);
    return *this;
  }

  ///
  /// NOTE: Using compiler-generated copy ctor, dtor, and copy assignment op.
  ///

  ///
  ///  Implicit conversion operator to (non-const) bare pointer.
  ///
  constexpr operator Real_type*() { return dptr; }

  ///
  ///  Implicit conversion operator to const bare pointer.
  ///
  constexpr operator const Real_type*() const { return dptr; }

  ///
  ///  "Explicit conversion operator" to (non-const) bare pointer,
  ///  consistent with boost shared ptr.
  ///
  constexpr Real_type* get() { return dptr; }

  ///
  ///  "Explicit conversion operator" to const bare pointer,
  ///  consistent with boost shared ptr.
  ///
  constexpr const Real_type* get() const { return dptr; }

  ///
  ///  Operator that enables implicit conversion from
  ///  RestrictAlignedRealPtr to RestrictAlignedRealConstPtr.
  ///
  constexpr operator ConstRestrictAlignedRealPtr()
  {
    return ConstRestrictAlignedRealPtr(dptr);
  }

  ///
  /// Compiler-specific bracket operators.
  ///

#if defined(RAJA_COMPILER_ICC)
  ///
  constexpr Real_type& operator[](Index_type i)
  {
#if __ICC < 1300  // use alignment intrinsic
    RAJA_ALIGN_DATA(dptr);
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
#else  // use alignment attribute
    return ((TDRAReal_ptr)dptr)[i];
#endif
  }

  ///
  constexpr const Real_type& operator[](Index_type i) const
  {
#if __ICC < 1300  // use alignment intrinsic
    RAJA_ALIGN_DATA(dptr);
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
#else  // use alignment attribute
    return ((TDRAReal_ptr)dptr)[i];
#endif
  }

#elif defined(RAJA_COMPILER_GNU)
  ///
  constexpr Real_type& operator[](Index_type i)
  {
#if 1  // NOTE: alignment instrinsic not available for older GNU compilers
    return ((Real_type * RAJA_RESTRICT) RAJA_ALIGN_DATA(dptr))[i];
#else
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
#endif
  }

  ///
  constexpr const Real_type& operator[](Index_type i) const
  {
#if 1  // NOTE: alignment instrinsic not available for older GNU compilers
    return ((Real_type * RAJA_RESTRICT) RAJA_ALIGN_DATA(dptr))[i];
#else
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
#endif
  }

#elif defined(RAJA_COMPILER_XLC)
  ///
  constexpr Real_type& operator[](Index_type i)
  {
    RAJA_ALIGN_DATA(dptr);
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
  }

  ///
  constexpr const Real_type& operator[](Index_type i) const
  {
    RAJA_ALIGN_DATA(dptr);
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
  }

#elif defined(RAJA_COMPILER_CLANG)
  ///
  constexpr Real_type& operator[](Index_type i)
  {
    return ((TDRAReal_ptr)dptr)[i];
  }

  ///
  const Real_type& operator[](Index_type i) const
  {
    return ((TDRAReal_ptr)dptr)[i];
  }

#else
#error RAJA compiler macro is undefined!

#endif

  ///
  /// + operator for (non-const) pointer arithmetic.
  ///
  constexpr Real_type* operator+(Index_type i) { return dptr + i; }

  ///
  /// + operator for const pointer arithmetic.
  ///
  constexpr const Real_type* operator+(Index_type i) const { return dptr + i; }

private:
  Real_type* dptr;
};

#if defined(RAJA_USE_COMPLEX)
/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted Complex_type const pointer.
 *
 ******************************************************************************
 */
class ConstRestrictComplexPtr
{
public:
  ///
  /// Ctors and assignment op.
  ///

  constexpr ConstRestrictComplexPtr() : dptr(0) { ; }

  constexpr ConstRestrictComplexPtr(const Complex_type* d) : dptr(d) { ; }

  constexpr ConstRestrictComplexPtr& operator=(const Complex_type* d)
  {
    ConstRestrictComplexPtr copy(d);
    std::swap(dptr, copy.dptr);
    return *this;
  }

  ///
  /// NOTE: Using compiler-generated copy ctor, dtor, and copy assignment op.
  ///

  ///
  ///  Implicit conversion operator to bare const pointer.
  ///
  constexpr operator const Complex_type*() const { return dptr; }

  ///
  ///  "Explicit conversion operator" to bare const pointer,
  ///  consistent with boost shared ptr.
  ///
  constexpr const Complex_type* get() const { return dptr; }

  ///
  ///  Bracket operator.
  ///
  constexpr const Complex_type& operator[](Index_type i) const
  {
    return ((const Complex_type* RAJA_RESTRICT)dptr)[i];
  }

  ///
  /// + operator for pointer arithmetic.
  ///
  constexpr const Complex_type* operator+(Index_type i) const
  {
    return dptr + i;
  }

private:
  const Complex_type* dptr;
};

/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted Complex_type (non-const) pointer.
 *
 ******************************************************************************
 */
class RestrictComplexPtr
{
public:
  ///
  /// Ctors and assignment op.
  ///

  constexpr RestrictComplexPtr() : dptr(0) { ; }

  constexpr RestrictComplexPtr(Complex_type* d) : dptr(d) { ; }

  constexpr RestrictComplexPtr& operator=(Complex_type* d)
  {
    RestrictComplexPtr copy(d);
    std::swap(dptr, copy.dptr);
    return *this;
  }

  ///
  /// NOTE: Using compiler-generated copy ctor, dtor, and copy assignment op.
  ///

  ///
  ///  Implicit conversion operator to (non-const) bare pointer.
  ///
  constexpr operator Complex_type*() { return dptr; }

  ///
  ///  Implicit conversion operator to const bare pointer.
  ///
  constexpr operator const Complex_type*() const { return dptr; }

  ///
  ///  "Explicit conversion operator" to (non-const) bare pointer,
  ///  consistent with boost shared ptr.
  ///
  constexpr Complex_type* get() { return dptr; }

  ///
  ///  "Explicit conversion operator" to const bare pointer,
  ///  consistent with boost shared ptr.
  ///
  constexpr const Complex_type* get() const { return dptr; }

  ///
  ///  Operator that enables implicit conversion from RestrictComplexPtr to
  ///  RestrictComplexConstPtr.
  ///
  constexpr operator ConstRestrictComplexPtr()
  {
    return ConstRestrictComplexPtr(dptr);
  }

  ///
  ///  (Non-const) bracket operator.
  ///
  constexpr Complex_type& operator[](Index_type i)
  {
    return ((Complex_type * RAJA_RESTRICT) dptr)[i];
  }

  ///
  ///  Const bracket operator.
  ///
  constexpr const Complex_type& operator[](Index_type i) const
  {
    return ((Complex_type * RAJA_RESTRICT) dptr)[i];
  }

  ///
  /// + operator for (non-const) pointer arithmetic.
  ///
  constexpr Complex_type* operator+(Index_type i) { return dptr + i; }

  ///
  /// + operator for const pointer arithmetic.
  ///
  constexpr const Complex_type* operator+(Index_type i) const
  {
    return dptr + i;
  }

private:
  Complex_type* dptr;
};
#endif  // defined(RAJA_USE_COMPLEX)

#endif  // defined(RAJA_USE_PTR_CLASS)

/*
 ******************************************************************************
 *
 * Finally, we define data pointer types based on definitions above and
 * -D value given at compile time.
 *
 ******************************************************************************
 */
#if defined(RAJA_USE_BARE_PTR)
using Real_ptr       = Real_type*;
using const_Real_ptr = const Real_type*;

#if defined(RAJA_USE_COMPLEX)
using Complex_ptr       = Complex_type*;
using const_Complex_ptr = const Complex_type*;
#endif

using UnalignedReal_ptr       = Real_type*;
using const_UnalignedReal_ptr = const Real_type*;

#elif defined(RAJA_USE_RESTRICT_PTR)
using Real_ptr          = Real_type* RAJA_RESTRICT;
using const_Real_ptr    = const Real_type* RAJA_RESTRICT;

#if defined(RAJA_USE_COMPLEX)
using Complex_ptr       = Complex_type* RAJA_RESTRICT;
using const_Complex_ptr = const Complex_type* RAJA_RESTRICT;
#endif

using UnalignedReal_ptr       = Real_type* RAJA_RESTRICT;
using const_UnalignedReal_ptr = const Real_type* RAJA_RESTRICT;

#elif defined(RAJA_USE_RESTRICT_ALIGNED_PTR)
using Real_ptr           = TDRAReal_ptr;
using const_Real_ptr     = const_TDRAReal_ptr;

#if defined(RAJA_USE_COMPLEX)
using Complex_ptr        = Complex_type* RAJA_RESTRICT;
using const_Complex_ptr  = const Complex_type* RAJA_RESTRICT;
#endif

using UnalignedReal_ptr       = Real_type* RAJA_RESTRICT;
using const_UnalignedReal_ptr = const Real_type* RAJA_RESTRICT;

#elif defined(RAJA_USE_PTR_CLASS)
using Real_ptr           = RestrictAlignedRealPtr;
using const_Real_ptr     = ConstRestrictAlignedRealPtr;

#if defined(RAJA_USE_COMPLEX)
using Complex_ptr        = RestrictComplexPtr;
using const_Complex_ptr  = ConstRestrictComplexPtr;
#endif

using UnalignedReal_ptr       = RestrictRealPtr;
using const_UnalignedReal_ptr = ConstRestrictRealPtr;

#else
#error RAJA pointer type is undefined!

#endif


namespace detail
{

/*!
 * \brief Abstracts access to memory using normal memory accesses.
 */
struct DefaultAccessor
{
  template<typename T>
  static RAJA_HOST_DEVICE RAJA_INLINE constexpr T get(T* ptr, size_t i)
  {
    return ptr[i];
  }

  template<typename T>
  static RAJA_HOST_DEVICE RAJA_INLINE constexpr void set(T* ptr,
                                                         size_t i,
                                                         T val)
  {
    ptr[i] = val;
  }
};

/*!
 * \brief Abstracts T into an equal or greater size array of integers whose
 * size is between min_integer_type_size and max_interger_type_size inclusive.
 */
template<typename T,
         size_t min_integer_type_size = 1,
         size_t max_integer_type_size = sizeof(unsigned long long)>
struct AsIntegerArray
{
  static_assert(min_integer_type_size <= max_integer_type_size,
                "incompatible min and max integer type size");
  using integer_type = std::conditional_t<
      ((alignof(T) >= alignof(unsigned long long) &&
        sizeof(unsigned long long) <= max_integer_type_size) ||
       sizeof(unsigned long) < min_integer_type_size),
      unsigned long long,
      std::conditional_t<
          ((alignof(T) >= alignof(unsigned long) &&
            sizeof(unsigned long) <= max_integer_type_size) ||
           sizeof(unsigned int) < min_integer_type_size),
          unsigned long,
          std::conditional_t<
              ((alignof(T) >= alignof(unsigned int) &&
                sizeof(unsigned int) <= max_integer_type_size) ||
               sizeof(unsigned short) < min_integer_type_size),
              unsigned int,
              std::conditional_t<
                  ((alignof(T) >= alignof(unsigned short) &&
                    sizeof(unsigned short) <= max_integer_type_size) ||
                   sizeof(unsigned char) < min_integer_type_size),
                  unsigned short,
                  std::conditional_t<((alignof(T) >= alignof(unsigned char) &&
                                       sizeof(unsigned char) <=
                                           max_integer_type_size)),
                                     unsigned char,
                                     void>>>>>;
  static_assert(!std::is_same<integer_type, void>::value,
                "could not find a compatible integer type");
  static_assert(sizeof(integer_type) >= min_integer_type_size,
                "integer_type smaller than min integer type size");
  static_assert(sizeof(integer_type) <= max_integer_type_size,
                "integer_type greater than max integer type size");

  static constexpr size_t num_integer_type =
      (sizeof(T) + sizeof(integer_type) - 1) / sizeof(integer_type);

  integer_type array[num_integer_type] = {0};

  AsIntegerArray() = default;

  RAJA_HOST_DEVICE constexpr size_t array_size() const
  {
    return num_integer_type;
  }

  RAJA_HOST_DEVICE T get_value() const
  {
    T value;
    memcpy(&value, &array[0], sizeof(T));
    return value;
  }

  RAJA_HOST_DEVICE void set_value(T value)
  {
    memcpy(&array[0], &value, sizeof(T));
  }
};

/*!
 * \brief Assign a new value to an object and restore the object's previous
 * value at the end of the current scope.
 */
template<typename T>
struct ScopedAssignment
{
  constexpr ScopedAssignment(T& val, T const& new_val)
      : m_ref_to_val(val),
        m_prev_val(std::move(val))
  {
    m_ref_to_val = new_val;
  }

  constexpr ScopedAssignment(T& val, T&& new_val)
      : m_ref_to_val(val),
        m_prev_val(std::move(val))
  {
    m_ref_to_val = std::move(new_val);
  }

  ScopedAssignment(ScopedAssignment const&)            = delete;
  ScopedAssignment(ScopedAssignment&&)                 = delete;
  ScopedAssignment& operator=(ScopedAssignment const&) = delete;
  ScopedAssignment& operator=(ScopedAssignment&&)      = delete;

  // constexpr in c++20
  ~ScopedAssignment() { m_ref_to_val = std::move(m_prev_val); }

private:
  T& m_ref_to_val;
  T m_prev_val;
};

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
