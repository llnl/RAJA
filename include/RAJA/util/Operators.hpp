/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA operator definitions
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

#ifndef RAJA_operators_HPP
#define RAJA_operators_HPP

#include "RAJA/config.hpp"

#include <stdint.h>
#include <cfloat>
#include <cstdint>
#include <ostream>
#include <type_traits>
#if defined(RAJA_CHECK_LIMITS)
#include <limits>
#endif

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/macros.hpp"

namespace RAJA
{

namespace operators
{

namespace detail
{

// truly associative (does not include fp add/multiply)
struct associative_tag
{};

// associative up to floating point rounding differences
struct fp_associative_tag : associative_tag
{};

// get associativity tag appropriate for the type
template<typename T>
using associative_or_fp_associative_tag =
    std::conditional_t<std::is_floating_point<std::decay_t<T>>::value,
                       fp_associative_tag,
                       associative_tag>;

template<typename Arg1, typename Arg2, typename Result>
struct binary_function
{
  using first_argument_type  = Arg1;
  using second_argument_type = Arg2;
  using result_type          = Result;
};

template<typename Argument, typename Result>
struct unary_function
{
  using argument_type = Argument;
  using result_type   = Result;
};

template<typename Arg1, typename Arg2>
struct comparison_function : public binary_function<Arg1, Arg2, bool>
{};

}  // namespace detail

namespace types
{

template<typename T>
struct is_unsigned_int
{
  static constexpr const bool value =
      std::is_unsigned<T>::value && std::is_integral<T>::value;
};

template<typename T>
struct is_signed_int
{
  static constexpr const bool value =
      !std::is_unsigned<T>::value && std::is_integral<T>::value;
};

// given a type, T, return a similar type whose size is >= sizeof(T)
/*!
        \brief type lookup to return the next largest similar type (or the same
   type)
*/
template<typename T, bool GPU = false>
struct larger
{};

template<>
struct larger<uint8_t>
{
  using type = uint16_t;
};

template<>
struct larger<uint16_t>
{
  using type = uint32_t;
};

template<>
struct larger<uint32_t>
{
  using type = uint64_t;
};

template<>
struct larger<int8_t>
{
  using type = int16_t;
};

template<>
struct larger<int16_t>
{
  using type = int32_t;
};

template<>
struct larger<int32_t>
{
  using type = int64_t;
};

template<>
struct larger<float>
{
  using type = double;
};

template<>
struct larger<double>
{
  using type = long double;
};

template<>
struct larger<double, true>
{
  using type = double;
};

namespace detail
{

template<typename T, bool isInt, bool isSigned, bool isFP, bool gpu = false>
struct largest
{};

template<typename T>
struct largest<T, true, false, false>
{
  using type = uint64_t;
};

template<typename T>
struct largest<T, true, true, false>
{
  using type = int64_t;
};

template<typename T>
struct largest<T, false, false, true, false>
{
  using type = long double;
};

template<typename T>
struct largest<T, false, false, true, true>
{
  using type = double;
};
}  // namespace detail

/*!
        \brief type lookup to return largest similar type. If running on GPU,
   pass 'true' as second template argument
*/
template<typename T, bool gpu = false>
struct largest
{
  using type = typename detail::largest<T,
                                        std::is_integral<T>::value,
                                        std::is_signed<T>::value,
                                        std::is_floating_point<T>::value,
                                        gpu>::type;
};

template<typename T>
struct size_of
{
  enum
  {
    value = sizeof(T)
  };
};

namespace detail
{

template<typename T, typename U, bool lhsLarger>
struct larger_of
{};

template<typename T, typename U>
struct larger_of<T, U, true>
{
  using type = T;
};

template<typename T, typename U>
struct larger_of<T, U, false>
{
  using type = U;
};
}  // namespace detail

template<typename T, typename U>
struct larger_of
{
  using type = typename detail::
      larger_of<T, U, (size_of<T>::value > size_of<U>::value)>::type;
};

}  // namespace types


template<typename T, typename Enable = void>
struct limits;

// limits for signed integer types
template<typename T>
struct limits<T,
              typename std::enable_if<std::is_integral<T>::value &&
                                      !std::is_unsigned<T>::value>::type>
{
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr T min()
  {
#ifdef RAJA_COMPILER_MSVC
#pragma warning(disable : 4309)
#endif
    return static_cast<T>(1llu << ((8llu * sizeof(T)) - 1llu));
#ifdef RAJA_COMPILER_MSVC
#pragma warning(default : 4309)
#endif
  }

  RAJA_INLINE RAJA_HOST_DEVICE static constexpr T max()
  {
#ifdef RAJA_COMPILER_MSVC
#pragma warning(disable : 4309)
#endif
    return static_cast<T>(~(1llu << ((8llu * sizeof(T)) - 1llu)));
#ifdef RAJA_COMPILER_MSVC
#pragma warning(default : 4309)
#endif
  }
};

// limits for signed integer types
template<typename T>
struct limits<T,
              typename std::enable_if<std::is_integral<T>::value &&
                                      std::is_unsigned<T>::value>::type>
{
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr T min()
  {
    return static_cast<T>(0);
  }

  RAJA_INLINE RAJA_HOST_DEVICE static constexpr T max()
  {
#ifdef RAJA_COMPILER_MSVC
#pragma warning(disable : 4309)
#endif
    return static_cast<T>(0xFFFFFFFFFFFFFFFF);
#ifdef RAJA_COMPILER_MSVC
#pragma warning(default : 4309)
#endif
  }
};

template<>
struct limits<float>
{
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr float min() { return -FLT_MAX; }

  RAJA_INLINE RAJA_HOST_DEVICE static constexpr float max() { return FLT_MAX; }
};

template<>
struct limits<double>
{
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr double min()
  {
    return -DBL_MAX;
  }

  RAJA_INLINE RAJA_HOST_DEVICE static constexpr double max() { return DBL_MAX; }
};

template<>
struct limits<long double>
{
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr long double min()
  {
    return -LDBL_MAX;
  }

  RAJA_INLINE RAJA_HOST_DEVICE static constexpr long double max()
  {
    return LDBL_MAX;
  }
};


#if defined(RAJA_CHECK_LIMITS)
template<typename T>
constexpr bool check()
{
  return limits<T>::min() == std::numeric_limits<T>::min() &&
         limits<T>::max() == std::numeric_limits<T>::max();
}

static_assert(check<char>(), "limits for char is broken");
static_assert(check<unsigned char>(), "limits for unsigned char is broken");
static_assert(check<short>(), "limits for short is broken");
static_assert(check<unsigned short>(), "limits for unsigned short is broken");
static_assert(check<int>(), "limits for int is broken");
static_assert(check<unsigned int>(), "limits for unsigned int is broken");
static_assert(check<long>(), "limits for long is broken");
static_assert(check<unsigned long>(), "limits for unsigned long is broken");
static_assert(check<long int>(), "limits for long int is broken");
static_assert(check<unsigned long int>(),
              "limits for unsigned long int is broken");
static_assert(check<long long>(), "limits for long long is broken");
static_assert(check<unsigned long long>(),
              "limits for unsigned long long is broken");
#endif

// Arithmetic

template<typename Ret, typename Arg1 = Ret, typename Arg2 = Arg1>
struct plus : public detail::binary_function<Arg1, Arg2, Ret>,
              detail::associative_or_fp_associative_tag<Ret>
{
  RAJA_HOST_DEVICE constexpr Ret operator()(const Arg1& lhs,
                                            const Arg2& rhs) const
  {
    return Ret {lhs} + rhs;
  }

  RAJA_HOST_DEVICE static constexpr Ret identity() { return Ret {0}; }

  friend inline std::ostream& operator<<(std::ostream& str, plus const&)
  {
    return str << "RAJA::operators::plus{}";
  }
};

template<typename Ret, typename Arg1 = Ret, typename Arg2 = Arg1>
struct minus : public detail::binary_function<Arg1, Arg2, Ret>
{
  RAJA_HOST_DEVICE constexpr Ret operator()(const Arg1& lhs,
                                            const Arg2& rhs) const
  {
    return Ret {lhs} - rhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str, minus const&)
  {
    return str << "RAJA::operators::minus{}";
  }
};

template<typename Ret, typename Arg1 = Ret, typename Arg2 = Arg1>
struct multiplies : public detail::binary_function<Arg1, Arg2, Ret>,
                    detail::associative_or_fp_associative_tag<Ret>
{

  RAJA_HOST_DEVICE constexpr Ret operator()(const Arg1& lhs,
                                            const Arg2& rhs) const
  {
    return Ret {lhs} * rhs;
  }

  RAJA_HOST_DEVICE static constexpr Ret identity() { return Ret {1}; }

  friend inline std::ostream& operator<<(std::ostream& str, multiplies const&)
  {
    return str << "RAJA::operators::multiplies{}";
  }
};

template<typename Ret, typename Arg1 = Ret, typename Arg2 = Arg1>
struct divides : public detail::binary_function<Arg1, Arg2, Ret>
{
  RAJA_HOST_DEVICE constexpr Ret operator()(const Arg1& lhs,
                                            const Arg2& rhs) const
  {
    return Ret {lhs} / rhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str, divides const&)
  {
    return str << "RAJA::operators::divides{}";
  }
};

template<typename Ret, typename Arg1 = Ret, typename Arg2 = Arg1>
struct modulus : public detail::binary_function<Arg1, Arg2, Ret>
{
  RAJA_HOST_DEVICE constexpr Ret operator()(const Arg1& lhs,
                                            const Arg2& rhs) const
  {
    return Ret {lhs} % rhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str, modulus const&)
  {
    return str << "RAJA::operators::modulus{}";
  }
};

// Conditions

template<typename Arg1, typename Arg2 = Arg1>
struct logical_and : public detail::comparison_function<Arg1, Arg2>,
                     detail::associative_tag
{
  RAJA_HOST_DEVICE constexpr bool operator()(const Arg1& lhs,
                                             const Arg2& rhs) const
  {
    return lhs && rhs;
  }

  RAJA_HOST_DEVICE static constexpr bool identity() { return true; }

  friend inline std::ostream& operator<<(std::ostream& str, logical_and const&)
  {
    return str << "RAJA::operators::logical_and{}";
  }
};

template<typename Arg1, typename Arg2 = Arg1>
struct logical_or : public detail::comparison_function<Arg1, Arg2>,
                    detail::associative_tag
{
  RAJA_HOST_DEVICE constexpr bool operator()(const Arg1& lhs,
                                             const Arg2& rhs) const
  {
    return lhs || rhs;
  }

  RAJA_HOST_DEVICE static constexpr bool identity() { return false; }

  friend inline std::ostream& operator<<(std::ostream& str, logical_or const&)
  {
    return str << "RAJA::operators::logical_or{}";
  }
};

template<typename T>
struct logical_not : public detail::unary_function<T, bool>
{
  RAJA_HOST_DEVICE constexpr bool operator()(const T& lhs) const
  {
    return !lhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str, logical_not const&)
  {
    return str << "RAJA::operators::logical_not{}";
  }
};

// Bitwise

template<typename Ret, typename Arg1 = Ret, typename Arg2 = Arg1>
struct bit_or : public detail::binary_function<Arg1, Arg2, Ret>
{
  RAJA_HOST_DEVICE constexpr Ret operator()(const Arg1& lhs,
                                            const Arg2& rhs) const
  {
    return lhs | rhs;
  }

  RAJA_HOST_DEVICE static constexpr Ret identity() { return Ret {0}; }

  friend inline std::ostream& operator<<(std::ostream& str, bit_or const&)
  {
    return str << "RAJA::operators::bit_or{}";
  }
};

template<typename Ret, typename Arg1 = Ret, typename Arg2 = Arg1>
struct bit_and : public detail::binary_function<Arg1, Arg2, Ret>
{
  RAJA_HOST_DEVICE constexpr Ret operator()(const Arg1& lhs,
                                            const Arg2& rhs) const
  {
    return lhs & rhs;
  }

  RAJA_HOST_DEVICE static constexpr Ret identity() { return ~Ret {0}; }

  friend inline std::ostream& operator<<(std::ostream& str, bit_and const&)
  {
    return str << "RAJA::operators::bit_and{}";
  }
};

template<typename Ret, typename Arg1 = Ret, typename Arg2 = Arg1>
struct bit_xor : public detail::binary_function<Arg1, Arg2, Ret>
{
  RAJA_HOST_DEVICE constexpr Ret operator()(const Arg1& lhs,
                                            const Arg2& rhs) const
  {
    return lhs ^ rhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str, bit_xor const&)
  {
    return str << "RAJA::operators::bit_xor{}";
  }
};

// comparison
/*!
 Checks if the candidate (rhs) value is strictly less than
 the current value (lhs); if so the candidate is returned.
 When this operator is used to cycle through an array
 this ensures that the location of the first min/max is kept.
*/
template<typename Ret, typename Arg1 = Ret, typename Arg2 = Arg1>
struct minimum : public detail::binary_function<Arg1, Arg2, Ret>,
                 detail::associative_tag
{
  RAJA_HOST_DEVICE constexpr Ret operator()(const Arg1& lhs,
                                            const Arg2& rhs) const
  {
    return (rhs < lhs) ? rhs : lhs;
  }

  RAJA_HOST_DEVICE static constexpr Ret identity()
  {
    return limits<Ret>::max();
  }

  friend inline std::ostream& operator<<(std::ostream& str, minimum const&)
  {
    return str << "RAJA::operators::minimum{}";
  }
};

template<typename Ret, typename Arg1 = Ret, typename Arg2 = Arg1>
struct maximum : public detail::binary_function<Arg1, Arg2, Ret>,
                 detail::associative_tag
{
  RAJA_HOST_DEVICE constexpr Ret operator()(const Arg1& lhs,
                                            const Arg2& rhs) const
  {
    return (lhs < rhs) ? rhs : lhs;
  }

  RAJA_HOST_DEVICE static constexpr Ret identity()
  {
    return limits<Ret>::min();
  }

  friend inline std::ostream& operator<<(std::ostream& str, maximum const&)
  {
    return str << "RAJA::operators::maximum{}";
  }
};

// Logical Comparison

template<typename Arg1, typename Arg2 = Arg1>
struct equal_to : public detail::comparison_function<Arg1, Arg2>
{
  RAJA_HOST_DEVICE constexpr bool operator()(const Arg1& lhs,
                                             const Arg2& rhs) const
  {
    return lhs == rhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str, equal_to const&)
  {
    return str << "RAJA::operators::equal_to{}";
  }
};

template<typename Arg1, typename Arg2 = Arg1>
struct not_equal_to : public detail::comparison_function<Arg1, Arg2>
{
  RAJA_HOST_DEVICE constexpr bool operator()(const Arg1& lhs,
                                             const Arg2& rhs) const
  {
    return lhs != rhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str, not_equal_to const&)
  {
    return str << "RAJA::operators::not_equal_to{}";
  }
};

template<typename Arg1, typename Arg2 = Arg1>
struct greater : public detail::comparison_function<Arg1, Arg2>
{
  RAJA_HOST_DEVICE constexpr bool operator()(const Arg1& lhs,
                                             const Arg2& rhs) const
  {
    return lhs > rhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str, greater const&)
  {
    return str << "RAJA::operators::greater{}";
  }
};

template<typename Arg1, typename Arg2 = Arg1>
struct less : public detail::comparison_function<Arg1, Arg2>
{
  RAJA_HOST_DEVICE constexpr bool operator()(const Arg1& lhs,
                                             const Arg2& rhs) const
  {
    return lhs < rhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str, less const&)
  {
    return str << "RAJA::operators::less{}";
  }
};

template<typename Arg1, typename Arg2 = Arg1>
struct greater_equal : public detail::comparison_function<Arg1, Arg2>
{
  RAJA_HOST_DEVICE constexpr bool operator()(const Arg1& lhs,
                                             const Arg2& rhs) const
  {
    return lhs >= rhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str,
                                         greater_equal const&)
  {
    return str << "RAJA::operators::greater_equal{}";
  }
};

template<typename Arg1, typename Arg2 = Arg1>
struct less_equal : public detail::comparison_function<Arg1, Arg2>
{
  RAJA_HOST_DEVICE constexpr bool operator()(const Arg1& lhs,
                                             const Arg2& rhs) const
  {
    return lhs <= rhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str, less_equal const&)
  {
    return str << "RAJA::operators::less_equal{}";
  }
};

// Filters

template<typename Ret, typename Orig = Ret>
struct identity : public detail::unary_function<Orig, Ret>
{
  RAJA_HOST_DEVICE constexpr Ret operator()(const Orig& lhs) const
  {
    return lhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str, identity const&)
  {
    return str << "RAJA::operators::identity{}";
  }
};

template<typename T, typename U>
struct project1st : public detail::binary_function<T, U, T>
{
  RAJA_HOST_DEVICE constexpr T operator()(const T& lhs,
                                          const U& RAJA_UNUSED_ARG(rhs)) const
  {
    return lhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str, project1st const&)
  {
    return str << "RAJA::operators::project1st{}";
  }
};

template<typename T, typename U = T>
struct project2nd : public detail::binary_function<T, U, U>
{
  RAJA_HOST_DEVICE constexpr U operator()(const T& RAJA_UNUSED_ARG(lhs),
                                          const U& rhs) const
  {
    return rhs;
  }

  friend inline std::ostream& operator<<(std::ostream& str, project2nd const&)
  {
    return str << "RAJA::operators::project2nd{}";
  }
};

// Type Traits

template<typename T>
struct is_associative
{
  static constexpr const bool value =
      std::is_base_of<detail::associative_tag, T>::value;
};

template<typename T>
struct is_fp_associative
{
  static constexpr const bool value =
      std::is_base_of<detail::fp_associative_tag, T>::value;
};

template<typename Arg1, typename Arg2 = Arg1>
struct safe_plus
    : public plus<Arg1,
                  Arg2,
                  typename types::larger<
                      typename types::larger_of<Arg1, Arg2>::type>::type>
{
  friend inline std::ostream& operator<<(std::ostream& str, safe_plus const&)
  {
    return str << "RAJA::operators::safe_plus{}";
  }
};

}  // namespace operators

namespace concepts
{

template<typename Function,
         typename Return,
         typename Arg1 = Return,
         typename Arg2 = Arg1>
struct BinaryFunction
    : DefineConcept(::RAJA::concepts::convertible_to<Return>(
          camp::val<Function>()(camp::val<Arg1>(), camp::val<Arg2>())))
{};

template<typename Function, typename Return, typename Arg = Return>
struct UnaryFunction : DefineConcept(::RAJA::concepts::convertible_to<Return>(
                           camp::val<Function>()(camp::val<Arg>())))
{};

namespace detail
{

template<typename Fun, typename Ret, typename T, typename U>
using is_binary_function =
    ::RAJA::concepts::requires_<BinaryFunction, Ret, T, U>;

template<typename Fun, typename Ret, typename T>
using is_unary_function = ::RAJA::concepts::requires_<UnaryFunction, Ret, T>;
}  // namespace detail

}  // namespace concepts

namespace type_traits
{
DefineTypeTraitFromConcept(is_binary_function, RAJA::concepts::BinaryFunction);
DefineTypeTraitFromConcept(is_unary_function, RAJA::concepts::UnaryFunction);
}  // namespace type_traits


}  // namespace RAJA

#endif  // closing endif for header file include guard
