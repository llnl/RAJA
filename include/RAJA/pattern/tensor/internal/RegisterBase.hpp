/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining SIMD/SIMT register operations.
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

#ifndef RAJA_pattern_tensor_RegisterBase_HPP
#define RAJA_pattern_tensor_RegisterBase_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "camp/camp.hpp"
#include "RAJA/pattern/tensor/TensorLayout.hpp"
#include "RAJA/pattern/tensor/internal/TensorRef.hpp"
#include "RAJA/util/BitMask.hpp"

#include "RAJA/policy/tensor/arch.hpp"

namespace RAJA
{
namespace expt
{
template<typename T, typename REGISTER_POLICY>
class Register;
}

namespace internal
{
namespace expt
{
class RegisterConcreteBase
{};

/*
 * Overload for:    arithmetic + TensorRegister

 */
template<
    typename LEFT,
    typename RIGHT,
    typename std::enable_if<std::is_arithmetic<LEFT>::value, bool>::type = true,
    typename std::enable_if<std::is_base_of<RegisterConcreteBase, RIGHT>::value,
                            bool>::type                                  = true>
RAJA_INLINE RAJA_HOST_DEVICE RIGHT operator+(LEFT const& lhs, RIGHT const& rhs)
{
  return RIGHT(lhs).add(rhs);
}

/*
 * Overload for:    arithmetic - TensorRegister

 */
template<
    typename LEFT,
    typename RIGHT,
    typename std::enable_if<std::is_arithmetic<LEFT>::value, bool>::type = true,
    typename std::enable_if<std::is_base_of<RegisterConcreteBase, RIGHT>::value,
                            bool>::type                                  = true>
RAJA_INLINE RAJA_HOST_DEVICE RIGHT operator-(LEFT const& lhs, RIGHT const& rhs)
{
  return RIGHT(lhs).subtract(rhs);
}

/*
 * Overload for:    arithmetic * TensorRegister

 */
template<
    typename LEFT,
    typename RIGHT,
    typename std::enable_if<std::is_arithmetic<LEFT>::value, bool>::type = true,
    typename std::enable_if<std::is_base_of<RegisterConcreteBase, RIGHT>::value,
                            bool>::type                                  = true>
RAJA_INLINE RAJA_HOST_DEVICE RIGHT operator*(LEFT const& lhs, RIGHT const& rhs)
{
  return rhs.scale(lhs);
}

/*
 * Overload for:    arithmetic / TensorRegister

 */
template<
    typename LEFT,
    typename RIGHT,
    typename std::enable_if<std::is_arithmetic<LEFT>::value, bool>::type = true,
    typename std::enable_if<std::is_base_of<RegisterConcreteBase, RIGHT>::value,
                            bool>::type                                  = true>
RAJA_INLINE RAJA_HOST_DEVICE RIGHT operator/(LEFT const& lhs, RIGHT const& rhs)
{
  return RIGHT(lhs).divide(rhs);
}


/*!
 * Register base class that provides some default behaviors and simplifies
 * the implementation of new register types.
 *
 * This uses CRTP to provide static polymorphism
 */
template<typename Derived>
class RegisterBase;

template<typename T, typename REGISTER_POLICY>
class RegisterBase<RAJA::expt::Register<T, REGISTER_POLICY>>
    : public RegisterConcreteBase
{
public:
  using self_type    = RAJA::expt::Register<T, REGISTER_POLICY>;
  using element_type = camp::decay<T>;

  using index_type = camp::idx_t;

  using int_element_type =
      typename RegisterTraits<REGISTER_POLICY, T>::int_element_type;
  using int_vector_type =
      RAJA::expt::Register<int_element_type, REGISTER_POLICY>;

private:
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type* getThis() { return static_cast<self_type*>(this); }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr self_type const* getThis() const
  {
    return static_cast<self_type const*>(this);
  }

public:
  RAJA_HOST_DEVICE

  RAJA_INLINE
  static constexpr bool is_root() { return true; }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr RegisterBase() {}

  RAJA_HOST_DEVICE

  RAJA_INLINE
  ~RegisterBase() {}

  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr RegisterBase(RegisterBase const&) {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr RegisterBase(self_type const&) {}

  /*!
   * @brief Broadcast scalar value to first N register elements
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  static self_type s_broadcast_n(element_type const& value, camp::idx_t N)
  {
    self_type x;
    for (camp::idx_t i = 0; i < N; ++i)
    {
      x.set(value, i);
    }
    return x;
  }

  /*!
   * @brief Extracts a scalar value and broadcasts to a new register
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type get_and_broadcast(int i) const
  {
    self_type x;
    x.broadcast(getThis()->get(i));
    return x;
  }

  /*!
   * @brief Generic gather operation for full vector.
   *
   * Must provide another register containing offsets of all values
   * to be loaded relative to supplied pointer.
   *
   * Offsets are element-wise, not byte-wise.
   *
   */
  template<typename T2>
  RAJA_HOST_DEVICE RAJA_INLINE self_type& gather(
      element_type const* ptr,
      RAJA::expt::Register<T2, REGISTER_POLICY> offsets)
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_load_strided_n++;
#endif
    for (camp::idx_t i = 0; i < self_type::s_num_elem; ++i)
    {
      getThis()->set(ptr[offsets.get(i)], i);
    }
    return *getThis();
  }

  /*!
   * @brief Generic gather operation for n-length subvector.
   *
   * Must provide another register containing offsets of all values
   * to be loaded relative to supplied pointer.
   *
   * Offsets are element-wise, not byte-wise.
   *
   */
  template<typename T2>
  RAJA_HOST_DEVICE RAJA_INLINE self_type& gather_n(
      element_type const* ptr,
      RAJA::expt::Register<T2, REGISTER_POLICY> const& offsets,
      camp::idx_t N)
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_load_strided_n++;
#endif
    for (camp::idx_t i = 0; i < N; ++i)
    {
      getThis()->set(ptr[offsets.get(i)], i);
    }
    return *getThis();
  }

  /*!
   * @brief Generic segmented load operation used for loading sub-matrices
   * from larger arrays.
   *
   * The default operation combines the s_segmented_offsets and gather
   * operations.
   *
   *
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& segmented_load(element_type const* ptr,
                            camp::idx_t segbits,
                            camp::idx_t stride_inner,
                            camp::idx_t stride_outer)
  {
    getThis()->gather(ptr, self_type::s_segmented_offsets(segbits, stride_inner,
                                                          stride_outer));
    return *getThis();
  }

  /*!
   * @brief Generic segmented load operation used for loading sub-matrices
   * from larger arrays where we load partial segments.
   *
   *
   *
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& segmented_load_nm(element_type const* ptr,
                               camp::idx_t segbits,
                               camp::idx_t stride_inner,
                               camp::idx_t stride_outer,
                               camp::idx_t num_inner,
                               camp::idx_t num_outer)
  {

    camp::idx_t num_segments = self_type::s_num_elem >> segbits;
    camp::idx_t seg_size     = 1 << segbits;

    camp::idx_t lane = 0;
    for (camp::idx_t seg = 0; seg < num_segments; ++seg)
    {
      for (camp::idx_t i = 0; i < seg_size; ++i)
      {

        if (seg >= num_outer || i >= num_inner)
        {
          getThis()->set(element_type(0), lane);
        }
        else
        {

          camp::idx_t offset = seg * stride_outer + i * stride_inner;

          element_type value = ptr[offset];

          getThis()->set(value, lane);
        }

        lane++;
      }
    }

    return *getThis();
  }

  /*!
   * @brief Generic scatter operation for full vector.
   *
   * Must provide another register containing offsets of all values
   * to be stored relative to supplied pointer.
   *
   * Offsets are element-wise, not byte-wise.
   *
   */
  template<typename T2>
  RAJA_HOST_DEVICE RAJA_INLINE self_type const& scatter(
      element_type* ptr,
      RAJA::expt::Register<T2, REGISTER_POLICY> const& offsets) const
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_load_strided_n++;
#endif
    for (camp::idx_t i = 0; i < self_type::s_num_elem; ++i)
    {
      ptr[offsets.get(i)] = getThis()->get(i);
    }
    return *getThis();
  }

  /*!
   * @brief Generic scatter operation for n-length subvector.
   *
   * Must provide another register containing offsets of all values
   * to be stored relative to supplied pointer.
   *
   * Offsets are element-wise, not byte-wise.
   *
   */
  template<typename T2>
  RAJA_HOST_DEVICE RAJA_INLINE self_type const& scatter_n(
      element_type* ptr,
      RAJA::expt::Register<T2, REGISTER_POLICY> const& offsets,
      camp::idx_t N) const
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_load_strided_n++;
#endif
    for (camp::idx_t i = 0; i < N; ++i)
    {
      ptr[offsets.get(i)] = getThis()->get(i);
    }
    return *getThis();
  }

  /*!
   * @brief Generic segmented load operation used for loading sub-matrices
   * from larger arrays.
   *
   * The default operation combines the s_segmented_offsets and gather
   * operations.
   *
   *
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& segmented_store(element_type* ptr,
                                   camp::idx_t segbits,
                                   camp::idx_t stride_inner,
                                   camp::idx_t stride_outer) const
  {
    getThis()->scatter(ptr, self_type::s_segmented_offsets(
                                segbits, stride_inner, stride_outer));
    return *getThis();
  }

  /*!
   * @brief Generic segmented load operation used for loading sub-matrices
   * from larger arrays where we load partial segments.
   *
   *
   *
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& segmented_store_nm(element_type* ptr,
                                      camp::idx_t segbits,
                                      camp::idx_t stride_inner,
                                      camp::idx_t stride_outer,
                                      camp::idx_t num_inner,
                                      camp::idx_t num_outer) const
  {

    camp::idx_t num_segments = self_type::s_num_elem >> segbits;
    camp::idx_t seg_size     = 1 << segbits;

    camp::idx_t lane = 0;
    for (camp::idx_t seg = 0; seg < num_segments; ++seg)
    {
      for (camp::idx_t i = 0; i < seg_size; ++i)
      {

        if (!(seg >= num_outer || i >= num_inner))
        {

          camp::idx_t offset = seg * stride_outer + i * stride_inner;

          ptr[offset] = getThis()->get(lane);
        }

        lane++;
      }
    }

    return *getThis();
  }

  /*!
   * @brief Set entire register to a single scalar value
   * @param value Value to set all register elements to
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& operator=(element_type value)
  {
    getThis()->broadcast(value);
    return *getThis();
  }

  /*!
   * @brief Set entire register to a single scalar value
   * @param value Value to set all register elements to
   */
  RAJA_SUPPRESS_HD_WARN
  template<typename T2>
  RAJA_HOST_DEVICE RAJA_INLINE self_type& operator=(
      RAJA::expt::Register<T2, RAJA::expt::scalar_register> const& value)
  {
    getThis()->broadcast(value.get(0));
    return *getThis();
  }

  /*!
   * @brief Assign one register to another
   * @param x register to copy
   * @return Value of (*this)
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& operator=(self_type const& x)
  {
    getThis()->copy(x);
    return *getThis();
  }

  /*!
   * @brief Add two registers
   * @param x register to add
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type operator+(self_type const& x) const { return getThis()->add(x); }

  /*!
   * @brief Add a register to this register
   * @param x register to add
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& operator+=(self_type const& x)
  {
    *getThis() = getThis()->add(x);
    return *getThis();
  }

  /*!
   * @brief Add scalar to this register
   * @param x scalar to add to this register
   * @return Value of (*this)+x
   *
   * This broadcasts the scalar to all lanes, then adds to this register
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type operator+(element_type const& x) const { return getThis()->add(x); }

  /*!
   * @brief Add a scalar to this register
   * @param x scalar to add to this register
   * @return Value of (*this)+x
   *
   * This broadcasts the scalar to all lanes, then adds to this register
   *
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& operator+=(element_type x)
  {
    *getThis() = getThis()->add(x);
    return *getThis();
  }

  /*!
   * @brief Negate the value of this register
   * @return Value of -(*this)
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type operator-() const { return self_type(0).subtract(*getThis()); }

  /*!
   * @brief Subtract two register registers
   * @param x register to subtract from this register
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type operator-(self_type const& x) const
  {
    return getThis()->subtract(x);
  }

  /*!
   * @brief Subtract a register from this register
   * @param x register to subtract from this register
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& operator-=(self_type const& x)
  {
    *getThis() = getThis()->subtract(x);
    return *getThis();
  }

  /*!
   * @brief Subtract scalar from this register
   * @param x register to subtract from this register
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type operator-(element_type const& x) const
  {
    return getThis()->subtract(x);
  }

  /*!
   * @brief Subtract a scalar from this register
   * @param x register to subtract from this register
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& operator-=(element_type const& x)
  {
    *getThis() = getThis()->subtract(x);
    return *getThis();
  }

  /*!
   * @brief Multiply two register registers, element wise
   * @param x register to subtract from this register
   * @return Value of (*this)+x
   */
  template<typename RHS>
  RAJA_HOST_DEVICE RAJA_INLINE self_type operator*(RHS const& rhs) const
  {
    return getThis()->multiply(rhs);
  }

  /*!
   * @brief Multiply a register with this register
   * @param x register to multiple with this register
   * @return Value of (*this)+x
   */
  template<typename RHS>
  RAJA_HOST_DEVICE RAJA_INLINE self_type& operator*=(RHS const& rhs)
  {
    *getThis() = getThis()->multiply(rhs);
    return *getThis();
  }

  /*!
   * @brief Divide two register registers, element wise
   * @param x register to subtract from this register
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type operator/(self_type const& x) const { return getThis()->divide(x); }

  /*!
   * @brief Divide this register by another register
   * @param x register to divide by
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& operator/=(self_type const& x)
  {
    *getThis() = getThis()->divide(x);
    return *getThis();
  }

  /*!
   * @brief Divide by a scalar, element wise
   * @param x Scalar to divide by
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type operator/(element_type const& x) const
  {
    return getThis()->divide(x);
  }

  /*!
   * @brief Divide this register by another register
   * @param x Scalar to divide by
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& operator/=(element_type const& x)
  {
    *getThis() = getThis()->divide(x);
    return *getThis();
  }

  /*!
   * @brief Divide n elements of this register by another register
   * @param x register to divide by
   * @param n Number of elements to divide
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide_n(self_type const& b, camp::idx_t n) const
  {
    self_type q(*getThis());
    for (camp::idx_t i = 0; i < n; ++i)
    {
      q.set(getThis()->get(i) / b.get(i), i);
    }
    return q;
  }

  /*!
   * @brief Divide n elements of this register by a scalar
   * @param x Scalar to divide by
   * @param n Number of elements to divide
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide_n(element_type const& b, camp::idx_t n) const
  {
    self_type q(*getThis());
    for (camp::idx_t i = 0; i < n; ++i)
    {
      q.set(getThis()->get(i) / b, i);
    }
    return q;
  }

  /*!
   * @brief Dot product of two registers
   * @param x Other register to dot with this register
   * @return Value of (*this) dot x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  element_type dot(self_type const& x) const
  {
    return getThis()->multiply(x).sum();
  }

  /*!
   * @brief Fused multiply add: fma(b, c) = (*this)*b+c
   *
   * Derived types can override this to implement intrinsic FMA's
   *
   * @param b Second product operand
   * @param c Sum operand
   * @return Value of (*this)*b+c
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type multiply_add(self_type const& b, self_type const& c) const
  {
    return (self_type(*getThis()) * self_type(b)) + self_type(c);
  }

  /*!
   * @brief Fused multiply subtract: fms(b, c) = (*this)*b-c
   *
   * Derived types can override this to implement intrinsic FMS's
   *
   * @param b Second product operand
   * @param c Subtraction operand
   * @return Value of (*this)*b-c
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type multiply_subtract(self_type const& b, self_type const& c) const
  {
    return getThis()->multiply_add(b, -c);
  }

  /*!
   * Multiply this tensor by a scalar value
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type scale(element_type c) const
  {
    return getThis()->multiply(self_type(c));
  }

  /*!
   * Minimum value across first N lanes of register
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  element_type min_n(camp::idx_t N) const { return getThis()->min(N); }

  /*!
   * Maximum value across first N lanes of register
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  element_type max_n(camp::idx_t N) const { return getThis()->max(N); }

  /*!
   * Provides vector-level building block for matrix transpose operations.
   *
   * This is a non-optimized reference version which will be used if
   * no architecture specialized version is supplied
   *
   * This is a permute-and-shuffle left operation
   *
   *           X=   x0  x1  x2  x3  x4  x5  x6  x7...
   *           Y=   y0  y1  y2  y3  y4  y5  y6  y7...
   *
   *  lvl=0    Z=   x0  y0  x2  y2  x4  y4  x6  y6...
   *  lvl=1    Z=   x0  x1  y0  y1  x4  x5  y4  y5...
   *  lvl=2    Z=   x0  x1  x2  x3  y0  y1  y2  y3...
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type transpose_shuffle_left(camp::idx_t lvl, self_type const& y) const
  {
    auto const& x = *getThis();

    self_type z;

    for (camp::idx_t i = 0; i < self_type::s_num_elem; ++i)
    {

      // extract value x or y
      camp::idx_t xy_select = (i >> lvl) & 0x1;


      z.set(xy_select == 0 ? x.get(i) : y.get(i - (1 << lvl)), i);
    }

    return z;
  }

  /*!
   * Provides vector-level building block for matrix transpose operations.
   *
   * This is a non-optimized reference version which will be used if
   * no architecture specialized version is supplied
   *
   * This is a permute-and-shuffle right operation
   *
   *           X=   x0  x1  x2  x3  x4  x5  x6  x7...
   *           Y=   y0  y1  y2  y3  y4  y5  y6  y7...
   *
   *  lvl=0    Z=   x1  y1  x3  y3  x5  y5  x7  y7...
   *  lvl=1    Z=   x2  x3  y2  y3  x6  x7  y6  y7...
   *  lvl=2    Z=   x4  x5  x6  x7  y4  y5  y6  y7...
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type transpose_shuffle_right(int lvl, self_type const& y) const
  {
    auto const& x = *getThis();

    self_type z;

    camp::idx_t i0 = 1 << lvl;

    for (camp::idx_t i = 0; i < self_type::s_num_elem; ++i)
    {

      // extract value x or y
      camp::idx_t xy_select = (i >> lvl) & 0x1;

      z.set(xy_select == 0 ? x.get(i0 + i) : y.get(i0 + i - (1 << lvl)), i);
    }

    return z;
  }

  /*!
   * Provides gather/scatter indices for segmented loads and stores
   *
   * THe number of segment bits (segbits) is specified, as well as the
   * stride between elements in a segment (stride_inner),
   * and the stride between segments (stride_outer)
   */
  RAJA_INLINE
  static int_vector_type s_segmented_offsets(camp::idx_t segbits,
                                             camp::idx_t stride_inner,
                                             camp::idx_t stride_outer)
  {
    int_vector_type result;

    camp::idx_t num_segments = self_type::s_num_elem >> segbits;
    camp::idx_t seg_size     = 1 << segbits;

    camp::idx_t lane = 0;
    for (camp::idx_t seg = 0; seg < num_segments; ++seg)
    {
      for (camp::idx_t i = 0; i < seg_size; ++i)
      {
        result.set(seg * stride_outer + i * stride_inner, lane);
        lane++;
      }
    }

    return result;
  }

  /*!
   * Sum elements within each segment, with segment size defined by segbits.
   * Stores each segments sum consecutively, but shifed to the
   * corresponding output_segment slot.
   *
   * Note: segment size is 1<<segbits elements
   *       number of segments is s_num_elem>>seg_bits
   *
   *
   *
   *
   *  Example:
   *
   *  Given input vector  X = x0, x1, x2, x3, x4, x5, x6, x7
   *
   *  segbits=0 is equivalent to the input vector,  since there are 8
   *      outputs, there is only 1 output segment
   *
   *      Result= x0, x1, x2, x3, x4, x5, x6, x7
   *
   *  segbits=1 sums neighboring pairs of values.  There are 4 output,
   *      so there are possible output segments.
   *
   *      output_segment=0:
   *      Result= x0+x1, x2+x3, x4+x5, x6+x7, 0, 0, 0, 0
   *
   *      output_segment=1:
   *      Result= 0, 0, 0, 0, x0+x1, x2+x3, x4+x5, x6+x7
   *
   *  and so on up to segbits=3, which is a full sum of x0..x7, and the
   *      output_segment denotes the vector position of the sum
   *
   */
  RAJA_INLINE
  self_type segmented_sum_inner(camp::idx_t segbits,
                                camp::idx_t output_segment) const
  {
    self_type result(0);

    // default implementation is dumb, just sum each value into
    // appropriate segment lane
    int output_offset = output_segment * self_type::s_num_elem >> segbits;

    for (camp::idx_t i = 0; i < self_type::s_num_elem; ++i)
    {
      auto value =
          getThis()->get(i) + result.get((i >> segbits) + output_offset);
      result.set(value, (i >> segbits) + output_offset);
    }

    return result;
  }

  /*!
   * Sum all segments as subvectors, with segment size defined by segbits
   *
   * Note: segment size is 1<<segbits elements
   *       number of segments is s_num_elem>>seg_bits
   *
   *
   *
   *
   *  Example:
   *
   *  Given input vector  X = x0, x1, x2, x3, x4, x5, x6, x7
   *
   *  segbits=0 the segments are size 1, which means that this is just a
   *      sum of all elements.  The output_segment determines where the
   *      result is placed.
   *
   *      output_segment=0:
   *      Result= x0+x1+x2+x3+x4+x5+x6+x7, 0, 0, 0, 0, 0, 0, 0, 0
   *
   *      output_segment=3:
   *      Result= 0, 0, x0+x1+x2+x3+x4+x5+x6+x7, 0, 0, 0, 0, 0, 0
   *
   *  segbits=1 the segments are 2-wide:
   *
   *      output_segment=0:
   *      Result= x0+x2+x4+x6, x1+x3+x5+x7, 0, 0, 0, 0, 0, 0
   *
   *      output_segment=1:
   *      Result= 0, 0, x0+x2+x4+x6, x1+x3+x5+x7, 0, 0, 0, 0
   *
   *  and so on up to segbits=3, which is just the original vector:
   *  segbits=3
   *      Result= x0, x1, x2, x3, x4, x5, x6, x7
   *
   */
  RAJA_INLINE
  self_type segmented_sum_outer(camp::idx_t segbits,
                                camp::idx_t output_segment) const
  {
    self_type result(0);

    // default implementation is dumb, just sum each value into
    // appropriate segment lane
    int output_offset = output_segment * (1 << segbits);

    for (camp::idx_t i = 0; i < self_type::s_num_elem; ++i)
    {
      camp::idx_t output_i = output_offset + (i & ((1 << segbits) - 1));
      auto value           = getThis()->get(i) + result.get(output_i);
      result.set(value, output_i);
    }

    return result;
  }

  RAJA_INLINE
  self_type segmented_divide_nm(self_type den,
                                camp::idx_t segbits,
                                camp::idx_t num_inner,
                                camp::idx_t num_outer) const
  {
    self_type result;

    camp::idx_t num_segments = self_type::s_num_elem >> segbits;
    camp::idx_t seg_size     = 1 << segbits;

    camp::idx_t lane = 0;
    for (camp::idx_t seg = 0; seg < num_segments; ++seg)
    {
      for (camp::idx_t i = 0; i < seg_size; ++i)
      {

        if (seg >= num_outer || i >= num_inner)
        {
          result.set(element_type(0), lane);
        }
        else
        {

          element_type div = getThis()->get(lane) / den.get(lane);

          result.set(div, lane);
        }

        lane++;
      }
    }

    return result;
  }

  /*!
   * Segmented dot product performs dot products
   * Note: segment size is 1<<segbits elements
   *       number of segments is s_num_elem>>seg_bits
   *
   *
   *  Example:
   *
   *  Given input vector  X = x0, x1, x2, x3, x4, x5, x6, x7
   *                      Y = y0, y1, y2, y3, y4, y5, y6, y7
   *
   *
   *  segbits=0 is equivalent to a vector multiply,  since there are 8
   *      outputs, there is only 1 output segment
   *
   *      Result= x0*y0, x1*y1, x2*y2, x3*y3, x4*y4, x5*y5, x6*y6, x7*y7
   *
   *  segbits=1 sums neighboring pairs of products.  There are 4 output,
   *      so there are possible output segments.
   *
   *      output_segment=0:
   *      Result= x0*y0+x1*y1, x2*y2+x3*y3, x4*y4+x5*y5, x6*y6+x7*y7, 0, 0, 0, 0
   *
   *      output_segment=1:
   *      Result= 0, 0, 0, 0, x0*y0+x1*y1, x2*y2+x3*y3, x4*y4+x5*y5, x6*y6+x7*y7
   *
   *  and so on up to segbits=3, which is a full dot-product of x and y, and the
   *      output_segment denotes the vector position of the result
   *
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type segmented_dot(camp::idx_t segbits,
                          camp::idx_t output_segment,
                          self_type const& x) const
  {
    return getThis()->multiply(x).segmented_sum_inner(segbits, output_segment);
  }

  /*!
   * Segmented broadcast copies a segment to all output segments of a vector
   *
   * Note: segment size is 1<<segbits elements
   *       number of segments is s_num_elem>>seg_bits
   *
   *
   *  Example:
   *
   *  Given input vector  X = x0, x1, x2, x3, x4, x5, x6, x7
   *
   *  segbits=0 means the input segment size is 1, so this selects the
   *      value at x[input_segmnet] and broadcasts it to the rest of the
   *      vector
   *
   *      input segments allowed are from 0 to 7, inclusive
   *
   *      input_segment=0
   *      Result= x0, x0, x0, x0, x0, x0, x0, x0
   *
   *      input_segment=5
   *      Result= x5, x5, x5, x5, x5, x5, x5, x5
   *
   *  segbits=1 means that the input segments are each pair of x values:
   *
   *      input segments allowed are from 0 to 3, inclusive
   *
   *      input_segment=0:
   *      Result= x0, x1, x0, x1, x0, x1, x0, x1
   *
   *      input_segment=1:
   *      Result= x2, x3, x2, x3, x2, x3, x2, x3
   *
   *      input_segment=3:
   *      Result= x6, x7, x6, x7, x6, x7, x6, x7
   *
   *  and so on up to segbits=2, the input segments are 4 wide:
   *
   *      input segments allowed are from 0 or 1
   *
   *      input_segment=0:
   *      Result= x0, x1, x2, x3, x0, x1, x2, x3
   *
   *      input_segment=1:
   *      Result= x4, x5, x6, x7, x4, x5, x6, x7
   *
   */
  RAJA_INLINE
  self_type segmented_broadcast_inner(camp::idx_t segbits,
                                      camp::idx_t input_segment) const
  {
    self_type result;

    camp::idx_t mask   = (1 << segbits) - 1;
    camp::idx_t offset = input_segment << segbits;

    // default implementation is dumb, just sum each value into
    // appropriate segment lane
    for (camp::idx_t i = 0; i < self_type::s_num_elem; ++i)
    {

      auto off = (i & mask) + offset;

      result.set(getThis()->get(off), i);
    }

    return result;
  }

  /*!
   * Segmented broadcast spreads a segment to all output segments of a vector
   *
   * Note: segment size is 1<<segbits elements
   *       number of segments is s_num_elem>>seg_bits
   *
   *
   *  Example:
   *
   *  Given input vector  X = x0, x1, x2, x3, x4, x5, x6, x7
   *
   *  segbits=0 means the input segment size is 1, so this selects the
   *      value at x[input_segmnet] and broadcasts it to the rest of the
   *      vector
   *
   *      input segments allowed are from 0 to 7, inclusive
   *
   *      input_segment=0
   *      Result= x0, x0, x0, x0, x0, x0, x0, x0
   *
   *      input_segment=5
   *      Result= x5, x5, x5, x5, x5, x5, x5, x5
   *
   *  segbits=1 means that the input segments are each pair of x values:
   *
   *      input segments allowed are from 0 to 3, inclusive
   *
   *      output_segment=0:
   *      Result= x0, x0, x0, x0, x1, x1, x1, x1
   *
   *      output_segment=1:
   *      Result= x2, x2, x2, x2, x3, x3, x3, x3
   *
   *      output_segment=3:
   *      Result= x6, x6, x6, x6, x7, x7, x7, x7
   */
  RAJA_INLINE
  self_type segmented_broadcast_outer(camp::idx_t segbits,
                                      camp::idx_t input_segment) const
  {
    self_type result;

    camp::idx_t offset = input_segment * (self_type::s_num_elem >> segbits);

    // default implementation is dumb, just sum each value into
    // appropriate segment lane
    for (camp::idx_t i = 0; i < self_type::s_num_elem; ++i)
    {

      auto off = (i >> segbits) + offset;

      result.set(getThis()->get(off), i);
    }

    return result;
  }

  /*!
   * @brief Converts to vector to a string
   *
   *
   */
  RAJA_INLINE
  std::string to_string() const
  {
    std::string s = "Register(" + std::to_string(self_type::s_num_elem) + ")[ ";

    //
    for (camp::idx_t i = 0; i < self_type::s_num_elem; ++i)
    {
      s += std::to_string(getThis()->get(i)) + " ";
    }

    s += " ]\n";

    return s;
  }
};


}  // namespace expt
}  // namespace internal
}  // namespace RAJA


#endif
