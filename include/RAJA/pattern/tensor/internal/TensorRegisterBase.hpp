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

#ifndef RAJA_pattern_tensor_TensorRegisterBase_HPP
#define RAJA_pattern_tensor_TensorRegisterBase_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "camp/camp.hpp"
#include "RAJA/pattern/tensor/TensorLayout.hpp"
#include "RAJA/pattern/tensor/internal/TensorRef.hpp"

namespace RAJA
{
namespace internal
{
namespace expt
{


namespace ET
{
class TensorExpressionConcreteBase;
}  // namespace ET

template<typename TENSOR, camp::idx_t DIM>
struct TensorDimSize
{
  static constexpr camp::idx_t value = TENSOR::s_dim_size(DIM);
};

/*
 * Tensor product helper class.
 *
 * This defines the default product operation between types when using the
 * operator*
 *
 */
template<typename LHS, typename RHS>
struct TensorDefaultOperation
{

  using multiply_type = decltype(LHS().multiply(RHS()));

  // default multiplication operator
  RAJA_HOST_DEVICE

  RAJA_INLINE
  static multiply_type multiply(LHS const& lhs, RHS const& rhs)
  {
    return lhs.multiply(rhs);
  }
};

template<typename REF_TYPE>
struct TensorRegisterStoreRef
{
  using self_type = TensorRegisterStoreRef<REF_TYPE>;
  REF_TYPE m_ref;

  RAJA_SUPPRESS_HD_WARN
  template<typename RHS>
  RAJA_HOST_DEVICE RAJA_INLINE self_type operator=(RHS const& rhs)
  {

    rhs.store_ref(m_ref);
    return *this;
  }
};

template<camp::idx_t N, camp::idx_t D>
struct DivideRoundUp
{
  static constexpr camp::idx_t value = (N % D) > 0 ? (1 + N / D) : (N / D);
};

class TensorRegisterConcreteBase
{};

/*!
 * TensorRegister base class that provides some default behaviors and simplifies
 * the implementation of new register types.
 *
 * This uses CRTP to provide static polymorphism
 */
template<typename Derived>
class TensorRegisterBase;

template<typename REGISTER_POLICY,
         typename T,
         typename LAYOUT,
         typename camp::idx_t... SIZES>
class TensorRegisterBase<
    RAJA::expt::
        TensorRegister<REGISTER_POLICY, T, LAYOUT, camp::idx_seq<SIZES...>>>
    : public TensorRegisterConcreteBase
{
public:
  using self_type = RAJA::expt::
      TensorRegister<REGISTER_POLICY, T, LAYOUT, camp::idx_seq<SIZES...>>;
  using element_type = camp::decay<T>;

  static constexpr camp::idx_t s_num_dims = sizeof...(SIZES);

  static constexpr camp::idx_t s_num_registers =
      DivideRoundUp<RAJA::product<camp::idx_t>(SIZES...),
                    RegisterTraits<REGISTER_POLICY, T>::s_num_elem>::value;

  using index_type = camp::idx_t;

  using register_type = RAJA::expt::Register<T, REGISTER_POLICY>;

  using register_policy = REGISTER_POLICY;

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

protected:
  register_type m_registers[s_num_registers];

public:
  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr TensorRegisterBase() {}

  RAJA_HOST_DEVICE

  RAJA_INLINE
  TensorRegisterBase(element_type c) { broadcast(c); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  TensorRegisterBase(self_type const& c) { copy(c); }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  ~TensorRegisterBase() {}

  /*
   * Overload for:    assignment of ET to a TensorRegister
   */
  template<typename RHS,
           typename std::enable_if<
               std::is_base_of<ET::TensorExpressionConcreteBase, RHS>::value,
               bool>::type = true>
  RAJA_INLINE RAJA_HOST_DEVICE TensorRegisterBase(RHS const& rhs)
  {
    // evaluate a single tile of the ET, storing in this TensorRegister
    *this = rhs.eval(self_type::s_get_default_tile());
  }

  template<typename... REGS>
  explicit RAJA_HOST_DEVICE RAJA_INLINE TensorRegisterBase(register_type reg0,
                                                           REGS const&... regs)
      : m_registers {reg0, regs...}
  {
    static_assert(1 + sizeof...(REGS) == s_num_registers,
                  "Incompatible number of registers");
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  static constexpr bool is_root() { return register_type::is_root(); }

  template<typename REF_TYPE>
  RAJA_HOST_DEVICE RAJA_INLINE static constexpr TensorRegisterStoreRef<REF_TYPE>
  create_et_store_ref(REF_TYPE const& ref)
  {
    return TensorRegisterStoreRef<REF_TYPE> {ref};
  }

  RAJA_SUPPRESS_HD_WARN
  template<typename REF_TYPE>
  RAJA_HOST_DEVICE RAJA_INLINE static self_type s_load_ref(REF_TYPE const& ref)
  {

    self_type value;

    value.load_ref(ref);
    return value;
  }

  /*!
   * Gets the size of the tensor
   * Since this is a vector, just the length of the vector in dim 0
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  static constexpr int s_dim_elem(int dim)
  {
    return (dim == 0) ? self_type::s_num_elem : 0;
  }

  /*!
   * Gets the default tile of this tensor
   * That tile always start at 0, and extends to the full tile sizes
   */

  RAJA_HOST_DEVICE

  RAJA_INLINE
  static constexpr StaticTensorTile<int,
                                    TENSOR_FULL,
                                    camp::int_seq<int, int(SIZES * 0)...>,
                                    camp::int_seq<int, int(SIZES)...>>
  s_get_default_tile()
  {
    return StaticTensorTile<int, TENSOR_FULL,
                            camp::int_seq<int, int(SIZES * 0)...>,
                            camp::int_seq<int, int(SIZES)...>>();
  }

  /*!
   * @brief convenience routine to allow Vector classes to use
   * camp::sink() across a variety of register types, and use things like
   * ternary operators
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr bool sink() const { return false; }

  /*!
   * Copy contents of another tensor
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& copy(self_type const& c)
  {
    for (camp::idx_t i = 0; i < s_num_registers; ++i)
    {
      m_registers[i] = c.vec(i);
    }
    return *getThis();
  }

  /*!
   * Sets all elements to zero
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& clear()
  {
    for (camp::idx_t i = 0; i < s_num_registers; ++i)
    {
      m_registers[i] = register_type(0);
    }


    return *getThis();
  }

  /*!
   * Copy contents of another matrix operator
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& broadcast(element_type v)
  {
    for (camp::idx_t i = 0; i < s_num_registers; ++i)
    {
      m_registers[i].broadcast(v);
    }
    return *getThis();
  }

  /*!
   * @brief Broadcast scalar value to first N register elements
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& broadcast_n(element_type const& value, camp::idx_t N)
  {
    for (camp::idx_t i = 0; i < N; ++i)
    {
      getThis()->set(value, i);
    }
    return *getThis();
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

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type add(self_type const& mat) const
  {
    self_type result;
    for (camp::idx_t i = 0; i < s_num_registers; ++i)
    {
      result.vec(i) = m_registers[i].add(mat.vec(i));
    }
    return result;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type subtract(self_type const& mat) const
  {
    self_type result;
    for (camp::idx_t i = 0; i < s_num_registers; ++i)
    {
      result.vec(i) = m_registers[i].subtract(mat.vec(i));
    }
    return result;
  }

  /*!
   * element-wise multiplication
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type multiply(self_type const& x) const
  {
    self_type result;
    for (camp::idx_t i = 0; i < s_num_registers; ++i)
    {
      result.vec(i) = m_registers[i].multiply(x.vec(i));
    }
    return result;
  }

  /*!
   * element-wise fused multiply add
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type multiply_add(self_type const& x, self_type const& add) const
  {
    self_type result;
    for (camp::idx_t i = 0; i < s_num_registers; ++i)
    {
      result.vec(i) = m_registers[i].multiply_add(x.vec(i), add.vec(i));
    }
    return result;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide(self_type const& mat) const
  {
    self_type result;
    for (camp::idx_t reg = 0; reg < s_num_registers; ++reg)
    {
      result.vec(reg) = m_registers[reg].divide(mat.vec(reg));
    }
    return result;
  }

  /*!
   * @brief Dot product of two vectors
   * @param x Other vector to dot with this vector
   * @return Value of (*this) dot x
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  element_type dot(self_type const& x) const
  {
    element_type result(0);

    for (camp::idx_t reg = 0; reg < s_num_registers; ++reg)
    {
      result += m_registers[reg].multiply(x.vec(reg)).sum();
    }

    return result;
  }

  /*!
   * @brief Set entire vector to a single scalar value
   * @param value Value to set all vector elements to
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& operator=(element_type value)
  {
    getThis()->broadcast(value);
    return *getThis();
  }

  /*!
   * @brief Set entire vector to a single scalar value
   * @param value Value to set all vector elements to
   */
  RAJA_SUPPRESS_HD_WARN
  template<typename T2>
  RAJA_HOST_DEVICE RAJA_INLINE self_type const& operator=(
      RAJA::expt::TensorRegister<RAJA::expt::scalar_register,
                                 T2,
                                 RAJA::expt::ScalarLayout,
                                 camp::idx_seq<>> const& value)
  {
    getThis()->broadcast(value.get(0));
    return *getThis();
  }

  /*!
   * @brief Assign one register to antoher
   * @param x Vector to copy
   * @return Value of (*this)
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& operator=(self_type const& x)
  {
    getThis()->copy(x);
    return *getThis();
  }

  /*!
   * @brief Add two vector registers
   * @param x Vector to add to this register
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type operator+(self_type const& x) const { return getThis()->add(x); }

  /*!
   * @brief Add a vector to this vector
   * @param x Vector to add to this register
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
   * @brief Add vector to a scalar
   * @param x scalar to add to this register
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type operator+(element_type const& x) const { return getThis()->add(x); }

  /*!
   * @brief Add a scalar to this vector
   * @param x scalar to add to this register
   * @return Value of (*this)+x
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
   * @brief Negate the value of this vector
   * @return Value of -(*this)
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type operator-() const { return self_type(0).subtract(*getThis()); }

  /*!
   * @brief Subtract two vector registers
   * @param x Vector to subtract from this register
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
   * @brief Subtract a vector from this vector
   * @param x Vector to subtract from this register
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
   * @param x Vector to subtract from this register
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
   * @brief Subtract a scalar from this vector
   * @param x Vector to subtract from this register
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
   * @brief Multiply two vector registers, element wise
   * @param x Vector to subtract from this register
   * @return Value of (*this)+x
   */
  template<typename RHS>
  RAJA_HOST_DEVICE RAJA_INLINE
      typename TensorDefaultOperation<self_type, RHS>::multiply_type
      operator*(RHS const& rhs) const
  {
    return TensorDefaultOperation<self_type, RHS>::multiply(*getThis(), rhs);
  }

  /*!
   * @brief Multiply a vector with this vector
   * @param x Vector to multiple with this register
   * @return Value of (*this)+x
   */
  template<typename RHS>
  RAJA_HOST_DEVICE RAJA_INLINE self_type& operator*=(RHS const& rhs)
  {
    *getThis() =
        TensorDefaultOperation<self_type, RHS>::multiply(*getThis(), rhs);
    return *getThis();
  }

  /*!
   * @brief Divide two vector registers, element wise
   * @param x Vector to subtract from this register
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type operator/(self_type const& x) const { return getThis()->divide(x); }

  /*!
   * @brief Divide this vector by another vector
   * @param x Vector to divide by
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
   * @brief Divide this vector by another vector
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
   * @brief Returns element wise minimum value tensor
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type vmin(self_type x) const
  {
    self_type result;
    for (camp::idx_t i = 0; i < s_num_registers; ++i)
    {
      result.vec(i) = m_registers[i].vmin(x.vec(i));
    }
    return result;
  }

  /*!
   * @brief Returns element wise maximum value tensor
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type vmax(self_type x) const
  {
    self_type result;
    for (camp::idx_t i = 0; i < s_num_registers; ++i)
    {
      result.vec(i) = m_registers[i].vmax(x.vec(i));
    }
    return result;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  register_type& vec(int i) { return m_registers[i]; }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr register_type const& vec(int i) const { return m_registers[i]; }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  register_type& get_register(int reg) { return m_registers[reg]; }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr register_type const& get_register(int reg) const
  {
    return m_registers[reg];
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
   * In-place add operation
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type& inplace_add(self_type x)
  {
    *getThis() = getThis()->add(x);
    return *getThis();
  }

  /*!
   * In-place sbutract operation
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type& inplace_subtract(self_type x)
  {
    *getThis() = getThis()->subtract(x);
    return *getThis();
  }

  /*!
   * In-place multiply operation
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type& inplace_multiply(self_type x)
  {
    *getThis() = getThis()->multiply(x);
    return *getThis();
  }

  /*!
   * In-place multiply-add operation
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type& inplace_multiply_add(self_type x, self_type y)
  {
    *getThis() = getThis()->multiply_add(x, y);
    return *getThis();
  }

  /*!
   * In-place multiply-subtract operation
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type& inplace_multiply_subtract(self_type x, self_type y)
  {
    *getThis() = getThis()->multiply_subtract(x, y);
    return *getThis();
  }

  /*!
   * In-place divide operation
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type& inplace_divide(self_type x)
  {
    *getThis() = getThis()->divide(x);
    return *getThis();
  }

  /*!
   * In-place scaling operation
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type& inplace_scale(element_type x)
  {
    *getThis() = getThis()->scale(x);
    return *getThis();
  }
};

}  // namespace expt

}  // namespace internal

}  // namespace RAJA


#endif
