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

#ifndef RAJA_pattern_tensor_ET_ExpressionTemplateBase_HPP
#define RAJA_pattern_tensor_ET_ExpressionTemplateBase_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/tensor/internal/TensorRef.hpp"

#include "RAJA/pattern/tensor/internal/ET/normalizeOperand.hpp"
#include "RAJA/pattern/tensor/internal/ET/BinaryOperatorTraits.hpp"

//#define RAJA_DEBUG_PRINT_ET_AST

namespace RAJA
{
namespace internal
{
namespace expt
{


class TensorRegisterConcreteBase;

namespace ET
{

//
// forward decls
//

template<typename TENSOR_REGISTER_TYPE, typename REF_TYPE>
class TensorLoadStore;


template<typename LHS_TYPE, typename RHS_TYPE>
class TensorMultiply;

template<typename LHS_TYPE, typename RHS_TYPE>
class TensorDivide;

template<typename TENSOR_TYPE>
class TensorNegate;

template<typename TENSOR_TYPE>
class TensorTranspose;

// provides a non-templated base-type for all ET's
// this allows using things like std::is_base_of
class TensorExpressionConcreteBase
{};

template<typename DERIVED_TYPE>
class TensorExpressionBase : public TensorExpressionConcreteBase
{
public:
  using self_type = DERIVED_TYPE;

private:
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type* getThis() { return static_cast<self_type*>(this); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr self_type const* getThis() const
  {
    return static_cast<self_type const*>(this);
  }

public:
  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr camp::idx_t getDimBegin(camp::idx_t) const { return 0; }

  RAJA_SUPPRESS_HD_WARN
  template<typename RHS>
  RAJA_INLINE RAJA_HOST_DEVICE TensorAdd<self_type, normalize_operand_t<RHS>>
  operator+(RHS const& rhs) const
  {
    return TensorAdd<self_type, normalize_operand_t<RHS>>(
        *getThis(), normalizeOperand(rhs));
  }

  RAJA_SUPPRESS_HD_WARN
  template<typename RHS>
  RAJA_INLINE RAJA_HOST_DEVICE
      TensorSubtract<self_type, normalize_operand_t<RHS>>
      operator-(RHS const& rhs) const
  {
    return TensorSubtract<self_type, normalize_operand_t<RHS>>(
        *getThis(), normalizeOperand(rhs));
  }

  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  TensorNegate<self_type> operator-() const
  {
    return TensorNegate<self_type>(*getThis());
  }

  RAJA_SUPPRESS_HD_WARN
  template<typename RHS>
  RAJA_INLINE RAJA_HOST_DEVICE
      TensorMultiply<self_type, normalize_operand_t<RHS>>
      operator*(RHS const& rhs) const
  {
    return TensorMultiply<self_type, normalize_operand_t<RHS>>(
        *getThis(), normalizeOperand(rhs));
  }

  RAJA_SUPPRESS_HD_WARN
  template<typename RHS>
  RAJA_INLINE RAJA_HOST_DEVICE TensorDivide<self_type, normalize_operand_t<RHS>>
  operator/(RHS const& rhs) const
  {
    return TensorDivide<self_type, normalize_operand_t<RHS>>(
        *getThis(), normalizeOperand(rhs));
  }

  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE

  RAJA_HOST_DEVICE
  TensorTranspose<self_type> transpose() const
  {
    return TensorTranspose<self_type>(*getThis());
  }
};


}  // namespace ET

}  // namespace expt
}  // namespace internal

}  // namespace RAJA


#endif
