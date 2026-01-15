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

#ifndef RAJA_pattern_tensor_ET_TensorMultiply_HPP
#define RAJA_pattern_tensor_ET_TensorMultiply_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/tensor/internal/ET/ExpressionTemplateBase.hpp"
#include "RAJA/pattern/tensor/internal/ET/MultiplyOperator.hpp"

namespace RAJA
{
namespace internal
{
namespace expt
{

namespace ET
{

// forward decl for FMA contraction
template<typename LEFT_OPERAND_TYPE,
         typename RIGHT_OPERAND_TYPE,
         typename ADD_TYPE>
class TensorMultiplyAdd;

template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
class TensorMultiply
    : public TensorExpressionBase<
          TensorMultiply<LEFT_OPERAND_TYPE, RIGHT_OPERAND_TYPE>>
{
public:
  using self_type = TensorMultiply<LEFT_OPERAND_TYPE, RIGHT_OPERAND_TYPE>;
  using left_operand_type  = LEFT_OPERAND_TYPE;
  using right_operand_type = RIGHT_OPERAND_TYPE;
  using multiply_op = MultiplyOperator<LEFT_OPERAND_TYPE, RIGHT_OPERAND_TYPE>;

  using element_type = typename LEFT_OPERAND_TYPE::element_type;
  using index_type   = typename LEFT_OPERAND_TYPE::index_type;

  using result_type                       = typename multiply_op::result_type;
  static constexpr camp::idx_t s_num_dims = multiply_op::s_num_dims;

private:
  left_operand_type m_left_operand;
  right_operand_type m_right_operand;

public:
  RAJA_INLINE

  RAJA_HOST_DEVICE
  TensorMultiply(left_operand_type const& left_operand,
                 right_operand_type const& right_operand)
      : m_left_operand {left_operand},
        m_right_operand {right_operand}
  {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr int getDimSize(int dim) const
  {
    return multiply_op::getDimSize(dim, m_left_operand, m_right_operand);
  }

  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE auto eval(TILE_TYPE const& tile) const
      -> decltype(multiply_op::multiply(tile, m_left_operand, m_right_operand))
  {
    return multiply_op::multiply(tile, m_left_operand, m_right_operand);
  }

  /*!
   * Returns the LHS of the operation, used to form contractions
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr left_operand_type const& getLeftOperand() const
  {
    return m_left_operand;
  }

  /*!
   * Returns the RHS of the operation, used to form contractions
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr right_operand_type const& getRightOperand() const
  {
    return m_right_operand;
  }

  /*!
   * operator+ overload that forms a FMA contraction
   */
  RAJA_SUPPRESS_HD_WARN
  template<typename ADD>
  RAJA_INLINE RAJA_HOST_DEVICE TensorMultiplyAdd<left_operand_type,
                                                 right_operand_type,
                                                 normalize_operand_t<ADD>>
  operator+(ADD const& add) const
  {
    return TensorMultiplyAdd<left_operand_type, right_operand_type,
                             normalize_operand_t<ADD>>(
        m_left_operand, m_right_operand, normalizeOperand(add));
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  void print_ast() const
  {
    printf("Multiply[");
    multiply_op::print_ast();
    printf("](");
    m_left_operand.print_ast();
    printf(", ");
    m_right_operand.print_ast();
    printf(")");
  }
};

/*
 * Overload for:    arithmetic * tensorexpression

 */
template<
    typename LHS,
    typename RHS,
    typename std::enable_if<std::is_arithmetic<LHS>::value, bool>::type = true,
    typename std::enable_if<
        std::is_base_of<TensorExpressionConcreteBase, RHS>::value,
        bool>::type = true>
RAJA_INLINE RAJA_HOST_DEVICE auto operator*(LHS const& left_operand,
                                            RHS const& right_operand)
    -> TensorMultiply<typename NormalizeOperandHelper<LHS>::return_type, RHS>
{
  return TensorMultiply<typename NormalizeOperandHelper<LHS>::return_type, RHS>(
      NormalizeOperandHelper<LHS>::normalize(left_operand), right_operand);
}

}  // namespace ET

}  // namespace expt
}  // namespace internal

}  // namespace RAJA


#endif
