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

#ifndef RAJA_pattern_tensor_ET_TensorAdd_HPP
#define RAJA_pattern_tensor_ET_TensorAdd_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/tensor/internal/ET/ExpressionTemplateBase.hpp"
#include "RAJA/pattern/tensor/internal/ET/BinaryOperatorTraits.hpp"

namespace RAJA
{
namespace internal
{
namespace expt
{


namespace ET
{


template<typename OPERATOR, typename LEFT_OPERAND, typename RIGHT_OPERAND>
class TensorBinaryOperator
    : public TensorExpressionBase<
          TensorBinaryOperator<OPERATOR, LEFT_OPERAND, RIGHT_OPERAND>>
{
public:
  using self_type = TensorBinaryOperator<OPERATOR, LEFT_OPERAND, RIGHT_OPERAND>;
  using operator_type      = OPERATOR;
  using left_operand_type  = LEFT_OPERAND;
  using right_operand_type = RIGHT_OPERAND;

  using element_type = typename LEFT_OPERAND::element_type;
  using index_type   = typename LEFT_OPERAND::index_type;

  using operator_traits = OperatorTraits<LEFT_OPERAND, RIGHT_OPERAND>;
  using result_type     = typename operator_traits::result_type;

  static constexpr camp::idx_t s_num_dims = operator_traits::s_num_dims;

private:
  left_operand_type m_left_operand;
  right_operand_type m_right_operand;

public:
  RAJA_INLINE

  RAJA_HOST_DEVICE
  TensorBinaryOperator(left_operand_type const& left,
                       right_operand_type const& right)
      : m_left_operand {left},
        m_right_operand {right}
  {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr auto getDimSize(camp::idx_t dim) const
      -> decltype(operator_traits::getDimSize(dim,
                                              m_left_operand,
                                              m_right_operand))
  {
    return operator_traits::getDimSize(dim, m_left_operand, m_right_operand);
  }

  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE auto eval(TILE_TYPE const& tile) const
      -> decltype(operator_type::eval(m_left_operand.eval(tile),
                                      m_right_operand.eval(tile)))
  {
    return operator_type::eval(m_left_operand.eval(tile),
                               m_right_operand.eval(tile));
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  void print_ast() const
  {
    operator_type::print_ast();
    printf("[");
    operator_type::print_ast();
    printf("](");
    m_left_operand.print_ast();
    printf(", ");
    m_right_operand.print_ast();
    printf(")");
  }
};

/*
 * Overload for:    arithmetic + tensorexpression

 */
template<typename LEFT_OPERAND,
         typename RIGHT_OPERAND,
         typename std::enable_if<std::is_arithmetic<LEFT_OPERAND>::value,
                                 bool>::type = true,
         typename std::enable_if<std::is_base_of<TensorExpressionConcreteBase,
                                                 RIGHT_OPERAND>::value,
                                 bool>::type = true>
RAJA_INLINE RAJA_HOST_DEVICE auto operator+(LEFT_OPERAND const& left,
                                            RIGHT_OPERAND const& right)
    -> TensorAdd<typename NormalizeOperandHelper<LEFT_OPERAND>::return_type,
                 RIGHT_OPERAND>
{
  return TensorAdd<typename NormalizeOperandHelper<LEFT_OPERAND>::return_type,
                   RIGHT_OPERAND>(
      NormalizeOperandHelper<LEFT_OPERAND>::normalize(left), right);
}

/*
 * Overload for:    arithmetic - tensorexpression

 */
template<typename LEFT_OPERAND,
         typename RIGHT_OPERAND,
         typename std::enable_if<std::is_arithmetic<LEFT_OPERAND>::value,
                                 bool>::type = true,
         typename std::enable_if<std::is_base_of<TensorExpressionConcreteBase,
                                                 RIGHT_OPERAND>::value,
                                 bool>::type = true>
RAJA_INLINE RAJA_HOST_DEVICE auto operator-(LEFT_OPERAND const& left,
                                            RIGHT_OPERAND const& right)
    -> TensorSubtract<
        typename NormalizeOperandHelper<LEFT_OPERAND>::return_type,
        RIGHT_OPERAND>
{
  return TensorSubtract<
      typename NormalizeOperandHelper<LEFT_OPERAND>::return_type,
      RIGHT_OPERAND>(NormalizeOperandHelper<LEFT_OPERAND>::normalize(left),
                     right);
}

//    /*
//     * Overload for:    arithmetic / tensorexpression
//
//     */
//    template<typename LEFT_OPERAND, typename RIGHT_OPERAND,
//      typename std::enable_if<std::is_arithmetic<LEFT_OPERAND>::value,
//      bool>::type = true, typename
//      std::enable_if<std::is_base_of<TensorExpressionConcreteBase,
//      RIGHT_OPERAND>::value, bool>::type = true>
//    RAJA_INLINE
//    RAJA_HOST_DEVICE
//    auto operator/(LEFT_OPERAND const &left, RIGHT_OPERAND const &right) ->
//    TensorDivide<typename NormalizeOperandHelper<LEFT_OPERAND>::return_type,
//    RIGHT_OPERAND>
//    {
//      return TensorDivide<typename
//      NormalizeOperandHelper<LEFT_OPERAND>::return_type,
//      RIGHT_OPERAND>(NormalizeOperandHelper<LEFT_OPERAND>::normalize(left),
//      right);
//    }


}  // namespace ET

}  // namespace expt
}  // namespace internal

}  // namespace RAJA


#endif
