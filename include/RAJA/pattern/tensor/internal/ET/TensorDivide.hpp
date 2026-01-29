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

#ifndef RAJA_pattern_tensor_ET_TensorDivide_HPP
#define RAJA_pattern_tensor_ET_TensorDivide_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/tensor/internal/ET/ExpressionTemplateBase.hpp"

namespace RAJA
{
namespace internal
{
namespace expt
{


namespace ET
{

template<typename LEFT_OPERAND_TYPE,
         typename RIGHT_OPERAND_TYPE,
         class ENABLE = void>
struct DivideOperator;

/*!
 * Specialization that provides dividing a scalar by a vector
 */
template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
struct DivideOperator<
    LEFT_OPERAND_TYPE,
    RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 0 &&
                            RIGHT_OPERAND_TYPE::s_num_dims == 1>::type>
{

  using result_type = typename RIGHT_OPERAND_TYPE::result_type;
  static constexpr camp::idx_t s_num_dims = RIGHT_OPERAND_TYPE::s_num_dims;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim,
                        LEFT_OPERAND_TYPE const&,
                        RIGHT_OPERAND_TYPE const& right)
  {
    return right.getDimSize(dim);
  }

  /*!
   * Evaluate operands and perform element-wise divide
   */
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static result_type divide(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right)
  {
    result_type numerator(left.eval(tile));

    if (tile.s_tensor_size == TENSOR_FULL)
    {
      return numerator.divide(right.eval(tile));
    }

    return numerator.divide_n(right.eval(tile), tile.m_size[0]);
  }
};

/*!
 * Specialization that provides dividing a vector by a scalar
 */
template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
struct DivideOperator<
    LEFT_OPERAND_TYPE,
    RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 1 &&
                            RIGHT_OPERAND_TYPE::s_num_dims == 0>::type>
{
  using result_type = typename LEFT_OPERAND_TYPE::result_type;
  static constexpr camp::idx_t s_num_dims = LEFT_OPERAND_TYPE::s_num_dims;


  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim,
                        LEFT_OPERAND_TYPE const& left,
                        RIGHT_OPERAND_TYPE const&)
  {
    return left.getDimSize(dim);
  }

  /*!
   * Evaluate operands and perform element-wise divide
   */
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static result_type divide(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right)
  {
    result_type denominator(right.eval(tile));

    if (tile.s_tensor_size == TENSOR_FULL)
    {
      return left.eval(tile).divide(denominator);
    }
    else
    {
      return left.eval(tile).divide_n(denominator, tile.m_size[0]);
    }
  }
};

/*!
 * Specialization that provides dividing a vector by a vector
 */
template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
struct DivideOperator<
    LEFT_OPERAND_TYPE,
    RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 1 &&
                            RIGHT_OPERAND_TYPE::s_num_dims == 1>::type>
{
  using result_type = typename LEFT_OPERAND_TYPE::result_type;
  static constexpr camp::idx_t s_num_dims = LEFT_OPERAND_TYPE::s_num_dims;


  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim,
                        LEFT_OPERAND_TYPE const& left,
                        RIGHT_OPERAND_TYPE const&)
  {
    return left.getDimSize(dim);
  }

  /*!
   * Evaluate operands and perform element-wise divide
   */
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static result_type divide(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right)
  {
    if (tile.s_tensor_size == TENSOR_FULL)
    {
      return left.eval(tile).divide(right.eval(tile));
    }
    else
    {
      return left.eval(tile).divide_n(right.eval(tile), tile.m_size[0]);
    }
  }
};

/*!
 * Specialization that provides dividing a scalar by a matrix
 */
template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
struct DivideOperator<
    LEFT_OPERAND_TYPE,
    RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 0 &&
                            RIGHT_OPERAND_TYPE::s_num_dims == 2>::type>
{

  using result_type = typename RIGHT_OPERAND_TYPE::result_type;
  static constexpr camp::idx_t s_num_dims = RIGHT_OPERAND_TYPE::s_num_dims;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim,
                        LEFT_OPERAND_TYPE const&,
                        RIGHT_OPERAND_TYPE const& right)
  {
    return right.getDimSize(dim);
  }

  /*!
   * Evaluate operands and perform element-wise divide
   */
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static result_type divide(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right)
  {
    result_type numerator(left.eval(tile));

    if (tile.s_tensor_size == TENSOR_FULL)
    {
      return numerator.divide(right.eval(tile));
    }

    return numerator.divide_nm(right.eval(tile), tile.m_size[0],
                               tile.m_size[1]);
  }
};

/*!
 * Specialization that provides dividing a vector by a scalar
 */
template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
struct DivideOperator<
    LEFT_OPERAND_TYPE,
    RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 2 &&
                            RIGHT_OPERAND_TYPE::s_num_dims == 0>::type>
{
  using result_type = typename LEFT_OPERAND_TYPE::result_type;
  static constexpr camp::idx_t s_num_dims = LEFT_OPERAND_TYPE::s_num_dims;


  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim,
                        LEFT_OPERAND_TYPE const& left,
                        RIGHT_OPERAND_TYPE const&)
  {
    return left.getDimSize(dim);
  }

  /*!
   * Evaluate operands and perform element-wise divide
   */
  RAJA_SUPPRESS_HD_WARN
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static result_type divide(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right)
  {
    result_type denominator(right.eval(tile));

    if (tile.s_tensor_size == TENSOR_FULL)
    {
      return left.eval(tile).divide(denominator);
    }
    else
    {
      return left.eval(tile).divide_nm(denominator, tile.m_size[0],
                                       tile.m_size[1]);
    }
  }
};

/*!
 * Specialization that provides dividing a vector by a vector
 */
template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
struct DivideOperator<
    LEFT_OPERAND_TYPE,
    RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 2 &&
                            RIGHT_OPERAND_TYPE::s_num_dims == 2>::type>
{
  using result_type = typename LEFT_OPERAND_TYPE::result_type;
  static constexpr camp::idx_t s_num_dims = LEFT_OPERAND_TYPE::s_num_dims;


  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim,
                        LEFT_OPERAND_TYPE const& left,
                        RIGHT_OPERAND_TYPE const&)
  {
    return left.getDimSize(dim);
  }

  /*!
   * Evaluate operands and perform element-wise divide
   */
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static result_type divide(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right)
  {
    if (tile.s_tensor_size == TENSOR_FULL)
    {
      return left.eval(tile).divide(right.eval(tile));
    }
    else
    {
      return left.eval(tile).divide_nm(right.eval(tile), tile.m_size[0],
                                       tile.m_size[1]);
    }
  }
};

template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
class TensorDivide : public TensorExpressionBase<
                         TensorDivide<LEFT_OPERAND_TYPE, RIGHT_OPERAND_TYPE>>
{
public:
  using self_type         = TensorDivide<LEFT_OPERAND_TYPE, RIGHT_OPERAND_TYPE>;
  using left_operand_type = LEFT_OPERAND_TYPE;
  using right_operand_type = RIGHT_OPERAND_TYPE;
  using element_type       = typename LEFT_OPERAND_TYPE::element_type;
  using index_type         = typename LEFT_OPERAND_TYPE::index_type;

  using divide_op   = DivideOperator<left_operand_type, right_operand_type>;
  using result_type = typename divide_op::result_type;
  static constexpr camp::idx_t s_num_dims = divide_op::s_num_dims;


private:
  left_operand_type m_left_operand;
  right_operand_type m_right_operand;

public:
  RAJA_INLINE

  RAJA_HOST_DEVICE
  TensorDivide(left_operand_type const& left_operand,
               right_operand_type const& right_operand)
      : m_left_operand {left_operand},
        m_right_operand {right_operand}
  {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr index_type getDimSize(index_type dim) const
  {
    return divide_op::getDimSize(dim, m_left_operand, m_right_operand);
  }

  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE result_type eval(TILE_TYPE const& tile) const
  {
    return divide_op::divide(tile, m_left_operand, m_right_operand);
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

  RAJA_INLINE

  RAJA_HOST_DEVICE
  void print_ast() const
  {
    printf("Divide(");
    m_left_operand.print_ast();
    printf(", ");
    m_right_operand.print_ast();
    printf(")");
  }
};

/*
 * Overload for:    arithmetic / tensorexpression

 */
template<
    typename LHS,
    typename RHS,
    typename std::enable_if<std::is_arithmetic<LHS>::value, bool>::type = true,
    typename std::enable_if<
        std::is_base_of<TensorExpressionConcreteBase, RHS>::value,
        bool>::type = true>
RAJA_INLINE RAJA_HOST_DEVICE auto operator/(LHS const& left_operand,
                                            RHS const& right_operand)
    -> TensorDivide<typename NormalizeOperandHelper<LHS>::return_type, RHS>
{
  return TensorDivide<typename NormalizeOperandHelper<LHS>::return_type, RHS>(
      NormalizeOperandHelper<LHS>::normalize(left_operand), right_operand);
}

}  // namespace ET

}  // namespace expt
}  // namespace internal

}  // namespace RAJA


#endif
