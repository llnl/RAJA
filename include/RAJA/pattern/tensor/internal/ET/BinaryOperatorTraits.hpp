/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header defining expression template behavior for operator*
 *
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

#ifndef RAJA_pattern_tensor_ET_BinaryOperator_HPP
#define RAJA_pattern_tensor_ET_BinaryOperator_HPP

namespace RAJA
{
namespace internal
{
namespace expt
{


namespace ET
{

struct TensorOperatorAdd
{

  template<typename LEFT, typename RIGHT>
  RAJA_INLINE RAJA_HOST_DEVICE static auto eval(LEFT const& left,
                                                RIGHT const& right)
      -> decltype(left + right)
  {
    return left + right;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static void print_ast() { printf("Add"); }
};

struct TensorOperatorSubtract
{

  template<typename LEFT, typename RIGHT>
  RAJA_INLINE RAJA_HOST_DEVICE static auto eval(LEFT const& left,
                                                RIGHT const& right)
      -> decltype(left - right)
  {
    return left - right;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static void print_ast() { printf("Subtract"); }
};


template<typename OPERATOR, typename LEFT_OPERAND, typename RIGHT_OPERAND>
class TensorBinaryOperator;

template<typename LHS, typename RHS>
using TensorAdd = TensorBinaryOperator<TensorOperatorAdd, LHS, RHS>;

template<typename LHS, typename RHS>
using TensorSubtract = TensorBinaryOperator<TensorOperatorSubtract, LHS, RHS>;

/*!
 * Provides default operations for add, subtract and divide
 *
 * For the most part, this is just element wise operations between
 * compatible tensors.
 *
 * There are specializations that handle when one operand is a scalar
 */
template<typename LHS_TYPE, typename RHS_TYPE, class ENABLE = void>
struct OperatorTraits
{

  using result_type                       = typename LHS_TYPE::result_type;
  static constexpr camp::idx_t s_num_dims = LHS_TYPE::s_num_dims;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static void print_ast() { printf("Elemental"); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim, LHS_TYPE const& lhs, RHS_TYPE const& rhs)
  {
    return dim == 0 ? lhs.getDimSize(0) : rhs.getDimSize(1);
  }
};

/*!
 * Specialization when the left operand is a scalar
 */
template<typename LHS_TYPE, typename RHS_TYPE>
struct OperatorTraits<LHS_TYPE,
                      RHS_TYPE,
                      typename std::enable_if<LHS_TYPE::s_num_dims == 0>::type>
{

  using result_type                       = typename RHS_TYPE::result_type;
  static constexpr camp::idx_t s_num_dims = RHS_TYPE::s_num_dims;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static void print_ast() { printf("Scalar"); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim, LHS_TYPE const&, RHS_TYPE const& rhs)
  {
    return rhs.getDimSize(dim);
  }
};

/*!
 * Specialization when the right operand is a scalar
 */
template<typename LHS_TYPE, typename RHS_TYPE>
struct OperatorTraits<LHS_TYPE,
                      RHS_TYPE,
                      typename std::enable_if<RHS_TYPE::s_num_dims == 0>::type>
{

  using result_type                       = typename LHS_TYPE::result_type;
  static constexpr camp::idx_t s_num_dims = LHS_TYPE::s_num_dims;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static void print_ast() { printf("Scalar"); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim, LHS_TYPE const& lhs, RHS_TYPE const&)
  {
    return lhs.getDimSize(dim);
  }
};


}  // namespace ET

}  // namespace expt
}  // namespace internal

}  // namespace RAJA


#endif
