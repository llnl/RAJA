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

#ifndef RAJA_pattern_tensor_ET_TensorLiteral_HPP
#define RAJA_pattern_tensor_ET_TensorLiteral_HPP

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


template<typename TENSOR_TYPE>
class TensorLiteral : public TensorExpressionBase<TensorLiteral<TENSOR_TYPE>>
{
public:
  using self_type    = TensorLiteral<TENSOR_TYPE>;
  using tensor_type  = TENSOR_TYPE;
  using element_type = typename TENSOR_TYPE::element_type;
  using result_type  = tensor_type;
  using index_type   = RAJA::Index_type;

  static constexpr camp::idx_t s_num_dims = result_type::s_num_dims;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr index_type getDimSize(index_type dim) const
  {
    return tensor_type::s_dim_elem(dim);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  explicit TensorLiteral(tensor_type const& value) : m_value {value} {}

  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE result_type eval(TILE_TYPE const&) const
  {
    return result_type(m_value);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  void print_ast() const { printf("TensorLiteral()"); }

private:
  tensor_type m_value;
};

/*
 * For TensorRegister nodes, we need to wrap this in a constant value ET node
 */
template<typename RHS>
struct NormalizeOperandHelper<
    RHS,
    typename std::enable_if<
        std::is_base_of<TensorRegisterConcreteBase, RHS>::value>::type>
{
  using return_type = TensorLiteral<RHS>;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr return_type normalize(RHS const& rhs)
  {
    return return_type(rhs);
  }
};

}  // namespace ET

}  // namespace expt
}  // namespace internal

}  // namespace RAJA


#endif
