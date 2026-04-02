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

#ifndef RAJA_pattern_tensor_ET_ScalarLiteral_HPP
#define RAJA_pattern_tensor_ET_ScalarLiteral_HPP

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


template<typename T>
class TensorScalarLiteral : public TensorExpressionBase<TensorScalarLiteral<T>>
{
public:
  using self_type    = TensorScalarLiteral<T>;
  using tensor_type  = RAJA::expt::ScalarRegister<T>;
  using element_type = T;
  using result_type  = T;
  using index_type   = RAJA::Index_type;

  static constexpr camp::idx_t s_num_dims = 0;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr index_type getDimSize(index_type) const { return 0; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  explicit constexpr TensorScalarLiteral(element_type const& value) noexcept
      : m_value {value}
  {}

  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE element_type eval(TILE_TYPE const&) const
  {
    return m_value;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  void print_ast() const { printf("ScalarLiteral(%e)", (double)m_value); }

private:
  element_type m_value;
};

/*
 * For arithmetic values, we need to wrap in a constant value ET node
 */
template<typename RHS>
struct NormalizeOperandHelper<
    RHS,
    typename std::enable_if<std::is_arithmetic<RHS>::value>::type>
{
  using return_type = TensorScalarLiteral<RHS>;

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
