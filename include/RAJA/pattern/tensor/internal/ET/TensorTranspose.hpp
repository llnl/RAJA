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

#ifndef RAJA_pattern_tensor_ET_TensorTranspose_HPP
#define RAJA_pattern_tensor_ET_TensorTranspose_HPP

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

template<typename ET_TYPE>
class TensorTranspose : public TensorExpressionBase<TensorTranspose<ET_TYPE>>
{
public:
  using self_type    = TensorTranspose<ET_TYPE>;
  using rhs_type     = ET_TYPE;
  using tensor_type  = typename ET_TYPE::result_type;
  using element_type = typename tensor_type::element_type;
  using index_type   = typename ET_TYPE::index_type;

  using result_type                       = tensor_type;
  using tile_type                         = typename ET_TYPE::tile_type;
  static constexpr camp::idx_t s_num_dims = ET_TYPE::s_num_dims;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  TensorTranspose(rhs_type const& tensor) : m_tensor {tensor} {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr index_type getDimSize(index_type dim) const
  {
    return m_tensor.getDimSize(dim);
  }

  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE result_type eval(TILE_TYPE const& tile) const
  {
    // transpose which tile we are returning
    TILE_TYPE trans_tile {{tile.m_begin[1], tile.m_begin[0]},
                          {tile.m_size[1], tile.m_size[0]}};

    // evaluate and return the transposed tile
    return m_tensor.eval(trans_tile).transpose();
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  void print_ast() const
  {
    printf("Transpose(");
    m_tensor.print_ast();
    printf(")");
  }

private:
  rhs_type m_tensor;
};


}  // namespace ET

}  // namespace expt
}  // namespace internal

}  // namespace RAJA


#endif
