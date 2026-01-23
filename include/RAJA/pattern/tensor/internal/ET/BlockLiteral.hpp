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

#ifndef RAJA_pattern_tensor_ET_BlockLiteral_HPP
#define RAJA_pattern_tensor_ET_BlockLiteral_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/tensor/internal/ET/ExpressionTemplateBase.hpp"
#include "RAJA/pattern/tensor/internal/TensorRef.hpp"

namespace RAJA
{
namespace internal
{
namespace expt
{


namespace ET
{


/*!
 * Temporary n-dimensional memory.
 *
 * STORAGE_TYPE defines the memory storage
 * TENSOR_TYPE defines what kind of tensor is returned by eval()
 */
template<typename STORAGE_TYPE, typename TENSOR_TYPE>
class BlockLiteral
    : public TensorExpressionBase<BlockLiteral<STORAGE_TYPE, TENSOR_TYPE>>
{
public:
  using self_type    = BlockLiteral<STORAGE_TYPE, TENSOR_TYPE>;
  using storage_type = STORAGE_TYPE;
  using tensor_type  = TENSOR_TYPE;
  using result_type  = TENSOR_TYPE;
  using ref_type     = typename STORAGE_TYPE::ref_type;
  using tile_type    = typename ref_type::tile_type;
  using index_type   = camp::idx_t;

  static constexpr camp::idx_t s_num_dims = result_type::s_num_dims;


private:
  storage_type m_storage;
  tile_type m_tile_origin;

public:
  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr index_type getDimSize(index_type dim) const
  {
    return storage_type::s_dim_elem(dim);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr BlockLiteral(tile_type tile_origin)
      : m_storage(),
        m_tile_origin(tile_origin)
  {}

  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE result_type eval(TILE_TYPE const& tile) const
  {
    result_type result;

    // load result from storage
    result.load_ref(merge_ref_tile(m_storage.get_ref(), tile - m_tile_origin));

    return result;
  }

  /*!
   *  Returns a ref that points at this data, shifted by its origin
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  ref_type get_ref()
  {

    // compute shifited origin ref
    return shift_tile_origin(m_storage.get_ref(), m_tile_origin);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  void print_ast() const { printf("BlockLiteral()"); }
};

//    /*
//     * For TensorRegister nodes, we need to wrap this in a constant value ET
//     node
//     */
//    template<typename RHS>
//    struct NormalizeOperandHelper<RHS,
//    typename
//    std::enable_if<std::is_base_of<RAJA::internal::TensorRegisterConcreteBase,
//    RHS>::value>::type>
//    {
//        using return_type = BlockLiteral<RHS>;
//
//        RAJA_INLINE
//        RAJA_HOST_DEVICE
//        static
//        constexpr
//        return_type normalize(RHS const &rhs){
//          return return_type(rhs);
//        }
//    };

}  // namespace ET

}  // namespace expt
}  // namespace internal

}  // namespace RAJA


#endif
