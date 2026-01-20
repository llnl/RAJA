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

#ifndef RAJA_pattern_tensor_ET_TensorLoadStore_HPP
#define RAJA_pattern_tensor_ET_TensorLoadStore_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/tensor/internal/ET/ExpressionTemplateBase.hpp"
#include "RAJA/pattern/tensor/internal/TensorTileExec.hpp"

namespace RAJA
{
namespace internal
{
namespace expt
{


namespace ET
{


template<typename STORAGE, typename LHS_TYPE, typename RHS_TYPE>
struct TensorStoreFunctor
{
  LHS_TYPE const& m_lhs;
  RHS_TYPE const& m_rhs;

  template<typename TILE_TYPE>
  RAJA_HOST_DEVICE RAJA_INLINE void operator()(TILE_TYPE const& tile) const
  {


    /*
     *
     * For recursive ET types, eval() produces a new ET, and
     * eval_lhs() produces a new TensorLoadStore.
     *
     */

    m_lhs.eval_lhs(tile) = m_rhs.eval(tile);
  }
};

template<typename STORAGE, typename LHS_TYPE, typename RHS_TYPE>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto makeTensorStoreFunctor(
    LHS_TYPE const& lhs,
    RHS_TYPE const& rhs) -> TensorStoreFunctor<STORAGE, LHS_TYPE, RHS_TYPE>
{
  return TensorStoreFunctor<STORAGE, LHS_TYPE, RHS_TYPE> {lhs, rhs};
}

template<typename TENSOR_TYPE, typename REF_TYPE>
class TensorLoadStore
    : public TensorExpressionBase<TensorLoadStore<TENSOR_TYPE, REF_TYPE>>
{
public:
  using self_type    = TensorLoadStore<TENSOR_TYPE, REF_TYPE>;
  using tensor_type  = TENSOR_TYPE;
  using element_type = typename TENSOR_TYPE::element_type;
  using index_type   = typename REF_TYPE::index_type;
  using ref_type     = REF_TYPE;
  using tile_type    = typename REF_TYPE::tile_type;
  using result_type  = TENSOR_TYPE;

  static constexpr camp::idx_t s_num_dims = result_type::s_num_dims;


private:
  ref_type m_ref;


public:
  RAJA_INLINE

  RAJA_HOST_DEVICE
  explicit TensorLoadStore(ref_type const& ref) : m_ref {ref} {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  TensorLoadStore(self_type const& rhs) : m_ref(rhs.m_ref) {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  void print() const
  {
    printf("TensorLoadStore: ");
    m_ref.m_tile.print();
  }

  //        RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& operator=(self_type const& rhs)
  {
    store(rhs);
    return *this;
  }

  //        RAJA_SUPPRESS_HD_WARN
  template<typename RHS>
  RAJA_HOST_DEVICE RAJA_INLINE self_type& operator=(RHS const& rhs)
  {

    store(normalizeOperand(rhs));

    return *this;
  }

  RAJA_SUPPRESS_HD_WARN
  template<typename RHS>
  RAJA_HOST_DEVICE RAJA_INLINE self_type& operator+=(RHS const& rhs)
  {
    store(normalizeOperand(rhs) + (*this));
    return *this;
  }

  RAJA_SUPPRESS_HD_WARN
  template<typename RHS>
  RAJA_HOST_DEVICE RAJA_INLINE self_type& operator-=(RHS const& rhs)
  {
    store(TensorSubtract<self_type, RHS>(*this, normalizeOperand(rhs)));
    return *this;
  }

  RAJA_SUPPRESS_HD_WARN
  template<typename RHS>
  RAJA_HOST_DEVICE RAJA_INLINE self_type operator*=(RHS const& rhs)
  {
    store(TensorMultiply<self_type, RHS>(*this, normalizeOperand(rhs)));
    return *this;
  }

  RAJA_SUPPRESS_HD_WARN
  template<typename RHS>
  RAJA_HOST_DEVICE RAJA_INLINE self_type operator/=(RHS const& rhs)
  {
    store(TensorDivide<self_type, RHS>(*this, normalizeOperand(rhs)));
    return *this;
  }

  RAJA_SUPPRESS_HD_WARN
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE auto eval(TILE_TYPE const& tile) const
      -> decltype(tensor_type::s_load_ref(merge_ref_tile(m_ref, tile)))
  {
    return tensor_type::s_load_ref(merge_ref_tile(m_ref, tile));
  }

  RAJA_SUPPRESS_HD_WARN
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE auto eval_lhs(TILE_TYPE const& tile) const
      -> decltype(TENSOR_TYPE::create_et_store_ref(merge_ref_tile(this->m_ref,
                                                                  tile)))
  {
    return TENSOR_TYPE::create_et_store_ref(merge_ref_tile(m_ref, tile));
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr index_type getDimSize(index_type dim) const
  {
    return m_ref.m_tile.m_size[dim];
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  void print_ast() const { printf("Load()"); }

private:
  RAJA_INLINE

  RAJA_HOST_DEVICE
  tile_type const& getTile() const { return m_ref.m_tile; }

  template<typename RHS>
  RAJA_INLINE RAJA_HOST_DEVICE void store(RHS const& rhs)
  {
#ifdef RAJA_DEBUG_PRINT_ET_AST
    printf("Store(");
    rhs.print_ast();
    printf(")\n");
#endif

    tensorTileExec<tensor_type>(
        m_ref.m_tile, makeTensorStoreFunctor<tensor_type>(*this, rhs));
  }
};


}  // namespace ET

}  // namespace expt
}  // namespace internal

}  // namespace RAJA


#endif
