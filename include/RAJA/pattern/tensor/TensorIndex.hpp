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

#ifndef RAJA_pattern_tensor_TensorIndex_HPP
#define RAJA_pattern_tensor_TensorIndex_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/index/IndexValue.hpp"

namespace RAJA
{
namespace expt
{


template<typename IDX,
         typename TENSOR_TYPE,
         camp::idx_t DIM,
         strip_index_type_t<IDX> INDEX_VALUE,
         strip_index_type_t<IDX> LENGTH_VALUE>
struct StaticTensorIndexInner;

template<typename INNER_TYPE>
struct StaticTensorIndex;

template<typename IDX, typename TENSOR_TYPE, camp::idx_t DIM>
class TensorIndex
{
public:
  using self_type   = TensorIndex<IDX, TENSOR_TYPE, DIM>;
  using value_type  = strip_index_type_t<IDX>;
  using index_type  = IDX;
  using tensor_type = TENSOR_TYPE;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr self_type all()
  {
    return self_type(index_type(-1), value_type(-1));
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr StaticTensorIndex<StaticTensorIndexInner<IDX,
                                                            TENSOR_TYPE,
                                                            DIM,
                                                            value_type(-1),
                                                            value_type(-1)>>
  static_all()
  {
    return StaticTensorIndex<StaticTensorIndexInner<
        IDX, TENSOR_TYPE, DIM, value_type(-1), value_type(-1)>>();
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr self_type range(index_type begin, index_type end)
  {
    return self_type(begin, value_type(stripIndexType(end - begin)));
  }

  template<value_type TBEGIN, value_type TEND>
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr StaticTensorIndex<
      StaticTensorIndexInner<IDX, TENSOR_TYPE, DIM, TBEGIN, TEND - TBEGIN>>
  static_range()
  {
    return StaticTensorIndex<
        StaticTensorIndexInner<IDX, TENSOR_TYPE, DIM, TBEGIN, TEND - TBEGIN>>();
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr TensorIndex() : m_index(index_type(0)), m_length(0) {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr TensorIndex(RAJA::TypedRangeSegment<IDX> const& seg)
      : m_index(*seg.begin()),
        m_length(seg.size())
  {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr TensorIndex(index_type value, value_type length)
      : m_index(value),
        m_length(length)
  {}

  template<typename T, camp::idx_t D>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr TensorIndex(
      TensorIndex<IDX, T, D> const& c)
      : m_index(*c),
        m_length(c.size())
  {}

  template<value_type IDX_VAL, value_type LEN_VAL>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr TensorIndex(
      StaticTensorIndex<
          StaticTensorIndexInner<IDX, TENSOR_TYPE, DIM, IDX_VAL, LEN_VAL>> const
          RAJA_UNUSED_ARG(&c))
      : m_index(IDX_VAL),
        m_length(LEN_VAL)
  {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr index_type const& operator*() const { return m_index; }

  // used in strip_by_value as a static cast
  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr explicit operator index_type() const
  {
    // return does not matter, but suppresses no-return warnings
    return m_index;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr index_type begin() const { return m_index; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr value_type size() const { return m_length; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr value_type dim() const { return DIM; }

private:
  index_type m_index;
  value_type m_length;
};

template<typename IDX,
         typename TENSOR_TYPE,
         camp::idx_t DIM,
         strip_index_type_t<IDX> INDEX_VALUE,
         strip_index_type_t<IDX> LENGTH_VALUE>
struct StaticTensorIndex<
    StaticTensorIndexInner<IDX, TENSOR_TYPE, DIM, INDEX_VALUE, LENGTH_VALUE>>
{

  using base_type   = TensorIndex<IDX, TENSOR_TYPE, DIM>;
  using value_type  = strip_index_type_t<IDX>;
  using index_type  = IDX;
  using tensor_type = TENSOR_TYPE;

  static const index_type s_index  = INDEX_VALUE;
  static const index_type s_length = LENGTH_VALUE;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr operator base_type() { return base_type(s_index, s_length); }
};

/*!
 * Index that specifies the starting element index of a Vector
 */
template<typename IDX, typename VECTOR_TYPE>
using VectorIndex = TensorIndex<IDX, VECTOR_TYPE, 0>;

/*!
 * Index that specifies the starting Row index of a matrix
 */
template<typename IDX, typename MATRIX_TYPE>
using RowIndex = TensorIndex<IDX, MATRIX_TYPE, 0>;

/*!
 * Index that specifies the starting Column index of a matrix
 */
template<typename IDX, typename MATRIX_TYPE>
using ColIndex = TensorIndex<IDX, MATRIX_TYPE, 1>;

/*!
 * Converts a Row index to a Column index
 */
template<typename IDX, typename MATRIX_TYPE>
RAJA_HOST_DEVICE RAJA_INLINE constexpr ColIndex<IDX, MATRIX_TYPE> toColIndex(
    RowIndex<IDX, MATRIX_TYPE> const& r)
{
  return ColIndex<IDX, MATRIX_TYPE>(*r, r.size());
}

/*!
 * Converts a Column index to a Row index
 */
template<typename IDX, typename MATRIX_TYPE>
RAJA_HOST_DEVICE RAJA_INLINE constexpr RowIndex<IDX, MATRIX_TYPE> toRowIndex(
    ColIndex<IDX, MATRIX_TYPE> const& c)
{
  return RowIndex<IDX, MATRIX_TYPE>(*c, c.size());
}

}  // namespace expt
}  // namespace RAJA

#include "RAJA/pattern/tensor/internal/TensorIndexTraits.hpp"

#endif
