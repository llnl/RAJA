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

#ifndef RAJA_pattern_tensor_TensorLayout_HPP
#define RAJA_pattern_tensor_TensorLayout_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "camp/camp.hpp"

namespace RAJA
{
namespace expt
{


template<camp::idx_t... DIM_SEQ>
struct TensorLayout : public camp::idx_seq<DIM_SEQ...>
{

  using seq_t = camp::idx_seq<DIM_SEQ...>;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr bool is_column_major() { return false; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr bool is_row_major() { return false; }
};

// specialization for Matrix layouts, where column vs row major matters
template<camp::idx_t S2, camp::idx_t S1>
struct TensorLayout<S2, S1> : public camp::idx_seq<S2, S1>
{
  using seq_t = camp::idx_seq<S2, S1>;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr bool is_column_major()
  {
    return S1 == 0;  // Rows are stride-1
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr bool is_row_major()
  {
    return S1 == 1;  // Columns are stride-1
  }
};

// 0d tensor (scalar) layout
using ScalarLayout = TensorLayout<>;

// 1d tensor (vector) layout
using VectorLayout = TensorLayout<0>;

// 2d tensor (matrix) layouts
using RowMajorLayout = TensorLayout<0, 1>;
using ColMajorLayout = TensorLayout<1, 0>;


}  // namespace expt
}  // namespace RAJA


#endif
