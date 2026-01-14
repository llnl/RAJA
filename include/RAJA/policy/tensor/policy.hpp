/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA simd policy definitions.
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

#ifndef RAJA_policy_tensor_policy_HPP
#define RAJA_policy_tensor_policy_HPP

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/config.hpp"

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///
namespace RAJA
{
namespace policy
{
namespace tensor
{

template<typename EXEC_POLICY,
         typename TENSOR_TYPE,
         camp::idx_t DIM,
         camp::idx_t TILE_SIZE>
struct tensor_exec : public EXEC_POLICY
{
  using exec_policy = EXEC_POLICY;
  using tensor_type = TENSOR_TYPE;

  static constexpr camp::idx_t s_tensor_dim = DIM;
  static constexpr camp::idx_t s_tile_size  = TILE_SIZE;
};


}  // end of namespace tensor

}  // end of namespace policy

namespace expt
{


template<typename TENSOR_TYPE, camp::idx_t TILE_SIZE = -1>
using vector_exec =
    policy::tensor::tensor_exec<RAJA::seq_exec, TENSOR_TYPE, 0, TILE_SIZE>;

template<typename TENSOR_TYPE, camp::idx_t TILE_SIZE = -1>
using matrix_row_exec =
    policy::tensor::tensor_exec<seq_exec, TENSOR_TYPE, 0, TILE_SIZE>;

template<typename TENSOR_TYPE, camp::idx_t TILE_SIZE = -1>
using matrix_col_exec =
    policy::tensor::tensor_exec<seq_exec, TENSOR_TYPE, 1, TILE_SIZE>;


}  //  namespace expt


}  // end of namespace RAJA

#endif
