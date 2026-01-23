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

#ifndef RAJA_policy_tensor_vectorregister_HPP
#define RAJA_policy_tensor_vectorregister_HPP

#include "RAJA/config.hpp"

namespace RAJA
{
namespace expt
{
// Convenience to describe VectorTensors
template<typename T,
         typename REGISTER_POLICY = default_register,
         camp::idx_t NUM_ELEM     = Register<T, REGISTER_POLICY>::s_num_elem>
using VectorRegister =
    TensorRegister<REGISTER_POLICY, T, VectorLayout, camp::idx_seq<NUM_ELEM>>;
}  // namespace expt

}  // namespace RAJA


#endif
