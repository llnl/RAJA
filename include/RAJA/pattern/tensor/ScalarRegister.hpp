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

#ifndef RAJA_policy_tensor_scalarregister_HPP
#define RAJA_policy_tensor_scalarregister_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/tensor/VectorRegister.hpp"
#include "RAJA/policy/tensor/arch.hpp"

namespace RAJA
{
namespace expt
{

// Convenience to describe ScalarTensors
template<typename T>
using ScalarRegister =
    TensorRegister<scalar_register, T, ScalarLayout, camp::idx_seq<>>;


}  // namespace expt
}  // namespace RAJA


#endif
