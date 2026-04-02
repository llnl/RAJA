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

#ifndef RAJA_pattern_tensor_MatrixRegister_HPP
#define RAJA_pattern_tensor_MatrixRegister_HPP

#include "camp/camp.hpp"
#include "RAJA/config.hpp"
#include "RAJA/policy/tensor/arch.hpp"
#include "RAJA/pattern/tensor/TensorRegister.hpp"

namespace RAJA
{
namespace expt
{
template<typename T,
         typename LAYOUT,
         typename REGISTER_POLICY = default_register>
using SquareMatrixRegister = TensorRegister<
    REGISTER_POLICY,
    T,
    LAYOUT,
    camp::idx_seq<
        RAJA::internal::expt::RegisterTraits<REGISTER_POLICY, T>::s_num_elem,
        RAJA::internal::expt::RegisterTraits<REGISTER_POLICY, T>::s_num_elem>>;

template<typename T,
         typename LAYOUT,
         camp::idx_t ROWS,
         camp::idx_t COLS,
         typename REGISTER_POLICY = default_register>
using RectMatrixRegister =
    TensorRegister<REGISTER_POLICY, T, LAYOUT, camp::idx_seq<ROWS, COLS>>;

}  // namespace expt
}  // namespace RAJA


#endif
