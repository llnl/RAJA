/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing SIMD abstractions for AVX512
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

// Check if the base AVX512 instructions are present
#ifdef __AVX512F__

#include <RAJA/policy/tensor/arch/avx512/traits.hpp>
#include <RAJA/policy/tensor/arch/avx512/avx512_int32.hpp>
#include <RAJA/policy/tensor/arch/avx512/avx512_int64.hpp>
#include <RAJA/policy/tensor/arch/avx512/avx512_float.hpp>
#include <RAJA/policy/tensor/arch/avx512/avx512_double.hpp>


#endif  // __AVX512F__
