/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing SIMD abstractions for AVX
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

#ifdef __AVX__

#include <RAJA/policy/tensor/arch/avx/traits.hpp>
#include <RAJA/policy/tensor/arch/avx/avx_int64.hpp>
#include <RAJA/policy/tensor/arch/avx/avx_int32.hpp>
#include <RAJA/policy/tensor/arch/avx/avx_float.hpp>
#include <RAJA/policy/tensor/arch/avx/avx_double.hpp>


#endif  // __AVX__
