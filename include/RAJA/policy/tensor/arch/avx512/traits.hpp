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

#ifndef RAJA_policy_tensor_arch_avx512_traits_HPP
#define RAJA_policy_tensor_arch_avx512_traits_HPP

namespace RAJA
{
namespace internal
{
namespace expt
{


template<>
struct RegisterTraits<RAJA::expt::avx512_register, int32_t>
{
  using element_type                      = int32_t;
  using register_policy                   = RAJA::expt::avx512_register;
  static constexpr camp::idx_t s_num_bits = 512;
  static constexpr camp::idx_t s_num_elem = 16;
  using int_element_type                  = int32_t;
};

template<>
struct RegisterTraits<RAJA::expt::avx512_register, int64_t>
{
  using element_type                      = int64_t;
  using register_policy                   = RAJA::expt::avx512_register;
  static constexpr camp::idx_t s_num_bits = 512;
  static constexpr camp::idx_t s_num_elem = 8;
  using int_element_type                  = int64_t;
};

template<>
struct RegisterTraits<RAJA::expt::avx512_register, float>
{
  using element_type                      = float;
  using register_policy                   = RAJA::expt::avx512_register;
  static constexpr camp::idx_t s_num_bits = 512;
  static constexpr camp::idx_t s_num_elem = 16;
  using int_element_type                  = int32_t;
};

template<>
struct RegisterTraits<RAJA::expt::avx512_register, double>
{
  using element_type                      = double;
  using register_policy                   = RAJA::expt::avx512_register;
  static constexpr camp::idx_t s_num_bits = 512;
  static constexpr camp::idx_t s_num_elem = 8;
  using int_element_type                  = int64_t;
};

}  // namespace expt
}  // namespace internal
}  // namespace RAJA

#endif  // guard


#endif  // __AVX512F__
