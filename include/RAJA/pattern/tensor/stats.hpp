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


// Place the following line before including RAJA to enable
// statistics on the Vector abstractions
// #define RAJA_ENABLE_VECTOR_STATS


#ifndef RAJA_pattern_simd_register_stats_HPP
#define RAJA_pattern_simd_register_stats_HPP

#include "RAJA/config.hpp"
#include "camp/camp.hpp"

namespace RAJA
{
namespace expt
{
struct tensor_stats
{
  static int indent;

  static camp::idx_t num_vector_copy;
  static camp::idx_t num_vector_copy_ctor;
  static camp::idx_t num_vector_broadcast_ctor;

  static camp::idx_t num_vector_load_packed;
  static camp::idx_t num_vector_load_packed_n;
  static camp::idx_t num_vector_load_strided;
  static camp::idx_t num_vector_load_strided_n;

  static camp::idx_t num_vector_store_packed;
  static camp::idx_t num_vector_store_packed_n;
  static camp::idx_t num_vector_store_strided;
  static camp::idx_t num_vector_store_strided_n;

  static camp::idx_t num_vector_broadcast;

  static camp::idx_t num_vector_get;
  static camp::idx_t num_vector_set;

  static camp::idx_t num_vector_add;
  static camp::idx_t num_vector_subtract;
  static camp::idx_t num_vector_multiply;
  static camp::idx_t num_vector_divide;

  static camp::idx_t num_vector_fma;
  static camp::idx_t num_vector_fms;

  static camp::idx_t num_vector_sum;
  static camp::idx_t num_vector_max;
  static camp::idx_t num_vector_min;
  static camp::idx_t num_vector_vmax;
  static camp::idx_t num_vector_vmin;
  static camp::idx_t num_vector_dot;


  static camp::idx_t num_matrix_mm_mult_row_row;
  static camp::idx_t num_matrix_mm_multacc_row_row;
  static camp::idx_t num_matrix_mm_mult_col_col;
  static camp::idx_t num_matrix_mm_multacc_col_col;

  static void resetVectorStats();
  static void printVectorStats();
};

}  // namespace expt
}  // namespace RAJA

#endif
