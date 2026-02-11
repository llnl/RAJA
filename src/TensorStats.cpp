//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/pattern/tensor/stats.hpp"
#include <stdio.h>

int RAJA::tensor_stats::indent = 0;

camp::idx_t RAJA::tensor_stats::num_vector_copy           = 0;
camp::idx_t RAJA::tensor_stats::num_vector_copy_ctor      = 0;
camp::idx_t RAJA::tensor_stats::num_vector_broadcast_ctor = 0;

camp::idx_t RAJA::tensor_stats::num_vector_load_packed    = 0;
camp::idx_t RAJA::tensor_stats::num_vector_load_packed_n  = 0;
camp::idx_t RAJA::tensor_stats::num_vector_load_strided   = 0;
camp::idx_t RAJA::tensor_stats::num_vector_load_strided_n = 0;

camp::idx_t RAJA::tensor_stats::num_vector_store_packed    = 0;
camp::idx_t RAJA::tensor_stats::num_vector_store_packed_n  = 0;
camp::idx_t RAJA::tensor_stats::num_vector_store_strided   = 0;
camp::idx_t RAJA::tensor_stats::num_vector_store_strided_n = 0;

camp::idx_t RAJA::tensor_stats::num_vector_broadcast = 0;

camp::idx_t RAJA::tensor_stats::num_vector_get = 0;
camp::idx_t RAJA::tensor_stats::num_vector_set = 0;

camp::idx_t RAJA::tensor_stats::num_vector_add      = 0;
camp::idx_t RAJA::tensor_stats::num_vector_subtract = 0;
camp::idx_t RAJA::tensor_stats::num_vector_multiply = 0;
camp::idx_t RAJA::tensor_stats::num_vector_divide   = 0;

camp::idx_t RAJA::tensor_stats::num_vector_fma = 0;
camp::idx_t RAJA::tensor_stats::num_vector_fms = 0;

camp::idx_t RAJA::tensor_stats::num_vector_sum  = 0;
camp::idx_t RAJA::tensor_stats::num_vector_max  = 0;
camp::idx_t RAJA::tensor_stats::num_vector_min  = 0;
camp::idx_t RAJA::tensor_stats::num_vector_vmax = 0;
camp::idx_t RAJA::tensor_stats::num_vector_vmin = 0;
camp::idx_t RAJA::tensor_stats::num_vector_dot  = 0;

camp::idx_t RAJA::tensor_stats::num_matrix_mm_mult_row_row    = 0;
camp::idx_t RAJA::tensor_stats::num_matrix_mm_multacc_row_row = 0;
camp::idx_t RAJA::tensor_stats::num_matrix_mm_mult_col_col    = 0;
camp::idx_t RAJA::tensor_stats::num_matrix_mm_multacc_col_col = 0;

void RAJA::tensor_stats::resetVectorStats()
{
  num_vector_copy           = 0;
  num_vector_copy_ctor      = 0;
  num_vector_broadcast_ctor = 0;

  num_vector_load_packed     = 0;
  num_vector_load_packed_n   = 0;
  num_vector_load_strided    = 0;
  num_vector_load_strided_n  = 0;
  num_vector_store_packed    = 0;
  num_vector_store_packed_n  = 0;
  num_vector_store_strided   = 0;
  num_vector_store_strided_n = 0;

  num_vector_broadcast = 0;

  num_vector_get = 0;
  num_vector_set = 0;

  num_vector_add      = 0;
  num_vector_subtract = 0;
  num_vector_multiply = 0;
  num_vector_divide   = 0;

  num_vector_fma  = 0;
  num_vector_fms  = 0;
  num_vector_sum  = 0;
  num_vector_max  = 0;
  num_vector_min  = 0;
  num_vector_vmax = 0;
  num_vector_vmin = 0;
  num_vector_dot  = 0;

  num_matrix_mm_mult_row_row    = 0;
  num_matrix_mm_multacc_row_row = 0;
  num_matrix_mm_mult_col_col    = 0;
  num_matrix_mm_multacc_col_col = 0;
}

#define PRINT_STAT(STAT)                                                       \
  if (STAT)                                                                    \
  {                                                                            \
    printf("  %-32s   %ld\n", #STAT, STAT);                                    \
  }

void RAJA::tensor_stats::printVectorStats()
{

  printf("RAJA SIMD Register Statistics:\n");

  PRINT_STAT(num_vector_copy);
  PRINT_STAT(num_vector_copy_ctor);
  PRINT_STAT(num_vector_broadcast_ctor);

  PRINT_STAT(num_vector_load_packed);
  PRINT_STAT(num_vector_load_packed_n);
  PRINT_STAT(num_vector_load_strided);
  PRINT_STAT(num_vector_load_strided_n);
  PRINT_STAT(num_vector_store_packed);
  PRINT_STAT(num_vector_store_packed_n);
  PRINT_STAT(num_vector_store_strided);
  PRINT_STAT(num_vector_store_strided_n);

  PRINT_STAT(num_vector_broadcast);
  PRINT_STAT(num_vector_get);
  PRINT_STAT(num_vector_set);

  PRINT_STAT(num_vector_add);
  PRINT_STAT(num_vector_subtract);
  PRINT_STAT(num_vector_multiply);
  PRINT_STAT(num_vector_divide);

  PRINT_STAT(num_vector_fma);
  PRINT_STAT(num_vector_fms);
  PRINT_STAT(num_vector_sum);
  PRINT_STAT(num_vector_max);
  PRINT_STAT(num_vector_min);
  PRINT_STAT(num_vector_vmax);
  PRINT_STAT(num_vector_vmin);
  PRINT_STAT(num_vector_dot);

  PRINT_STAT(num_matrix_mm_mult_row_row);
  PRINT_STAT(num_matrix_mm_multacc_row_row);
  PRINT_STAT(num_matrix_mm_mult_col_col);
  PRINT_STAT(num_matrix_mm_multacc_col_col);
}
