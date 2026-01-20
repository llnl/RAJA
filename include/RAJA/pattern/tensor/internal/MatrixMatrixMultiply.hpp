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

#ifndef RAJA_pattern_tensor_internal_MatrixMatrixMultiply_HPP
#define RAJA_pattern_tensor_internal_MatrixMatrixMultiply_HPP

#include "camp/camp.hpp"
#include "RAJA/config.hpp"
#include "RAJA/pattern/tensor/MatrixRegister.hpp"

namespace RAJA
{
namespace internal
{
namespace expt
{


template<typename MATA, typename MATB>
struct MatrixMatrixMultiplyHelper;

/**
 *
 * Row-Major * Row-Major ==> Row-Major
 *
 */
template<typename T,
         typename REGISTER_POLICY,
         camp::idx_t N_SIZE,
         camp::idx_t M_SIZE,
         camp::idx_t M2_SIZE,
         camp::idx_t O_SIZE>
struct MatrixMatrixMultiplyHelper<
    RAJA::expt::TensorRegister<REGISTER_POLICY,
                               T,
                               RAJA::expt::RowMajorLayout,
                               camp::idx_seq<N_SIZE, M_SIZE>>,
    RAJA::expt::TensorRegister<REGISTER_POLICY,
                               T,
                               RAJA::expt::RowMajorLayout,
                               camp::idx_seq<M2_SIZE, O_SIZE>>>
{

  static_assert(M_SIZE == M2_SIZE,
                "Matrices are not compatible for multiplication");

  using left_type = RAJA::expt::TensorRegister<REGISTER_POLICY,
                                               T,
                                               RAJA::expt::RowMajorLayout,
                                               camp::idx_seq<N_SIZE, M_SIZE>>;

  using right_type = RAJA::expt::TensorRegister<REGISTER_POLICY,
                                                T,
                                                RAJA::expt::RowMajorLayout,
                                                camp::idx_seq<M_SIZE, O_SIZE>>;

  using result_type = RAJA::expt::TensorRegister<REGISTER_POLICY,
                                                 T,
                                                 RAJA::expt::RowMajorLayout,
                                                 camp::idx_seq<N_SIZE, O_SIZE>>;

  using register_type = typename result_type::register_type;

  static constexpr camp::idx_t s_elements_per_register =
      left_type::s_elements_per_register;
  static constexpr camp::idx_t s_A_minor_dim_registers =
      left_type::s_minor_dim_registers;
  static constexpr camp::idx_t s_B_minor_dim_registers =
      right_type::s_minor_dim_registers;
  static constexpr camp::idx_t s_C_minor_dim_registers =
      result_type::s_minor_dim_registers;

  /*
   * Matrix B (and C) has 1 more more registers per row
   *
   */
  template<typename dummy = void>
  RAJA_HOST_DEVICE static RAJA_INLINE
      typename std::enable_if<(s_C_minor_dim_registers != 0), dummy>::type
      multiply_accumulate(left_type const& A,
                          right_type const& B,
                          result_type& C)
  {
#if defined(RAJA_ENABLE_VECTOR_STATS) && !defined(__CUDA_ARCH__)
    RAJA::tensor_stats::num_matrix_mm_multacc_row_row++;
#endif

    constexpr camp::idx_t num_bc_reg_per_row = s_C_minor_dim_registers;

    RAJA_UNROLL
    for (camp::idx_t c_reg = 0; c_reg < result_type::s_num_registers; ++c_reg)
    {
      camp::idx_t bc_col_reg = c_reg % num_bc_reg_per_row;
      camp::idx_t ac_row     = c_reg / num_bc_reg_per_row;

      RAJA_UNROLL
      for (camp::idx_t a_col = 0; a_col < M_SIZE; ++a_col)
      {
        camp::idx_t b_reg = a_col * num_bc_reg_per_row + bc_col_reg;

        C.get_register(c_reg) =
            register_type(A.get(ac_row, a_col))
                .multiply_add(B.get_register(b_reg), C.get_register(c_reg));
      }
    }
  }

  /*
   * Matrix B (and C) have less than one register per row
   *
   */
  template<typename dummy = void>
  RAJA_HOST_DEVICE RAJA_INLINE static
      typename std::enable_if<(s_C_minor_dim_registers == 0), dummy>::type
      multiply_accumulate(left_type const& A,
                          right_type const& B,
                          result_type& C)
  {
    constexpr camp::idx_t bc_segbits              = result_type::s_segbits;
    constexpr camp::idx_t a_segments_per_register = 1 << bc_segbits;

    RAJA_UNROLL
    for (camp::idx_t ac_row = 0; ac_row < N_SIZE; ++ac_row)
    {
      camp::idx_t c_reg     = ac_row / result_type::s_major_dim_per_register;
      camp::idx_t c_segment = ac_row % result_type::s_major_dim_per_register;
      register_type c_tmp;

      RAJA_UNROLL
      for (camp::idx_t b_reg = 0; b_reg < right_type::s_num_registers; ++b_reg)
      {

        camp::idx_t a_segment = ac_row * right_type::s_num_registers + b_reg;
        camp::idx_t a_reg     = a_segment / a_segments_per_register;
        camp::idx_t a_reg_segment = a_segment % a_segments_per_register;

        auto a_tmp = A.get_register(a_reg).segmented_broadcast_outer(
            bc_segbits, a_reg_segment);

        if (b_reg == 0)
        {

          c_tmp = a_tmp.multiply(B.get_register(b_reg));
        }
        else
        {
          c_tmp = a_tmp.multiply_add(B.get_register(b_reg), c_tmp);
        }
      }

      C.get_register(c_reg) += c_tmp.segmented_sum_outer(bc_segbits, c_segment);
    }
  }

  RAJA_HOST_DEVICE
  static RAJA_INLINE void multiply(left_type const& A,
                                   right_type const& B,
                                   result_type& C)
  {
    C = result_type(0);
    multiply_accumulate(A, B, C);
  }
};

/**
 *
 * Column-Major * Column-Major ==> Column-Major
 *
 */
template<typename T,
         typename REGISTER_POLICY,
         camp::idx_t N_SIZE,
         camp::idx_t M_SIZE,
         camp::idx_t M2_SIZE,
         camp::idx_t O_SIZE>
struct MatrixMatrixMultiplyHelper<
    RAJA::expt::TensorRegister<REGISTER_POLICY,
                               T,
                               RAJA::expt::ColMajorLayout,
                               camp::idx_seq<N_SIZE, M_SIZE>>,
    RAJA::expt::TensorRegister<REGISTER_POLICY,
                               T,
                               RAJA::expt::ColMajorLayout,
                               camp::idx_seq<M2_SIZE, O_SIZE>>>
{

  using self_type = MatrixMatrixMultiplyHelper<
      RAJA::expt::TensorRegister<REGISTER_POLICY,
                                 T,
                                 RAJA::expt::ColMajorLayout,
                                 camp::idx_seq<N_SIZE, M_SIZE>>,
      RAJA::expt::TensorRegister<REGISTER_POLICY,
                                 T,
                                 RAJA::expt::ColMajorLayout,
                                 camp::idx_seq<M2_SIZE, O_SIZE>>>;

  static_assert(M_SIZE == M2_SIZE,
                "Matrices are not compatible for multiplication");

  using left_type = RAJA::expt::TensorRegister<REGISTER_POLICY,
                                               T,
                                               RAJA::expt::ColMajorLayout,
                                               camp::idx_seq<N_SIZE, M_SIZE>>;

  using right_type = RAJA::expt::TensorRegister<REGISTER_POLICY,
                                                T,
                                                RAJA::expt::ColMajorLayout,
                                                camp::idx_seq<M_SIZE, O_SIZE>>;

  using result_type = RAJA::expt::TensorRegister<REGISTER_POLICY,
                                                 T,
                                                 RAJA::expt::ColMajorLayout,
                                                 camp::idx_seq<N_SIZE, O_SIZE>>;

  using register_type = typename result_type::register_type;

  static constexpr camp::idx_t s_elements_per_register =
      left_type::s_elements_per_register;
  static constexpr camp::idx_t s_A_minor_dim_registers =
      left_type::s_minor_dim_registers;
  static constexpr camp::idx_t s_B_minor_dim_registers =
      right_type::s_minor_dim_registers;
  static constexpr camp::idx_t s_C_minor_dim_registers =
      result_type::s_minor_dim_registers;

  /*
   * Matrix A (and C) has 1 more more registers per column
   *
   */
  template<typename dummy = void>
  RAJA_HOST_DEVICE static RAJA_INLINE
      typename std::enable_if<(s_C_minor_dim_registers != 0), dummy>::type
      multiply_accumulate(left_type const& A,
                          right_type const& B,
                          result_type& C)
  {

#if defined(RAJA_ENABLE_VECTOR_STATS) && !defined(__CUDA_ARCH__)
    RAJA::tensor_stats::num_matrix_mm_multacc_row_row++;
#endif


    constexpr camp::idx_t num_ac_reg_per_col = s_C_minor_dim_registers;

    RAJA_UNROLL
    for (camp::idx_t c_reg = 0; c_reg < result_type::s_num_registers; ++c_reg)
    {
      camp::idx_t ac_row_reg = c_reg % num_ac_reg_per_col;
      camp::idx_t bc_col     = c_reg / num_ac_reg_per_col;

      RAJA_UNROLL
      for (camp::idx_t b_row = 0; b_row < M_SIZE; ++b_row)
      {
        camp::idx_t a_reg = b_row * num_ac_reg_per_col + ac_row_reg;

        C.get_register(c_reg) =
            register_type(B.get(b_row, bc_col))
                .multiply_add(A.get_register(a_reg), C.get_register(c_reg));
      }
    }
  }

  /*
   * Matrix A (and C) have less than one register per column
   *
   */
  template<typename dummy = void>
  RAJA_HOST_DEVICE RAJA_INLINE static
      typename std::enable_if<(s_C_minor_dim_registers == 0), dummy>::type
      multiply_accumulate(left_type const& A,
                          right_type const& B,
                          result_type& C)
  {
    constexpr camp::idx_t ac_segbits              = result_type::s_segbits;
    constexpr camp::idx_t b_segments_per_register = 1 << ac_segbits;

    camp::idx_t bc_col = 0;

    RAJA_UNROLL
    for (camp::idx_t c_reg = 0;
         c_reg < N_SIZE / result_type::s_major_dim_per_register; ++c_reg)
    {

      RAJA_UNROLL
      for (camp::idx_t c_segment = 0;
           c_segment < result_type::s_major_dim_per_register; ++c_segment)
      {

        register_type c_tmp;

        RAJA_UNROLL
        for (camp::idx_t a_reg = 0; a_reg < right_type::s_num_registers;
             ++a_reg)
        {


          camp::idx_t b_segment = bc_col * right_type::s_num_registers + a_reg;
          camp::idx_t b_reg     = b_segment / b_segments_per_register;
          camp::idx_t b_reg_segment = b_segment % b_segments_per_register;

          register_type b_tmp = B.get_register(b_reg).segmented_broadcast_outer(
              ac_segbits, b_reg_segment);

          if (a_reg == 0)
          {
            c_tmp = b_tmp.multiply(A.get_register(a_reg));
          }
          else
          {
            c_tmp = b_tmp.multiply_add(A.get_register(a_reg), c_tmp);
          }
        }

        C.get_register(c_reg) +=
            c_tmp.segmented_sum_outer(ac_segbits, c_segment);

        ++bc_col;
      }  // c_segment
    }    // c_reg
  }

  RAJA_HOST_DEVICE
  static RAJA_INLINE void multiply(left_type const& A,
                                   right_type const& B,
                                   result_type& C)
  {
    C = result_type(0);
    self_type::multiply_accumulate(A, B, C);
  }
};


}  // namespace expt
}  // namespace internal
}  // namespace RAJA


#endif
