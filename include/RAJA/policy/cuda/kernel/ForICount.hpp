/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for CUDA statement executors.
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


#ifndef RAJA_policy_cuda_kernel_ForICount_HPP
#define RAJA_policy_cuda_kernel_ForICount_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/cuda/kernel/internal.hpp"

namespace RAJA
{

namespace internal
{

/*
 * Executor for work sharing inside CudaKernel.
 * Provides a direct unchecked mapping.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 * Meets all sync requirements
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         typename IndexMapper,
         kernel_sync_requirement sync,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<
        ArgumentId,
        ParamId,
        RAJA::policy::cuda::
            cuda_indexer<iteration_mapping::DirectUnchecked, sync, IndexMapper>,
        EnclosedStmts...>,
    Types>
    : CudaStatementExecutor<
          Data,
          statement::For<ArgumentId,
                         RAJA::policy::cuda::cuda_indexer<
                             iteration_mapping::DirectUnchecked,
                             sync,
                             IndexMapper>,
                         EnclosedStmts...>,
          Types>
{

  using Base = CudaStatementExecutor<
      Data,
      statement::For<
          ArgumentId,
          RAJA::policy::cuda::cuda_indexer<iteration_mapping::DirectUnchecked,
                                           sync,
                                           IndexMapper>,
          EnclosedStmts...>,
      Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // grid stride loop
    const diff_t i = IndexMapper::template index<diff_t>();

    // Assign the index to the argument and param
    data.template assign_offset<ArgumentId>(i);
    data.template assign_param<ParamId>(i);

    // execute enclosed statements
    enclosed_stmts_t::exec(data, thread_active);
  }
};

/*
 * Executor for work sharing inside CudaKernel.
 * Provides a direct mapping.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 * Meets all sync requirements
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         typename IndexMapper,
         kernel_sync_requirement sync,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<
        ArgumentId,
        ParamId,
        RAJA::policy::cuda::
            cuda_indexer<iteration_mapping::Direct, sync, IndexMapper>,
        EnclosedStmts...>,
    Types>
    : CudaStatementExecutor<
          Data,
          statement::For<
              ArgumentId,
              RAJA::policy::cuda::
                  cuda_indexer<iteration_mapping::Direct, sync, IndexMapper>,
              EnclosedStmts...>,
          Types>
{

  using Base = CudaStatementExecutor<
      Data,
      statement::For<ArgumentId,
                     RAJA::policy::cuda::cuda_indexer<iteration_mapping::Direct,
                                                      sync,
                                                      IndexMapper>,
                     EnclosedStmts...>,
      Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // grid stride loop
    const diff_t len = segment_length<ArgumentId>(data);
    const diff_t i   = IndexMapper::template index<diff_t>();

    // execute enclosed statements if any thread will
    // but mask off threads without work
    const bool have_work = (i < len);

    // Assign the index to the argument and param
    data.template assign_offset<ArgumentId>(i);
    data.template assign_param<ParamId>(i);

    // execute enclosed statements
    enclosed_stmts_t::exec(data, thread_active && have_work);
  }
};

/*
 * Executor for work sharing inside CudaKernel.
 * Provides a strided loop.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 * Meets all sync requirements
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         typename IndexMapper,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<
        ArgumentId,
        ParamId,
        RAJA::policy::cuda::cuda_indexer<
            iteration_mapping::StridedLoop<named_usage::unspecified>,
            kernel_sync_requirement::sync,
            IndexMapper>,
        EnclosedStmts...>,
    Types>
    : public CudaStatementExecutor<
          Data,
          statement::For<
              ArgumentId,
              RAJA::policy::cuda::cuda_indexer<
                  iteration_mapping::StridedLoop<named_usage::unspecified>,
                  kernel_sync_requirement::sync,
                  IndexMapper>,
              EnclosedStmts...>,
          Types>
{

  using Base = CudaStatementExecutor<
      Data,
      statement::For<
          ArgumentId,
          RAJA::policy::cuda::cuda_indexer<
              iteration_mapping::StridedLoop<named_usage::unspecified>,
              kernel_sync_requirement::sync,
              IndexMapper>,
          EnclosedStmts...>,
      Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // grid stride loop
    const diff_t len      = segment_length<ArgumentId>(data);
    const diff_t i_init   = IndexMapper::template index<diff_t>();
    const diff_t i_stride = IndexMapper::template size<diff_t>();

    // Iterate through in chunks
    // threads will have the same numbers of iterations
    for (diff_t ii = 0; ii < len; ii += i_stride)
    {
      const diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      const bool have_work = (i < len);

      // Assign the index to the argument and param
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }
  }
};

/*
 * Executor for work sharing inside CudaKernel.
 * Provides a strided loop.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 * Meets no sync requirements
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         typename IndexMapper,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<
        ArgumentId,
        ParamId,
        RAJA::policy::cuda::cuda_indexer<
            iteration_mapping::StridedLoop<named_usage::unspecified>,
            kernel_sync_requirement::none,
            IndexMapper>,
        EnclosedStmts...>,
    Types>
    : public CudaStatementExecutor<
          Data,
          statement::For<
              ArgumentId,
              RAJA::policy::cuda::cuda_indexer<
                  iteration_mapping::StridedLoop<named_usage::unspecified>,
                  kernel_sync_requirement::none,
                  IndexMapper>,
              EnclosedStmts...>,
          Types>
{

  using Base = CudaStatementExecutor<
      Data,
      statement::For<
          ArgumentId,
          RAJA::policy::cuda::cuda_indexer<
              iteration_mapping::StridedLoop<named_usage::unspecified>,
              kernel_sync_requirement::none,
              IndexMapper>,
          EnclosedStmts...>,
      Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // grid stride loop
    const diff_t len      = segment_length<ArgumentId>(data);
    const diff_t i_init   = IndexMapper::template index<diff_t>();
    const diff_t i_stride = IndexMapper::template size<diff_t>();

    // Iterate through one at a time
    // threads will have the different numbers of iterations
    for (diff_t i = i_init; i < len; i += i_stride)
    {

      // Assign the index to the argument and param
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
  }
};

/*
 * Executor for sequential loops inside of a CudaKernel.
 *
 * This is specialized since it need to execute the loop immediately.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, seq_exec, EnclosedStmts...>,
    Types>
    : CudaStatementExecutor<
          Data,
          statement::ForICount<
              ArgumentId,
              RAJA::policy::cuda::cuda_indexer<
                  iteration_mapping::StridedLoop<named_usage::unspecified>,
                  kernel_sync_requirement::none,
                  cuda::IndexGlobal<named_dim::x,
                                    named_usage::ignored,
                                    named_usage::ignored>>,
              EnclosedStmts...>,
          Types>
{};

/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         typename Mask,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<ArgumentId,
                         ParamId,
                         RAJA::cuda_warp_masked_direct<Mask>,
                         EnclosedStmts...>,
    Types>
    : public CudaStatementExecutor<
          Data,
          statement::For<ArgumentId,
                         RAJA::cuda_warp_masked_direct<Mask>,
                         EnclosedStmts...>,
          Types>
{

  using Base =
      CudaStatementExecutor<Data,
                            statement::For<ArgumentId,
                                           RAJA::cuda_warp_masked_direct<Mask>,
                                           EnclosedStmts...>,
                            Types>;

  using typename Base::diff_t;

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  static_assert(mask_t::max_masked_size <=
                    RAJA::policy::cuda::device_constants.WARP_SIZE,
                "BitMask is too large for CUDA warp size");

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    const diff_t len = segment_length<ArgumentId>(data);

    const diff_t i = mask_t::maskValue((diff_t)threadIdx.x);

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);
    data.template assign_param<ParamId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i < len));
  }
};

/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         typename Mask,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<ArgumentId,
                         ParamId,
                         RAJA::cuda_warp_masked_loop<Mask>,
                         EnclosedStmts...>,
    Types>
    : public CudaStatementExecutor<
          Data,
          statement::For<ArgumentId,
                         RAJA::cuda_warp_masked_loop<Mask>,
                         EnclosedStmts...>,
          Types>
{

  using Base =
      CudaStatementExecutor<Data,
                            statement::For<ArgumentId,
                                           RAJA::cuda_warp_masked_loop<Mask>,
                                           EnclosedStmts...>,
                            Types>;

  using typename Base::diff_t;

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  static_assert(mask_t::max_masked_size <=
                    RAJA::policy::cuda::device_constants.WARP_SIZE,
                "BitMask is too large for CUDA warp size");

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // masked size strided loop
    const diff_t len      = segment_length<ArgumentId>(data);
    const diff_t i_init   = mask_t::maskValue((diff_t)threadIdx.x);
    const diff_t i_stride = (diff_t)mask_t::max_masked_size;

    // Iterate through grid stride of chunks
    for (diff_t ii = 0; ii < len; ii += i_stride)
    {
      const diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      bool have_work = i < len;

      // Assign the x thread to the argument and param
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }
  }
};

/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         typename Mask,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<ArgumentId,
                         ParamId,
                         RAJA::cuda_thread_masked_direct<Mask>,
                         EnclosedStmts...>,
    Types>
    : public CudaStatementExecutor<
          Data,
          statement::For<ArgumentId,
                         RAJA::cuda_thread_masked_direct<Mask>,
                         EnclosedStmts...>,
          Types>
{

  using Base = CudaStatementExecutor<
      Data,
      statement::For<ArgumentId,
                     RAJA::cuda_thread_masked_direct<Mask>,
                     EnclosedStmts...>,
      Types>;

  using typename Base::diff_t;

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    const diff_t len = segment_length<ArgumentId>(data);

    const diff_t i = mask_t::maskValue((diff_t)threadIdx.x);

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);
    data.template assign_param<ParamId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i < len));
  }
};

/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         typename Mask,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<ArgumentId,
                         ParamId,
                         RAJA::cuda_thread_masked_loop<Mask>,
                         EnclosedStmts...>,
    Types>
    : public CudaStatementExecutor<
          Data,
          statement::For<ArgumentId,
                         RAJA::cuda_thread_masked_loop<Mask>,
                         EnclosedStmts...>,
          Types>
{

  using Base =
      CudaStatementExecutor<Data,
                            statement::For<ArgumentId,
                                           RAJA::cuda_thread_masked_loop<Mask>,
                                           EnclosedStmts...>,
                            Types>;

  using typename Base::diff_t;

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // masked size strided loop
    const diff_t len      = segment_length<ArgumentId>(data);
    const diff_t i_init   = mask_t::maskValue((diff_t)threadIdx.x);
    const diff_t i_stride = (diff_t)mask_t::max_masked_size;

    // Iterate through grid stride of chunks
    for (diff_t ii = 0; ii < len; ii += i_stride)
    {
      const diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      bool have_work = i < len;

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }
  }
};

}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_policy_cuda_kernel_ForICount_HPP */
