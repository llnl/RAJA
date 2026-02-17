/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for SYCL statement executors.
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


#ifndef RAJA_policy_sycl_kernel_ForICount_HPP
#define RAJA_policy_sycl_kernel_ForICount_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/sycl/kernel/internal.hpp"

namespace RAJA
{

namespace internal
{


/*
 * Executor for local work sharing loop inside SyclKernel.
 * Mapping directly from local id to indices
 * Assigns the loop iterate to offset ArgumentId
 * Assigns the loop count to param ParamId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         int ThreadDim,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<
    Data,
    statement::ForICount<ArgumentId,
                         ParamId,
                         RAJA::sycl_local_012_direct<ThreadDim>,
                         EnclosedStmts...>,
    Types>
    : public SyclStatementExecutor<
          Data,
          statement::For<ArgumentId,
                         RAJA::sycl_local_012_direct<ThreadDim>,
                         EnclosedStmts...>,
          Types>
{

  using Base = SyclStatementExecutor<
      Data,
      statement::For<ArgumentId,
                     RAJA::sycl_local_012_direct<ThreadDim>,
                     EnclosedStmts...>,
      Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    diff_t len = segment_length<ArgumentId>(data);
    auto i     = item.get_local_id(ThreadDim);

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);
    data.template assign_param<ParamId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, item, thread_active && (i < len));
  }
};

/*
 * Executor for local work sharing loop inside SyclKernel.
 * Assigns the loop index to offset ArgumentId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         typename Mask,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<
    Data,
    statement::ForICount<ArgumentId,
                         ParamId,
                         RAJA::sycl_local_masked_direct<Mask>,
                         EnclosedStmts...>,
    Types>
    : public SyclStatementExecutor<
          Data,
          statement::For<ArgumentId,
                         RAJA::sycl_local_masked_direct<Mask>,
                         EnclosedStmts...>,
          Types>
{

  using Base =
      SyclStatementExecutor<Data,
                            statement::For<ArgumentId,
                                           RAJA::sycl_local_masked_direct<Mask>,
                                           EnclosedStmts...>,
                            Types>;

  using typename Base::diff_t;

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      SyclStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    diff_t len = segment_length<ArgumentId>(data);
    auto i0    = item.get_local_id(0);
    diff_t i   = mask_t::maskValue(i0);

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);
    data.template assign_param<ParamId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, item, thread_active && (i < len));
  }
};

/*
 * Executor for local work sharing loop inside SyclKernel.
 * Assigns the loop index to offset ArgumentId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         typename Mask,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<
    Data,
    statement::ForICount<ArgumentId,
                         ParamId,
                         RAJA::sycl_local_masked_loop<Mask>,
                         EnclosedStmts...>,
    Types>
    : public SyclStatementExecutor<
          Data,
          statement::For<ArgumentId,
                         RAJA::sycl_local_masked_loop<Mask>,
                         EnclosedStmts...>,
          Types>
{

  using Base =
      SyclStatementExecutor<Data,
                            statement::For<ArgumentId,
                                           RAJA::sycl_local_masked_loop<Mask>,
                                           EnclosedStmts...>,
                            Types>;

  using typename Base::diff_t;

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      SyclStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    // masked size strided loop
    diff_t len      = segment_length<ArgumentId>(data);
    auto i0         = item.get_local_id(0);
    diff_t i_init   = mask_t::maskValue(i0);
    diff_t i_stride = (diff_t)mask_t::max_masked_size;

    // Iterate through grid stride of chunks
    for (diff_t ii = 0; ii < len; ii += i_stride)
    {
      diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      bool have_work = i < len;

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active && have_work);
    }
  }
};

/*
 * Executor for thread work sharing loop inside SyclKernel.
 * Provides a block-stride loop (stride of blockDim.xyz) for
 * each thread in xyz.
 * Assigns the loop iterate to offset ArgumentId
 * Assigns the loop offset to param ParamId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         int ThreadDim,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<
    Data,
    statement::ForICount<ArgumentId,
                         ParamId,
                         RAJA::sycl_local_012_loop<ThreadDim>,
                         EnclosedStmts...>,
    Types>
    : public SyclStatementExecutor<
          Data,
          statement::For<ArgumentId,
                         RAJA::sycl_local_012_loop<ThreadDim>,
                         EnclosedStmts...>,
          Types>
{

  using Base =
      SyclStatementExecutor<Data,
                            statement::For<ArgumentId,
                                           RAJA::sycl_local_012_loop<ThreadDim>,
                                           EnclosedStmts...>,
                            Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    // block stride loop
    diff_t len    = segment_length<ArgumentId>(data);
    auto i_init   = item.get_local_id(ThreadDim);
    auto i_stride = item.get_local_range(ThreadDim);

    // Iterate through grid stride of chunks
    for (diff_t ii = 0; ii < len; ii += i_stride)
    {
      diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      bool have_work = i < len;

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active && have_work);
    }
  }
};

/*
 * Executor for group work sharing inside SyclKernel.
 * Provides a direct mapping of each block in 012.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         int BlockDim,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<
    Data,
    statement::ForICount<ArgumentId,
                         ParamId,
                         RAJA::sycl_group_012_direct<BlockDim>,
                         EnclosedStmts...>,
    Types>
    : public SyclStatementExecutor<
          Data,
          statement::For<ArgumentId,
                         RAJA::sycl_group_012_direct<BlockDim>,
                         EnclosedStmts...>,
          Types>
{

  using Base = SyclStatementExecutor<
      Data,
      statement::For<ArgumentId,
                     RAJA::sycl_group_012_direct<BlockDim>,
                     EnclosedStmts...>,
      Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    // grid stride loop
    diff_t len = segment_length<ArgumentId>(data);
    auto i     = item.get_group(BlockDim);

    if (i < len)
    {

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active);
    }
  }
};

/*
 * Executor for group work sharing inside SyclKernel.
 * Provides a group-stride loop for
 * each block in 012.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         int BlockDim,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<
    Data,
    statement::ForICount<ArgumentId,
                         ParamId,
                         RAJA::sycl_group_012_loop<BlockDim>,
                         EnclosedStmts...>,
    Types>
    : public SyclStatementExecutor<
          Data,
          statement::For<ArgumentId,
                         RAJA::sycl_group_012_loop<BlockDim>,
                         EnclosedStmts...>,
          Types>
{

  using Base =
      SyclStatementExecutor<Data,
                            statement::For<ArgumentId,
                                           RAJA::sycl_group_012_loop<BlockDim>,
                                           EnclosedStmts...>,
                            Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    // grid stride loop
    diff_t len    = segment_length<ArgumentId>(data);
    auto i_init   = item.get_group(BlockDim);
    auto i_stride = item.get_group_range(BlockDim);

    // Iterate through grid stride of chunks
    for (diff_t i = i_init; i < len; i += i_stride)
    {

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active);
    }
  }
};

/*
 * Executor for sequential loops inside of a SyclKernel.
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
struct SyclStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, seq_exec, EnclosedStmts...>,
    Types>
    : public SyclStatementExecutor<
          Data,
          statement::For<ArgumentId, seq_exec, EnclosedStmts...>,
          Types>
{

  using Base = SyclStatementExecutor<
      Data,
      statement::For<ArgumentId, seq_exec, EnclosedStmts...>,
      Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    diff_t len = segment_length<ArgumentId>(data);

    for (diff_t i = 0; i < len; ++i)
    {
      // Assign i to the argument
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active);
    }
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_policy_sycl_kernel_ForICount_HPP */
