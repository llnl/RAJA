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


#ifndef RAJA_policy_cuda_kernel_For_HPP
#define RAJA_policy_cuda_kernel_For_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/cuda/kernel/internal.hpp"

namespace RAJA
{

namespace internal
{

/*
 * Executor for work sharing inside CudaKernel.
 * Mapping without checking from IndexMapper to indices
 * Assigns the loop index to offset ArgumentId
 * Meets all sync requirements
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename IndexMapper,
         kernel_sync_requirement sync,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::For<
        ArgumentId,
        RAJA::policy::cuda::
            cuda_indexer<iteration_mapping::DirectUnchecked, sync, IndexMapper>,
        EnclosedStmts...>,
    Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  using DimensionCalculator = RAJA::internal::KernelDimensionCalculator<
      RAJA::policy::cuda::
          cuda_indexer<iteration_mapping::DirectUnchecked, sync, IndexMapper>>;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    const diff_t i = IndexMapper::template index<diff_t>();

    // Assign the index to the argument
    data.template assign_offset<ArgumentId>(i);

    // execute enclosed statements
    enclosed_stmts_t::exec(data, thread_active);
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {
    const diff_t len = segment_length<ArgumentId>(data);

    LaunchDims dims = DimensionCalculator::get_dimensions(len);

    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);

    return combine(dims, enclosed_dims);
  }
};

/*
 * Executor for work sharing inside CudaKernel.
 * Mapping directly from IndexMapper to indices
 * Assigns the loop index to offset ArgumentId
 * Meets all sync requirements
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename IndexMapper,
         kernel_sync_requirement sync,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::For<ArgumentId,
                   RAJA::policy::cuda::cuda_indexer<iteration_mapping::Direct,
                                                    sync,
                                                    IndexMapper>,
                   EnclosedStmts...>,
    Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  using DimensionCalculator = RAJA::internal::KernelDimensionCalculator<
      RAJA::policy::cuda::
          cuda_indexer<iteration_mapping::Direct, sync, IndexMapper>>;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    const diff_t len = segment_length<ArgumentId>(data);
    const diff_t i   = IndexMapper::template index<diff_t>();

    // execute enclosed statements if any thread will
    // but mask off threads without work
    const bool have_work = (i < len);

    // Assign the index to the argument
    data.template assign_offset<ArgumentId>(i);

    // execute enclosed statements
    enclosed_stmts_t::exec(data, thread_active && have_work);
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {
    const diff_t len = segment_length<ArgumentId>(data);

    LaunchDims dims = DimensionCalculator::get_dimensions(len);

    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);

    return combine(dims, enclosed_dims);
  }
};

/*
 * Executor for work sharing inside CudaKernel.
 * Provides a strided loop for IndexMapper.
 * Assigns the loop index to offset ArgumentId.
 * Meets all sync requirements
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename IndexMapper,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::For<ArgumentId,
                   RAJA::policy::cuda::cuda_indexer<
                       iteration_mapping::StridedLoop<named_usage::unspecified>,
                       kernel_sync_requirement::sync,
                       IndexMapper>,
                   EnclosedStmts...>,
    Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  using DimensionCalculator = RAJA::internal::KernelDimensionCalculator<
      RAJA::policy::cuda::cuda_indexer<
          iteration_mapping::StridedLoop<named_usage::unspecified>,
          kernel_sync_requirement::sync,
          IndexMapper>>;

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

      // Assign the index to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {
    diff_t len = segment_length<ArgumentId>(data);

    LaunchDims dims = DimensionCalculator::get_dimensions(len);

    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);

    return combine(dims, enclosed_dims);
  }
};

/*
 * Executor for work sharing inside CudaKernel.
 * Provides a strided loop for IndexMapper.
 * Assigns the loop index to offset ArgumentId.
 * Meets no sync requirements
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename IndexMapper,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::For<ArgumentId,
                   RAJA::policy::cuda::cuda_indexer<
                       iteration_mapping::StridedLoop<named_usage::unspecified>,
                       kernel_sync_requirement::none,
                       IndexMapper>,
                   EnclosedStmts...>,
    Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  using DimensionCalculator = RAJA::internal::KernelDimensionCalculator<
      RAJA::policy::cuda::cuda_indexer<
          iteration_mapping::StridedLoop<named_usage::unspecified>,
          kernel_sync_requirement::none,
          IndexMapper>>;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // grid stride loop
    const diff_t len      = segment_length<ArgumentId>(data);
    const diff_t i_init   = IndexMapper::template index<diff_t>();
    const diff_t i_stride = IndexMapper::template size<diff_t>();

    // Iterate through one at a time
    // threads will have different numbers of iterations
    for (diff_t i = i_init; i < len; i += i_stride)
    {

      // Assign the index to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {
    const diff_t len = segment_length<ArgumentId>(data);

    LaunchDims dims = DimensionCalculator::get_dimensions(len);

    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);

    return combine(dims, enclosed_dims);
  }
};

/*
 * Executor for sequential loops inside of a CudaKernel.
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::For<ArgumentId, seq_exec, EnclosedStmts...>,
    Types>
    : CudaStatementExecutor<
          Data,
          statement::For<
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
         typename Mask,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<Data,
                             statement::For<ArgumentId,
                                            RAJA::cuda_warp_masked_direct<Mask>,
                                            EnclosedStmts...>,
                             Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  static_assert(mask_t::max_masked_size <=
                    RAJA::policy::cuda::device_constants.WARP_SIZE,
                "BitMask is too large for CUDA warp size");

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    const diff_t len = segment_length<ArgumentId>(data);

    const diff_t i = mask_t::maskValue((diff_t)threadIdx.x);

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i < len));
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {
    // Get enclosed statements
    LaunchDims dims = enclosed_stmts_t::calculateDimensions(data);

    // we always get EXACTLY one warp by allocating one warp in the X
    // dimension
    const diff_t len = RAJA::policy::cuda::device_constants.WARP_SIZE;

    // request one thread per element in the segment
    set_cuda_dim<named_dim::x>(dims.dims.threads, len);

    // since we are direct-mapping, we REQUIRE len
    set_cuda_dim<named_dim::x>(dims.min_dims.threads, len);

    return (dims);
  }
};

/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename Mask,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<Data,
                             statement::For<ArgumentId,
                                            RAJA::cuda_warp_masked_loop<Mask>,
                                            EnclosedStmts...>,
                             Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  using DimensionCalculator =
      RAJA::internal::KernelDimensionCalculator<cuda_warp_loop>;

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

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {
    diff_t len = segment_length<ArgumentId>(data);

    LaunchDims dims = DimensionCalculator::get_dimensions(len);

    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);

    return combine(dims, enclosed_dims);
  }
};

/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from raw threadIdx.x
 * Assigns the loop index to offset ArgumentId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename Mask,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<
    Data,
    statement::For<ArgumentId,
                   RAJA::cuda_thread_masked_direct<Mask>,
                   EnclosedStmts...>,
    Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  using DimensionCalculator = RAJA::internal::KernelDimensionCalculator<
      cuda_thread_size_x_direct<mask_t::max_input_size>>;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    const diff_t len = segment_length<ArgumentId>(data);

    const diff_t i = mask_t::maskValue((diff_t)threadIdx.x);

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i < len));
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {
    const diff_t len = segment_length<ArgumentId>(data);

    LaunchDims dims = DimensionCalculator::get_dimensions(len);

    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);

    return combine(dims, enclosed_dims);
  }
};

/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename Mask,
         typename... EnclosedStmts,
         typename Types>
struct CudaStatementExecutor<Data,
                             statement::For<ArgumentId,
                                            RAJA::cuda_thread_masked_loop<Mask>,
                                            EnclosedStmts...>,
                             Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  using DimensionCalculator = RAJA::internal::KernelDimensionCalculator<
      cuda_thread_size_x_loop<mask_t::max_input_size>>;

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

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {
    diff_t len = segment_length<ArgumentId>(data);

    LaunchDims dims = DimensionCalculator::get_dimensions(len);

    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);

    return combine(dims, enclosed_dims);
  }
};

}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_policy_cuda_kernel_For_HPP */
