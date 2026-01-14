/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for SYCL tiled executors.
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


#ifndef RAJA_policy_sycl_kernel_TileTCount_HPP
#define RAJA_policy_sycl_kernel_TileTCount_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>
#include <type_traits>

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/Tile.hpp"
#include "RAJA/pattern/kernel/internal.hpp"

namespace RAJA
{
namespace internal
{

/*!
 * A specialized RAJA::kernel sycl_impl executor for statement::TileTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         typename TPol,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<
    Data,
    statement::
        TileTCount<ArgumentId, ParamId, TPol, seq_exec, EnclosedStmts...>,
    Types>
    : public SyclStatementExecutor<
          Data,
          statement::Tile<ArgumentId, TPol, seq_exec, EnclosedStmts...>,
          Types>
{

  using Base = SyclStatementExecutor<
      Data,
      statement::Tile<ArgumentId, TPol, seq_exec, EnclosedStmts...>,
      Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto& segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t        = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    diff_t chunk_size = TPol::chunk_size;

    // compute trip count
    diff_t len = segment.end() - segment.begin();

    // Iterate through tiles
    for (diff_t i = 0, t = 0; i < len; i += chunk_size, ++t)
    {

      // Assign our new tiled segment
      segment = orig_segment.slice(i, chunk_size);
      data.template assign_param<ParamId>(t);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active);
    }

    // Set range back to original values
    segment = orig_segment;
  }
};

/*!
 * A specialized RAJA::kernel sycl_impl executor for statement::TileTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         camp::idx_t chunk_size,
         int BlockDim,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<
    Data,
    statement::TileTCount<ArgumentId,
                          ParamId,
                          RAJA::tile_fixed<chunk_size>,
                          sycl_group_012_direct<BlockDim>,
                          EnclosedStmts...>,
    Types>
    : public SyclStatementExecutor<
          Data,
          statement::Tile<ArgumentId,
                          RAJA::tile_fixed<chunk_size>,
                          sycl_group_012_direct<BlockDim>,
                          EnclosedStmts...>,
          Types>
{

  using Base =
      SyclStatementExecutor<Data,
                            statement::Tile<ArgumentId,
                                            RAJA::tile_fixed<chunk_size>,
                                            sycl_group_012_direct<BlockDim>,
                                            EnclosedStmts...>,
                            Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto& segment = camp::get<ArgumentId>(data.segment_tuple);

    using segment_t = camp::decay<decltype(segment)>;

    // compute trip count
    diff_t len = segment.end() - segment.begin();
    // diff_t t = get_sycl_dim<BlockDim>(blockIdx);
    diff_t t = item.get_group(BlockDim);
    diff_t i = t * chunk_size;

    // check have a chunk
    if (i < len)
    {

      // Keep copy of original segment, so we can restore it
      segment_t orig_segment = segment;

      // Assign our new tiled segment
      segment = orig_segment.slice(i, chunk_size);
      data.template assign_param<ParamId>(t);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active);

      // Set range back to original values
      segment = orig_segment;
    }
  }
};

/*!
 * A specialized RAJA::kernel sycl_impl executor for statement::TileTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         camp::idx_t chunk_size,
         int BlockDim,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<
    Data,
    statement::TileTCount<ArgumentId,
                          ParamId,
                          RAJA::tile_fixed<chunk_size>,
                          sycl_group_012_loop<BlockDim>,
                          EnclosedStmts...>,
    Types>
    : public SyclStatementExecutor<
          Data,
          statement::Tile<ArgumentId,
                          RAJA::tile_fixed<chunk_size>,
                          sycl_group_012_loop<BlockDim>,
                          EnclosedStmts...>,
          Types>
{

  using Base =
      SyclStatementExecutor<Data,
                            statement::Tile<ArgumentId,
                                            RAJA::tile_fixed<chunk_size>,
                                            sycl_group_012_loop<BlockDim>,
                                            EnclosedStmts...>,
                            Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto& segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t        = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    // compute trip count
    diff_t len      = segment.end() - segment.begin();
    diff_t t_init   = item.get_group(BlockDim);
    diff_t i_init   = t_init * chunk_size;
    diff_t t_stride = item.get_group_range(BlockDim);
    diff_t i_stride = t_stride * chunk_size;

    // Iterate through grid stride of chunks
    for (diff_t i = i_init, t = t_init; i < len; i += i_stride, t += t_stride)
    {

      // Assign our new tiled segment
      segment = orig_segment.slice(i, chunk_size);
      data.template assign_param<ParamId>(t);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active);
    }

    // Set range back to original values
    segment = orig_segment;
  }
};

/*!
 * A specialized RAJA::kernel sycl_impl executor for statement::TileTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         camp::idx_t chunk_size,
         int ThreadDim,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<
    Data,
    statement::TileTCount<ArgumentId,
                          ParamId,
                          RAJA::tile_fixed<chunk_size>,
                          sycl_local_012_direct<ThreadDim>,
                          EnclosedStmts...>,
    Types>
    : public SyclStatementExecutor<
          Data,
          statement::Tile<ArgumentId,
                          RAJA::tile_fixed<chunk_size>,
                          sycl_local_012_direct<ThreadDim>,
                          EnclosedStmts...>,
          Types>
{

  using Base =
      SyclStatementExecutor<Data,
                            statement::Tile<ArgumentId,
                                            RAJA::tile_fixed<chunk_size>,
                                            sycl_local_012_direct<ThreadDim>,
                                            EnclosedStmts...>,
                            Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto& segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t        = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    // compute trip count
    diff_t len = segment.end() - segment.begin();
    // diff_t t = get_sycl_dim<ThreadDim>(threadIdx);
    diff_t t = item.get_local_id(ThreadDim);
    diff_t i = t * chunk_size;

    // execute enclosed statements if any thread will
    // but mask off threads without work
    bool have_work = i < len;

    // Assign our new tiled segment
    diff_t slice_size = have_work ? chunk_size : 0;
    segment           = orig_segment.slice(i, slice_size);
    data.template assign_param<ParamId>(t);

    // execute enclosed statements
    enclosed_stmts_t::exec(data, item, thread_active && have_work);

    // Set range back to original values
    segment = orig_segment;
  }
};

/*!
 * A specialized RAJA::kernel sycl_impl executor for statement::TileTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         camp::idx_t chunk_size,
         int ThreadDim,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<
    Data,
    statement::TileTCount<ArgumentId,
                          ParamId,
                          RAJA::tile_fixed<chunk_size>,
                          sycl_local_012_loop<ThreadDim>,
                          EnclosedStmts...>,
    Types>
    : public SyclStatementExecutor<
          Data,
          statement::Tile<ArgumentId,
                          RAJA::tile_fixed<chunk_size>,
                          sycl_local_012_loop<ThreadDim>,
                          EnclosedStmts...>,
          Types>
{

  using Base =
      SyclStatementExecutor<Data,
                            statement::Tile<ArgumentId,
                                            RAJA::tile_fixed<chunk_size>,
                                            sycl_local_012_loop<ThreadDim>,
                                            EnclosedStmts...>,
                            Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto& segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t        = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    // compute trip count
    diff_t len = segment_length<ArgumentId>(data);
    //    diff_t t_init = get_sycl_dim<ThreadDim>(threadIdx);
    diff_t t_init = item.get_local_id(ThreadDim);
    diff_t i_init = t_init * chunk_size;
    //    diff_t t_stride = get_sycl_dim<ThreadDim>(blockDim);
    diff_t t_stride = item.get_local_range(ThreadDim);
    diff_t i_stride = t_stride * chunk_size;

    // Iterate through grid stride of chunks
    for (diff_t ii = 0, t = t_init; ii < len; ii += i_stride, t += t_stride)
    {
      diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      bool have_work = i < len;

      // Assign our new tiled segment
      diff_t slice_size = have_work ? chunk_size : 0;
      segment           = orig_segment.slice(i, slice_size);
      data.template assign_param<ParamId>(t);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active && have_work);
    }

    // Set range back to original values
    segment = orig_segment;
  }
};

}  // end namespace internal
}  // end namespace RAJA

#endif  // RAJA_ENABLE_SYCL
#endif  /* RAJA_policy_sycl_kernel_TileTCount_HPP */
