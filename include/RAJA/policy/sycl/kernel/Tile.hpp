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


#ifndef RAJA_policy_sycl_kernel_Tile_HPP
#define RAJA_policy_sycl_kernel_Tile_HPP

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
 * A specialized RAJA::kernel sycl_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 *
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename TPol,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<
    Data,
    statement::Tile<ArgumentId, TPol, seq_exec, EnclosedStmts...>,
    Types>
{

  using stmt_list_t      = StatementList<EnclosedStmts...>;
  using enclosed_stmts_t = SyclStatementListExecutor<Data, stmt_list_t, Types>;
  using diff_t           = segment_diff_type<ArgumentId, Data>;

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
    for (diff_t i = 0; i < len; i += chunk_size)
    {

      // Assign our new tiled segment
      segment = orig_segment.slice(i, chunk_size);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active);
    }

    // Set range back to original values
    segment = orig_segment;
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {

    // privatize data, so we can mess with the segments
    using data_t        = camp::decay<Data>;
    data_t private_data = data;

    // Get original segment
    auto& segment = camp::get<ArgumentId>(private_data.segment_tuple);

    // restrict to first tile
    segment = segment.slice(0, TPol::chunk_size);

    // compute dimensions of children with segment restricted to tile
    LaunchDims enclosed_dims =
        enclosed_stmts_t::calculateDimensions(private_data);

    return enclosed_dims;
  }
};

/*!
 * A specialized RAJA::kernel sycl_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 *
 */
template<typename Data,
         camp::idx_t ArgumentId,
         camp::idx_t chunk_size,
         int BlockDim,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<Data,
                             statement::Tile<ArgumentId,
                                             RAJA::tile_fixed<chunk_size>,
                                             sycl_group_012_direct<BlockDim>,
                                             EnclosedStmts...>,
                             Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = SyclStatementListExecutor<Data, stmt_list_t, Types>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto& segment = camp::get<ArgumentId>(data.segment_tuple);

    using segment_t = camp::decay<decltype(segment)>;

    // compute trip count
    diff_t len = segment.end() - segment.begin();
    // diff_t i = get_sycl_dim<BlockDim>(blockIdx) * chunk_size; // TODO
    diff_t i =
        item.get_group(BlockDim) *
        chunk_size;  // get_sycl_dim<BlockDim>(blockIdx) * chunk_size; // TODO

    // check have chunk
    if (i < len)
    {

      // Keep copy of original segment, so we can restore it
      segment_t orig_segment = segment;

      // Assign our new tiled segment
      segment = orig_segment.slice(i, chunk_size);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active);

      // Set range back to original values
      segment = orig_segment;
    }
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {

    // Compute how many blocks
    diff_t len        = segment_length<ArgumentId>(data);
    diff_t num_blocks = len / chunk_size;
    if (num_blocks * chunk_size < len)
    {
      num_blocks++;
    }

    LaunchDims dims;
    set_sycl_dim<BlockDim>(dims.group, num_blocks);

    // since we are direct-mapping, we REQUIRE len
    set_sycl_dim<BlockDim>(dims.min_groups, num_blocks);


    // privatize data, so we can mess with the segments
    using data_t        = camp::decay<Data>;
    data_t private_data = data;

    // Get original segment
    auto& segment = camp::get<ArgumentId>(private_data.segment_tuple);

    // restrict to first tile
    segment = segment.slice(0, chunk_size);


    LaunchDims enclosed_dims =
        enclosed_stmts_t::calculateDimensions(private_data);

    return dims.max(enclosed_dims);
  }
};

/*!
 * A specialized RAJA::kernel sycl_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 *
 */
template<typename Data,
         camp::idx_t ArgumentId,
         camp::idx_t chunk_size,
         int BlockDim,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<Data,
                             statement::Tile<ArgumentId,
                                             RAJA::tile_fixed<chunk_size>,
                                             sycl_group_012_loop<BlockDim>,
                                             EnclosedStmts...>,
                             Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = SyclStatementListExecutor<Data, stmt_list_t, Types>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

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
    diff_t i_init   = item.get_group(BlockDim) * chunk_size;        // TODO
    diff_t i_stride = item.get_group_range(BlockDim) * chunk_size;  // TODO

    // Iterate through grid stride of chunks
    for (diff_t i = i_init; i < len; i += i_stride)
    {

      // Assign our new tiled segment
      segment = orig_segment.slice(i, chunk_size);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active);
    }

    // Set range back to original values
    segment = orig_segment;
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {

    // Compute how many blocks
    diff_t len        = segment_length<ArgumentId>(data);
    diff_t num_blocks = len / chunk_size;
    if (num_blocks * chunk_size < len)
    {
      num_blocks++;
    }

    LaunchDims dims;
    set_sycl_dim<BlockDim>(dims.group, num_blocks);


    // privatize data, so we can mess with the segments
    using data_t        = camp::decay<Data>;
    data_t private_data = data;

    // Get original segment
    auto& segment = camp::get<ArgumentId>(private_data.segment_tuple);

    // restrict to first tile
    segment = segment.slice(0, chunk_size);


    LaunchDims enclosed_dims =
        enclosed_stmts_t::calculateDimensions(private_data);

    return dims.max(enclosed_dims);
  }
};

/*!
 * A specialized RAJA::kernel sycl_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 *
 */
template<typename Data,
         camp::idx_t ArgumentId,
         camp::idx_t chunk_size,
         int ThreadDim,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<Data,
                             statement::Tile<ArgumentId,
                                             RAJA::tile_fixed<chunk_size>,
                                             sycl_local_012_direct<ThreadDim>,
                                             EnclosedStmts...>,
                             Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = SyclStatementListExecutor<Data, stmt_list_t, Types>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

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
    diff_t i   = item.get_local_id(ThreadDim) * chunk_size;

    // execute enclosed statements if any thread will
    // but mask off threads without work
    bool have_work = i < len;

    // Assign our new tiled segment
    diff_t slice_size = have_work ? chunk_size : 0;
    segment           = orig_segment.slice(i, slice_size);

    // execute enclosed statements
    enclosed_stmts_t::exec(data, item, thread_active && have_work);

    // Set range back to original values
    segment = orig_segment;
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {

    // Compute how many blocks
    diff_t len         = segment_length<ArgumentId>(data);
    diff_t num_threads = len / chunk_size;
    if (num_threads * chunk_size < len)
    {
      num_threads++;
    }

    LaunchDims dims;
    set_sycl_dim<ThreadDim>(dims.local, num_threads);
    set_sycl_dim<ThreadDim>(dims.min_locals, num_threads);

    // privatize data, so we can mess with the segments
    using data_t        = camp::decay<Data>;
    data_t private_data = data;

    // Get original segment
    auto& segment = camp::get<ArgumentId>(private_data.segment_tuple);

    // restrict to first tile
    segment = segment.slice(0, chunk_size);


    LaunchDims enclosed_dims =
        enclosed_stmts_t::calculateDimensions(private_data);

    return (dims.max(enclosed_dims));
  }
};

/*!
 * A specialized RAJA::kernel sycl_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 *
 */
template<typename Data,
         camp::idx_t ArgumentId,
         camp::idx_t chunk_size,
         int ThreadDim,
         typename... EnclosedStmts,
         typename Types>
struct SyclStatementExecutor<Data,
                             statement::Tile<ArgumentId,
                                             RAJA::tile_fixed<chunk_size>,
                                             sycl_local_012_loop<ThreadDim>,
                                             EnclosedStmts...>,
                             Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = SyclStatementListExecutor<Data, stmt_list_t, Types>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

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
    diff_t len      = segment_length<ArgumentId>(data);
    diff_t i_init   = item.get_local_id(ThreadDim) * chunk_size;
    diff_t i_stride = item.get_group_range(ThreadDim) * chunk_size;

    // Iterate through grid stride of chunks
    for (diff_t ii = 0; ii < len; ii += i_stride)
    {
      diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      bool have_work = i < len;

      // Assign our new tiled segment
      diff_t slice_size = have_work ? chunk_size : 0;
      segment           = orig_segment.slice(i, slice_size);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active && have_work);
    }

    // Set range back to original values
    segment = orig_segment;
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {

    // Compute how many blocks
    diff_t len         = segment_length<ArgumentId>(data);
    diff_t num_threads = len / chunk_size;
    if (num_threads * chunk_size < len)
    {
      num_threads++;
    }
    num_threads = std::max(num_threads, (diff_t)1);

    LaunchDims dims;
    set_sycl_dim<ThreadDim>(dims.local, num_threads);
    set_sycl_dim<ThreadDim>(dims.min_locals, 1);

    // privatize data, so we can mess with the segments
    using data_t        = camp::decay<Data>;
    data_t private_data = data;

    // Get original segment
    auto& segment = camp::get<ArgumentId>(private_data.segment_tuple);

    // restrict to first tile
    segment = segment.slice(0, chunk_size);


    LaunchDims enclosed_dims =
        enclosed_stmts_t::calculateDimensions(private_data);

    return (dims.max(enclosed_dims));
  }
};


}  // end namespace internal
}  // end namespace RAJA

#endif  // RAJA_ENABLE_SYCL
#endif  /* RAJA_policy_sycl_kernel_Tile_HPP */
