/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for HIP tiled executors.
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


#ifndef RAJA_policy_hip_kernel_TileTCount_HPP
#define RAJA_policy_hip_kernel_TileTCount_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

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
 * A specialized RAJA::kernel hip_impl executor for statement::TileTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 * Meets all sync requirements
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         camp::idx_t chunk_size,
         typename IndexMapper,
         kernel_sync_requirement sync,
         typename... EnclosedStmts,
         typename Types>
struct HipStatementExecutor<
    Data,
    statement::TileTCount<
        ArgumentId,
        ParamId,
        RAJA::tile_fixed<chunk_size>,
        RAJA::policy::hip::
            hip_indexer<iteration_mapping::DirectUnchecked, sync, IndexMapper>,
        EnclosedStmts...>,
    Types>
    : public HipStatementExecutor<
          Data,
          statement::Tile<
              ArgumentId,
              RAJA::tile_fixed<chunk_size>,
              RAJA::policy::hip::hip_indexer<iteration_mapping::DirectUnchecked,
                                             sync,
                                             IndexMapper>,
              EnclosedStmts...>,
          Types>
{

  using Base = HipStatementExecutor<
      Data,
      statement::Tile<
          ArgumentId,
          RAJA::tile_fixed<chunk_size>,
          RAJA::policy::hip::hip_indexer<iteration_mapping::DirectUnchecked,
                                         sync,
                                         IndexMapper>,
          EnclosedStmts...>,
      Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto& segment = camp::get<ArgumentId>(data.segment_tuple);

    using segment_t = camp::decay<decltype(segment)>;

    // compute trip count
    const diff_t t = IndexMapper::template index<diff_t>();
    const diff_t i = t * static_cast<diff_t>(chunk_size);

    // Keep copy of original segment, so we can restore it
    segment_t orig_segment = segment;

    // Assign our new tiled segment
    segment = orig_segment.slice(i, static_cast<diff_t>(chunk_size));
    data.template assign_param<ParamId>(t);

    // execute enclosed statements
    enclosed_stmts_t::exec(data, thread_active);

    // Set range back to original values
    segment = orig_segment;
  }
};

/*!
 * A specialized RAJA::kernel hip_impl executor for statement::TileTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 * Meets all sync requirements
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         camp::idx_t chunk_size,
         typename IndexMapper,
         kernel_sync_requirement sync,
         typename... EnclosedStmts,
         typename Types>
struct HipStatementExecutor<
    Data,
    statement::TileTCount<
        ArgumentId,
        ParamId,
        RAJA::tile_fixed<chunk_size>,
        RAJA::policy::hip::
            hip_indexer<iteration_mapping::Direct, sync, IndexMapper>,
        EnclosedStmts...>,
    Types>
    : public HipStatementExecutor<
          Data,
          statement::Tile<
              ArgumentId,
              RAJA::tile_fixed<chunk_size>,
              RAJA::policy::hip::
                  hip_indexer<iteration_mapping::Direct, sync, IndexMapper>,
              EnclosedStmts...>,
          Types>
{

  using Base = HipStatementExecutor<
      Data,
      statement::Tile<ArgumentId,
                      RAJA::tile_fixed<chunk_size>,
                      RAJA::policy::hip::hip_indexer<iteration_mapping::Direct,
                                                     sync,
                                                     IndexMapper>,
                      EnclosedStmts...>,
      Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto& segment = camp::get<ArgumentId>(data.segment_tuple);

    using segment_t = camp::decay<decltype(segment)>;

    // compute trip count
    const diff_t len = segment.end() - segment.begin();
    const diff_t t   = IndexMapper::template index<diff_t>();
    const diff_t i   = t * static_cast<diff_t>(chunk_size);

    // execute enclosed statements if any thread will
    // but mask off threads without work
    const bool have_work = (i < len);

    // Keep copy of original segment, so we can restore it
    segment_t orig_segment = segment;

    // Assign our new tiled segment
    segment = orig_segment.slice(i, static_cast<diff_t>(chunk_size));
    data.template assign_param<ParamId>(t);

    // execute enclosed statements
    enclosed_stmts_t::exec(data, thread_active && have_work);

    // Set range back to original values
    segment = orig_segment;
  }
};

/*!
 * A specialized RAJA::kernel hip_impl executor for statement::TileTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 * Meets all sync requirements
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         camp::idx_t chunk_size,
         typename IndexMapper,
         typename... EnclosedStmts,
         typename Types>
struct HipStatementExecutor<
    Data,
    statement::TileTCount<
        ArgumentId,
        ParamId,
        RAJA::tile_fixed<chunk_size>,
        RAJA::policy::hip::hip_indexer<
            iteration_mapping::StridedLoop<named_usage::unspecified>,
            kernel_sync_requirement::sync,
            IndexMapper>,
        EnclosedStmts...>,
    Types>
    : public HipStatementExecutor<
          Data,
          statement::Tile<
              ArgumentId,
              RAJA::tile_fixed<chunk_size>,
              RAJA::policy::hip::hip_indexer<
                  iteration_mapping::StridedLoop<named_usage::unspecified>,
                  kernel_sync_requirement::sync,
                  IndexMapper>,
              EnclosedStmts...>,
          Types>
{

  using Base = HipStatementExecutor<
      Data,
      statement::Tile<
          ArgumentId,
          RAJA::tile_fixed<chunk_size>,
          RAJA::policy::hip::hip_indexer<
              iteration_mapping::StridedLoop<named_usage::unspecified>,
              kernel_sync_requirement::sync,
              IndexMapper>,
          EnclosedStmts...>,
      Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto& segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t        = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    // compute trip count
    const diff_t len      = segment.end() - segment.begin();
    const diff_t t_init   = IndexMapper::template index<diff_t>();
    const diff_t i_init   = t_init * static_cast<diff_t>(chunk_size);
    const diff_t t_stride = IndexMapper::template size<diff_t>();
    const diff_t i_stride = t_stride * static_cast<diff_t>(chunk_size);

    // Iterate through in chunks
    // threads will have the same numbers of iterations
    for (diff_t ii = 0, t = t_init; ii < len; ii += i_stride, t += t_stride)
    {
      const diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      const bool have_work = (i < len);

      // Assign our new tiled segment
      segment = orig_segment.slice(i, static_cast<diff_t>(chunk_size));
      data.template assign_param<ParamId>(t);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }

    // Set range back to original values
    segment = orig_segment;
  }
};

/*!
 * A specialized RAJA::kernel hip_impl executor for statement::TileTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 * Meets no sync requirements
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         camp::idx_t chunk_size,
         typename IndexMapper,
         typename... EnclosedStmts,
         typename Types>
struct HipStatementExecutor<
    Data,
    statement::TileTCount<
        ArgumentId,
        ParamId,
        RAJA::tile_fixed<chunk_size>,
        RAJA::policy::hip::hip_indexer<
            iteration_mapping::StridedLoop<named_usage::unspecified>,
            kernel_sync_requirement::none,
            IndexMapper>,
        EnclosedStmts...>,
    Types>
    : public HipStatementExecutor<
          Data,
          statement::Tile<
              ArgumentId,
              RAJA::tile_fixed<chunk_size>,
              RAJA::policy::hip::hip_indexer<
                  iteration_mapping::StridedLoop<named_usage::unspecified>,
                  kernel_sync_requirement::none,
                  IndexMapper>,
              EnclosedStmts...>,
          Types>
{

  using Base = HipStatementExecutor<
      Data,
      statement::Tile<
          ArgumentId,
          RAJA::tile_fixed<chunk_size>,
          RAJA::policy::hip::hip_indexer<
              iteration_mapping::StridedLoop<named_usage::unspecified>,
              kernel_sync_requirement::none,
              IndexMapper>,
          EnclosedStmts...>,
      Types>;

  using typename Base::diff_t;
  using typename Base::enclosed_stmts_t;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto& segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t        = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    // compute trip count
    const diff_t len      = segment.end() - segment.begin();
    const diff_t t_init   = IndexMapper::template index<diff_t>();
    const diff_t i_init   = t_init * static_cast<diff_t>(chunk_size);
    const diff_t t_stride = IndexMapper::template size<diff_t>();
    const diff_t i_stride = t_stride * static_cast<diff_t>(chunk_size);

    // Iterate through one at a time
    // threads will have the different numbers of iterations
    for (diff_t i = i_init, t = t_init; i < len; i += i_stride, t += t_stride)
    {

      // Assign our new tiled segment
      segment = orig_segment.slice(i, static_cast<diff_t>(chunk_size));
      data.template assign_param<ParamId>(t);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }

    // Set range back to original values
    segment = orig_segment;
  }
};

/*!
 * A specialized RAJA::kernel hip_impl executor for statement::TileTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 */
template<typename Data,
         camp::idx_t ArgumentId,
         typename ParamId,
         typename TPol,
         typename... EnclosedStmts,
         typename Types>
struct HipStatementExecutor<
    Data,
    statement::
        TileTCount<ArgumentId, ParamId, TPol, seq_exec, EnclosedStmts...>,
    Types>
    : HipStatementExecutor<
          Data,
          statement::TileTCount<
              ArgumentId,
              ParamId,
              TPol,
              RAJA::policy::hip::hip_indexer<
                  iteration_mapping::StridedLoop<named_usage::unspecified>,
                  kernel_sync_requirement::none,
                  hip::IndexGlobal<named_dim::x,
                                   named_usage::ignored,
                                   named_usage::ignored>>,
              EnclosedStmts...>,
          Types>
{};

}  // end namespace internal
}  // end namespace RAJA

#endif  // RAJA_ENABLE_HIP
#endif  /* RAJA_policy_hip_kernel_TileTCount_HPP */
