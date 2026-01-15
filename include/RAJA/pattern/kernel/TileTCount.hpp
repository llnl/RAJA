/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for tile wrapper and iterator.
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

#ifndef RAJA_pattern_kernel_TileTCount_HPP
#define RAJA_pattern_kernel_TileTCount_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{
namespace statement
{


/*!
 * A RAJA::kernel statement that implements a tiling (or blocking) loop.
 * Assigns the tile index to param ParamId
 *
 */
template<camp::idx_t ArgumentId,
         typename ParamId,
         typename TilePolicy,
         typename ExecPolicy,
         typename... EnclosedStmts>
struct TileTCount : public internal::Statement<ExecPolicy, EnclosedStmts...>
{
  static_assert(std::is_base_of<RAJA::expt::detail::ParamBase, ParamId>::value,
                "Inappropriate ParamId, ParamId must be of type "
                "RAJA::Statement::Param< # >");
  using tile_policy_t = TilePolicy;
  using exec_policy_t = ExecPolicy;
};


}  // end namespace statement

namespace internal
{

/*!
 * A generic RAJA::kernel forall_impl tile wrapper for statement::ForTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 */
template<camp::idx_t ArgumentId,
         typename ParamId,
         typename Data,
         typename Types,
         typename... EnclosedStmts>
struct TileTCountWrapper : public GenericWrapper<Data, Types, EnclosedStmts...>
{

  using Base = GenericWrapper<Data, Types, EnclosedStmts...>;
  using Base::Base;
  using privatizer = NestedPrivatizer<TileTCountWrapper>;

  template<typename InSegmentIndexType>
  RAJA_INLINE void operator()(InSegmentIndexType si)
  {
    // Assign the tile's segment to the tuple
    camp::get<ArgumentId>(Base::data.segment_tuple) = si.s;

    // Assign the tile's index
    Base::data.template assign_param<ParamId>(si.i);

    // Execute enclosed statements
    Base::exec();
  }
};

/*!
 * A generic RAJA::kernel forall_impl executor for statement::TileTCount
 *
 *
 */
template<camp::idx_t ArgumentId,
         typename ParamId,
         typename TPol,
         typename EPol,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<
    statement::TileTCount<ArgumentId, ParamId, TPol, EPol, EnclosedStmts...>,
    Types>
{


  template<typename Data>
  static RAJA_INLINE void exec(Data& data)
  {
    // Get the segment we are going to tile
    auto const& segment = camp::get<ArgumentId>(data.segment_tuple);

    // Get the tiling policies chunk size
    auto chunk_size = TPol::chunk_size;

    // Create a tile iterator, needs to survive until the forall is
    // done executing.
    IterableTiler<decltype(segment)> tiled_iterable(segment, chunk_size);

    // Wrap in case forall_impl needs to thread_privatize
    TileTCountWrapper<ArgumentId, ParamId, Data, Types, EnclosedStmts...>
        tile_wrapper(data);

    // Loop over tiles, executing enclosed statement list
    auto r = resources::get_resource<EPol>::type::get_default();
    forall_impl(r, EPol {}, tiled_iterable, tile_wrapper,
                RAJA::expt::get_empty_forall_param_pack());

    // Set range back to original values
    camp::get<ArgumentId>(data.segment_tuple) = tiled_iterable.it;
  }
};


}  // end namespace internal
}  // end namespace RAJA

#endif /* RAJA_pattern_kernel_HPP */
