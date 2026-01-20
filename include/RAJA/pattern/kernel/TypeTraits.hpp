/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing type traits needed by kernel implementation
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

#ifndef RAJA_pattern_kernel_type_traits_HPP
#define RAJA_pattern_kernel_type_traits_HPP

#include "RAJA/pattern/params/reducer.hpp"
#include "RAJA/pattern/kernel/For.hpp"
#include "RAJA/pattern/kernel/ForICount.hpp"
#include "RAJA/pattern/kernel/Tile.hpp"
#include "RAJA/pattern/kernel/TileTCount.hpp"

namespace RAJA
{
namespace internal
{

template<typename T>
struct loop_data_has_reducers : std::false_type
{};

template<typename SegmentTuple,
         typename ParamTuple,
         typename Resource,
         typename... Bodies>
struct loop_data_has_reducers<
    LoopData<SegmentTuple, ParamTuple, Resource, Bodies...>>
    : RAJA::expt::tuple_contains_Reducers<ParamTuple>
{};

template<typename T>
struct is_wrapper_with_reducers : std::false_type
{};

template<camp::idx_t ArgumentId,
         typename Data,
         typename Types,
         typename... EnclosedStmts>
struct is_wrapper_with_reducers<
    ForWrapper<ArgumentId, Data, Types, EnclosedStmts...>>
    : loop_data_has_reducers<camp::decay<Data>>
{};

template<camp::idx_t ArgumentId,
         typename ParamId,
         typename Data,
         typename Types,
         typename... EnclosedStmts>
struct is_wrapper_with_reducers<
    ForICountWrapper<ArgumentId, ParamId, Data, Types, EnclosedStmts...>>
    : loop_data_has_reducers<camp::decay<Data>>
{};

template<camp::idx_t ArgumentId,
         typename Data,
         typename Types,
         typename... EnclosedStmts>
struct is_wrapper_with_reducers<
    TileWrapper<ArgumentId, Data, Types, EnclosedStmts...>>
    : loop_data_has_reducers<camp::decay<Data>>
{};

template<camp::idx_t ArgumentId,
         typename Data,
         typename Types,
         typename... EnclosedStmts>
struct is_wrapper_with_reducers<
    TileTCountWrapper<ArgumentId, Data, Types, EnclosedStmts...>>
    : loop_data_has_reducers<camp::decay<Data>>
{};

}  // namespace internal
}  // namespace RAJA

#endif
