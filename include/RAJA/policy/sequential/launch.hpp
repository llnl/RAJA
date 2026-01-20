/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::Teams::seq
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

#ifndef RAJA_pattern_launch_sequential_HPP
#define RAJA_pattern_launch_sequential_HPP

#include "RAJA/pattern/launch/launch_core.hpp"
#include "RAJA/policy/sequential/policy.hpp"
#include "RAJA/pattern/params/forall.hpp"

namespace RAJA
{

template<>
struct LaunchExecute<RAJA::null_launch_t>
{
  template<typename BODY>
  static void exec(LaunchContext const& RAJA_UNUSED_ARG(ctx),
                   BODY const& RAJA_UNUSED_ARG(body))
  {
    RAJA_ABORT_OR_THROW("NULL Launch");
  }
};

template<>
struct LaunchExecute<RAJA::seq_launch_t>
{
  template<typename BODY, typename ReduceParams>
  static concepts::enable_if_t<
      resources::EventProxy<resources::Resource>,
      RAJA::expt::type_traits::is_ForallParamPack<ReduceParams>>
  exec(RAJA::resources::Resource res,
       LaunchParams const& launch_params,
       BODY const& body,
       ReduceParams& launch_reducers)
  {
    using EXEC_POL = RAJA::seq_exec;
    EXEC_POL pol {};
    constexpr bool is_parampack_empty =
        RAJA::expt::type_traits::is_ForallParamPack_empty<ReduceParams>::value;
    if constexpr (!is_parampack_empty)
    {
      expt::ParamMultiplexer::parampack_init(pol, launch_reducers);
    }

    LaunchContext ctx;
    char* kernel_local_mem = new char[launch_params.shared_mem_size];
    ctx.shared_mem_ptr     = kernel_local_mem;

    if constexpr (!is_parampack_empty)
    {
      expt::invoke_body(launch_reducers, body, ctx);
    }
    else
    {
      body(ctx);
    }

    delete[] kernel_local_mem;
    ctx.shared_mem_ptr = nullptr;
    if constexpr (!is_parampack_empty)
    {
      expt::ParamMultiplexer::parampack_resolve(pol, launch_reducers);
    }

    return resources::EventProxy<resources::Resource>(res);
  }
};

template<typename SEGMENT>
struct LoopExecute<seq_exec, SEGMENT>
{

  RAJA_SUPPRESS_HD_WARN
  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(SEGMENT const& segment,
                                                BODY const& body)
  {

    const int len = segment.end() - segment.begin();
    for (int i = 0; i < len; i++)
    {

      body(*(segment.begin() + i));
    }
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {

    const int len = segment.end() - segment.begin();
    for (int i = 0; i < len; i++)
    {
      body(*(segment.begin() + i));
    }
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {

    // block stride loop
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int j = 0; j < len1; j++)
    {
      for (int i = 0; i < len0; i++)
      {

        body(*(segment0.begin() + i), *(segment1.begin() + j));
      }
    }
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {

    // block stride loop
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int k = 0; k < len2; k++)
    {
      for (int j = 0; j < len1; j++)
      {
        for (int i = 0; i < len0; i++)
        {
          body(*(segment0.begin() + i), *(segment1.begin() + j),
               *(segment2.begin() + k));
        }
      }
    }
  }
};

template<typename SEGMENT>
struct LoopICountExecute<seq_exec, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {
    const int len = segment.end() - segment.begin();
    for (int i = 0; i < len; i++)
    {
      body(*(segment.begin() + i), i);
    }
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {

    // block stride loop
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int j = 0; j < len1; j++)
    {
      for (int i = 0; i < len0; i++)
      {

        body(*(segment0.begin() + i), *(segment1.begin() + j), i, j);
      }
    }
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {

    // block stride loop
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int k = 0; k < len2; k++)
    {
      for (int j = 0; j < len1; j++)
      {
        for (int i = 0; i < len0; i++)
        {
          body(*(segment0.begin() + i), *(segment1.begin() + j),
               *(segment2.begin() + k), i, j, k);
        }
      }
    }
  }
};

// Tile Execute + variants

template<typename SEGMENT>
struct TileExecute<seq_exec, SEGMENT>
{

  template<typename TILE_T, typename BODY>
  static RAJA_HOST_DEVICE RAJA_INLINE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = 0; tx < len; tx += tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};

template<typename SEGMENT>
struct TileTCountExecute<seq_exec, SEGMENT>
{

  template<typename TILE_T, typename BODY>
  static RAJA_HOST_DEVICE RAJA_INLINE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = 0, bx = 0; tx < len; tx += tile_size, bx++)
    {
      body(segment.slice(tx, tile_size), bx);
    }
  }
};

}  // namespace RAJA
#endif
