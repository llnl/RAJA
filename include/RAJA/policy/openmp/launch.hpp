/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::launch::openmp
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

#ifndef RAJA_pattern_launch_openmp_HPP
#define RAJA_pattern_launch_openmp_HPP

#include "RAJA/pattern/launch/launch_core.hpp"
#include "RAJA/policy/openmp/policy.hpp"

namespace RAJA
{

template<>
struct LaunchExecute<RAJA::omp_launch_t>
{


  template<typename ReduceParams, typename BODY>
  static concepts::enable_if_t<
      resources::EventProxy<resources::Resource>,
      RAJA::expt::type_traits::is_ForallParamPack<ReduceParams>>
  exec(RAJA::resources::Resource res,
       LaunchParams const& launch_params,
       BODY const& body,
       ReduceParams& f_params)
  {
    using RAJA::internal::thread_privatize;
    constexpr bool has_reducers =
        !RAJA::expt::type_traits::is_ForallParamPack_empty<ReduceParams>::value;
    using EXEC_POL = RAJA::omp_launch_t;
    EXEC_POL pol {};
    using BodyType = decltype(thread_privatize(body));

    auto parallel_section = [&](ReduceParams& f_params, auto func) {
      LaunchContext ctx;
      auto loop_body = thread_privatize(body);
      static_assert(std::is_invocable<decltype(func), ReduceParams&, BodyType&,
                                      LaunchContext&>::value,
                    "Internal RAJA error: Check the parallel kernel passed to "
                    "OpenMP Parallel section in openmp/launch.hpp");

      ctx.shared_mem_ptr = (char*)malloc(launch_params.shared_mem_size);

      func(f_params, loop_body, ctx);

      free(ctx.shared_mem_ptr);
      ctx.shared_mem_ptr = nullptr;
    };

    // Init reducers if present
    expt::ParamMultiplexer::parampack_init(pol, f_params);

    // reducer object must be named f_params as expected by macro below
    if constexpr (has_reducers)
    {
      RAJA_OMP_DECLARE_REDUCTION_COMBINE;
#pragma omp parallel reduction(combine : f_params)
      {
        // This "extra lambda" has to be declared within the scope of the OpenMP
        // pragma so that the reduction parameter pack it operates on is the
        // version tracked by the combine OpenMP syntax
        auto parallel_kernel = [&](ReduceParams& f_params, BodyType& body,
                                   LaunchContext& ctx) {
          expt::invoke_body(f_params, body.get_priv(), ctx);
        };
        parallel_section(f_params, parallel_kernel);
      }
    }
    else
    {
      RAJA::region<RAJA::omp_parallel_region>([&]() {
        auto parallel_kernel = [&](ReduceParams&, BodyType& body,
                                   LaunchContext& ctx) {
          body.get_priv()(ctx);
        };
        parallel_section(f_params, parallel_kernel);
      });
    }
    // Resolve reducers if present
    expt::ParamMultiplexer::parampack_resolve(pol, f_params);

    return resources::EventProxy<resources::Resource>(res);
  }
};

template<typename SEGMENT>
struct LoopExecute<omp_parallel_for_exec, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {

    int len = segment.end() - segment.begin();
    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);
#pragma omp for
      for (int i = 0; i < len; i++)
      {

        loop_body.get_priv()(*(segment.begin() + i));
      }
    });
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for
      for (int j = 0; j < len1; j++)
      {
        for (int i = 0; i < len0; i++)
        {

          loop_body.get_priv()(*(segment0.begin() + i),
                               *(segment1.begin() + j));
        }
      }
    });
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for
      for (int k = 0; k < len2; k++)
      {
        for (int j = 0; j < len1; j++)
        {
          for (int i = 0; i < len0; i++)
          {
            loop_body.get_priv()(*(segment0.begin() + i),
                                 *(segment1.begin() + j),
                                 *(segment2.begin() + k));
          }
        }
      }
    });
  }
};

template<typename SEGMENT>
struct LoopExecute<omp_for_exec, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {

    int len = segment.end() - segment.begin();
#pragma omp for
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

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp for
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

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp for
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

//
// Return local index
//
template<typename SEGMENT>
struct LoopICountExecute<omp_for_exec, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {

    int len = segment.end() - segment.begin();

#pragma omp for
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

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp for
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

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp for
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

// policy for perfectly nested loops
struct omp_parallel_nested_for_exec;

template<typename SEGMENT>
struct LoopExecute<omp_parallel_nested_for_exec, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for RAJA_COLLAPSE(2)
      for (int j = 0; j < len1; j++)
      {
        for (int i = 0; i < len0; i++)
        {

          loop_body.get_priv()(*(segment0.begin() + i),
                               *(segment1.begin() + j));
        }
      }
    });
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for RAJA_COLLAPSE(3)
      for (int k = 0; k < len2; k++)
      {
        for (int j = 0; j < len1; j++)
        {
          for (int i = 0; i < len0; i++)
          {
            loop_body.get_priv()(*(segment0.begin() + i),
                                 *(segment1.begin() + j),
                                 *(segment2.begin() + k));
          }
        }
      }
    });
  }
};

// Return local index
template<typename SEGMENT>
struct LoopICountExecute<omp_parallel_nested_for_exec, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for RAJA_COLLAPSE(2)
      for (int j = 0; j < len1; j++)
      {
        for (int i = 0; i < len0; i++)
        {

          loop_body.get_priv()(*(segment0.begin() + i), *(segment1.begin() + j),
                               i, j);
        }
      }
    });
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for RAJA_COLLAPSE(3)
      for (int k = 0; k < len2; k++)
      {
        for (int j = 0; j < len1; j++)
        {
          for (int i = 0; i < len0; i++)
          {
            loop_body.get_priv()(*(segment0.begin() + i),
                                 *(segment1.begin() + j),
                                 *(segment2.begin() + k), i, j, k);
          }
        }
      }
    });
  }
};

template<typename SEGMENT>
struct TileExecute<omp_parallel_for_exec, SEGMENT>
{

  template<typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {

    int len = segment.end() - segment.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for
      for (int i = 0; i < len; i += tile_size)
      {
        loop_body.get_priv()(segment.slice(i, tile_size));
      }
    });
  }
};

template<typename SEGMENT>
struct TileTCountExecute<omp_parallel_for_exec, SEGMENT>
{

  template<typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {

    const int len      = segment.end() - segment.begin();
    const int numTiles = (len - 1) / tile_size + 1;

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp parallel for
      for (int i = 0; i < numTiles; i++)
      {
        const int i_tile_size = i * tile_size;
        loop_body.get_priv()(segment.slice(i_tile_size, tile_size), i);
      }
    });
  }
};

template<typename SEGMENT>
struct TileExecute<omp_for_exec, SEGMENT>
{

  template<typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {

    int len = segment.end() - segment.begin();
#pragma omp for
    for (int i = 0; i < len; i += tile_size)
    {
      body(segment.slice(i, tile_size));
    }
  }
};

template<typename SEGMENT>
struct TileTCountExecute<omp_for_exec, SEGMENT>
{

  template<typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {

    const int len      = segment.end() - segment.begin();
    const int numTiles = (len - 1) / tile_size + 1;

#pragma omp for
    for (int i = 0; i < numTiles; i++)
    {
      const int i_tile_size = i * tile_size;
      body(segment.slice(i_tile_size, tile_size), i);
    }
  }
};

}  // namespace RAJA
#endif
