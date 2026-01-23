/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::launch::sycl
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

#ifndef RAJA_pattern_launch_sycl_HPP
#define RAJA_pattern_launch_sycl_HPP

#include "RAJA/pattern/launch/launch_core.hpp"
#include "RAJA/pattern/detail/privatizer.hpp"
#include "RAJA/policy/sycl/policy.hpp"
#include "RAJA/policy/sycl/MemUtils_SYCL.hpp"
//#include "RAJA/policy/sycl/raja_syclerrchk.hpp"
#include "RAJA/util/resource.hpp"

namespace RAJA
{

template<bool async>
struct LaunchExecute<RAJA::sycl_launch_t<async, 0>>
{
  template<typename LoopBody, typename ReduceParams>
  static concepts::enable_if_t<
      resources::EventProxy<resources::Resource>,
      RAJA::expt::type_traits::is_ForallParamPack<ReduceParams>>
  exec(RAJA::resources::Resource res,
       const LaunchParams& launch_params,
       LoopBody&& loop_body,
       ReduceParams launch_reducers)
  {
    using EXEC_POL  = RAJA::sycl_launch_t<async, 0>;
    using LOOP_BODY = camp::decay<LoopBody>;
    // Deduce at compile time if lbody is trivially constructible and if user
    // has supplied parameters.  These will be used to determine which sycl
    // launch to configure below.
    constexpr bool is_parampack_empty =
        RAJA::expt::type_traits::is_ForallParamPack_empty<ReduceParams>::value;
    constexpr bool is_lbody_trivially_copyable =
        std::is_trivially_copyable<LOOP_BODY>::value;
    EXEC_POL pol {};

    /*Get the queue from concrete resource */
    ::sycl::queue* q = res.get<camp::resources::Sycl>().get_queue();

    if constexpr (!is_parampack_empty)
    {
      RAJA::expt::ParamMultiplexer::parampack_init(pol, launch_reducers);
    }

    //
    // Compute the number of blocks and threads
    //
    const ::sycl::range<3> blockSize(launch_params.threads.value[2],
                                     launch_params.threads.value[1],
                                     launch_params.threads.value[0]);

    const ::sycl::range<3> gridSize(
        launch_params.threads.value[2] * launch_params.teams.value[2],
        launch_params.threads.value[1] * launch_params.teams.value[1],
        launch_params.threads.value[0] * launch_params.teams.value[0]);

    // Only launch kernel if we have something to iterate over
    constexpr int zero = 0;
    if (launch_params.threads.value[0] <= zero ||
        launch_params.threads.value[1] <= zero ||
        launch_params.threads.value[2] <= zero ||
        launch_params.teams.value[0] <= zero ||
        launch_params.teams.value[1] <= zero ||
        launch_params.teams.value[2] <= zero)
    {
      return resources::EventProxy<resources::Resource>(res);
    }


    using LOOP_BODY  = camp::decay<LoopBody>;
    LOOP_BODY* lbody = nullptr;
    //
    // Kernel body is nontrivially copyable, create space on device and copy
    // to Workaround until "is_device_copyable" is supported
    //
    if constexpr (!is_lbody_trivially_copyable)
    {
      lbody = (LOOP_BODY*)::sycl::malloc_device(sizeof(LOOP_BODY), *q);
      q->memcpy(lbody, &loop_body, sizeof(LOOP_BODY)).wait();
    }
    // Both the parallel_for call, combinations, and resolution are all
    // unique to the parameter case, so we make a constexpr branch here
    if constexpr (!is_parampack_empty)
    {
      auto combiner = [](ReduceParams x, ReduceParams y) {
        RAJA::expt::ParamMultiplexer::parampack_combine(EXEC_POL {}, x, y);
        return x;
      };

      ReduceParams* res = ::sycl::malloc_shared<ReduceParams>(1, *q);
      RAJA::expt::ParamMultiplexer::parampack_init(pol, *res);
      auto reduction = ::sycl::reduction(res, launch_reducers, combiner);

      q->submit([&](::sycl::handler& h) {
         auto s_vec =
             ::sycl::local_accessor<char, 1>(launch_params.shared_mem_size, h);

         h.parallel_for(
             ::sycl::nd_range<3>(gridSize, blockSize), reduction,
             [=](::sycl::nd_item<3> itm, auto& red) {
               LaunchContext ctx;
               ctx.itm = &itm;

               // Point to shared memory
               ctx.shared_mem_ptr =
                   s_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

               ReduceParams fp;
               RAJA::expt::ParamMultiplexer::parampack_init(pol, fp);
               if constexpr (is_lbody_trivially_copyable)
               {
                 RAJA::expt::invoke_body(fp, loop_body, ctx);
               }
               else
               {
                 RAJA::expt::invoke_body(fp, *lbody, ctx);
               }

               red.combine(fp);
             });
       }).wait();  // Need to wait for completion to free memory

      RAJA::expt::ParamMultiplexer::parampack_combine(pol, launch_reducers,
                                                      *res);
      ::sycl::free(res, *q);
      ::sycl::free(lbody, *q);
      RAJA::expt::ParamMultiplexer::parampack_resolve(pol, launch_reducers);
    }
    else
    {
      q->submit([&](::sycl::handler& h) {
        auto s_vec =
            ::sycl::local_accessor<char, 1>(launch_params.shared_mem_size, h);

        h.parallel_for(
            ::sycl::nd_range<3>(gridSize, blockSize),
            [=](::sycl::nd_item<3> itm) {
              LaunchContext ctx;
              ctx.itm = &itm;

              // Point to shared memory
              ctx.shared_mem_ptr =
                  s_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
              if constexpr (is_lbody_trivially_copyable)
              {
                loop_body(ctx);
              }
              else
              {
                (*lbody)(ctx);
              }
            });
      });

      if (!async)
      {
        q->wait();
      }
    }

    return resources::EventProxy<resources::Resource>(res);
  }
};

/*
   SYCL global thread mapping
*/
template<int... DIM>
struct sycl_global_item;

using sycl_global_item_0 = sycl_global_item<0>;
using sycl_global_item_1 = sycl_global_item<1>;
using sycl_global_item_2 = sycl_global_item<2>;

template<typename SEGMENT, int DIM>
struct LoopExecute<sycl_global_item<DIM>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = ctx.itm->get_group(DIM) * ctx.itm->get_local_range(DIM) +
                     ctx.itm->get_local_id(DIM);

      if (tx < len) body(*(segment.begin() + tx));
    }
  }
};

using sycl_global_item_01 = sycl_global_item<0, 1>;
using sycl_global_item_02 = sycl_global_item<0, 2>;
using sycl_global_item_10 = sycl_global_item<1, 0>;
using sycl_global_item_12 = sycl_global_item<1, 2>;
using sycl_global_item_20 = sycl_global_item<2, 0>;
using sycl_global_item_21 = sycl_global_item<2, 1>;

template<typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_global_item<DIM0, DIM1>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment0,
                                           SEGMENT const& segment1,
                                           BODY const& body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = ctx.itm->get_group(DIM0) * ctx.itm->get_local_range(DIM0) +
                     ctx.itm->get_local_id(DIM0);

      const int ty = ctx.itm->get_group(DIM1) * ctx.itm->get_local_range(DIM1) +
                     ctx.itm->get_local_id(DIM1);


      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty));
    }
  }
};

using sycl_global_item_012 = sycl_global_item<0, 1, 2>;
using sycl_global_item_021 = sycl_global_item<0, 2, 1>;
using sycl_global_item_102 = sycl_global_item<1, 0, 2>;
using sycl_global_item_120 = sycl_global_item<1, 2, 0>;
using sycl_global_item_201 = sycl_global_item<2, 0, 1>;
using sycl_global_item_210 = sycl_global_item<2, 1, 0>;

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_global_item<DIM0, DIM1, DIM2>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment0,
                                           SEGMENT const& segment1,
                                           SEGMENT const& segment2,
                                           BODY const& body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = ctx.itm->get_group(DIM0) * ctx.itm->get_local_range(DIM0) +
                     ctx.itm->get_local_id(DIM0);

      const int ty = ctx.itm->get_group(DIM1) * ctx.itm->get_local_range(DIM1) +
                     ctx.itm->get_local_id(DIM1);

      const int tz = ctx.itm->get_group(DIM2) * ctx.itm->get_local_range(DIM2) +
                     ctx.itm->get_local_id(DIM2);

      if (tx < len0 && ty < len1 && tz < len2)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty),
             *(segment1.begin() + ty));
    }
  }
};

/*
Reshape threads in a block into a 1D iteration space
*/
template<int... dim>
struct sycl_flatten_group_local_direct
{};

using sycl_flatten_group_local_01_direct =
    sycl_flatten_group_local_direct<0, 1>;
using sycl_flatten_group_local_02_direct =
    sycl_flatten_group_local_direct<0, 2>;
using sycl_flatten_group_local_10_direct =
    sycl_flatten_group_local_direct<1, 0>;
using sycl_flatten_group_local_12_direct =
    sycl_flatten_group_local_direct<1, 2>;
using sycl_flatten_group_local_20_direct =
    sycl_flatten_group_local_direct<2, 0>;
using sycl_flatten_group_local_21_direct =
    sycl_flatten_group_local_direct<2, 1>;

using sycl_flatten_group_local_012_direct =
    sycl_flatten_group_local_direct<0, 1, 2>;
using sycl_flatten_group_local_021_direct =
    sycl_flatten_group_local_direct<0, 2, 1>;
using sycl_flatten_group_local_102_direct =
    sycl_flatten_group_local_direct<1, 0, 2>;
using sycl_flatten_group_local_120_direct =
    sycl_flatten_group_local_direct<1, 2, 0>;
using sycl_flatten_group_local_201_direct =
    sycl_flatten_group_local_direct<2, 0, 1>;
using sycl_flatten_group_local_210_direct =
    sycl_flatten_group_local_direct<2, 1, 0>;

template<int... dim>
struct sycl_flatten_group_local_loop
{};

using sycl_flatten_group_local_01_loop = sycl_flatten_group_local_loop<0, 1>;
using sycl_flatten_group_local_02_loop = sycl_flatten_group_local_loop<0, 2>;
using sycl_flatten_group_local_10_loop = sycl_flatten_group_local_loop<1, 0>;
using sycl_flatten_group_local_12_loop = sycl_flatten_group_local_loop<1, 2>;
using sycl_flatten_group_local_20_loop = sycl_flatten_group_local_loop<2, 0>;
using sycl_flatten_group_local_21_loop = sycl_flatten_group_local_loop<2, 1>;

using sycl_flatten_group_local_012_loop =
    sycl_flatten_group_local_loop<0, 1, 2>;
using sycl_flatten_group_local_021_loop =
    sycl_flatten_group_local_loop<0, 2, 1>;
using sycl_flatten_group_local_102_loop =
    sycl_flatten_group_local_loop<1, 0, 2>;
using sycl_flatten_group_local_120_loop =
    sycl_flatten_group_local_loop<1, 2, 0>;
using sycl_flatten_group_local_201_loop =
    sycl_flatten_group_local_loop<2, 0, 1>;
using sycl_flatten_group_local_210_loop =
    sycl_flatten_group_local_loop<2, 1, 0>;

template<typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_flatten_group_local_direct<DIM0, DIM1>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx  = ctx.itm->get_local_id(DIM0);
      const int ty  = ctx.itm->get_local_id(DIM1);
      const int bx  = ctx.itm->get_local_range(DIM0);
      const int tid = tx + bx * ty;

      if (tid < len) body(*(segment.begin() + tid));
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_flatten_group_local_loop<DIM0, DIM1>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {
    const int len = segment.end() - segment.begin();

    const int tx = ctx.itm->get_local_id(DIM0);
    const int ty = ctx.itm->get_local_id(DIM1);

    const int bx = ctx.itm->get_local_range(DIM0);
    const int by = ctx.itm->get_local_range(DIM1);

    for (int tid = tx + bx * ty; tid < len; tid += bx * by)
    {
      body(*(segment.begin() + tid));
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_flatten_group_local_direct<DIM0, DIM1, DIM2>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {
    const int len = segment.end() - segment.begin();
    {
      const int tx = ctx.itm->get_local_id(DIM0);
      const int ty = ctx.itm->get_local_id(DIM1);
      const int tz = ctx.itm->get_local_id(DIM2);
      const int bx = ctx.itm->get_local_range(DIM0);
      const int by = ctx.itm->get_local_range(DIM1);

      const int tid = tx + bx * (ty + by * tz);

      if (tid < len) body(*(segment.begin() + tid));
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_flatten_group_local_loop<DIM0, DIM1, DIM2>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {
    const int len = segment.end() - segment.begin();

    const int tx = ctx.itm->get_local_id(DIM0);
    const int ty = ctx.itm->get_local_id(DIM1);
    const int tz = ctx.itm->get_local_id(DIM2);
    const int bx = ctx.itm->get_local_range(DIM0);
    const int by = ctx.itm->get_local_range(DIM1);
    const int bz = ctx.itm->get_local_range(DIM2);

    for (int tid = tx + bx * (ty + by * tz); tid < len; tid += bx * by * bz)
    {
      body(*(segment.begin() + tid));
    }
  }
};

/*
  SYCL thread loops with block strides
*/
template<typename SEGMENT, int DIM>
struct LoopExecute<sycl_local_012_loop<DIM>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = ctx.itm->get_local_id(DIM); tx < len;
         tx += ctx.itm->get_local_range(DIM))
    {
      body(*(segment.begin() + tx));
    }
  }
};

/*
  SYCL thread direct mappings
*/
template<typename SEGMENT, int DIM>
struct LoopExecute<sycl_local_012_direct<DIM>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = ctx.itm->get_local_id(DIM);
      if (tx < len) body(*(segment.begin() + tx));
    }
  }
};

/*
  SYCL block loops with grid strides
*/
template<typename SEGMENT, int DIM>
struct LoopExecute<sycl_group_012_loop<DIM>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = ctx.itm->get_group(DIM); bx < len;
         bx += ctx.itm->get_group_range(DIM))
    {
      body(*(segment.begin() + bx));
    }
  }
};

/*
  SYCL block direct mappings
*/
template<typename SEGMENT, int DIM>
struct LoopExecute<sycl_group_012_direct<DIM>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int bx = ctx.itm->get_group(DIM);
      if (bx < len) body(*(segment.begin() + bx));
    }
  }
};

/*
  SYCL thread loops with block strides + Return Index
*/
template<typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_local_012_loop<DIM>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = ctx.itm->get_local_id(DIM); tx < len;
         tx += ctx.itm->get_local_range(DIM))
    {
      body(*(segment.begin() + tx), tx);
    }
  }
};

/*
  SYCL thread direct mappings
*/
template<typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_local_012_direct<DIM>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = ctx.itm->get_local_id(DIM);
      if (tx < len) body(*(segment.begin() + tx), tx);
    }
  }
};

/*
  SYCL block loops with grid strides
*/
template<typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_group_012_loop<DIM>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = ctx.itm->get_group(DIM); bx < len;
         bx += ctx.itm->get_group_range(DIM))
    {
      body(*(segment.begin() + bx), bx);
    }
  }
};

/*
  SYCL block direct mappings
*/
template<typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_group_012_direct<DIM>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int bx = ctx.itm->get_group(DIM);
      if (bx < len) body(*(segment.begin() + bx), bx);
    }
  }
};

// perfectly nested sycl direct policies
using sycl_group_01_nested_direct = sycl_group_012_direct<0, 1>;
using sycl_group_02_nested_direct = sycl_group_012_direct<0, 2>;
using sycl_group_10_nested_direct = sycl_group_012_direct<1, 0>;
using sycl_group_12_nested_direct = sycl_group_012_direct<1, 2>;
using sycl_group_20_nested_direct = sycl_group_012_direct<2, 0>;
using sycl_group_21_nested_direct = sycl_group_012_direct<2, 1>;

using sycl_group_012_nested_direct = sycl_group_012_direct<0, 1, 2>;
using sycl_group_021_nested_direct = sycl_group_012_direct<0, 2, 1>;
using sycl_group_102_nested_direct = sycl_group_012_direct<1, 0, 2>;
using sycl_group_120_nested_direct = sycl_group_012_direct<1, 2, 0>;
using sycl_group_201_nested_direct = sycl_group_012_direct<2, 0, 1>;
using sycl_group_210_nested_direct = sycl_group_012_direct<2, 1, 0>;

template<typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_group_012_direct<DIM0, DIM1>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment0,
                                           SEGMENT const& segment1,
                                           BODY const& body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = ctx.itm->get_group(DIM0);
      const int ty = ctx.itm->get_group(DIM1);
      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty));
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_group_012_direct<DIM0, DIM1, DIM2>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment0,
                                           SEGMENT const& segment1,
                                           SEGMENT const& segment2,
                                           BODY const& body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = ctx.itm->get_group(DIM0);
      const int ty = ctx.itm->get_group(DIM1);
      const int tz = ctx.itm->get_group(DIM2);
      if (tx < len0 && ty < len1 && tz < len2)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty),
             *(segment2.begin() + tz));
    }
  }
};

/*
  Perfectly nested sycl direct policies
  Return local index
*/
template<typename SEGMENT, int DIM0, int DIM1>
struct LoopICountExecute<sycl_group_012_direct<DIM0, DIM1>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment0,
                                           SEGMENT const& segment1,
                                           BODY const& body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = ctx.itm->get_group(DIM0);
      const int ty = ctx.itm->get_group(DIM1);
      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty), tx, ty);
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopICountExecute<sycl_group_012_direct<DIM0, DIM1, DIM2>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment0,
                                           SEGMENT const& segment1,
                                           SEGMENT const& segment2,
                                           BODY const& body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = ctx.itm->get_group(DIM0);
      const int ty = ctx.itm->get_group(DIM1);
      const int tz = ctx.itm->get_group(DIM2);
      if (tx < len0 && ty < len1 && tz < len2)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty),
             *(segment2.begin() + tz), tx, ty, tz);
    }
  }
};

// perfectly nested sycl loop policies
using sycl_group_01_nested_loop = sycl_group_012_loop<0, 1>;
using sycl_group_02_nested_loop = sycl_group_012_loop<0, 2>;
using sycl_group_10_nested_loop = sycl_group_012_loop<1, 0>;
using sycl_group_12_nested_loop = sycl_group_012_loop<1, 2>;
using sycl_group_20_nested_loop = sycl_group_012_loop<2, 0>;
using sycl_group_21_nested_loop = sycl_group_012_loop<2, 1>;

using sycl_group_012_nested_loop = sycl_group_012_loop<0, 1, 2>;
using sycl_group_021_nested_loop = sycl_group_012_loop<0, 2, 1>;
using sycl_group_102_nested_loop = sycl_group_012_loop<1, 0, 2>;
using sycl_group_120_nested_loop = sycl_group_012_loop<1, 2, 0>;
using sycl_group_201_nested_loop = sycl_group_012_loop<2, 0, 1>;
using sycl_group_210_nested_loop = sycl_group_012_loop<2, 1, 0>;

template<typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_group_012_loop<DIM0, DIM1>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment0,
                                           SEGMENT const& segment1,
                                           BODY const& body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {

      for (int bx = ctx.itm->get_group(DIM0); bx < len0;
           bx += ctx.itm->get_group_range(DIM0))
      {
        for (int by = ctx.itm->get_group(DIM1); by < len1;
             bx += ctx.itm->get_group_range(DIM1))
        {
          body(*(segment0.begin() + bx), *(segment1.begin() + by));
        }
      }
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_group_012_loop<DIM0, DIM1, DIM2>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment0,
                                           SEGMENT const& segment1,
                                           SEGMENT const& segment2,
                                           BODY const& body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int bx = ctx.itm->get_group(DIM0); bx < len0;
         bx += ctx.itm->get_group_range(DIM0))
    {

      for (int by = ctx.itm->get_group(DIM1); by < len1;
           by += ctx.itm->get_group_range(DIM1))
      {

        for (int bz = ctx.itm->get_group(DIM2); bz < len2;
             bz += ctx.itm->get_group_range(DIM2))
        {

          body(*(segment0.begin() + bx), *(segment1.begin() + by),
               *(segment2.begin() + bz));
        }
      }
    }
  }
};

/*
  perfectly nested sycl loop policies + returns local index
*/
template<typename SEGMENT, int DIM0, int DIM1>
struct LoopICountExecute<sycl_group_012_loop<DIM0, DIM1>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment0,
                                           SEGMENT const& segment1,
                                           BODY const& body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {

      for (int bx = ctx.itm->get_group(DIM0); bx < len0;
           bx += ctx.itm->get_group_range(DIM0))
      {
        for (int by = ctx.itm->get_group(DIM0); by < len1;
             by += ctx.itm->get_group_range(DIM1))
        {

          body(*(segment0.begin() + bx), *(segment1.begin() + by), bx, by);
        }
      }
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopICountExecute<sycl_group_012_loop<DIM0, DIM1, DIM2>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           SEGMENT const& segment0,
                                           SEGMENT const& segment1,
                                           SEGMENT const& segment2,
                                           BODY const& body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int bx = ctx.itm->get_group(DIM0); bx < len0;
         bx += ctx.itm->get_group_range(DIM0))
    {

      for (int by = ctx.itm->get_group(DIM0); by < len1;
           by += ctx.itm->get_group_range(DIM0))
      {

        for (int bz = ctx.itm->get_group(DIM0); bz < len2;
             bz += ctx.itm->get_group_range(DIM0))
        {

          body(*(segment0.begin() + bx), *(segment1.begin() + by),
               *(segment2.begin() + bz), bx, by, bz);
        }
      }
    }
  }
};

template<typename SEGMENT, int DIM>
struct TileExecute<sycl_local_012_loop<DIM>, SEGMENT>
{

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           TILE_T tile_size,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = ctx.itm->get_local_id(DIM) * tile_size; tx < len;
         tx += ctx.itm->get_local_range(DIM) * tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};

template<typename SEGMENT, int DIM>
struct TileExecute<sycl_local_012_direct<DIM>, SEGMENT>
{

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           TILE_T tile_size,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    int tx = ctx.itm->get_local_id(DIM) * tile_size;
    if (tx < len)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};

template<typename SEGMENT, int DIM>
struct TileExecute<sycl_group_012_loop<DIM>, SEGMENT>
{

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           TILE_T tile_size,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = ctx.itm->get_group(DIM) * tile_size;

         tx < len;

         tx += ctx.itm->get_group_range(DIM) * tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};

template<typename SEGMENT, int DIM>
struct TileExecute<sycl_group_012_direct<DIM>, SEGMENT>
{

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           TILE_T tile_size,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    int tx = ctx.itm->get_group(DIM) * tile_size;
    if (tx < len)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};

// Tile execute + return index
template<typename SEGMENT, int DIM>
struct TileTCountExecute<sycl_local_012_loop<DIM>, SEGMENT>
{

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           TILE_T tile_size,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = ctx.itm->get_local_id(DIM) * tile_size; tx < len;
         tx += ctx.itm->get_local_range(DIM) * tile_size)
    {
      body(segment.slice(tx, tile_size), tx / tile_size);
    }
  }
};

template<typename SEGMENT, int DIM>
struct TileTCountExecute<sycl_local_012_direct<DIM>, SEGMENT>
{

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           TILE_T tile_size,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    int tx = ctx.itm->get_local_id(DIM) * tile_size;
    if (tx < len)
    {
      body(segment.slice(tx, tile_size), tx / tile_size);
    }
  }
};

template<typename SEGMENT, int DIM>
struct TileTCountExecute<sycl_group_012_loop<DIM>, SEGMENT>
{

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           TILE_T tile_size,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = ctx.itm->get_group(DIM) * tile_size; bx < len;
         bx += ctx.itm->get_group_range(DIM) * tile_size)
    {
      body(segment.slice(bx, tile_size), bx / tile_size);
    }
  }
};

template<typename SEGMENT, int DIM>
struct TileTCountExecute<sycl_group_012_direct<DIM>, SEGMENT>
{

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const& ctx,
                                           TILE_T tile_size,
                                           SEGMENT const& segment,
                                           BODY const& body)
  {

    const int len = segment.end() - segment.begin();

    int bx = ctx.itm->get_group(DIM) * tile_size;
    if (bx < len)
    {
      body(segment.slice(bx, tile_size), bx / tile_size);
    }
  }
};

}  // namespace RAJA
#endif
