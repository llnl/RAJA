/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          execution via SYCL kernel launch.
 *
 *          These methods should work on any platform that supports
 *          SYCL devices.
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

#ifndef RAJA_forall_sycl_HPP
#define RAJA_forall_sycl_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <algorithm>
#include <chrono>

#include "RAJA/util/sycl_compat.hpp"

#include "RAJA/pattern/forall.hpp"

#include "RAJA/pattern/params/forall.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/sycl/MemUtils_SYCL.hpp"
#include "RAJA/policy/sycl/policy.hpp"

#include "RAJA/index/IndexSet.hpp"

#include "RAJA/util/resource.hpp"

namespace RAJA
{

namespace policy
{

namespace sycl
{

namespace impl
{

/*!
 ******************************************************************************
 *
 * \brief calculate gridDim from length of iteration and blockDim
 *
 ******************************************************************************
 */
RAJA_INLINE
::sycl::range<1> getGridDim(size_t len, size_t block_size)
{
  size_t size = {block_size * ((len + block_size - 1) / block_size)};
  ::sycl::range<1> gridSize(size);
  return gridSize;
}

}  // namespace impl

//
////////////////////////////////////////////////////////////////////////
//
// Function templates for SYCL execution over iterables.
//
////////////////////////////////////////////////////////////////////////
//


template<typename Iterable,
         typename LoopBody,
         size_t BlockSize,
         bool Async,
         typename ForallParam>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Sycl>,
    RAJA::expt::type_traits::is_ForallParamPack<ForallParam>>
forall_impl(resources::Sycl& sycl_res,
            sycl_exec<BlockSize, Async> const& pol,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam f_params)

{
  using Iterator = camp::decay<decltype(std::begin(iter))>;
  using IndexType =
      camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;
  using EXEC_POL  = camp::decay<decltype(pol)>;
  using LOOP_BODY = camp::decay<LoopBody>;
  // Deduce at compile time if lbody is trivially constructible and if user
  // has supplied parameters.  These will be used to determine which sycl launch
  // to configure below.
  constexpr bool is_parampack_empty =
      RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>::value;
  constexpr bool is_lbody_trivially_copyable =
      std::is_trivially_copyable<LoopBody>::value;

  //
  // Compute the requested iteration space size
  //
  Iterator begin = std::begin(iter);
  Iterator end   = std::end(iter);
  IndexType len  = std::distance(begin, end);

  // Return immediately if there is no work to be done
  if (len <= 0 || BlockSize <= 0)
  {
    return resources::EventProxy<resources::Sycl>(sycl_res);
  }

  //
  // Compute the number of blocks
  //
  sycl_dim_t blockSize {BlockSize};
  sycl_dim_t gridSize = impl::getGridDim(static_cast<size_t>(len), BlockSize);

  ::sycl::queue* q  = sycl_res.get_queue();
  LOOP_BODY* lbody  = nullptr;
  Iterator* d_begin = nullptr;

  if constexpr (!is_parampack_empty)
  {
    RAJA::expt::ParamMultiplexer::parampack_init(pol, f_params);
  }
  if constexpr (!is_lbody_trivially_copyable)
  {
    //
    // Setup shared memory buffers
    // Kernel body is nontrivially copyable, create space on device and copy to
    // Workaround until "is_device_copyable" is supported
    //
    lbody = (LOOP_BODY*)::sycl::malloc_device(sizeof(LoopBody), *q);
    q->memcpy(lbody, &loop_body, sizeof(LOOP_BODY)).wait();

    d_begin = (Iterator*)::sycl::malloc_device(sizeof(Iterator), *q);
    q->memcpy(d_begin, &begin, sizeof(Iterator)).wait();
  }

  // Both the parallel_for call, combinations, and resolution are all
  // unique to the parameter case, so we make a constexpr branch here
  if constexpr (!is_parampack_empty)
  {
    auto combiner = [](ForallParam x, ForallParam y) {
      RAJA::expt::ParamMultiplexer::parampack_combine(EXEC_POL {}, x, y);
      return x;
    };

    ForallParam* res = ::sycl::malloc_shared<ForallParam>(1, *q);
    RAJA::expt::ParamMultiplexer::parampack_init(pol, *res);
    auto reduction = ::sycl::reduction(res, f_params, combiner);

    q->submit([&](::sycl::handler& h) {
      h.parallel_for(::sycl::range<1>(len), reduction,
                     [=](::sycl::item<1> it, auto& red) {
                       ForallParam fp;
                       RAJA::expt::ParamMultiplexer::parampack_init(pol, fp);
                       IndexType ii = it.get_id(0);
                       if (ii < len)
                       {
                         if constexpr (is_lbody_trivially_copyable)
                         {
                           RAJA::expt::invoke_body(fp, loop_body, begin[ii]);
                         }
                         else
                         {
                           RAJA::expt::invoke_body(fp, *lbody, (*d_begin)[ii]);
                         }
                       }
                       red.combine(fp);
                     });
    });

    q->wait();
    RAJA::expt::ParamMultiplexer::parampack_combine(pol, f_params, *res);
    ::sycl::free(res, *q);
    RAJA::expt::ParamMultiplexer::parampack_resolve(pol, f_params);
  }
  // Note: separate branches
  else
  {
    q->submit([&](::sycl::handler& h) {
      h.parallel_for(::sycl::nd_range<1> {gridSize, blockSize},
                     [=](::sycl::nd_item<1> it) {
                       IndexType ii = it.get_global_id(0);
                       if (ii < len)
                       {
                         if constexpr (is_lbody_trivially_copyable)
                         {
                           loop_body(begin[ii]);
                         }
                         else
                         {
                           (*lbody)((*d_begin)[ii]);
                         }
                       }
                     });
    });

    if (!Async)
    {
      q->wait();
    }
  }


  // If we had to allocate device memory, free it
  if constexpr (!is_lbody_trivially_copyable)
  {
    ::sycl::free(lbody, *q);
    ::sycl::free(d_begin, *q);
  }


  return resources::EventProxy<resources::Sycl>(sycl_res);
}

//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// using the explicitly named segment iteration policy and execute
// segments as SYCL kernels.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         SYCL execution for segments.
 *
 ******************************************************************************
 */
template<typename LoopBody,
         size_t BlockSize,
         bool Async,
         typename... SegmentTypes>
RAJA_INLINE resources::EventProxy<resources::Sycl> forall_impl(
    resources::Sycl& r,
    ExecPolicy<seq_segit, sycl_exec<BlockSize, Async>>,
    const TypedIndexSet<SegmentTypes...>& iset,
    LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi)
  {
    iset.segmentCall(r, isi, detail::CallForall(), sycl_exec<BlockSize, true>(),
                     loop_body);
  }  // iterate over segments of index set

  if (!Async)
  {
    ::sycl::queue* q = r.get_queue();
    q->wait();
  }

  return resources::EventProxy<resources::Sycl>(r);
}

}  // namespace sycl

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_SYCL guard

#endif  // closing endif for header file include guard
