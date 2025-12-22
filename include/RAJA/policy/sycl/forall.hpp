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
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
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

#include "RAJA/internal/fault_tolerance.hpp"

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
         typename ForallParam,
         typename std::enable_if<std::is_trivially_copyable<LoopBody> {},
                                 bool>::type = true>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Sycl>,
    RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
    RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>
forall_impl(resources::Sycl& sycl_res,
            sycl_exec<BlockSize, Async>,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam)
{

  using Iterator = camp::decay<decltype(std::begin(iter))>;
  using IndexType =
      camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;

  //
  // Compute the requested iteration space size
  //
  Iterator begin = std::begin(iter);
  Iterator end   = std::end(iter);
  IndexType len  = std::distance(begin, end);

  // Only launch kernel if we have something to iterate over
  if (len > 0 && BlockSize > 0)
  {
    // Note: We could fix an incorrect workgroup size.
    //       It would change what was specified.
    //       For now, leave the device compiler to error with invalid WG size.

    //
    // Compute the number of blocks
    //
    sycl_dim_t blockSize {BlockSize};
    sycl_dim_t gridSize = impl::getGridDim(static_cast<size_t>(len), BlockSize);

    ::sycl::queue* q = sycl_res.get_queue();

    q->submit([&](::sycl::handler& h) {
      h.parallel_for(::sycl::nd_range<1> {gridSize, blockSize},
                     [=](::sycl::nd_item<1> it) {
                       IndexType ii = it.get_global_id(0);
                       if (ii < len)
                       {
                         loop_body(begin[ii]);
                       }
                     });
    });

    if (!Async)
    {
      q->wait();
    }
  }

  return resources::EventProxy<resources::Sycl>(sycl_res);
}

//
// Define if parampack is empty, avoids ambigous definitions.
//

template<typename Iterable,
         typename LoopBody,
         size_t BlockSize,
         bool Async,
         typename ForallParam,
         typename std::enable_if<!std::is_trivially_copyable<LoopBody> {},
                                 bool>::type = true>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Sycl>,
    RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
    RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>
forall_impl(resources::Sycl& sycl_res,
            sycl_exec<BlockSize, Async>,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam)
{
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType =
      camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;

  //
  // Compute the requested iteration space size
  //
  Iterator begin = std::begin(iter);
  Iterator end   = std::end(iter);
  IndexType len  = std::distance(begin, end);

  // Only launch kernel if we have something to iterate over
  if (len > 0 && BlockSize > 0)
  {

    // Note: We could fix an incorrect workgroup size.
    //       It would change what was specified.
    //       For now, leave the device compiler to error with invalid WG size.

    //
    // Compute the number of blocks
    //
    sycl_dim_t blockSize {BlockSize};
    sycl_dim_t gridSize = impl::getGridDim(static_cast<size_t>(len), BlockSize);

    ::sycl::queue* q = sycl_res.get_queue();

    LOOP_BODY* lbody;
    Iterator* beg;

        //
    // Setup shared memory buffers
    // Kernel body is nontrivially copyable, create space on device and copy to
    // Workaround until "is_device_copyable" is supported
    //
    lbody = (LOOP_BODY*)::sycl::malloc_device(sizeof(LOOP_BODY), *q);
    q->memcpy(lbody, &loop_body, sizeof(LOOP_BODY)).wait();

    beg = (Iterator*)::sycl::malloc_device(sizeof(Iterator), *q);
    q->memcpy(beg, &begin, sizeof(Iterator)).wait();

    q->submit([&](::sycl::handler& h) {
       h.parallel_for(::sycl::nd_range<1> {gridSize, blockSize},
                      [=](::sycl::nd_item<1> it) {
                        Index_type ii = it.get_global_id(0);

                        if (ii < len)
                        {
                          (*lbody)((*beg)[ii]);
                        }
                      });
     }).wait();  // Need to wait for completion to free memory

    // Free our device memory
    ::sycl::free(lbody, *q);
    ::sycl::free(beg, *q);

      }

  return resources::EventProxy<resources::Sycl>(sycl_res);
}

template<typename Iterable,
         typename LoopBody,
         size_t BlockSize,
         bool Async,
         typename ForallParam,
         typename std::enable_if<std::is_trivially_copyable<LoopBody> {},
                                 bool>::type = true>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Sycl>,
    RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
    concepts::negate<
        RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>>
forall_impl(resources::Sycl& sycl_res,
            sycl_exec<BlockSize, Async> const& pol,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam f_params)

{
  using Iterator = camp::decay<decltype(std::begin(iter))>;
  using IndexType =
      camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;
  using EXEC_POL = camp::decay<decltype(pol)>;

  //
  // Compute the requested iteration space size
  //
  Iterator begin = std::begin(iter);
  Iterator end   = std::end(iter);
  IndexType len  = std::distance(begin, end);

  RAJA::expt::ParamMultiplexer::parampack_init(pol, f_params);

  // Only launch kernel if we have something to iterate over
  if (len > 0 && BlockSize > 0)
  {

    //
    // Compute the number of blocks
    //
    sycl_dim_t blockSize {BlockSize};
    sycl_dim_t gridSize = impl::getGridDim(static_cast<size_t>(len), BlockSize);
    RAJA_UNUSED_VAR(gridSize);

    ::sycl::queue* q = sycl_res.get_queue();

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
                         RAJA::expt::invoke_body(fp, loop_body, begin[ii]);
                       }
                       red.combine(fp);
                     });
    });

    q->wait();
    RAJA::expt::ParamMultiplexer::parampack_combine(pol, f_params, *res);
    ::sycl::free(res, *q);
  }
  RAJA::expt::ParamMultiplexer::parampack_resolve(pol, f_params);

  return resources::EventProxy<resources::Sycl>(sycl_res);
}

template<typename Iterable,
         typename LoopBody,
         size_t BlockSize,
         bool Async,
         typename ForallParam,
         typename std::enable_if<!std::is_trivially_copyable<LoopBody> {},
                                 bool>::type = true>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Sycl>,
    RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
    concepts::negate<
        RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>>
forall_impl(resources::Sycl& sycl_res,
            sycl_exec<BlockSize, Async> const& pol,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam f_params)

{
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType =
      camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;
  using EXEC_POL = camp::decay<decltype(pol)>;

  //
  // Compute the requested iteration space size
  //
  Iterator begin = std::begin(iter);
  Iterator end   = std::end(iter);
  IndexType len  = std::distance(begin, end);

  RAJA::expt::ParamMultiplexer::parampack_init(pol, f_params);

  // Only launch kernel if we have something to iterate over
  if (len > 0 && BlockSize > 0)
  {
    //
    // Compute the number of blocks
    //
    sycl_dim_t blockSize {BlockSize};
    sycl_dim_t gridSize = impl::getGridDim(static_cast<size_t>(len), BlockSize);
    RAJA_UNUSED_VAR(gridSize);

    ::sycl::queue* q = sycl_res.get_queue();

    auto combiner = [](ForallParam x, ForallParam y) {
      RAJA::expt::ParamMultiplexer::parampack_combine(EXEC_POL {}, x, y);
      return x;
    };

    // START
    //
    LOOP_BODY* lbody;
    Iterator* beg;
        //
    // Setup shared memory buffers
    // Kernel body is nontrivially copyable, create space on device and copy to
    // Workaround until "is_device_copyable" is supported
    //
    lbody = (LOOP_BODY*)::sycl::malloc_device(sizeof(LOOP_BODY), *q);
    q->memcpy(lbody, &loop_body, sizeof(LOOP_BODY)).wait();

    beg = (Iterator*)::sycl::malloc_device(sizeof(Iterator), *q);
    q->memcpy(beg, &begin, sizeof(Iterator)).wait();

    ForallParam* res = ::sycl::malloc_shared<ForallParam>(1, *q);
    RAJA::expt::ParamMultiplexer::parampack_init(pol, *res);
    auto reduction = ::sycl::reduction(res, f_params, combiner);

    q->submit([&](::sycl::handler& h) {
       h.parallel_for(::sycl::range<1>(len), reduction,
                      [=](::sycl::item<1> it, auto& red) {
                        Index_type ii = it.get_id(0);
                        ForallParam fp;
                        RAJA::expt::ParamMultiplexer::parampack_init(pol, fp);
                        if (ii < len)
                        {
                          RAJA::expt::invoke_body(fp, *lbody, (*beg)[ii]);
                        }
                        red.combine(fp);
                      });
     }).wait();  // Need to wait for completion to free memory
    RAJA::expt::ParamMultiplexer::parampack_combine(pol, f_params, *res);
    // Free our device memory
    ::sycl::free(res, *q);
    ::sycl::free(lbody, *q);
    ::sycl::free(beg, *q);

      }
  RAJA::expt::ParamMultiplexer::parampack_resolve(pol, f_params);

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
