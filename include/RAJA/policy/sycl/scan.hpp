/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_scan_sycl_HPP
#define RAJA_scan_sycl_HPP

#include <cstddef>
#include <iterator>
#include "RAJA/config.hpp"
#include "camp/resource/sycl.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "RAJA/policy/sycl/policy.hpp"

namespace RAJA
{
namespace impl
{
namespace scan
{

template <size_t BLOCK_SIZE, bool Async, typename InputIter, typename Function>
RAJA_INLINE 
camp::resources::EventProxy<camp::resources::Sycl> 
inclusive_inplace(
    camp::resources::Sycl sycl_res,
    ::RAJA::policy::sycl::sycl_exec<BLOCK_SIZE, Async>,
    InputIter begin,
    InputIter end,
    Function binary_op)
{
    ::sycl::queue* sycl_queue = sycl_res.get_queue();
 
    // using valueT = typename std::remove_reference<decltype(*begin)>::type;
    using valueT = typename std::iterator_traits<InputIter>::value_type;

     // Calculate the size of the input range
    size_t size = std::distance(begin, end);

    ::sycl::buffer<valueT, 1> buffer(begin, end);
    ::sycl::buffer<valueT, 1> tempAccBuff(begin, ::sycl::range<1>(size));

    int iterations = 0;
    for (size_t ii = size >> 1; ii > 0; ii >>= 1) {
        iterations++;
    }
    if ((size & (size - 1)) != 0) {
        iterations++;
    }

    auto buffPtr = &buffer;
    auto tempPtr = &tempAccBuff;

    if (iterations % 2 == 0) { 
        tempPtr = &buffer;
        buffPtr = &tempAccBuff;
    } 

    int ii = 1;
    do {
        // Submit the kernel to the SYCL queue  
        sycl_queue->submit([&](::sycl::handler& cgh) {
            auto buffAccessor = buffPtr->get_access(cgh);
            auto tempAccessor = tempPtr->get_access(cgh);
            // outBuffAccessor = outBuff->get_access<::sycl::access::mode::read(cgh);

            cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::item<1> idx) {
                size_t td = 1 << (ii - 1);
                size_t thisID = idx.get_id(0);
                if (thisID < size and thisID >= td) { 
                    tempAccessor[thisID] = binary_op(buffAccessor[thisID - td], buffAccessor[thisID]);
                } else {
                    tempAccessor[thisID] = buffAccessor[thisID];
                }
            });
        });

        std::swap(buffPtr, tempPtr);
        ii++;
    } while (  ii <= iterations);

  if (!Async) { sycl_res.wait(); }
  return camp::resources::EventProxy<camp::resources::Sycl>(sycl_res);
}

template <size_t BLOCK_SIZE,
          bool Async,
          typename InputIter,
          typename Function,
          typename TT>
RAJA_INLINE 
resources::EventProxy<resources::Sycl> 
exclusive_inplace(
    resources::Sycl sycl_res,
    ::RAJA::policy::sycl::sycl_exec<BLOCK_SIZE, Async> exec,
    InputIter begin,
    InputIter end,
    Function binary_op,
    TT initVal)
{
    ::sycl::queue* sycl_queue = sycl_res.get_queue();
    
    // using valueT = typename std::remove_reference<decltype(*begin)>::type;
    using valueT = typename std::iterator_traits<InputIter>::value_type;


     // Calculate the size of the input range
    size_t size = std::distance(begin, end);

    ::sycl::buffer<valueT, 1> buffer(begin, end);
    ::sycl::buffer<valueT, 1> tempAccBuff(begin, ::sycl::range<1>(size));

    int iterations = 0;
    for (size_t ii = size >> 1; ii > 0; ii >>= 1) {
        iterations++;
    }
    if ((size & (size - 1)) != 0) {
        iterations++;
    }

    auto buffPtr = &buffer;
    auto tempPtr = &tempAccBuff;

    if (iterations % 2 != 0) { 
        tempPtr = &buffer;
        buffPtr = &tempAccBuff;
    } 

    // Submit the kernel to the SYCL queue  
    sycl_queue->submit([&](::sycl::handler& cgh) {
        auto inAccessor = buffPtr->get_access(cgh);
        auto outAccessor = tempPtr->get_access(cgh);
        // outBuffAccessor = outBuff->get_access<::sycl::access::mode::read(cgh);

        cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::item<1> idx) {
            size_t thisID = idx.get_id(0);
            // size_t td = 1 << (ii - 1);
            if (thisID > 0) { 
                outAccessor[thisID] = inAccessor[thisID - 1];
            } else {
                outAccessor[thisID] = initVal;
            }
        });
    });

    std::swap(buffPtr, tempPtr);

    int ii = 1;
    do {
        // Submit the kernel to the SYCL queue  
        sycl_queue->submit([&](::sycl::handler& cgh) {
            auto buffAccessor = buffPtr->get_access(cgh);
            auto tempAccessor = tempPtr->get_access(cgh);
            // outBuffAccessor = outBuff->get_access<::sycl::access::mode::read(cgh);

            cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::item<1> idx) {
                size_t td = 1 << (ii - 1);
                size_t thisID = idx.get_id(0);
                if (thisID < size and thisID >= td) { 
                    tempAccessor[thisID] = binary_op(buffAccessor[thisID - td], buffAccessor[thisID]);
                } else {
                    tempAccessor[thisID] = buffAccessor[thisID];
                }
            });
        });

        std::swap(buffPtr, tempPtr);
        ii++;
    } while (  ii <= iterations);

  if(!Async) { sycl_res.wait(); }
  return camp::resources::EventProxy<camp::resources::Sycl>(sycl_res);
    return inclusive_inplace(sycl_res, exec, begin, end, binary_op);
}

template <size_t BLOCK_SIZE,
          bool Async,
          typename InputIter,
          typename OutputIter,
          typename Function>
RAJA_INLINE 
resources::EventProxy<resources::Sycl> 
inclusive(
    resources::Sycl sycl_res,
    ::RAJA::policy::sycl::sycl_exec<BLOCK_SIZE, Async> exec,
    InputIter begin,
    InputIter end,
    OutputIter out,
    Function binary_op)
{
    // using valueT = typename std::remove_reference<decltype(*out)>::type;
    using valueT = typename std::iterator_traits<OutputIter>::value_type;
    sycl_res.memcpy(out, begin, std::distance(begin, end) * sizeof(valueT));
    return inclusive_inplace(sycl_res, exec, out, out + std::distance(begin, end), binary_op);
}

template <size_t BLOCK_SIZE,
          bool Async,
          typename InputIter,
          typename OutputIter,
          typename Function,
          typename TT>
RAJA_INLINE 
resources::EventProxy<resources::Sycl> 
exclusive(
    resources::Sycl sycl_res,
    ::RAJA::policy::sycl::sycl_exec<BLOCK_SIZE, Async> exec,
    InputIter begin,
    InputIter end,
    OutputIter out,
    Function binary_op,
    TT initVal)
{
    // using valueT = typename std::remove_reference<decltype(*out)>::type;
    using valueT = typename std::iterator_traits<OutputIter>::value_type;

    sycl_res.memcpy(out, begin, std::distance(begin, end) * sizeof(valueT));
    return exclusive_inplace(sycl_res, exec, out, out + std::distance(begin, end), binary_op, initVal);
}

}  // namespace scan
}  // namespace impl
}  // namespace RAJA

#endif  // closing endif for RAJA enable Sycl guard

#endif  // closing endif for header include guard