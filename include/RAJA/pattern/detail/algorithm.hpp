/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA algorithm definitions.
 *
 *          Definitions in this file will propagate to all RAJA header files.
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

#ifndef RAJA_pattern_detail_algorithm_HPP
#define RAJA_pattern_detail_algorithm_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "camp/helpers.hpp"

#include <iterator>

namespace RAJA
{

namespace detail
{

template<typename Iter>
using IterVal = typename ::std::iterator_traits<Iter>::value_type;

template<typename Iter>
using IterRef = typename ::std::iterator_traits<Iter>::reference;

template<typename Iter>
using IterDiff = typename ::std::iterator_traits<Iter>::difference_type;

template<typename Container>
using ContainerIter = camp::iterator_from<Container>;

template<typename Container>
using ContainerVal =
    camp::decay<decltype(*camp::val<camp::iterator_from<Container>>())>;

template<typename Container>
using ContainerRef = decltype(*camp::val<camp::iterator_from<Container>>());

template<typename Container>
using ContainerDiff =
    camp::decay<decltype(camp::val<camp::iterator_from<Container>>() -
                         camp::val<camp::iterator_from<Container>>())>;

template<typename DiffType, typename CountType>
RAJA_INLINE DiffType firstIndex(DiffType n,
                                CountType num_threads,
                                CountType thread_id)
{
  return (static_cast<size_t>(n) * thread_id) / num_threads;
}

}  // end namespace detail

/*!
    \brief swap values at iterators lhs and rhs
*/
template<typename Iter>
RAJA_HOST_DEVICE RAJA_INLINE void safe_iter_swap(Iter lhs, Iter rhs)
{
#ifdef RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE
  using camp::safe_swap;
  safe_swap(*lhs, *rhs);
#else
  using std::iter_swap;
  iter_swap(lhs, rhs);
#endif
}

/*!
    \brief returns iterator to next item
*/
template<typename Iter>
RAJA_HOST_DEVICE RAJA_INLINE Iter next(Iter it)
{
  ++it;
  return it;
}

/*!
    \brief returns iterator to next item
*/
template<typename Iter>
RAJA_HOST_DEVICE RAJA_INLINE Iter prev(Iter it)
{
  --it;
  return it;
}

}  // end namespace RAJA

#endif
