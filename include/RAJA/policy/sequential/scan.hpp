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
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_scan_sequential_HPP
#define RAJA_scan_sequential_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <functional>
#include <iterator>

#include "RAJA/util/macros.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/policy/sequential/policy.hpp"

namespace RAJA
{
namespace impl
{
namespace scan
{
/*!
        \brief explicit inclusive inplace scan given range, function, and
   initial value
*/
template<typename ExecPolicy, typename Iter, typename BinFn>
RAJA_INLINE concepts::enable_if_t<resources::EventProxy<resources::Host>,
                                  type_traits::is_sequential_policy<ExecPolicy>>
inclusive_inplace(resources::Host host_res,
                  const ExecPolicy&,
                  Iter begin,
                  Iter end,
                  BinFn f)
{
  using ValueT = typename std::remove_reference<decltype(*begin)>::type;
  ValueT agg   = *begin;

  for (Iter i = ++begin; i != end; ++i)
  {
    agg = f(agg, *i);
    *i  = agg;
  }

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template<typename ExecPolicy, typename Iter, typename BinFn, typename T>
RAJA_INLINE concepts::enable_if_t<resources::EventProxy<resources::Host>,
                                  type_traits::is_sequential_policy<ExecPolicy>>
exclusive_inplace(resources::Host host_res,
                  const ExecPolicy&,
                  Iter begin,
                  Iter end,
                  BinFn f,
                  T v)
{
  using std::distance;
  const auto n    = distance(begin, end);
  using DistanceT = typename std::remove_const<decltype(n)>::type;

  using ValueT = typename std::remove_reference<decltype(*begin)>::type;
  ValueT agg   = v;

  for (DistanceT i = 0; i < n; ++i)
  {
    auto t   = begin[i];
    begin[i] = agg;
    agg      = f(agg, t);
  }

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template<typename ExecPolicy, typename Iter, typename OutIter, typename BinFn>
RAJA_INLINE concepts::enable_if_t<resources::EventProxy<resources::Host>,
                                  type_traits::is_sequential_policy<ExecPolicy>>
inclusive(resources::Host host_res,
          const ExecPolicy&,
          const Iter begin,
          const Iter end,
          OutIter out,
          BinFn f)
{
  using ValueT = typename std::remove_reference<decltype(*out)>::type;
  ValueT agg   = *begin;
  *out++       = agg;

  for (Iter i = begin + 1; i != end; ++i)
  {
    agg    = f(agg, *i);
    *out++ = agg;
  }

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template<typename ExecPolicy,
         typename Iter,
         typename OutIter,
         typename BinFn,
         typename T>
RAJA_INLINE concepts::enable_if_t<resources::EventProxy<resources::Host>,
                                  type_traits::is_sequential_policy<ExecPolicy>>
exclusive(resources::Host host_res,
          const ExecPolicy&,
          const Iter begin,
          const Iter end,
          OutIter out,
          BinFn f,
          T v)
{
  using ValueT = typename std::remove_reference<decltype(*out)>::type;
  ValueT agg   = v;
  OutIter o    = out;
  *o++         = v;

  for (Iter i = begin; i != end - 1; ++i, ++o)
  {
    agg = f(agg, *i);
    *o  = agg;
  }

  return resources::EventProxy<resources::Host>(host_res);
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif
