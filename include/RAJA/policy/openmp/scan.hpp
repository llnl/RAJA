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

#ifndef RAJA_scan_openmp_HPP
#define RAJA_scan_openmp_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>
#include <vector>

#include <omp.h>

#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/sequential/scan.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"

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
template<typename Policy, typename Iter, typename BinFn>
RAJA_INLINE concepts::enable_if_t<resources::EventProxy<resources::Host>,
                                  type_traits::is_openmp_policy<Policy>>
inclusive_inplace(resources::Host host_res,
                  const Policy&,
                  Iter begin,
                  Iter end,
                  BinFn f)
{
  using RAJA::detail::firstIndex;
  using std::distance;
  using Value     = typename ::std::iterator_traits<Iter>::value_type;
  const auto n    = distance(begin, end);
  using DistanceT = typename std::remove_const<decltype(n)>::type;
  const int p0    = std::min(n, static_cast<DistanceT>(omp_get_max_threads()));
  ::std::vector<Value> sums(p0, Value());
#pragma omp parallel num_threads(p0)
  {
    const int p               = omp_get_num_threads();
    const int pid             = omp_get_thread_num();
    const DistanceT idx_begin = firstIndex(n, p, pid);
    const DistanceT idx_end   = firstIndex(n, p, pid + 1);
    if (idx_begin != idx_end)
    {
      inclusive_inplace(host_res, ::RAJA::seq_exec {}, begin + idx_begin,
                        begin + idx_end, f);
      sums[pid] = begin[idx_end - 1];
    }
#pragma omp barrier
#pragma omp single
    exclusive_inplace(host_res, ::RAJA::seq_exec {}, sums.data(),
                      sums.data() + p, f, BinFn::identity());
    for (auto i = idx_begin; i < idx_end; ++i)
    {
      begin[i] = f(begin[i], sums[pid]);
    }
  }

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template<typename Policy, typename Iter, typename BinFn, typename ValueT>
RAJA_INLINE concepts::enable_if_t<resources::EventProxy<resources::Host>,
                                  type_traits::is_openmp_policy<Policy>>
exclusive_inplace(resources::Host host_res,
                  const Policy&,
                  Iter begin,
                  Iter end,
                  BinFn f,
                  ValueT v)
{
  using RAJA::detail::firstIndex;
  using std::distance;
  using Value     = typename ::std::iterator_traits<Iter>::value_type;
  const auto n    = distance(begin, end);
  using DistanceT = typename std::remove_const<decltype(n)>::type;
  const int p0    = std::min(n, static_cast<DistanceT>(omp_get_max_threads()));
  ::std::vector<Value> sums(p0, v);
#pragma omp parallel num_threads(p0)
  {
    const int p               = omp_get_num_threads();
    const int pid             = omp_get_thread_num();
    const DistanceT idx_begin = firstIndex(n, p, pid);
    const DistanceT idx_end   = firstIndex(n, p, pid + 1);
    const Value init          = ((pid == 0) ? v : *(begin + idx_begin - 1));
#pragma omp barrier
    if (idx_begin != idx_end)
    {
      exclusive_inplace(host_res, seq_exec {}, begin + idx_begin,
                        begin + idx_end, f, init);
      sums[pid] = begin[idx_end - 1];
    }
#pragma omp barrier
#pragma omp single
    exclusive_inplace(host_res, seq_exec {}, sums.data(), sums.data() + p, f,
                      BinFn::identity());
    for (auto i = idx_begin; i < idx_end; ++i)
    {
      begin[i] = f(begin[i], sums[pid]);
    }
  }

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template<typename Policy, typename Iter, typename OutIter, typename BinFn>
RAJA_INLINE concepts::enable_if_t<resources::EventProxy<resources::Host>,
                                  type_traits::is_openmp_policy<Policy>>
inclusive(resources::Host host_res,
          const Policy& exec,
          Iter begin,
          Iter end,
          OutIter out,
          BinFn f)
{
  using std::distance;
  ::std::copy(begin, end, out);
  return inclusive_inplace(host_res, exec, out, out + distance(begin, end), f);
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template<typename Policy,
         typename Iter,
         typename OutIter,
         typename BinFn,
         typename ValueT>
RAJA_INLINE concepts::enable_if_t<resources::EventProxy<resources::Host>,
                                  type_traits::is_openmp_policy<Policy>>
exclusive(resources::Host host_res,
          const Policy& exec,
          Iter begin,
          Iter end,
          OutIter out,
          BinFn f,
          ValueT v)
{
  using std::distance;
  ::std::copy(begin, end, out);
  return exclusive_inplace(host_res, exec, out, out + distance(begin, end), f,
                           v);
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif
