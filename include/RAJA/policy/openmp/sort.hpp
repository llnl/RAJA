/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA sort declarations.
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

#ifndef RAJA_sort_openmp_HPP
#define RAJA_sort_openmp_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <functional>
#include <iterator>

#include <omp.h>

#include "RAJA/util/macros.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/sequential/sort.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"

namespace RAJA
{
namespace impl
{
namespace sort
{

namespace detail
{
namespace openmp
{

// this number is arbitrary
constexpr int get_min_iterates_per_task() { return 128; }

#if defined(RAJA_ENABLE_OPENMP_TASK_INTERNAL)
/*!
        \brief sort given range using sorter and comparison function
               by spawning tasks
*/
template<typename Sorter, typename Iter, typename Compare>
inline void sort_task(Sorter sorter,
                      Iter begin,
                      RAJA::detail::IterDiff<Iter> i_begin,
                      RAJA::detail::IterDiff<Iter> i_end,
                      RAJA::detail::IterDiff<Iter> iterates_per_task,
                      Compare comp)
{
  using diff_type   = RAJA::detail::IterDiff<Iter>;
  const diff_type n = i_end - i_begin;

  if (n <= iterates_per_task)
  {

    sorter(begin + i_begin, begin + i_end, comp);
  }
  else
  {

    const diff_type i_middle = i_begin + n / 2;

#pragma omp task
    sort_task(sorter, begin, i_begin, i_middle, iterates_per_task, comp);

#pragma omp task
    sort_task(sorter, begin, i_middle, i_end, iterates_per_task, comp);

#pragma omp taskwait

    // std::inplace_merge(begin + i_begin, begin + i_middle, begin + i_end,
    // comp);
    RAJA::detail::inplace_merge(begin + i_begin, begin + i_middle,
                                begin + i_end, comp);
  }
}

#else

/*!
        \brief sort given range using sorter and comparison function
               by manually assigning work to threads
*/
template<typename Sorter, typename Iter, typename Compare>
inline void sort_parallel_region(Sorter sorter,
                                 Iter begin,
                                 RAJA::detail::IterDiff<Iter> n,
                                 Compare comp)
{
  using RAJA::detail::firstIndex;
  using diff_type = RAJA::detail::IterDiff<Iter>;

  const diff_type num_threads = omp_get_num_threads();

  const diff_type thread_id = omp_get_thread_num();

  const diff_type i_begin = firstIndex(n, num_threads, thread_id);
  {
    const diff_type i_end = firstIndex(n, num_threads, thread_id + 1);

    // this thread sorts range [i_begin, i_end)
    sorter(begin + i_begin, begin + i_end, comp);
  }

  // hierarchically merge ranges
  for (diff_type middle_offset = 1; middle_offset < num_threads;
       middle_offset *= 2)
  {

    diff_type end_offset = 2 * middle_offset;

    const diff_type i_middle = firstIndex(
        n, num_threads, std::min(thread_id + middle_offset, num_threads));
    const diff_type i_end = firstIndex(
        n, num_threads, std::min(thread_id + end_offset, num_threads));

#pragma omp barrier

    if (thread_id % end_offset == 0)
    {

      // this thread merges ranges [i_begin, i_middle) and [i_middle, i_end)
      // std::inplace_merge(begin + i_begin, begin + i_middle, begin + i_end,
      // comp);
      RAJA::detail::inplace_merge(begin + i_begin, begin + i_middle,
                                  begin + i_end, comp);
    }
  }
}

#endif


/*!
        \brief sort given range using sorter and comparison function
*/
template<typename Sorter, typename Iter, typename Compare>
inline void sort(Sorter sorter, Iter begin, Iter end, Compare comp)
{
  using diff_type = RAJA::detail::IterDiff<Iter>;

  constexpr diff_type min_iterates_per_task = get_min_iterates_per_task();

  const diff_type n = end - begin;

  if (n <= min_iterates_per_task)
  {

    sorter(begin, end, comp);
  }
  else
  {

    const diff_type max_threads = omp_get_max_threads();

#if defined(RAJA_ENABLE_OPENMP_TASK_INTERNAL)

    const diff_type iterates_per_task =
        std::max(n / (2 * max_threads), min_iterates_per_task);

    const diff_type requested_num_threads =
        std::min((n + iterates_per_task - 1) / iterates_per_task, max_threads);
    RAJA_UNUSED_VAR(requested_num_threads);  // avoid warning in hip device code

#pragma omp parallel num_threads(static_cast <int>(requested_num_threads))
#pragma omp master
    {
      sort_task(sorter, begin, 0, n, iterates_per_task, comp);
    }

#else

    const diff_type requested_num_threads = std::min(
        (n + min_iterates_per_task - 1) / min_iterates_per_task, max_threads);
    RAJA_UNUSED_VAR(requested_num_threads);  // avoid warning in hip device code

#pragma omp parallel num_threads(static_cast <int>(requested_num_threads))
    {
      sort_parallel_region(sorter, begin, n, comp);
    }

#endif
  }
}

}  // namespace openmp

}  // namespace detail

/*!
        \brief sort given range using comparison function
*/
template<typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_openmp_policy<ExecPolicy>>
unstable(resources::Host host_res,
         const ExecPolicy&,
         Iter begin,
         Iter end,
         Compare comp)
{
  detail::openmp::sort(detail::UnstableSorter {}, begin, end, comp);

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief stable sort given range using comparison function
*/
template<typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_openmp_policy<ExecPolicy>>
stable(resources::Host host_res,
       const ExecPolicy&,
       Iter begin,
       Iter end,
       Compare comp)
{
  detail::openmp::sort(detail::StableSorter {}, begin, end, comp);

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief sort given range of pairs using comparison function on keys
*/
template<typename ExecPolicy,
         typename KeyIter,
         typename ValIter,
         typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_openmp_policy<ExecPolicy>>
unstable_pairs(resources::Host host_res,
               const ExecPolicy&,
               KeyIter keys_begin,
               KeyIter keys_end,
               ValIter vals_begin,
               Compare comp)
{
  auto begin    = RAJA::zip(keys_begin, vals_begin);
  auto end      = RAJA::zip(keys_end, vals_begin + (keys_end - keys_begin));
  using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
  detail::openmp::sort(detail::UnstableSorter {}, begin, end,
                       RAJA::compare_first<zip_ref>(comp));

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief stable sort given range of pairs using comparison function on
   keys
*/
template<typename ExecPolicy,
         typename KeyIter,
         typename ValIter,
         typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_openmp_policy<ExecPolicy>>
stable_pairs(resources::Host host_res,
             const ExecPolicy&,
             KeyIter keys_begin,
             KeyIter keys_end,
             ValIter vals_begin,
             Compare comp)
{
  auto begin    = RAJA::zip(keys_begin, vals_begin);
  auto end      = RAJA::zip(keys_end, vals_begin + (keys_end - keys_begin));
  using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
  detail::openmp::sort(detail::StableSorter {}, begin, end,
                       RAJA::compare_first<zip_ref>(comp));

  return resources::EventProxy<resources::Host>(host_res);
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif
