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

#ifndef RAJA_sort_sequential_HPP
#define RAJA_sort_sequential_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <functional>
#include <iterator>

#include "RAJA/util/macros.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/util/zip.hpp"

#include "RAJA/util/sort.hpp"

#include "RAJA/policy/sequential/policy.hpp"

namespace RAJA
{
namespace impl
{
namespace sort
{

namespace detail
{

/*!
    \brief Functional that performs an unstable sort with the
           given arguments, uses RAJA::intro_sort
*/
struct UnstableSorter
{
  template<typename... Args>
  RAJA_INLINE void operator()(Args&&... args) const
  {
    RAJA::detail::intro_sort(std::forward<Args>(args)...);
  }
};

/*!
    \brief Functional that performs a stable sort with the
           given arguments, calls RAJA::merge_sort
*/
struct StableSorter
{
  template<typename... Args>
  RAJA_INLINE void operator()(Args&&... args) const
  {
    RAJA::detail::merge_sort(std::forward<Args>(args)...);
  }
};

}  // namespace detail

/*!
        \brief sort given range using comparison function
*/
template<typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_sequential_policy<ExecPolicy>>
unstable(resources::Host host_res,
         const ExecPolicy&,
         Iter begin,
         Iter end,
         Compare comp)
{
  detail::UnstableSorter {}(begin, end, comp);

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief stable sort given range using comparison function
*/
template<typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_sequential_policy<ExecPolicy>>
stable(resources::Host host_res,
       const ExecPolicy&,
       Iter begin,
       Iter end,
       Compare comp)
{
  detail::StableSorter {}(begin, end, comp);

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
                      type_traits::is_sequential_policy<ExecPolicy>>
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
  detail::UnstableSorter {}(begin, end, RAJA::compare_first<zip_ref>(comp));

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
                      type_traits::is_sequential_policy<ExecPolicy>>
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
  detail::StableSorter {}(begin, end, RAJA::compare_first<zip_ref>(comp));

  return resources::EventProxy<resources::Host>(host_res);
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif
