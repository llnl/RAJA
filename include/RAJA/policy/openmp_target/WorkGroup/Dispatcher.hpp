/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA workgroup Dispatcher.
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

#ifndef RAJA_openmp_target_WorkGroup_Dispatcher_HPP
#define RAJA_openmp_target_WorkGroup_Dispatcher_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/openmp_target/policy.hpp"

#include "RAJA/pattern/WorkGroup/Dispatcher.hpp"

namespace RAJA
{

namespace detail
{

namespace omp_target
{

// create the value in a target region using the factory, map the value
// back, and return the value created in the target region
template<typename Factory>
inline auto get_value(Factory factory)
{
  typename std::decay_t<Factory>::value_type value;

#pragma omp target map(tofrom : value) map(to : factory)
  {
    value = factory();
  }

  return value;
}

// get the device value and store it so it can be used
// multiple times
template<typename Factory>
inline auto get_cached_value(Factory&& factory)
{
  static auto value = get_value(std::forward<Factory>(factory));
  return value;
}

}  // namespace omp_target

/*!
 * Populate and return a Dispatcher object that can be used in omp target
 * regions
 */
template<typename T, typename Dispatcher_T>
inline const Dispatcher_T* get_Dispatcher(omp_target_work const&)
{
  static Dispatcher_T dispatcher {
      Dispatcher_T::template makeDispatcher<T>([](auto&& factory) {
        return omp_target::get_cached_value(
            std::forward<decltype(factory)>(factory));
      })};
  return &dispatcher;
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
