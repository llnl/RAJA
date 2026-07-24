//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __RAJA_TEST_REDUCER_API_HPP__
#define __RAJA_TEST_REDUCER_API_HPP__

#include "RAJA/RAJA.hpp"

#include "camp/camp.hpp"

#include <utility>

template <typename PolicyList>
struct ReducerApi;

template <RAJA::Policy... Ps>
struct ReducerApi<RAJA::PolicyList<Ps...>>
{
  template <typename REDUCER, typename... Args>
  static REDUCER make(Args&&... args)
  {
    return REDUCER(Ps..., std::forward<Args>(args)...);
  }

  template <typename REDUCER, typename... Args>
  static void reset(REDUCER& reducer, Args&&... args)
  {
    reducer.reset(Ps..., std::forward<Args>(args)...);
  }
};

struct LegacyReducerApi
{
  template <typename EXEC_POLICY>
  using type = ReducerApi<RAJA::PolicyList<>>;
};

struct RuntimeReducerApi
{
  template <typename EXEC_POLICY>
  using type =
      ReducerApi<RAJA::PolicyList<RAJA::policy_of<EXEC_POLICY>::value>>;
};

using LegacyReducerApiList = camp::list<LegacyReducerApi>;
using RuntimeReducerApiList = camp::list<RuntimeReducerApi>;

#endif  // __RAJA_TEST_REDUCER_API_HPP__
