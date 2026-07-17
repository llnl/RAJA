//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_BASIC_REDUCE_INTERFACE_HPP__
#define __TEST_FORALL_BASIC_REDUCE_INTERFACE_HPP__

#include "RAJA/RAJA.hpp"

#include "camp/camp.hpp"

#include <utility>

struct LegacyReducerApi
{
  template <typename EXEC_POLICY, typename REDUCER, typename... Args>
  static REDUCER make(Args&&... args)
  {
    return REDUCER(std::forward<Args>(args)...);
  }

  template <typename EXEC_POLICY, typename REDUCER, typename... Args>
  static void reset(REDUCER& reducer, Args&&... args)
  {
    reducer.reset(std::forward<Args>(args)...);
  }
};

struct RuntimePolicyReducerApi
{
  template <typename EXEC_POLICY, typename REDUCER, typename... Args>
  static REDUCER make(Args&&... args)
  {
    return REDUCER(RAJA::policy_of<EXEC_POLICY>::value,
                   std::forward<Args>(args)...);
  }

  template <typename EXEC_POLICY, typename REDUCER, typename... Args>
  static void reset(REDUCER& reducer, Args&&... args)
  {
    reducer.reset(RAJA::policy_of<EXEC_POLICY>::value,
                  std::forward<Args>(args)...);
  }
};

using LegacyReducerApiList = camp::list<LegacyReducerApi>;
using RuntimePolicyReducerApiList = camp::list<RuntimePolicyReducerApi>;

#endif  // __TEST_FORALL_BASIC_REDUCE_INTERFACE_HPP__
