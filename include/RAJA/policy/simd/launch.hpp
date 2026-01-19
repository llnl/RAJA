/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::Teams::simd
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

#ifndef RAJA_pattern_launch_simd_HPP
#define RAJA_pattern_launch_simd_HPP

#include "RAJA/pattern/launch/launch_core.hpp"
#include "RAJA/policy/simd/policy.hpp"

namespace RAJA
{

template<typename SEGMENT>
struct LoopExecute<simd_exec, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {

    const int len = segment.end() - segment.begin();
    RAJA_SIMD
    for (int i = 0; i < len; i++)
    {
      body(*(segment.begin() + i));
    }
  }
};

template<typename SEGMENT>
struct LoopICountExecute<simd_exec, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {

    const int len = segment.end() - segment.begin();
    RAJA_SIMD
    for (int i = 0; i < len; i++)
    {
      body(*(segment.begin() + i), i);
    }
  }
};

}  // namespace RAJA
#endif
