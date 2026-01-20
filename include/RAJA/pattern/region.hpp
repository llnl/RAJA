/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing the RAJA Region API call
 *
 *             \code region<exec_policy>(loop body ); \endcode
 *
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

#ifndef RAJA_region_HPP
#define RAJA_region_HPP

#include "RAJA/policy/sequential/region.hpp"

namespace RAJA
{

template<typename ExecutionPolicy, typename LoopBody>
void region(LoopBody&& loop_body)
{
  region_impl(ExecutionPolicy(), loop_body);
}

template<typename ExecutionPolicy, typename OuterBody, typename InnerBody>
void region(OuterBody&& outer_body, InnerBody&& inner_body)
{
  region_impl(ExecutionPolicy(), outer_body, inner_body);
}

}  // namespace RAJA


#endif  // closing endif for header file include guard
