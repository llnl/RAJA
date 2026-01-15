/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel statement collapse struct.
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


#ifndef RAJA_pattern_kernel_Collapse_HPP
#define RAJA_pattern_kernel_Collapse_HPP

namespace RAJA
{

namespace statement
{


template<typename ExecPolicy, typename ForList, typename... EnclosedStmts>
struct Collapse : public internal::ForList,
                  public internal::CollapseBase,
                  public internal::Statement<ExecPolicy, EnclosedStmts...>
{};


}  // namespace statement
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
