/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for dependency graph node class.
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

#include <iostream>
#include <string>

#include "RAJA/internal/DepGraphNode.hpp"

namespace RAJA
{

void DepGraphNode::print(std::ostream& os) const
{
  os << "DepGraphNode : sem, reload value = " << m_semaphore_value << " , "
     << m_semaphore_reload_value << std::endl;

  os << "     num dep tasks = " << m_num_dep_tasks;
  if (m_num_dep_tasks > 0)
  {
    os << " ( ";
    for (int jj = 0; jj < m_num_dep_tasks; ++jj)
    {
      os << m_dep_task[jj] << "  ";
    }
    os << " )";
  }
  os << std::endl;
}

}  // namespace RAJA
