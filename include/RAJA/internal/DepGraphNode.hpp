/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining simple class to manage scheduling
 *          of nodes in a task dependency graph.
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

#ifndef RAJA_DepGraphNode_HPP
#define RAJA_DepGraphNode_HPP

#include "RAJA/config.hpp"

#include <atomic>
#include <cstdlib>
#include <iosfwd>
#include <thread>

#include "RAJA/util/types.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Class defining a simple semephore-based data structure for
 *         managing a node in a dependency graph.
 *
 ******************************************************************************
 */
class RAJA_ALIGNED_ATTR(256) DepGraphNode
{
public:
  ///
  /// Constants for total number of allowable dependent tasks
  /// and alignment of semaphore value (should match cache
  /// coherence page size?).
  ///
  /// These values may need to be set differently for different
  /// algorithms and platforms. We haven't determined the best defaults yet!
  ///
  static const int _MaxDepTasks_ = 8;

  ///
  /// Default ctor initializes node to default state.
  ///
  DepGraphNode()
      : m_num_dep_tasks(0),
        m_semaphore_reload_value(0),
        m_semaphore_value(0)
  {}

  ///
  /// Get/set semaphore value; i.e., the current number of (unsatisfied)
  /// dependencies that must be satisfied before this task can execute.
  ///
  std::atomic<int>& semaphoreValue() { return m_semaphore_value; }

  ///
  /// Get/set semaphore "reload" value; i.e., the total number of external
  /// task dependencies that must be satisfied before this task can execute.
  ///
  int& semaphoreReloadValue() { return m_semaphore_reload_value; }

  ///
  /// Ready this task to be used again
  ///
  void reset() { m_semaphore_value.store(m_semaphore_reload_value); }

  ///
  /// Satisfy one incoming dependency
  ///
  void satisfyOne()
  {
    if (m_semaphore_value > 0)
    {
      --m_semaphore_value;
    }
  }

  ///
  /// Wait for all dependencies to be satisfied
  ///
  void wait()
  {
    while (m_semaphore_value > 0)
    {
      // TODO: an efficient wait would be better here, but the standard
      // promise/future is not good enough
      std::this_thread::yield();
    }
  }

  ///
  /// Get/set the number of "forward-dependencies" for this task; i.e., the
  /// number of external tasks that cannot execute until this task completes.
  ///
  int& numDepTasks() { return m_num_dep_tasks; }

  ///
  /// Get/set the forward dependency task number associated with the given
  /// index for this task. This is used to notify the appropriate external
  /// dependencies when this task completes.
  ///
  int& depTaskNum(int tidx) { return m_dep_task[tidx]; }

  ///
  /// Print task graph object node data to given output stream.
  ///
  void print(std::ostream& os) const;

private:
  int m_dep_task[_MaxDepTasks_];
  int m_num_dep_tasks;
  int m_semaphore_reload_value;
  std::atomic<int> m_semaphore_value;
};

}  // namespace RAJA

#endif  // closing endif for header file include guard
