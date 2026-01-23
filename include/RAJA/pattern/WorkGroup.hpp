/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA WorkPool and WorkGroup declarations.
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

#ifndef RAJA_PATTERN_WorkGroup_HPP
#define RAJA_PATTERN_WorkGroup_HPP

#include "RAJA/config.hpp"

#include "RAJA/pattern/WorkGroup/WorkStorage.hpp"
#include "RAJA/pattern/WorkGroup/WorkRunner.hpp"

#include "RAJA/internal/get_platform.hpp"
#include "RAJA/util/plugins.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  xargs alias.
 *
 * Usage example:
 *
 * \verbatim

   WorkPool<WorkGroup_policy, Index_type, xargs<int*, int>, Allocator>
 pool(allocator);

   pool.enqueue(..., [=] (Index_type i, int* xarg0, int xarg1) {
      xarg0[i] = xarg1;
   });

   WorkGroup<WorkGroup_policy, Index_type, xargs<int*, int>, Allocator> group =
 pool.instantiate();

   int* xarg0 = ...;
   int xarg1 = ...;
   WorkSite<WorkGroup_policy, Index_type, xargs<int*, int>, Allocator> site =
 group.run(xarg0, xarg1);

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename... Args>
using xargs = camp::list<Args...>;

namespace detail
{

template<typename T>
struct is_xargs
{
  static constexpr bool value = false;
};

template<typename... Args>
struct is_xargs<xargs<Args...>>
{
  static constexpr bool value = true;
};

}  // namespace detail

//
// Forward declarations for WorkPool and WorkGroup templates.
// Actual classes appear in forall_*.hxx header files.
//

/*!
 ******************************************************************************
 *
 * \brief  WorkPool class template.
 *
 * The WorkPool object is the first member of the workgroup constructs. It
 * takes loops via enqueue and stores the loops so the loops can be run later.
 * The WorkPool creates a WorkGroup object with the enqueued collection of
 * loops via instantiate. The WorkPool can then be reused to enqueue more loops.
 * The WorkPool attempts to optimize storage usage by remembering the max number
 * of loops and the max storage size previously used and automatically reserving
 * that amount when it is reused.
 *
 * Usage example:
 *
 * \verbatim

   WorkPool<WorkGroup_policy, Index_type, xargs<>, Allocator> pool(allocator);

   Real_ptr data = ...;

   pool.enqueue( ..., [=] (Index_type i) {
      data[i] = 1;
   });

   WorkGroup<WorkGroup_policy, Index_type, xargs<>, Allocator> group =
 pool.instantiate();

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename WORKGROUP_POLICY_T,
         typename INDEX_T,
         typename EXTRA_ARGS_T,
         typename ALLOCATOR_T>
struct WorkPool
{
  static_assert(
      RAJA::pattern_is<WORKGROUP_POLICY_T, RAJA::Pattern::workgroup>::value,
      "WorkPool: WORKGROUP_POLICY_T must be a workgroup policy");
  static_assert(detail::is_xargs<EXTRA_ARGS_T>::value,
                "WorkPool: EXTRA_ARGS_T must be a RAJA::xargs<...> type");
};

/*!
 ******************************************************************************
 *
 * \brief  WorkGroup class template. Owns loops from an instantiated WorkPool.
 *
 * The WorkGroup object is the second member of the workgroup constructs. It
 * is created by a WorkPool and stores a collection of loops so they can be
 * run. When the WorkGroup is run it creates a WorkSite object with any per run
 * data. Because the WorkGroup owns a collection of loops it must not be
 * destroyed before that collection of loops has finished running. The
 * WorkGroup can be used to run its collection of loops multiple times.
 *
 * Usage example:
 *
 * \verbatim

   WorkGroup<WorkGroup_policy, Index_type, xargs<>, Allocator> group =
 pool.instantiate();

   WorkSite<WorkGroup_policy, Index_type, xargs<>, Allocator> site =
 group.run();

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename WORKGROUP_POLICY_T,
         typename INDEX_T,
         typename EXTRA_ARGS_T,
         typename ALLOCATOR_T>
struct WorkGroup
{
  static_assert(
      RAJA::pattern_is<WORKGROUP_POLICY_T, RAJA::Pattern::workgroup>::value,
      "WorkGroup: WORKGROUP_POLICY_T must be a workgroup policy");
  static_assert(detail::is_xargs<EXTRA_ARGS_T>::value,
                "WorkGroup: EXTRA_ARGS_T must be a RAJA::xargs<...> type");
};

/*!
 ******************************************************************************
 *
 * \brief  WorkSite class template. Owns per run objects from a single run of
 *         a WorkGroup.
 *
 * The WorkSite object is the third member of the workgroup constructs. It is
 * created by a WorkGroup when calling run and stores any data needed for that
 * run of that WorkGroup. Because the WorkSite owns data used for the running
 * of a collection of loops it must not be destroyed before that collection
 * of loops has finished running.
 *
 * Usage example:
 *
 * \verbatim

   WorkSite<WorkGroup_policy, Index_type, xargs<>, Allocator> site =
 group.run();

   site.synchronize();

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename WORKGROUP_POLICY_T,
         typename INDEX_T,
         typename EXTRA_ARGS_T,
         typename ALLOCATOR_T>
struct WorkSite
{
  static_assert(
      RAJA::pattern_is<WORKGROUP_POLICY_T, RAJA::Pattern::workgroup>::value,
      "WorkSite: WORKGROUP_POLICY_T must be a workgroup policy");
  static_assert(detail::is_xargs<EXTRA_ARGS_T>::value,
                "WorkSite: EXTRA_ARGS_T must be a RAJA::xargs<...> type");
};

template<typename EXEC_POLICY_T,
         typename ORDER_POLICY_T,
         typename STORAGE_POLICY_T,
         typename DISPATCH_POLICY_T,
         typename INDEX_T,
         typename... Args,
         typename ALLOCATOR_T>
struct WorkPool<WorkGroupPolicy<EXEC_POLICY_T,
                                ORDER_POLICY_T,
                                STORAGE_POLICY_T,
                                DISPATCH_POLICY_T>,
                INDEX_T,
                xargs<Args...>,
                ALLOCATOR_T>
{
  using exec_policy     = EXEC_POLICY_T;
  using order_policy    = ORDER_POLICY_T;
  using storage_policy  = STORAGE_POLICY_T;
  using dispatch_policy = DISPATCH_POLICY_T;
  using policy          = WorkGroupPolicy<exec_policy,
                                 order_policy,
                                 storage_policy,
                                 dispatch_policy>;
  using index_type      = INDEX_T;
  using xarg_type       = xargs<Args...>;
  using Allocator       = ALLOCATOR_T;

  using workgroup_type = WorkGroup<policy, index_type, xarg_type, Allocator>;
  using worksite_type  = WorkSite<policy, index_type, xarg_type, Allocator>;

private:
  using workrunner_type = detail::WorkRunner<exec_policy,
                                             order_policy,
                                             dispatch_policy,
                                             Allocator,
                                             index_type,
                                             Args...>;
  using storage_type =
      detail::WorkStorage<storage_policy,
                          Allocator,
                          typename workrunner_type::dispatcher_type>;

  friend workgroup_type;
  friend worksite_type;

public:
  using resource_type = typename workrunner_type::resource_type;

  explicit WorkPool(Allocator const& aloc) : m_storage(aloc) {}

  WorkPool(WorkPool const&)            = delete;
  WorkPool& operator=(WorkPool const&) = delete;

  WorkPool(WorkPool&&)            = default;
  WorkPool& operator=(WorkPool&&) = default;

  size_t num_loops() const { return m_storage.size(); }

  size_t storage_bytes() const { return m_storage.storage_size(); }

  void reserve(size_t num_loops, size_t storage_bytes)
  {
    m_storage.reserve(num_loops, storage_bytes);
  }

  template<typename segment_T, typename loop_T>
  inline void enqueue(segment_T&& seg, loop_T&& loop_body)
  {
    {
      // ignore zero length loops
      using std::begin;
      using std::end;
      if (begin(seg) == end(seg)) return;
    }
    if (m_storage.begin() == m_storage.end())
    {
      // perform auto-reserve on reuse
      reserve(m_max_num_loops, m_max_storage_bytes);
    }

    util::PluginContext context {
        util::make_context<exec_policy>(std::string())};
    util::callPreCapturePlugins(context);

    using RAJA::util::trigger_updates_before;
    auto body = trigger_updates_before(loop_body);

    m_runner.enqueue(m_storage, std::forward<segment_T>(seg), std::move(body));

    util::callPostCapturePlugins(context);
  }

  inline workgroup_type instantiate();

  void clear()
  {
    // storage is about to be destroyed
    // but it was never used so no synchronization necessary
    m_storage.clear();
    m_runner.clear();
  }

  ~WorkPool() { clear(); }

private:
  storage_type m_storage;
  size_t m_max_num_loops     = 0;
  size_t m_max_storage_bytes = 0;

  workrunner_type m_runner;
};

template<typename EXEC_POLICY_T,
         typename ORDER_POLICY_T,
         typename STORAGE_POLICY_T,
         typename DISPATCH_POLICY_T,
         typename INDEX_T,
         typename... Args,
         typename ALLOCATOR_T>
struct WorkGroup<WorkGroupPolicy<EXEC_POLICY_T,
                                 ORDER_POLICY_T,
                                 STORAGE_POLICY_T,
                                 DISPATCH_POLICY_T>,
                 INDEX_T,
                 xargs<Args...>,
                 ALLOCATOR_T>
{
  using exec_policy     = EXEC_POLICY_T;
  using order_policy    = ORDER_POLICY_T;
  using storage_policy  = STORAGE_POLICY_T;
  using dispatch_policy = DISPATCH_POLICY_T;
  using policy          = WorkGroupPolicy<exec_policy,
                                 order_policy,
                                 storage_policy,
                                 dispatch_policy>;
  using index_type      = INDEX_T;
  using xarg_type       = xargs<Args...>;
  using Allocator       = ALLOCATOR_T;

  using workpool_type = WorkPool<policy, index_type, xarg_type, Allocator>;
  using worksite_type = WorkSite<policy, index_type, xarg_type, Allocator>;

private:
  using storage_type    = typename workpool_type::storage_type;
  using workrunner_type = typename workpool_type::workrunner_type;

  friend workpool_type;
  friend worksite_type;

public:
  using resource_type = typename workpool_type::resource_type;

  WorkGroup(WorkGroup const&)            = delete;
  WorkGroup& operator=(WorkGroup const&) = delete;

  WorkGroup(WorkGroup&&)            = default;
  WorkGroup& operator=(WorkGroup&&) = default;

  inline worksite_type run(resource_type r, Args...);

  worksite_type run(Args... args)
  {
    auto r = resource_type::get_default();
    return run(r, std::move(args)...);
  }

  void clear()
  {
    // storage is about to be destroyed
    // TODO: synchronize
    m_storage.clear();
    m_runner.clear();
  }

  ~WorkGroup() { clear(); }

private:
  storage_type m_storage;
  workrunner_type m_runner;

  WorkGroup(storage_type&& storage, workrunner_type&& runner)
      : m_storage(std::move(storage)),
        m_runner(std::move(runner))
  {}
};

template<typename EXEC_POLICY_T,
         typename ORDER_POLICY_T,
         typename STORAGE_POLICY_T,
         typename DISPATCH_POLICY_T,
         typename INDEX_T,
         typename... Args,
         typename ALLOCATOR_T>
struct WorkSite<WorkGroupPolicy<EXEC_POLICY_T,
                                ORDER_POLICY_T,
                                STORAGE_POLICY_T,
                                DISPATCH_POLICY_T>,
                INDEX_T,
                xargs<Args...>,
                ALLOCATOR_T>
{
  using exec_policy     = EXEC_POLICY_T;
  using order_policy    = ORDER_POLICY_T;
  using storage_policy  = STORAGE_POLICY_T;
  using dispatch_policy = DISPATCH_POLICY_T;
  using policy          = WorkGroupPolicy<exec_policy,
                                 order_policy,
                                 storage_policy,
                                 dispatch_policy>;
  using index_type      = INDEX_T;
  using xarg_type       = xargs<Args...>;
  using Allocator       = ALLOCATOR_T;

  using workpool_type  = WorkPool<policy, index_type, xarg_type, Allocator>;
  using workgroup_type = WorkGroup<policy, index_type, xarg_type, Allocator>;

private:
  using workrunner_type = typename workgroup_type::workrunner_type;
  using per_run_storage = typename workrunner_type::per_run_storage;

  friend workpool_type;
  friend workgroup_type;

public:
  using resource_type = typename workpool_type::resource_type;

  WorkSite(WorkSite const&)            = delete;
  WorkSite& operator=(WorkSite const&) = delete;

  WorkSite(WorkSite&&)            = default;
  WorkSite& operator=(WorkSite&&) = default;

  resource_type get_resource() const { return m_resource; }

  void clear()
  {
    // resources is about to be released
    // TODO: synchronize
  }

  ~WorkSite() { clear(); }

private:
  per_run_storage m_run_storage;
  resource_type m_resource;

  explicit WorkSite(resource_type r, per_run_storage&& run_storage)
      : m_run_storage(std::move(run_storage)),
        m_resource(r)
  {}
};

template<typename EXEC_POLICY_T,
         typename ORDER_POLICY_T,
         typename STORAGE_POLICY_T,
         typename DISPATCH_POLICY_T,
         typename INDEX_T,
         typename... Args,
         typename ALLOCATOR_T>
inline typename WorkPool<WorkGroupPolicy<EXEC_POLICY_T,
                                         ORDER_POLICY_T,
                                         STORAGE_POLICY_T,
                                         DISPATCH_POLICY_T>,
                         INDEX_T,
                         xargs<Args...>,
                         ALLOCATOR_T>::workgroup_type
WorkPool<WorkGroupPolicy<EXEC_POLICY_T,
                         ORDER_POLICY_T,
                         STORAGE_POLICY_T,
                         DISPATCH_POLICY_T>,
         INDEX_T,
         xargs<Args...>,
         ALLOCATOR_T>::instantiate()
{
  // update max sizes to auto-reserve on reuse
  m_max_num_loops     = std::max(m_storage.size(), m_max_num_loops);
  m_max_storage_bytes = std::max(m_storage.storage_size(), m_max_storage_bytes);

  // move storage into workgroup
  return workgroup_type {std::move(m_storage), std::move(m_runner)};
}

template<typename EXEC_POLICY_T,
         typename ORDER_POLICY_T,
         typename STORAGE_POLICY_T,
         typename DISPATCH_POLICY_T,
         typename INDEX_T,
         typename... Args,
         typename ALLOCATOR_T>
inline typename WorkGroup<WorkGroupPolicy<EXEC_POLICY_T,
                                          ORDER_POLICY_T,
                                          STORAGE_POLICY_T,
                                          DISPATCH_POLICY_T>,
                          INDEX_T,
                          xargs<Args...>,
                          ALLOCATOR_T>::worksite_type
WorkGroup<
    WorkGroupPolicy<EXEC_POLICY_T,
                    ORDER_POLICY_T,
                    STORAGE_POLICY_T,
                    DISPATCH_POLICY_T>,
    INDEX_T,
    xargs<Args...>,
    ALLOCATOR_T>::run(typename WorkGroup<WorkGroupPolicy<EXEC_POLICY_T,
                                                         ORDER_POLICY_T,
                                                         STORAGE_POLICY_T,
                                                         DISPATCH_POLICY_T>,
                                         INDEX_T,
                                         xargs<Args...>,
                                         ALLOCATOR_T>::resource_type r,
                      Args... args)
{
  util::PluginContext context {
      util::make_context<EXEC_POLICY_T>(std::string())};
  util::callPreLaunchPlugins(context);

  // move any per run storage into worksite
  worksite_type site(r,
                     m_runner.run(m_storage, r, std::forward<Args>(args)...));

  util::callPostLaunchPlugins(context);

  return site;
}

}  // namespace RAJA

#endif  // closing endif for header file include guard
