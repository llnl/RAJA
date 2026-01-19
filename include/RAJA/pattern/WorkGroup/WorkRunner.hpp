/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA WorkStorage.
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

#ifndef RAJA_PATTERN_WORKGROUP_WorkRunner_HPP
#define RAJA_PATTERN_WORKGROUP_WorkRunner_HPP

#include "RAJA/config.hpp"

#include <utility>
#include <type_traits>

#include "RAJA/policy/sequential/policy.hpp"

#include "RAJA/pattern/forall.hpp"

#include "RAJA/pattern/WorkGroup/Dispatcher.hpp"
#include "RAJA/policy/WorkGroup.hpp"

namespace RAJA
{

namespace detail
{

/*!
 * A body and args holder for storing loops that are being executed in foralls
 */
template<typename LoopBody, typename... Args>
struct HoldBodyArgs_base
{
  // NOTE: This constructor is disabled when body_in is not LoopBody
  // to avoid it conflicting with the copy and move constructors
  template<typename body_in,
           typename = typename std::enable_if<
               std::is_same<LoopBody, camp::decay<body_in>>::value>::type>
  HoldBodyArgs_base(body_in&& body, Args... args)
      : m_body(std::forward<body_in>(body)),
        m_arg_tuple(std::forward<Args>(args)...)
  {}

protected:
  LoopBody m_body;
  camp::tuple<Args...> m_arg_tuple;
};

/*!
 * A body and args holder for storing loops that are being executed in foralls
 * that run on the host
 */
template<typename LoopBody, typename index_type, typename... Args>
struct HoldBodyArgs_host : HoldBodyArgs_base<LoopBody, Args...>
{
  using base = HoldBodyArgs_base<LoopBody, Args...>;
  using base::base;

  RAJA_INLINE void operator()(index_type i) const
  {
    invoke(i, camp::make_idx_seq_t<sizeof...(Args)> {});
  }

  template<camp::idx_t... Is>
  RAJA_INLINE void invoke(index_type i, camp::idx_seq<Is...>) const
  {
    this->m_body(i, get<Is>(this->m_arg_tuple)...);
  }
};

/*!
 * A body and args holder for storing loops that are being executed in foralls
 * that run on the device
 */
template<typename LoopBody, typename index_type, typename... Args>
struct HoldBodyArgs_device : HoldBodyArgs_base<LoopBody, Args...>
{
  using base = HoldBodyArgs_base<LoopBody, Args...>;
  using base::base;

  RAJA_DEVICE RAJA_INLINE void operator()(index_type i) const
  {
    invoke(i, camp::make_idx_seq_t<sizeof...(Args)> {});
  }

  template<camp::idx_t... Is>
  RAJA_DEVICE RAJA_INLINE void invoke(index_type i, camp::idx_seq<Is...>) const
  {
    this->m_body(i, get<Is>(this->m_arg_tuple)...);
  }
};

/*!
 * A body and segment holder for storing loops that will be executed as foralls
 */
template<typename ExecutionPolicy,
         typename Segment_type,
         typename LoopBody,
         typename index_type,
         typename... Args>
struct HoldForall
{
  using resource_type = typename resources::get_resource<ExecutionPolicy>::type;
  using HoldBodyArgs  = typename std::conditional<
      !type_traits::is_device_exec_policy<ExecutionPolicy>::value,
      HoldBodyArgs_host<LoopBody, index_type, Args...>,
      HoldBodyArgs_device<LoopBody, index_type, Args...>>::type;

  template<typename segment_in, typename body_in>
  HoldForall(segment_in&& segment, body_in&& body)
      : m_segment(std::forward<segment_in>(segment)),
        m_body(std::forward<body_in>(body))
  {}

  RAJA_INLINE void operator()(resource_type r, Args... args) const
  {
    wrap::forall(r, ExecutionPolicy(), m_segment,
                 HoldBodyArgs {m_body, std::forward<Args>(args)...});
  }

private:
  Segment_type m_segment;
  LoopBody m_body;
};


/*!
 * A class that handles running work in a work container
 */
template<typename EXEC_POLICY_T,
         typename ORDER_POLICY_T,
         typename DISPATCH_POLICY_T,
         typename ALLOCATOR_T,
         typename INDEX_T,
         typename... Args>
struct WorkRunner;

/*!
 * Base class describing storage for ordered runners using forall
 */
template<typename FORALL_EXEC_POLICY,
         typename EXEC_POLICY_T,
         typename ORDER_POLICY_T,
         typename DISPATCH_POLICY_T,
         typename ALLOCATOR_T,
         typename INDEX_T,
         typename... Args>
struct WorkRunnerForallOrdered_base
{
  using exec_policy     = EXEC_POLICY_T;
  using order_policy    = ORDER_POLICY_T;
  using dispatch_policy = DISPATCH_POLICY_T;
  using Allocator       = ALLOCATOR_T;
  using index_type      = INDEX_T;
  using resource_type =
      typename resources::get_resource<FORALL_EXEC_POLICY>::type;

  using forall_exec_policy = FORALL_EXEC_POLICY;

  // The type that will hold the segment and loop body in work storage
  struct holder_type
  {
    template<typename T>
    using type =
        HoldForall<forall_exec_policy,
                   typename camp::at<T, camp::num<0>>::type,  // segment_type
                   typename camp::at<T, camp::num<1>>::type,  // loop_type
                   index_type,
                   Args...>;
  };

  ///
  template<typename T>
  using holder_type_t = typename holder_type::template type<T>;

  // The policy indicating where the call function is invoked
  // in this case the values are called on the host in a loop
  using dispatcher_exec_policy = RAJA::seq_work;

  // The Dispatcher policy with holder_types used internally to handle the
  // ranges and callables passed in by the user.
  using dispatcher_holder_policy =
      dispatcher_transform_types_t<dispatch_policy, holder_type>;

  using dispatcher_type = Dispatcher<Platform::host,
                                     dispatcher_holder_policy,
                                     void,
                                     resource_type,
                                     Args...>;

  WorkRunnerForallOrdered_base() = default;

  WorkRunnerForallOrdered_base(WorkRunnerForallOrdered_base const&) = delete;
  WorkRunnerForallOrdered_base& operator=(WorkRunnerForallOrdered_base const&) =
      delete;

  WorkRunnerForallOrdered_base(WorkRunnerForallOrdered_base&&) = default;
  WorkRunnerForallOrdered_base& operator=(WorkRunnerForallOrdered_base&&) =
      default;

  // runner interfaces with storage to enqueue so the runner can get
  // information from the segment and loop at enqueue time
  template<typename WorkContainer, typename segment_T, typename loop_T>
  inline void enqueue(WorkContainer& storage, segment_T&& seg, loop_T&& loop)
  {
    using holder =
        holder_type_t<camp::list<camp::decay<segment_T>, camp::decay<loop_T>>>;

    storage.template emplace<holder>(
        get_Dispatcher<holder, dispatcher_type>(dispatcher_exec_policy {}),
        std::forward<segment_T>(seg), std::forward<loop_T>(loop));
  }

  // clear any state so ready to be destroyed or reused
  void clear() {}

  // no extra storage required here
  using per_run_storage = int;
};

/*!
 * Runs work in a storage container in order using forall
 */
template<typename FORALL_EXEC_POLICY,
         typename EXEC_POLICY_T,
         typename ORDER_POLICY_T,
         typename DISPATCH_POLICY_T,
         typename ALLOCATOR_T,
         typename INDEX_T,
         typename... Args>
struct WorkRunnerForallOrdered
    : WorkRunnerForallOrdered_base<FORALL_EXEC_POLICY,
                                   EXEC_POLICY_T,
                                   ORDER_POLICY_T,
                                   DISPATCH_POLICY_T,
                                   ALLOCATOR_T,
                                   INDEX_T,
                                   Args...>
{
  using base = WorkRunnerForallOrdered_base<FORALL_EXEC_POLICY,
                                            EXEC_POLICY_T,
                                            ORDER_POLICY_T,
                                            DISPATCH_POLICY_T,
                                            ALLOCATOR_T,
                                            INDEX_T,
                                            Args...>;
  using base::base;

  // run the loops using forall in the order that they were enqueued
  template<typename WorkContainer>
  typename base::per_run_storage run(WorkContainer const& storage,
                                     typename base::resource_type r,
                                     Args... args) const
  {
    using value_type = typename WorkContainer::value_type;

    typename base::per_run_storage run_storage {};

    auto end = storage.end();
    for (auto iter = storage.begin(); iter != end; ++iter)
    {
      value_type::host_call(&*iter, r, args...);
    }

    return run_storage;
  }
};

/*!
 * Runs work in a storage container in reverse order using forall
 */
template<typename FORALL_EXEC_POLICY,
         typename EXEC_POLICY_T,
         typename ORDER_POLICY_T,
         typename DISPATCH_POLICY_T,
         typename ALLOCATOR_T,
         typename INDEX_T,
         typename... Args>
struct WorkRunnerForallReverse
    : WorkRunnerForallOrdered_base<FORALL_EXEC_POLICY,
                                   EXEC_POLICY_T,
                                   ORDER_POLICY_T,
                                   DISPATCH_POLICY_T,
                                   ALLOCATOR_T,
                                   INDEX_T,
                                   Args...>
{
  using base = WorkRunnerForallOrdered_base<FORALL_EXEC_POLICY,
                                            EXEC_POLICY_T,
                                            ORDER_POLICY_T,
                                            DISPATCH_POLICY_T,
                                            ALLOCATOR_T,
                                            INDEX_T,
                                            Args...>;
  using base::base;

  // run the loops using forall in the reverse order to the order they were
  // enqueued
  template<typename WorkContainer>
  typename base::per_run_storage run(WorkContainer const& storage,
                                     typename base::resource_type r,
                                     Args... args) const
  {
    using value_type = typename WorkContainer::value_type;

    typename base::per_run_storage run_storage {};

    auto begin = storage.begin();
    for (auto iter = storage.end(); iter != begin; --iter)
    {
      value_type::host_call(&*(iter - 1), r, args...);
    }

    return run_storage;
  }
};

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
