/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA WorkStruct for workgroup.
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

#ifndef RAJA_PATTERN_WORKGROUP_WorkStruct_HPP
#define RAJA_PATTERN_WORKGROUP_WorkStruct_HPP

#include "RAJA/config.hpp"

#include <utility>
#include <cstddef>

#include "RAJA/pattern/WorkGroup/Dispatcher.hpp"

namespace RAJA
{

namespace detail
{

/*!
 * A struct that gives a generic way to layout memory for different loops
 */
template<size_t size, typename Dispatcher_T>
struct WorkStruct;

/*!
 * Generic struct used to layout memory for structs of unknown size.
 * Assumptions for any size (checked in construct):
 *   offsetof(GenericWorkStruct<>, obj) == offsetof(WorkStruct<size>, obj)
 *   sizeof(GenericWorkStruct) <= sizeof(WorkStruct<size>)
 */
template<typename Dispatcher_T>
using GenericWorkStruct = WorkStruct<RAJA_MAX_ALIGN, Dispatcher_T>;

template<size_t size,
         Platform platform,
         typename dispatch_policy,
         typename DispatcherID,
         typename... CallArgs>
struct WorkStruct<
    size,
    Dispatcher<platform, dispatch_policy, DispatcherID, CallArgs...>>
{
  using dispatcher_type =
      Dispatcher<platform, dispatch_policy, DispatcherID, CallArgs...>;

  // construct a WorkStruct with a value of type holder from the args and
  // check a variety of constraints at compile time
  template<typename holder, typename... holder_ctor_args>
  static RAJA_INLINE void construct(void* ptr,
                                    const dispatcher_type* dispatcher,
                                    holder_ctor_args&&... ctor_args)
  {
    using true_value_type = WorkStruct<sizeof(holder), dispatcher_type>;
    using value_type      = GenericWorkStruct<dispatcher_type>;

    static_assert(sizeof(holder) <= sizeof(true_value_type::obj),
                  "holder must fit in WorkStruct::obj");
    static_assert(std::is_standard_layout<true_value_type>::value,
                  "WorkStruct must be a standard layout type");
    static_assert(std::is_standard_layout<value_type>::value,
                  "GenericWorkStruct must be a standard layout type");
    static_assert(
        offsetof(value_type, obj) == offsetof(true_value_type, obj),
        "WorkStruct and GenericWorkStruct must have obj at the same offset");
    static_assert(sizeof(value_type) <= sizeof(true_value_type),
                  "WorkStruct must not be smaller than GenericWorkStruct");
    true_value_type* value_ptr = static_cast<true_value_type*>(ptr);

    value_ptr->dispatcher = dispatcher;
    value_ptr->invoke     = dispatcher->invoke;
    new (&value_ptr->obj) holder(std::forward<holder_ctor_args>(ctor_args)...);
  }

  // move construct in dst from the value in src and destroy the value in src
  static RAJA_INLINE void move_destroy(WorkStruct* value_dst,
                                       WorkStruct* value_src)
  {
    value_dst->dispatcher = value_src->dispatcher;
    value_dst->invoke     = value_src->invoke;
    value_dst->dispatcher->move_construct_destroy(&value_dst->obj,
                                                  &value_src->obj);
  }

  // destroy the value ptr
  static RAJA_INLINE void destroy(WorkStruct* value_ptr)
  {
    value_ptr->dispatcher->destroy(&value_ptr->obj);
  }

  // invoke the call operator of the value ptr with args
  static RAJA_INLINE void host_call(const WorkStruct* value_ptr,
                                    CallArgs... args)
  {
    value_ptr->invoke(&value_ptr->obj, std::forward<CallArgs>(args)...);
  }

  ///
  // invoke the call operator of the value ptr with args
  static RAJA_DEVICE RAJA_INLINE void device_call(const WorkStruct* value_ptr,
                                                  CallArgs... args)
  {
    value_ptr->invoke(&value_ptr->obj, std::forward<CallArgs>(args)...);
  }

  const dispatcher_type* dispatcher;
  typename dispatcher_type::invoker_type invoke;
  typename std::aligned_storage<size, RAJA_MAX_ALIGN>::type obj;
};

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
