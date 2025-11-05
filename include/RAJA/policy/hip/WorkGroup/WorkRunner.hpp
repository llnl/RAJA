/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA WorkRunner class specializations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_hip_WorkGroup_WorkRunner_HPP
#define RAJA_hip_WorkGroup_WorkRunner_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/MemUtils_HIP.hpp"

#include "RAJA/pattern/WorkGroup/WorkRunner.hpp"

namespace RAJA
{

namespace detail
{

/*!
 * Runs work in a storage container in order
 * and returns any per run resources
 */
template<size_t BLOCK_SIZE,
         bool Async,
         typename DISPATCH_POLICY_T,
         typename ALLOCATOR_T,
         typename INDEX_T,
         typename... Args>
struct WorkRunner<RAJA::hip_work<BLOCK_SIZE, Async>,
                  RAJA::ordered,
                  DISPATCH_POLICY_T,
                  ALLOCATOR_T,
                  INDEX_T,
                  Args...>
    : WorkRunnerForallOrdered<RAJA::hip_exec_async<BLOCK_SIZE>,
                              RAJA::hip_work<BLOCK_SIZE, Async>,
                              RAJA::ordered,
                              DISPATCH_POLICY_T,
                              ALLOCATOR_T,
                              INDEX_T,
                              Args...>
{
  using base = WorkRunnerForallOrdered<RAJA::hip_exec_async<BLOCK_SIZE>,
                                       RAJA::hip_work<BLOCK_SIZE, Async>,
                                       RAJA::ordered,
                                       DISPATCH_POLICY_T,
                                       ALLOCATOR_T,
                                       INDEX_T,
                                       Args...>;
  using base::base;
  using IndexType       = INDEX_T;
  using per_run_storage = typename base::per_run_storage;

  ///
  /// run the loops in the given work container in order using forall
  /// run all loops asynchronously and synchronize after is necessary
  ///
  template<typename WorkContainer>
  per_run_storage run(WorkContainer const& storage,
                      typename base::resource_type r,
                      Args... args) const
  {
    per_run_storage run_storage =
        base::run(storage, r, std::forward<Args>(args)...);

    IndexType num_loops = std::distance(std::begin(storage), std::end(storage));

    // Only synchronize if we had something to iterate over
    if (num_loops > 0 && BLOCK_SIZE > 0)
    {
      if (!Async)
      {
        RAJA::hip::synchronize(r);
      }
    }

    return run_storage;
  }
};

/*!
 * Runs work in a storage container in reverse order
 * and returns any per run resources
 */
template<size_t BLOCK_SIZE,
         bool Async,
         typename DISPATCH_POLICY_T,
         typename ALLOCATOR_T,
         typename INDEX_T,
         typename... Args>
struct WorkRunner<RAJA::hip_work<BLOCK_SIZE, Async>,
                  RAJA::reverse_ordered,
                  DISPATCH_POLICY_T,
                  ALLOCATOR_T,
                  INDEX_T,
                  Args...>
    : WorkRunnerForallReverse<RAJA::hip_exec_async<BLOCK_SIZE>,
                              RAJA::hip_work<BLOCK_SIZE, Async>,
                              RAJA::reverse_ordered,
                              DISPATCH_POLICY_T,
                              ALLOCATOR_T,
                              INDEX_T,
                              Args...>
{
  using base = WorkRunnerForallReverse<RAJA::hip_exec_async<BLOCK_SIZE>,
                                       RAJA::hip_work<BLOCK_SIZE, Async>,
                                       RAJA::reverse_ordered,
                                       DISPATCH_POLICY_T,
                                       ALLOCATOR_T,
                                       INDEX_T,
                                       Args...>;
  using base::base;
  using IndexType       = INDEX_T;
  using per_run_storage = typename base::per_run_storage;

  ///
  /// run the loops in the given work container in reverse order using forall
  /// run all loops asynchronously and synchronize after is necessary
  ///
  template<typename WorkContainer>
  per_run_storage run(WorkContainer const& storage,
                      typename base::resource_type r,
                      Args... args) const
  {
    per_run_storage run_storage =
        base::run(storage, r, std::forward<Args>(args)...);

    IndexType num_loops = std::distance(std::begin(storage), std::end(storage));

    // Only synchronize if we had something to iterate over
    if (num_loops > 0 && BLOCK_SIZE > 0)
    {
      if (!Async)
      {
        RAJA::hip::synchronize(r);
      }
    }

    return run_storage;
  }
};

/*!
 * A body and segment holder for storing loops that will be executed
 * on the device
 */
template<typename Segment_type,
         typename LoopBody,
         typename index_type,
         typename... Args>
struct HoldHipDeviceXThreadblockLoop
{
  template<typename segment_in, typename body_in>
  HoldHipDeviceXThreadblockLoop(segment_in&& segment, body_in&& body)
      : m_segment(std::forward<segment_in>(segment)),
        m_body(std::forward<body_in>(body))
  {}

  RAJA_DEVICE RAJA_INLINE void operator()(Args... args) const
  {
    // TODO:: decide when to run hooks, may bypass this and use impl directly
    // TODO:: decide whether or not to privatize the loop body
    const index_type i_begin = threadIdx.x + blockIdx.x * blockDim.x;
    const index_type stride  = blockDim.x * gridDim.x;
    const auto begin         = m_segment.begin();
    const auto end           = m_segment.end();
    const index_type len(end - begin);
    for (index_type i = i_begin; i < len; i += stride)
    {
      m_body(begin[i], std::forward<Args>(args)...);
    }
  }

private:
  Segment_type m_segment;
  LoopBody m_body;
};

template<size_t BLOCK_SIZE,
         typename StorageIter,
         typename value_type,
         typename index_type,
         typename... Args>
__launch_bounds__(BLOCK_SIZE, 1) __global__
    void hip_unordered_y_block_global(const StorageIter iter, Args... args)
{
  const index_type i_loop = blockIdx.y;
  // TODO: cache pointer to value_type in shared memory
  // TODO: cache holder (value_type::obj) in shared memory
  value_type::device_call(&iter[i_loop], args...);
}

/*!
 * Runs work in a storage container out of order with loops mapping to
 * hip blocks in the y direction and iterations mapping to threads in
 * the x direction, with the number of threads in the x dimension determined
 * by the average number of iterates per loop
 */
template<size_t BLOCK_SIZE,
         bool Async,
         typename DISPATCH_POLICY_T,
         typename ALLOCATOR_T,
         typename INDEX_T,
         typename... Args>
struct WorkRunner<
    RAJA::hip_work<BLOCK_SIZE, Async>,
    RAJA::policy::hip::unordered_hip_loop_y_block_iter_x_threadblock_average,
    DISPATCH_POLICY_T,
    ALLOCATOR_T,
    INDEX_T,
    Args...>
{
  using exec_policy = RAJA::hip_work<BLOCK_SIZE, Async>;
  using order_policy =
      RAJA::policy::hip::unordered_hip_loop_y_block_iter_x_threadblock_average;
  using dispatch_policy = DISPATCH_POLICY_T;
  using Allocator       = ALLOCATOR_T;
  using index_type      = INDEX_T;
  using resource_type   = resources::Hip;

  // The type that will hold the segment and loop body in work storage
  struct holder_type
  {
    template<typename T>
    using type = HoldHipDeviceXThreadblockLoop<
        typename camp::at<T, camp::num<0>>::type,  // ITERABLE
        typename camp::at<T, camp::num<1>>::type,  // LOOP_BODY
        index_type,
        Args...>;
  };

  ///
  template<typename T>
  using holder_type_t = typename holder_type::template type<T>;

  // The policy indicating where the call function is invoked
  // in this case the values are called on the device
  using dispatcher_exec_policy = exec_policy;

  // The Dispatcher policy with holder_types used internally to handle the
  // ranges and callables passed in by the user.
  using dispatcher_holder_policy =
      dispatcher_transform_types_t<dispatch_policy, holder_type>;

  using dispatcher_type = Dispatcher<Platform::hip,
                                     dispatcher_holder_policy,
                                     RAJA::hip_work<BLOCK_SIZE, true>,
                                     Args...>;

  WorkRunner() = default;

  WorkRunner(WorkRunner const&)            = delete;
  WorkRunner& operator=(WorkRunner const&) = delete;

  WorkRunner(WorkRunner&& o) : m_total_iterations(o.m_total_iterations)
  {
    o.m_total_iterations = 0;
  }

  WorkRunner& operator=(WorkRunner&& o)
  {
    m_total_iterations = o.m_total_iterations;

    o.m_total_iterations = 0;
    return *this;
  }

  // runner interfaces with storage to enqueue so the runner can get
  // information from the segment and loop at enqueue time
  template<typename WorkContainer, typename Iterable, typename LoopBody>
  inline void enqueue(WorkContainer& storage,
                      Iterable&& iter,
                      LoopBody&& loop_body)
  {
    using Iterator  = camp::decay<decltype(std::begin(iter))>;
    using LOOP_BODY = camp::decay<LoopBody>;
    using ITERABLE  = camp::decay<Iterable>;
    using IndexType =
        camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;

    using holder = holder_type_t<camp::list<ITERABLE, LOOP_BODY>>;

    // using true_value_type = typename WorkContainer::template
    // true_value_type<holder>;

    Iterator begin = std::begin(iter);
    Iterator end   = std::end(iter);
    IndexType len  = std::distance(begin, end);

    // Only launch kernel if we have something to iterate over
    if (len > 0 && BLOCK_SIZE > 0)
    {

      m_total_iterations += len;

      //
      // TODO: Privatize the loop_body, using make_launch_body to setup
      // reductions
      //
      // LOOP_BODY body = RAJA::hip::make_launch_body(func,
      //     gridSize, blockSize, shmem, stream,
      //     std::forward<LoopBody>(loop_body));

      storage.template emplace<holder>(
          get_Dispatcher<holder, dispatcher_type>(dispatcher_exec_policy {}),
          std::forward<Iterable>(iter), std::forward<LoopBody>(loop_body));
    }
  }

  // no extra storage required here
  using per_run_storage = int;

  template<typename WorkContainer>
  per_run_storage run(WorkContainer const& storage,
                      resource_type r,
                      Args... args) const
  {
    using Iterator   = camp::decay<decltype(std::begin(storage))>;
    using IndexType  = camp::decay<decltype(std::distance(std::begin(storage),
                                                          std::end(storage)))>;
    using value_type = typename WorkContainer::value_type;

    per_run_storage run_storage {};

    auto func = hip_unordered_y_block_global<BLOCK_SIZE, Iterator, value_type,
                                             index_type, Args...>;

    //
    // Compute the requested iteration space size
    //
    Iterator begin      = std::begin(storage);
    Iterator end        = std::end(storage);
    IndexType num_loops = std::distance(begin, end);

    // Only launch kernel if we have something to iterate over
    if (num_loops > 0 && BLOCK_SIZE > 0)
    {

      index_type average_iterations =
          m_total_iterations / static_cast<index_type>(num_loops);

      //
      // Compute the number of blocks
      //
      constexpr index_type block_size = static_cast<index_type>(BLOCK_SIZE);
      hip_dim_t blockSize {static_cast<hip_dim_member_t>(block_size), 1, 1};
      hip_dim_t gridSize {
          static_cast<hip_dim_member_t>((average_iterations + block_size - 1) /
                                        block_size),
          static_cast<hip_dim_member_t>(num_loops), 1};

      RAJA_FT_BEGIN;

      //
      // Setup shared memory buffers
      //
      size_t shmem = 0;

      {
        //
        // Launch the kernel
        //
        void* func_args[] = {(void*)&begin, (void*)&args...};
        RAJA::hip::launch((const void*)func, gridSize, blockSize, func_args,
                          shmem, r, Async);
      }

      RAJA_FT_END;
    }

    return run_storage;
  }

  // clear any state so ready to be destroyed or reused
  void clear() { m_total_iterations = 0; }

private:
  index_type m_total_iterations = 0;
};

#if !defined(RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL)

/// leave unsupported runner types incomplete
template<size_t BLOCK_SIZE,
         bool Async,
         typename ALLOCATOR_T,
         typename INDEX_T,
         typename... Args>
struct WorkRunner<
    RAJA::hip_work<BLOCK_SIZE, Async>,
    RAJA::policy::hip::unordered_hip_loop_y_block_iter_x_threadblock_average,
    RAJA::indirect_function_call_dispatch,
    ALLOCATOR_T,
    INDEX_T,
    Args...>;
///
template<size_t BLOCK_SIZE,
         bool Async,
         typename ALLOCATOR_T,
         typename INDEX_T,
         typename... Args>
struct WorkRunner<
    RAJA::hip_work<BLOCK_SIZE, Async>,
    RAJA::policy::hip::unordered_hip_loop_y_block_iter_x_threadblock_average,
    RAJA::indirect_virtual_function_dispatch,
    ALLOCATOR_T,
    INDEX_T,
    Args...>;

#endif

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
