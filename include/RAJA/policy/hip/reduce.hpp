/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for HIP execution.
 *
 *          These methods should work on any platform that supports
 *          HIP devices.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_hip_reduce_HPP
#define RAJA_hip_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <type_traits>

#include <hip/hip_runtime.h>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/SoAArray.hpp"
#include "RAJA/util/SoAPtr.hpp"
#include "RAJA/util/basic_mempool.hpp"
#include "RAJA/util/mutex.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/util/reduce.hpp"

#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/intrinsics.hpp"
#include "RAJA/policy/hip/atomic.hpp"
#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"

namespace RAJA
{

namespace reduce
{

namespace hip
{

//! atomic operator version of Combiner object
template<typename Combiner>
struct atomic;

template<typename T>
struct atomic<sum<T>>
{
  RAJA_DEVICE RAJA_INLINE void operator()(T& val, const T v)
  {
    RAJA::atomicAdd(RAJA::hip_atomic {}, &val, v);
  }
};

template<typename T>
struct atomic<min<T>>
{
  RAJA_DEVICE RAJA_INLINE void operator()(T& val, const T v)
  {
    RAJA::atomicMin(RAJA::hip_atomic {}, &val, v);
  }
};

template<typename T>
struct atomic<max<T>>
{
  RAJA_DEVICE RAJA_INLINE void operator()(T& val, const T v)
  {
    RAJA::atomicMax(RAJA::hip_atomic {}, &val, v);
  }
};

template<typename T>
struct atomic<and_bit<T>>
{
  RAJA_DEVICE RAJA_INLINE void operator()(T& val, const T v)
  {
    RAJA::atomicAnd(RAJA::hip_atomic {}, &val, v);
  }
};

template<typename T>
struct atomic<or_bit<T>>
{
  RAJA_DEVICE RAJA_INLINE void operator()(T& val, const T v)
  {
    RAJA::atomicOr(RAJA::hip_atomic {}, &val, v);
  }
};

template<typename T>
struct hip_atomic_available
{
  static constexpr const bool value =
      (std::is_integral<T>::value && (4 == sizeof(T) || 8 == sizeof(T))) ||
      std::is_same<T, float>::value || std::is_same<T, double>::value;
};

}  // namespace hip

}  // namespace reduce

namespace hip
{

namespace impl
{

//! reduce values in grid into thread 0 of last running block
//  returns true if put reduced value in val
template<typename Combiner,
         typename Accessor,
         int replication,
         int atomic_stride,
         typename T,
         typename TempIterator>
RAJA_DEVICE RAJA_INLINE int grid_reduce_last_block(T& val,
                                                   T identity,
                                                   TempIterator in_device_mem,
                                                   unsigned int* device_count)
{
  typename TempIterator::template rebind_accessor<Accessor> device_mem(
      in_device_mem);

  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;
  int numThreads = blockDim.x * blockDim.y * blockDim.z;

  int blockId = blockIdx.x + gridDim.x * blockIdx.y +
                (gridDim.x * gridDim.y) * blockIdx.z;
  int numBlocks = gridDim.x * gridDim.y * gridDim.z;

  int replicationId = blockId % replication;
  int slotId        = blockId / replication;

  int maxNumSlots       = (numBlocks + replication - 1) / replication;
  unsigned int numSlots = (numBlocks / replication) +
                          ((replicationId < (numBlocks % replication)) ? 1 : 0);

  int atomicOffset = replicationId * atomic_stride;
  int beginSlots   = replicationId * maxNumSlots;
  int blockSlot    = beginSlots + slotId;

  T temp = block_reduce<Combiner>(val, identity);

  if (numSlots <= 1u)
  {
    if (threadId == 0)
    {
      val = temp;
    }
    return (threadId == 0) ? replicationId : replication;
  }

  // one thread per block writes to device_mem
  __shared__ bool isLastBlock;
  if (threadId == 0)
  {
    device_mem.set(blockSlot, temp);
    // ensure write visible to all threadblocks
    Accessor::fence_release();
    // increment counter, (wraps back to zero if old count == (numSlots-1))
    unsigned int old_count =
        ::atomicInc(&device_count[atomicOffset], (numSlots - 1));
    isLastBlock = (old_count == (numSlots - 1));
  }

  // returns non-zero value if any thread passes in a non-zero value
  __syncthreads();

  // last block accumulates values from device_mem
  if (isLastBlock)
  {
    temp = identity;
    Accessor::fence_acquire();

    for (unsigned int i = threadId; i < numSlots; i += numThreads)
    {
      Combiner {}(temp, device_mem.get(beginSlots + i));
    }

    temp = block_reduce<Combiner>(temp, identity);

    // one thread returns value
    if (threadId == 0)
    {
      val = temp;
    }
  }

  return (isLastBlock && threadId == 0) ? replicationId : replication;
}

namespace expt
{

template<typename ThreadIterationGetter, typename Combiner, typename T>
RAJA_DEVICE RAJA_INLINE T block_reduce(T val, T identity)
{
  const int numThreads = ThreadIterationGetter::size();
  const int threadId   = ThreadIterationGetter::index();

  const int warpId  = threadId % RAJA::policy::hip::device_constants.WARP_SIZE;
  const int warpNum = threadId / RAJA::policy::hip::device_constants.WARP_SIZE;

  T temp = val;

  if (numThreads % RAJA::policy::hip::device_constants.WARP_SIZE == 0)
  {

    // reduce each warp
    for (int i = 1; i < RAJA::policy::hip::device_constants.WARP_SIZE; i *= 2)
    {
      T rhs = RAJA::hip::impl::shfl_xor_sync(temp, i);
      temp  = Combiner {}(temp, rhs);
    }
  }
  else
  {

    // reduce each warp
    for (int i = 1; i < RAJA::policy::hip::device_constants.WARP_SIZE; i *= 2)
    {
      int srcLane = threadId ^ i;
      T rhs       = RAJA::hip::impl::shfl_sync(temp, srcLane);
      // only add from threads that exist (don't double count own value)
      if (srcLane < numThreads)
      {
        temp = Combiner {}(temp, rhs);
      }
    }
  }

  static_assert(RAJA::policy::hip::device_constants.MAX_WARPS <=
                    RAJA::policy::hip::device_constants.WARP_SIZE,
                "Max Warps must be less than or equal to Warp Size for this "
                "algorithm to work");

  // reduce per warp values
  if (numThreads > RAJA::policy::hip::device_constants.WARP_SIZE)
  {

    // Need to separate declaration and initialization for clang-hip
    __shared__ unsigned char tmpsd[sizeof(
        RAJA::detail::SoAArray<T,
                               RAJA::policy::hip::device_constants.MAX_WARPS>)];

    // Partial placement new: Should call new(tmpsd) here but recasting memory
    // to avoid calling constructor/destructor in shared memory.
    RAJA::detail::SoAArray<T, RAJA::policy::hip::device_constants.MAX_WARPS>*
        sd = reinterpret_cast<RAJA::detail::SoAArray<
            T, RAJA::policy::hip::device_constants.MAX_WARPS>*>(tmpsd);

    // write per warp values to shared memory
    if (warpId == 0)
    {
      sd->set(warpNum, temp);
    }

    __syncthreads();

    if (warpNum == 0)
    {

      // read per warp values
      if (warpId * RAJA::policy::hip::device_constants.WARP_SIZE < numThreads)
      {
        temp = sd->get(warpId);
      }
      else
      {
        temp = identity;
      }

      for (int i = 1; i < RAJA::policy::hip::device_constants.MAX_WARPS; i *= 2)
      {
        T rhs = RAJA::hip::impl::shfl_xor_sync(temp, i);
        temp  = Combiner {}(temp, rhs);
      }
    }

    __syncthreads();
  }

  return temp;
}

template<typename GlobalIterationGetter, typename OP, typename T>
RAJA_DEVICE RAJA_INLINE void grid_reduce(
    T* device_target,
    T val,
    RAJA::detail::SoAPtr<T, RAJA::hip::device_mempool_type> device_mem,
    unsigned int* device_count)
{
  using BlockIterationGetter =
      typename get_index_block<GlobalIterationGetter>::type;
  using ThreadIterationGetter =
      typename get_index_thread<GlobalIterationGetter>::type;

  const int numBlocks            = BlockIterationGetter::size();
  const int numThreads           = ThreadIterationGetter::size();
  const unsigned int wrap_around = numBlocks - 1;

  const int blockId  = BlockIterationGetter::index();
  const int threadId = ThreadIterationGetter::index();

  T temp = block_reduce<ThreadIterationGetter, OP>(val, OP::identity());

  // one thread per block writes to device_mem
  bool lastBlock = false;
  if (threadId == 0)
  {
    device_mem.set(blockId, temp);
    // ensure write visible to all threadblocks
    __threadfence();
    // increment counter, (wraps back to zero if old count == wrap_around)
    unsigned int old_count = ::atomicInc(device_count, wrap_around);
    lastBlock              = (old_count == wrap_around);
  }

  // returns non-zero value if any thread passes in a non-zero value
  lastBlock = __syncthreads_or(lastBlock);

  // last block accumulates values from device_mem
  if (lastBlock)
  {
    temp = OP::identity();
    __threadfence();

    for (int i = threadId; i < numBlocks; i += numThreads)
    {
      temp = OP {}(temp, device_mem.get(i));
    }

    temp = block_reduce<ThreadIterationGetter, OP>(temp, OP::identity());

    // one thread returns value
    if (threadId == 0)
    {
      *device_target = temp;
    }
  }
}

}  //  namespace expt

//! reduce values in grid into thread 0 of last running block
//  returns true if put reduced value in val
template<typename Combiner,
         typename Accessor,
         int replication,
         int atomic_stride,
         typename T>
RAJA_DEVICE RAJA_INLINE int grid_reduce_atomic_device_init(
    T& val,
    T identity,
    T* device_mem,
    unsigned int* device_count)
{
  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;

  int blockId = blockIdx.x + gridDim.x * blockIdx.y +
                (gridDim.x * gridDim.y) * blockIdx.z;
  int numBlocks = gridDim.x * gridDim.y * gridDim.z;

  int replicationId = (blockId % replication);
  int atomicOffset  = replicationId * atomic_stride;

  unsigned int numSlots = (numBlocks / replication) +
                          ((replicationId < (numBlocks % replication)) ? 1 : 0);

  if (numSlots <= 1u)
  {
    T temp = block_reduce<Combiner>(val, identity);
    if (threadId == 0)
    {
      val = temp;
    }
    return (threadId == 0) ? replicationId : replication;
  }

  // the first block of each replication initializes device_mem
  if (threadId == 0)
  {
    unsigned int old_val = ::atomicCAS(&device_count[atomicOffset], 0u, 1u);
    if (old_val == 0u)
    {
      Accessor::set(device_mem, atomicOffset, identity);
      Accessor::fence_release();
      ::atomicAdd(&device_count[atomicOffset], 1u);
    }
  }

  T temp = block_reduce<Combiner>(val, identity);

  // one thread per block performs an atomic on device_mem
  bool isLastBlock = false;
  if (threadId == 0)
  {
    // wait for device_mem to be initialized
    while (::atomicAdd(&device_count[atomicOffset], 0u) < 2u)
      ;
    Accessor::fence_acquire();
    RAJA::reduce::hip::atomic<Combiner> {}(device_mem[atomicOffset], temp);
    Accessor::fence_release();
    // increment counter, (wraps back to zero if old count == (numSlots+1))
    unsigned int old_count =
        ::atomicInc(&device_count[atomicOffset], (numSlots + 1));
    isLastBlock = (old_count == (numSlots + 1));

    // the last block for each replication gets the value from device_mem
    if (isLastBlock)
    {
      Accessor::fence_acquire();
      val = Accessor::get(device_mem, atomicOffset);
    }
  }

  return isLastBlock ? replicationId : replication;
}

//! reduce values in block into thread 0 and atomically combines into device_mem
template<typename Combiner, int replication, int atomic_stride, typename T>
RAJA_DEVICE RAJA_INLINE void grid_reduce_atomic_host_init(T& val,
                                                          T identity,
                                                          T* device_mem)
{
  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;

  int blockId = blockIdx.x + gridDim.x * blockIdx.y +
                (gridDim.x * gridDim.y) * blockIdx.z;

  int replicationId = (blockId % replication);
  int atomicOffset  = replicationId * atomic_stride;

  T temp = block_reduce<Combiner>(val, identity);

  // one thread per block performs an atomic on device_mem
  if (threadId == 0 && temp != identity)
  {
    RAJA::reduce::hip::atomic<Combiner> {}(device_mem[atomicOffset], temp);
  }
}

}  // namespace impl

//! Object that manages pinned memory buffers for reduction results
//  use one per reducer object
template<typename T, size_t num_slots, typename mempool>
class PinnedTally
{
public:
  //! Object put in Pinned memory with value and pointer to next Node
  struct Node
  {
    Node* next;
    T values[num_slots];
  };

  //! Object per resource to keep track of pinned memory nodes
  struct ResourceNode
  {
    ResourceNode* next;
    ::RAJA::resources::Hip res;
    Node* node_list;
  };

  //! Iterator over resources used by reducer
  class ResourceIterator
  {
  public:
    ResourceIterator() = delete;

    ResourceIterator(ResourceNode* rn) : m_rn(rn) {}

    const ResourceIterator& operator++()
    {
      m_rn = m_rn->next;
      return *this;
    }

    ResourceIterator operator++(int)
    {
      ResourceIterator ret = *this;
      this->operator++();
      return ret;
    }

    ::RAJA::resources::Hip& operator*() { return m_rn->res; }

    bool operator==(const ResourceIterator& rhs) const
    {
      return m_rn == rhs.m_rn;
    }

    bool operator!=(const ResourceIterator& rhs) const
    {
      return !this->operator==(rhs);
    }

  private:
    ResourceNode* m_rn;
  };

  //! Iterator over all values generated by reducer
  class ResourceNodeIterator
  {
  public:
    ResourceNodeIterator() = delete;

    ResourceNodeIterator(ResourceNode* rn, Node* n) : m_rn(rn), m_n(n) {}

    const ResourceNodeIterator& operator++()
    {
      if (m_n->next)
      {
        m_n = m_n->next;
      }
      else if (m_rn->next)
      {
        m_rn = m_rn->next;
        m_n  = m_rn->node_list;
      }
      else
      {
        m_rn = nullptr;
        m_n  = nullptr;
      }
      return *this;
    }

    ResourceNodeIterator operator++(int)
    {
      ResourceNodeIterator ret = *this;
      this->operator++();
      return ret;
    }

    auto operator*() -> T (&)[num_slots] { return m_n->values; }

    bool operator==(const ResourceNodeIterator& rhs) const
    {
      return m_n == rhs.m_n;
    }

    bool operator!=(const ResourceNodeIterator& rhs) const
    {
      return !this->operator==(rhs);
    }

  private:
    ResourceNode* m_rn;
    Node* m_n;
  };

  PinnedTally() : resource_list(nullptr) {}

  PinnedTally(const PinnedTally&) = delete;

  //! get begin iterator over resources
  ResourceIterator resourceBegin() { return {resource_list}; }

  //! get end iterator over resources
  ResourceIterator resourceEnd() { return {nullptr}; }

  //! get begin iterator over values
  ResourceNodeIterator begin()
  {
    return {resource_list, resource_list ? resource_list->node_list : nullptr};
  }

  //! get end iterator over values
  ResourceNodeIterator end() { return {nullptr, nullptr}; }

  //! get new value for use in resource
  auto new_value(::RAJA::resources::Hip res) -> T (&)[num_slots]
  {
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(m_mutex);
#endif
    ResourceNode* rn = resource_list;
    while (rn)
    {
      if (rn->res.get_stream() == res.get_stream()) break;
      rn = rn->next;
    }
    if (!rn)
    {
      rn            = (ResourceNode*)malloc(sizeof(ResourceNode));
      rn->next      = resource_list;
      rn->res       = res;
      rn->node_list = nullptr;
      resource_list = rn;
    }
    Node* n       = mempool::getInstance().template malloc<Node>(1);
    n->next       = rn->node_list;
    rn->node_list = n;
    return n->values;
  }

  //! synchronize all resources used
  void synchronize_resources()
  {
    auto end = resourceEnd();
    for (auto r = resourceBegin(); r != end; ++r)
    {
      ::RAJA::hip::synchronize(*r);
    }
  }

  //! all values used in all resources
  void free_list()
  {
    while (resource_list)
    {
      ResourceNode* rn = resource_list;
      while (rn->node_list)
      {
        Node* n       = rn->node_list;
        rn->node_list = n->next;
        mempool::getInstance().free(n);
      }
      resource_list = rn->next;
      free(rn);
    }
  }

  ~PinnedTally() { free_list(); }

#if defined(RAJA_ENABLE_OPENMP)
  omp::mutex m_mutex;
#endif

private:
  ResourceNode* resource_list;
};

//
//////////////////////////////////////////////////////////////////////
//
// Reduction classes.
//
//////////////////////////////////////////////////////////////////////
//

//! Reduction data for Hip Offload -- stores value, host pointer, and device
//! pointer
template<typename Combiner,
         typename Accessor,
         typename T,
         size_t replication,
         size_t atomic_stride>
struct ReduceLastBlock_Data
{
  using tally_mempool_type = pinned_mempool_type;
  using data_mempool_type  = device_mempool_type;
  using count_mempool_type = device_zeroed_mempool_type;

  static constexpr size_t tally_slots = replication;

  mutable T value;
  T identity;
  unsigned int* device_count;
  RAJA::detail::SoAPtr<T, data_mempool_type> device;
  bool own_device_ptr;

  ReduceLastBlock_Data() : ReduceLastBlock_Data(T(), T()) {};

  /*! \brief create from a default value and offload information
   *
   *  allocates PinnedTally to hold device values
   */

  ReduceLastBlock_Data(T initValue, T identity_)
      : value {initValue},
        identity {identity_},
        device_count {nullptr},
        device {},
        own_device_ptr {false}
  {}

  RAJA_HOST_DEVICE
  ReduceLastBlock_Data(const ReduceLastBlock_Data& other)
      : value {other.identity},
        identity {other.identity},
        device_count {other.device_count},
        device {other.device},
        own_device_ptr {false}
  {}

  ReduceLastBlock_Data& operator=(const ReduceLastBlock_Data&) = default;

  //! initialize output to identity to ensure never read
  //  uninitialized memory
  T* init_grid_vals(T (&output)[tally_slots])
  {
    for (size_t r = 0; r < tally_slots; ++r)
    {
      output[r] = identity;
    }
    return &output[0];
  }

  //! reduce values in grid to single value, store in output
  RAJA_DEVICE
  void grid_reduce(T* output)
  {
    T temp = value;
    size_t replicationId =
        impl::grid_reduce_last_block<Combiner, Accessor, replication,
                                     atomic_stride>(temp, identity, device,
                                                    device_count);
    if (replicationId != replication)
    {
      output[replicationId] = temp;
    }
  }

  //! check and setup for device
  //  allocate device pointers and get a new result buffer from the pinned tally
  bool setupForDevice()
  {
    bool act = !device.allocated() && setupReducers();
    if (act)
    {
      hip_dim_t gridDim  = currentGridDim();
      size_t numBlocks   = gridDim.x * gridDim.y * gridDim.z;
      size_t maxNumSlots = (numBlocks + replication - 1) / replication;
      device.allocate(maxNumSlots * replication);
      device_count =
          count_mempool_type::getInstance().template malloc<unsigned int>(
              replication * atomic_stride);
      own_device_ptr = true;
    }
    return act;
  }

  //! if own resources teardown device setup
  //  free device pointers
  bool teardownForDevice()
  {
    bool act = own_device_ptr;
    if (act)
    {
      device.deallocate();
      count_mempool_type::getInstance().free(device_count);
      device_count   = nullptr;
      own_device_ptr = false;
    }
    return act;
  }
};

//! Reduction data for Hip Offload -- stores value, host pointer
template<typename Combiner,
         typename T,
         size_t replication,
         size_t atomic_stride>
struct ReduceAtomicHostInit_Data
{
  using tally_mempool_type = device_pinned_mempool_type;

  static constexpr size_t tally_slots = replication * atomic_stride;

  mutable T value;
  T identity;
  bool is_setup;
  bool own_device_ptr;

  ReduceAtomicHostInit_Data() : ReduceAtomicHostInit_Data(T(), T()) {}

  ReduceAtomicHostInit_Data(T initValue, T identity_)
      : value {initValue},
        identity {identity_},
        is_setup {false},
        own_device_ptr {false}
  {}

  RAJA_HOST_DEVICE
  ReduceAtomicHostInit_Data(const ReduceAtomicHostInit_Data& other)
      : value {other.identity},
        identity {other.identity},
        is_setup {other.is_setup},
        own_device_ptr {false}
  {}

  ReduceAtomicHostInit_Data& operator=(const ReduceAtomicHostInit_Data&) =
      default;

  //! initialize output to identity to ensure never read
  //  uninitialized memory
  T* init_grid_vals(T (&output)[tally_slots])
  {
    for (size_t r = 0; r < tally_slots; ++r)
    {
      output[r] = identity;
    }
    return &output[0];
  }

  //! reduce values in grid to single value, store in output
  RAJA_DEVICE
  void grid_reduce(T* output)
  {
    T temp = value;

    impl::grid_reduce_atomic_host_init<Combiner, replication, atomic_stride>(
        temp, identity, output);
  }

  //! check and setup for device
  //  allocate device pointers and get a new result buffer from the pinned tally
  bool setupForDevice()
  {
    bool act = !is_setup && setupReducers();
    if (act)
    {
      is_setup       = true;
      own_device_ptr = true;
    }
    return act;
  }

  //! if own resources teardown device setup
  //  free device pointers
  bool teardownForDevice()
  {
    bool act = own_device_ptr;
    if (act)
    {
      is_setup       = false;
      own_device_ptr = false;
    }
    return act;
  }
};

//! Reduction data for Hip Offload -- stores value, host pointer
template<typename Combiner,
         typename Accessor,
         typename T,
         size_t replication,
         size_t atomic_stride>
struct ReduceAtomicDeviceInit_Data
{
  using tally_mempool_type = pinned_mempool_type;
  using data_mempool_type  = device_mempool_type;
  using count_mempool_type = device_zeroed_mempool_type;

  static constexpr size_t tally_slots = replication;

  mutable T value;
  T identity;
  unsigned int* device_count;
  T* device;
  bool own_device_ptr;

  ReduceAtomicDeviceInit_Data() : ReduceAtomicDeviceInit_Data(T(), T()) {}

  ReduceAtomicDeviceInit_Data(T initValue, T identity_)
      : value {initValue},
        identity {identity_},
        device_count {nullptr},
        device {nullptr},
        own_device_ptr {false}
  {}

  RAJA_HOST_DEVICE
  ReduceAtomicDeviceInit_Data(const ReduceAtomicDeviceInit_Data& other)
      : value {other.identity},
        identity {other.identity},
        device_count {other.device_count},
        device {other.device},
        own_device_ptr {false}
  {}

  ReduceAtomicDeviceInit_Data& operator=(const ReduceAtomicDeviceInit_Data&) =
      default;

  //! initialize output to identity to ensure never read
  //  uninitialized memory
  T* init_grid_vals(T (&output)[tally_slots])
  {
    for (size_t r = 0; r < tally_slots; ++r)
    {
      output[r] = identity;
    }
    return &output[0];
  }

  //! reduce values in grid to single value, store in output
  RAJA_DEVICE
  void grid_reduce(T* output)
  {
    T temp = value;

    size_t replicationId =
        impl::grid_reduce_atomic_device_init<Combiner, Accessor, replication,
                                             atomic_stride>(
            temp, identity, device, device_count);
    if (replicationId != replication)
    {
      output[replicationId] = temp;
    }
  }

  //! check and setup for device
  //  allocate device pointers and get a new result buffer from the pinned tally
  bool setupForDevice()
  {
    bool act = !device && setupReducers();
    if (act)
    {
      device = data_mempool_type::getInstance().template malloc<T>(
          replication * atomic_stride);
      device_count =
          count_mempool_type::getInstance().template malloc<unsigned int>(
              replication * atomic_stride);
      own_device_ptr = true;
    }
    return act;
  }

  //! if own resources teardown device setup
  //  free device pointers
  bool teardownForDevice()
  {
    bool act = own_device_ptr;
    if (act)
    {
      data_mempool_type::getInstance().free(device);
      device = nullptr;
      count_mempool_type::getInstance().free(device_count);
      device_count   = nullptr;
      own_device_ptr = false;
    }
    return act;
  }
};

//! Hip Reduction entity -- generalize on reduction, and type
template<typename Combiner, typename T, typename tuning>
class Reduce
{
  static constexpr size_t replication =
      (tuning::replication > 0) ? tuning::replication : 32;
  static constexpr size_t atomic_stride =
      (tuning::atomic_stride > 0)
          ? tuning::atomic_stride
          : ((policy::hip::device_constants
                  .ATOMIC_DESTRUCTIVE_INTERFERENCE_SIZE > sizeof(T))
                 ? RAJA_DIVIDE_CEILING_INT(
                       policy::hip::device_constants
                           .ATOMIC_DESTRUCTIVE_INTERFERENCE_SIZE,
                       sizeof(T))
                 : 1);

  using Accessor = std::conditional_t<
      (tuning::comm_mode == block_communication_mode::block_fence),
      impl::AccessorDeviceScopeUseBlockFence,
      std::conditional_t<(tuning::comm_mode ==
                          block_communication_mode::device_fence),
                         impl::AccessorDeviceScopeUseDeviceFence,
                         void>>;

  static constexpr bool atomic_policy =
      (tuning::algorithm ==
       reduce_algorithm::init_device_combine_atomic_block) ||
      (tuning::algorithm == reduce_algorithm::init_host_combine_atomic_block);
  static constexpr bool atomic_available =
      RAJA::reduce::hip::hip_atomic_available<T>::value;

  //! hip reduction data storage class and folding algorithm
  using reduce_data_type = std::conditional_t<
      (tuning::algorithm == reduce_algorithm::combine_last_block) ||
          (atomic_policy && !atomic_available),
      hip::ReduceLastBlock_Data<Combiner,
                                Accessor,
                                T,
                                replication,
                                atomic_stride>,
      std::conditional_t<
          atomic_available,
          std::conditional_t<
              (tuning::algorithm ==
               reduce_algorithm::init_device_combine_atomic_block),
              hip::ReduceAtomicDeviceInit_Data<Combiner,
                                               Accessor,
                                               T,
                                               replication,
                                               atomic_stride>,
              std::conditional_t<
                  (tuning::algorithm ==
                   reduce_algorithm::init_host_combine_atomic_block),
                  hip::ReduceAtomicHostInit_Data<Combiner,
                                                 T,
                                                 replication,
                                                 atomic_stride>,
                  void>>,
          void>>;

  static constexpr size_t tally_slots = reduce_data_type::tally_slots;

  using TallyType = PinnedTally<T,
                                tally_slots,
                                typename reduce_data_type::tally_mempool_type>;

  //! union to hold either pointer to PinnedTally or pointer to value
  //  only use list before setup for device and only use val_ptr after
  union tally_u
  {
    TallyType* list;
    T* val_ptr;
    constexpr tally_u(TallyType* l) : list(l) {};
    constexpr tally_u(T* v_ptr) : val_ptr(v_ptr) {};
  };

public:
  Reduce() : Reduce(T(), Combiner::identity()) {}

  //! create a reduce object
  //  the original object's parent is itself
  explicit Reduce(T init_val, T identity_ = Combiner::identity())
      : parent {this},
        tally_or_val_ptr {new TallyType},
        val(init_val, identity_)
  {}

  void reset(T in_val, T identity_ = Combiner::identity())
  {
    operator T();  // syncs device
    val = reduce_data_type(in_val, identity_);
  }

  //! copy and on host attempt to setup for device
  //  init val_ptr to avoid uninitialized read caused by host copy of
  //  reducer in host device lambda not being used on device.
  RAJA_HOST_DEVICE
  Reduce(const Reduce& other)
#if !defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
      : parent {other.parent},
#else
      : parent {&other},
#endif
        tally_or_val_ptr {other.tally_or_val_ptr},
        val(other.val)
  {
#if !defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
    if (parent)
    {
      if (val.setupForDevice())
      {
        tally_or_val_ptr.val_ptr = val.init_grid_vals(
            tally_or_val_ptr.list->new_value(currentResource()));
        parent = nullptr;
      }
    }
#endif
  }

  //! apply reduction upon destruction and cleanup resources owned by this copy
  //  on device store in pinned buffer on host
  RAJA_HOST_DEVICE
  ~Reduce()
  {
#if !defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
    if (parent == this)
    {
      delete tally_or_val_ptr.list;
      tally_or_val_ptr.list = nullptr;
    }
    else if (parent)
    {
      if (val.value != val.identity)
      {
#if defined(RAJA_ENABLE_OPENMP)
        lock_guard<omp::mutex> lock(tally_or_val_ptr.list->m_mutex);
#endif
        parent->combine(val.value);
      }
    }
    else
    {
      if (val.teardownForDevice())
      {
        tally_or_val_ptr.val_ptr = nullptr;
      }
    }
#else
    if (!parent->parent)
    {
      val.grid_reduce(tally_or_val_ptr.val_ptr);
    }
    else
    {
      parent->combine(val.value);
    }
#endif
  }

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    auto n   = tally_or_val_ptr.list->begin();
    auto end = tally_or_val_ptr.list->end();
    if (n != end)
    {
      tally_or_val_ptr.list->synchronize_resources();
      ::RAJA::detail::HighAccuracyReduce<T, typename Combiner::operator_type>
          reducer(std::move(val.value));
      for (; n != end; ++n)
      {
        T(&values)[tally_slots] = *n;
        for (size_t r = 0; r < tally_slots; ++r)
        {
          reducer.combine(std::move(values[r]));
        }
      }
      val.value = reducer.get_and_clear();
      tally_or_val_ptr.list->free_list();
    }
    return val.value;
  }

  //! alias for operator T()
  T get() { return operator T(); }

  //! apply reduction (const version) -- still combines internal values
  RAJA_HOST_DEVICE
  void combine(T other) const { Combiner {}(val.value, other); }

  /*!
   *  \return reference to the local value
   */
  T& local() const { return val.value; }

  T get_combined() const { return val.value; }

private:
  const Reduce* parent;
  tally_u tally_or_val_ptr;
  reduce_data_type val;
};

}  // end namespace hip

//! specialization of ReduceSum for hip_reduce
template<typename tuning, typename T>
class ReduceSum<RAJA::policy::hip::hip_reduce_policy<tuning>, T>
    : public hip::Reduce<RAJA::reduce::sum<T>, T, tuning>
{

public:
  using Base = hip::Reduce<RAJA::reduce::sum<T>, T, tuning>;
  using Base::Base;

  //! enable operator+= for ReduceSum -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceSum& operator+=(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

//! specialization of ReduceBitOr for hip_reduce
template<typename tuning, typename T>
class ReduceBitOr<RAJA::policy::hip::hip_reduce_policy<tuning>, T>
    : public hip::Reduce<RAJA::reduce::or_bit<T>, T, tuning>
{

public:
  using Base = hip::Reduce<RAJA::reduce::or_bit<T>, T, tuning>;
  using Base::Base;

  //! enable operator|= for ReduceBitOr -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceBitOr& operator|=(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

//! specialization of ReduceBitAnd for hip_reduce
template<typename tuning, typename T>
class ReduceBitAnd<RAJA::policy::hip::hip_reduce_policy<tuning>, T>
    : public hip::Reduce<RAJA::reduce::and_bit<T>, T, tuning>
{

public:
  using Base = hip::Reduce<RAJA::reduce::and_bit<T>, T, tuning>;
  using Base::Base;

  //! enable operator&= for ReduceBitAnd -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceBitAnd& operator&=(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

//! specialization of ReduceMin for hip_reduce
template<typename tuning, typename T>
class ReduceMin<RAJA::policy::hip::hip_reduce_policy<tuning>, T>
    : public hip::Reduce<RAJA::reduce::min<T>, T, tuning>
{

public:
  using Base = hip::Reduce<RAJA::reduce::min<T>, T, tuning>;
  using Base::Base;

  //! enable min() for ReduceMin -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceMin& min(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

//! specialization of ReduceMax for hip_reduce
template<typename tuning, typename T>
class ReduceMax<RAJA::policy::hip::hip_reduce_policy<tuning>, T>
    : public hip::Reduce<RAJA::reduce::max<T>, T, tuning>
{

public:
  using Base = hip::Reduce<RAJA::reduce::max<T>, T, tuning>;
  using Base::Base;

  //! enable max() for ReduceMax -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceMax& max(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

//! specialization of ReduceMinLoc for hip_reduce
template<typename tuning, typename T, typename IndexType>
class ReduceMinLoc<RAJA::policy::hip::hip_reduce_policy<tuning>, T, IndexType>
    : public hip::Reduce<
          RAJA::reduce::min<RAJA::reduce::detail::ValueLoc<T, IndexType>>,
          RAJA::reduce::detail::ValueLoc<T, IndexType>,
          tuning>
{

public:
  using value_type     = RAJA::reduce::detail::ValueLoc<T, IndexType>;
  using Combiner       = RAJA::reduce::min<value_type>;
  using NonLocCombiner = RAJA::reduce::min<T>;
  using Base           = hip::Reduce<Combiner, value_type, tuning>;
  using Base::Base;

  //! constructor requires a default value for the reducer
  ReduceMinLoc(T init_val,
               IndexType init_idx,
               T identity_val = NonLocCombiner::identity(),
               IndexType identity_idx =
                   RAJA::reduce::detail::DefaultLoc<IndexType>().value())
      : Base(value_type(init_val, init_idx),
             value_type(identity_val, identity_idx))
  {}

  //! reset requires a default value for the reducer
  // this must be here to hide Base::reset
  void reset(T init_val,
             IndexType init_idx,
             T identity_val = NonLocCombiner::identity(),
             IndexType identity_idx =
                 RAJA::reduce::detail::DefaultLoc<IndexType>().value())
  {
    Base::reset(value_type(init_val, init_idx),
                value_type(identity_val, identity_idx));
  }

  //! reducer function; updates the current instance's state
  RAJA_HOST_DEVICE
  const ReduceMinLoc& minloc(T rhs, IndexType loc) const
  {
    this->combine(value_type(rhs, loc));
    return *this;
  }

  //! Get the calculated reduced value
  IndexType getLoc() { return Base::get().getLoc(); }

  //! Get the calculated reduced value
  operator T() { return Base::get(); }

  //! Get the calculated reduced value
  T get() { return Base::get(); }
};

//! specialization of ReduceMaxLoc for hip_reduce
template<typename tuning, typename T, typename IndexType>
class ReduceMaxLoc<RAJA::policy::hip::hip_reduce_policy<tuning>, T, IndexType>
    : public hip::Reduce<
          RAJA::reduce::max<
              RAJA::reduce::detail::ValueLoc<T, IndexType, false>>,
          RAJA::reduce::detail::ValueLoc<T, IndexType, false>,
          tuning>
{
public:
  using value_type     = RAJA::reduce::detail::ValueLoc<T, IndexType, false>;
  using Combiner       = RAJA::reduce::max<value_type>;
  using NonLocCombiner = RAJA::reduce::max<T>;
  using Base           = hip::Reduce<Combiner, value_type, tuning>;
  using Base::Base;

  //! constructor requires a default value for the reducer
  ReduceMaxLoc(T init_val,
               IndexType init_idx,
               T identity_val = NonLocCombiner::identity(),
               IndexType identity_idx =
                   RAJA::reduce::detail::DefaultLoc<IndexType>().value())
      : Base(value_type(init_val, init_idx),
             value_type(identity_val, identity_idx))
  {}

  //! reset requires a default value for the reducer
  // this must be here to hide Base::reset
  void reset(T init_val,
             IndexType init_idx,
             T identity_val = NonLocCombiner::identity(),
             IndexType identity_idx =
                 RAJA::reduce::detail::DefaultLoc<IndexType>().value())
  {
    Base::reset(value_type(init_val, init_idx),
                value_type(identity_val, identity_idx));
  }

  //! reducer function; updates the current instance's state
  RAJA_HOST_DEVICE
  const ReduceMaxLoc& maxloc(T rhs, IndexType loc) const
  {
    this->combine(value_type(rhs, loc));
    return *this;
  }

  //! Get the calculated reduced value
  IndexType getLoc() { return Base::get().getLoc(); }

  //! Get the calculated reduced value
  operator T() { return Base::get(); }

  //! Get the calculated reduced value
  T get() { return Base::get(); }
};

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
