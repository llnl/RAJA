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

#ifndef RAJA_hip_multi_reduce_HPP
#define RAJA_hip_multi_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <type_traits>
#include <limits>
#include <utility>
#include <vector>

#include "hip/hip_runtime.h"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/math.hpp"
#include "RAJA/util/mutex.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/util/reduce.hpp"
#include "RAJA/util/OffsetOperators.hpp"

#include "RAJA/pattern/detail/multi_reduce.hpp"
#include "RAJA/pattern/multi_reduce.hpp"

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/intrinsics.hpp"

#if defined(RAJA_ENABLE_DESUL_ATOMICS)
#include "RAJA/policy/desul/atomic.hpp"
#else
#include "RAJA/policy/hip/atomic.hpp"
#endif

#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"

namespace RAJA
{

namespace hip
{

namespace impl
{


//
//////////////////////////////////////////////////////////////////////
//
// MultiReduction algorithms.
//
//////////////////////////////////////////////////////////////////////
//

//! combine value into global memory
template<typename Combiner,
         typename GetTallyIndex,
         typename T,
         typename GetTallyOffset>
RAJA_DEVICE RAJA_INLINE void block_multi_reduce_combine_global_atomic(
    int RAJA_UNUSED_ARG(num_bins),
    T identity,
    int bin,
    T value,
    T* tally_mem,
    GetTallyOffset get_tally_offset,
    int tally_replication,
    int tally_bins)
{
  if (value == identity)
  {
    return;
  }

  int tally_index =
      GetTallyIndex::template index<int>();  // globalWarpId by default
  int tally_rep = ::RAJA::power_of_2_mod(tally_index, tally_replication);
  int tally_offset =
      get_tally_offset(bin, tally_bins, tally_rep, tally_replication);
  RAJA::reduce::hip::atomic<Combiner> {}(tally_mem[tally_offset], value);
}

//! initialize shared memory
template<typename T>
RAJA_DEVICE RAJA_INLINE void block_multi_reduce_init_shmem(
    int num_bins,
    T identity,
    T* shared_mem,
    int shared_replication)
{
  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;
  int numThreads = blockDim.x * blockDim.y * blockDim.z;

  for (int shmem_offset = threadId;
       shmem_offset < shared_replication * num_bins; shmem_offset += numThreads)
  {
    shared_mem[shmem_offset] = identity;
  }
  __syncthreads();
}

//! combine value into shared memory
template<typename Combiner,
         typename GetSharedIndex,
         typename T,
         typename GetSharedOffset>
RAJA_DEVICE RAJA_INLINE void block_multi_reduce_combine_shmem_atomic(
    int num_bins,
    T identity,
    int bin,
    T value,
    T* shared_mem,
    GetSharedOffset get_shared_offset,
    int shared_replication)
{
  if (value == identity)
  {
    return;
  }

  int shared_index =
      GetSharedIndex::template index<int>();  // threadId by default
  int shared_rep = ::RAJA::power_of_2_mod(shared_index, shared_replication);
  int shmem_offset =
      get_shared_offset(bin, num_bins, shared_rep, shared_replication);

  RAJA::reduce::hip::atomic<Combiner> {}(shared_mem[shmem_offset], value);
}

//! combine value into shared memory
template<typename Combiner,
         typename T,
         typename GetSharedOffset,
         typename GetTallyOffset>
RAJA_DEVICE RAJA_INLINE void grid_multi_reduce_shmem_to_global_atomic(
    int num_bins,
    T identity,
    T* shared_mem,
    GetSharedOffset get_shared_offset,
    int shared_replication,
    T* tally_mem,
    GetTallyOffset get_tally_offset,
    int tally_replication,
    int tally_bins)
{
  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;
  int numThreads = blockDim.x * blockDim.y * blockDim.z;

  int blockId = blockIdx.x + gridDim.x * blockIdx.y +
                (gridDim.x * gridDim.y) * blockIdx.z;

  __syncthreads();
  for (int bin = threadId; bin < num_bins; bin += numThreads)
  {

    T value = identity;
    for (int shared_rep = 0; shared_rep < shared_replication; ++shared_rep)
    {
      int shmem_offset =
          get_shared_offset(bin, num_bins, shared_rep, shared_replication);
      Combiner {}(value, shared_mem[shmem_offset]);
    }

    if (value != identity)
    {
      int tally_rep = ::RAJA::power_of_2_mod(blockId, tally_replication);
      int tally_offset =
          get_tally_offset(bin, tally_bins, tally_rep, tally_replication);
      RAJA::reduce::hip::atomic<Combiner> {}(tally_mem[tally_offset], value);
    }
  }
}

}  // namespace impl

//
//////////////////////////////////////////////////////////////////////
//
// MultiReduction classes.
//
//////////////////////////////////////////////////////////////////////
//

//! MultiReduction data for Hip Offload -- stores value, host pointer
template<typename Combiner, typename T, typename tuning>
struct MultiReduceGridAtomicHostInit_TallyData
{
  //! setup permanent settings, allocate and initialize tally memory
  template<typename Container>
  MultiReduceGridAtomicHostInit_TallyData(Container const& container,
                                          T const& identity)
      : m_tally_mem(nullptr),
        m_identity(identity),
        m_num_bins(container.size()),
        m_tally_bins(get_tally_bins(m_num_bins)),
        m_tally_replication(get_tally_replication())
  {
    m_tally_mem = create_tally(container, identity, m_num_bins, m_tally_bins,
                               m_tally_replication);
  }

  MultiReduceGridAtomicHostInit_TallyData() = delete;
  MultiReduceGridAtomicHostInit_TallyData(
      MultiReduceGridAtomicHostInit_TallyData const&) = default;
  MultiReduceGridAtomicHostInit_TallyData(
      MultiReduceGridAtomicHostInit_TallyData&&) = delete;
  MultiReduceGridAtomicHostInit_TallyData& operator=(
      MultiReduceGridAtomicHostInit_TallyData const&) = default;
  MultiReduceGridAtomicHostInit_TallyData& operator=(
      MultiReduceGridAtomicHostInit_TallyData&&) = delete;
  ~MultiReduceGridAtomicHostInit_TallyData()     = default;

  //! reset permanent settings, reallocate and reset tally memory
  template<typename Container>
  void reset_permanent(Container const& container, T const& identity)
  {
    int new_num_bins = container.size();
    if (new_num_bins != m_num_bins)
    {
      teardown_permanent();
      m_num_bins          = new_num_bins;
      m_tally_bins        = get_tally_bins(m_num_bins);
      m_tally_replication = get_tally_replication();
      m_tally_mem = create_tally(container, identity, m_num_bins, m_tally_bins,
                                 m_tally_replication);
    }
    else
    {
      {
        int tally_rep = 0;
        int bin       = 0;
        for (auto const& value : container)
        {
          m_tally_mem[GetTallyOffset {}(bin, m_tally_bins, tally_rep,
                                        m_tally_replication)] = value;
          ++bin;
        }
      }
      for (int tally_rep = 1; tally_rep < m_tally_replication; ++tally_rep)
      {
        for (int bin = 0; bin < m_num_bins; ++bin)
        {
          m_tally_mem[GetTallyOffset {}(bin, m_tally_bins, tally_rep,
                                        m_tally_replication)] = identity;
        }
      }
    }
    m_identity = identity;
  }

  //! teardown permanent settings, free tally memory
  void teardown_permanent()
  {
    destroy_tally(m_tally_mem, m_num_bins, m_tally_bins, m_tally_replication);
  }

  //! get value for bin, assumes synchronization occurred elsewhere
  T get(int bin) const
  {
    ::RAJA::detail::HighAccuracyReduce<T, typename Combiner::operator_type>
        reducer(m_identity);
    for (int tally_rep = 0; tally_rep < m_tally_replication; ++tally_rep)
    {
      int tally_offset =
          GetTallyOffset {}(bin, m_tally_bins, tally_rep, m_tally_replication);
      reducer.combine(m_tally_mem[tally_offset]);
    }
    return reducer.get_and_clear();
  }

  int num_bins() const { return m_num_bins; }

  T identity() const { return m_identity; }

private:
  static constexpr size_t s_tally_alignment = std::max(
      size_t(
          policy::hip::device_constants.ATOMIC_DESTRUCTIVE_INTERFERENCE_SIZE),
      size_t(RAJA::DATA_ALIGN));
  static constexpr size_t s_tally_bunch_size =
      RAJA_DIVIDE_CEILING_INT(s_tally_alignment, sizeof(T));

  using tally_mempool_type = device_pinned_mempool_type;
  using tally_tuning       = typename tuning::GlobalAtomicReplicationTuning;
  using TallyAtomicReplicationConcretizer =
      typename tally_tuning::AtomicReplicationConcretizer;
  using GetTallyOffset_rebind_rebunch = typename tally_tuning::OffsetCalculator;
  using GetTallyOffset_rebind =
      typename GetTallyOffset_rebind_rebunch::template rebunch<
          s_tally_bunch_size>;

  static int get_tally_bins(int num_bins)
  {
    return RAJA_DIVIDE_CEILING_INT(num_bins, s_tally_bunch_size) *
           s_tally_bunch_size;
  }

  static int get_tally_replication()
  {
    int min_tally_replication = 1;
#if defined(RAJA_ENABLE_OPENMP)
    min_tally_replication = omp_get_max_threads();
#endif

    struct
    {
      int func_min_global_replication;
    } func_data {min_tally_replication};

    return TallyAtomicReplicationConcretizer {}
        .template get_global_replication<int>(func_data);
  }

  template<typename Container>
  static T* create_tally(Container const& container,
                         T const& identity,
                         int num_bins,
                         int tally_bins,
                         int tally_replication)
  {
    if (num_bins == size_t(0))
    {
      return nullptr;
    }

    T* tally_mem = tally_mempool_type::getInstance().template malloc<T>(
        tally_replication * tally_bins, s_tally_alignment);

    if (tally_replication > 0)
    {
      {
        int tally_rep = 0;
        int bin       = 0;
        for (auto const& value : container)
        {
          int tally_offset =
              GetTallyOffset {}(bin, tally_bins, tally_rep, tally_replication);
          new (&tally_mem[tally_offset]) T(value);
          ++bin;
        }
      }
      for (int tally_rep = 1; tally_rep < tally_replication; ++tally_rep)
      {
        for (int bin = 0; bin < num_bins; ++bin)
        {
          int tally_offset =
              GetTallyOffset {}(bin, tally_bins, tally_rep, tally_replication);
          new (&tally_mem[tally_offset]) T(identity);
        }
      }
    }
    return tally_mem;
  }

  static void destroy_tally(T*& tally_mem,
                            int num_bins,
                            int tally_bins,
                            int tally_replication)
  {
    if (num_bins == size_t(0))
    {
      return;
    }

    for (int tally_rep = tally_replication + 1; tally_rep > 0; --tally_rep)
    {
      for (int bin = num_bins; bin > 0; --bin)
      {
        int tally_offset = GetTallyOffset {}(bin - 1, tally_bins, tally_rep - 1,
                                             tally_replication);
        tally_mem[tally_offset].~T();
      }
    }
    tally_mempool_type::getInstance().free(tally_mem);
    tally_mem = nullptr;
  }

protected:
  using GetTallyIndex  = typename tally_tuning::ReplicationIndexer;
  using GetTallyOffset = typename GetTallyOffset_rebind::template rebind<int>;

  T* m_tally_mem;
  T m_identity;
  int m_num_bins;
  int m_tally_bins;
  int m_tally_replication;  // power of 2, at least the max number of omp
                            // threads
};

//! MultiReduction data for Hip Offload -- stores value, host pointer
template<typename Combiner, typename T, typename tuning>
struct MultiReduceGridAtomicHostInit_Data
    : MultiReduceGridAtomicHostInit_TallyData<Combiner, T, tuning>
{
  using TallyData =
      MultiReduceGridAtomicHostInit_TallyData<Combiner, T, tuning>;

  //! defer to tally data for some functions
  using TallyData::get;
  using TallyData::identity;
  using TallyData::num_bins;
  using TallyData::reset_permanent;
  using TallyData::TallyData;
  using TallyData::teardown_permanent;

  //! setup per launch, do nothing
  void setup_launch(size_t RAJA_UNUSED_ARG(block_size)) {}

  //! teardown per launch, do nothing
  void teardown_launch() {}

  //! setup on device, do nothing
  RAJA_DEVICE
  void setup_device() {}

  //! finalize on device, do nothing
  RAJA_DEVICE
  void finalize_device() {}

  //! combine value on device, combine a value into the tally atomically
  RAJA_DEVICE
  void combine_device(int bin, T value)
  {
    impl::block_multi_reduce_combine_global_atomic<Combiner, GetTallyIndex>(
        m_num_bins, m_identity, bin, value, m_tally_mem, GetTallyOffset {},
        m_tally_replication, m_tally_bins);
  }

  //! combine value on host, combine a value into the tally
  void combine_host(int bin, T value)
  {
    int tally_rep = 0;
#if defined(RAJA_ENABLE_OPENMP)
    tally_rep = omp_get_thread_num();
#endif
    int tally_offset =
        GetTallyOffset {}(bin, m_tally_bins, tally_rep, m_tally_replication);
    Combiner {}(m_tally_mem[tally_offset], value);
  }

private:
  using typename TallyData::GetTallyIndex;
  using typename TallyData::GetTallyOffset;

  using TallyData::m_identity;
  using TallyData::m_num_bins;
  using TallyData::m_tally_bins;
  using TallyData::m_tally_mem;
  using TallyData::m_tally_replication;
};

//! MultiReduction data for Hip Offload -- stores value, host pointer
template<typename Combiner, typename T, typename tuning>
struct MultiReduceBlockThenGridAtomicHostInit_Data
    : MultiReduceGridAtomicHostInit_TallyData<Combiner, T, tuning>
{
  using TallyData =
      MultiReduceGridAtomicHostInit_TallyData<Combiner, T, tuning>;

  //! setup permanent settings, defer to tally data
  template<typename Container>
  MultiReduceBlockThenGridAtomicHostInit_Data(Container const& container,
                                              T const& identity)
      : TallyData(container, identity),
        m_shared_offset(s_shared_offset_unknown),
        m_shared_replication(0)
  {}

  MultiReduceBlockThenGridAtomicHostInit_Data() = delete;
  MultiReduceBlockThenGridAtomicHostInit_Data(
      MultiReduceBlockThenGridAtomicHostInit_Data const&) = default;
  MultiReduceBlockThenGridAtomicHostInit_Data(
      MultiReduceBlockThenGridAtomicHostInit_Data&&) = delete;
  MultiReduceBlockThenGridAtomicHostInit_Data& operator=(
      MultiReduceBlockThenGridAtomicHostInit_Data const&) = default;
  MultiReduceBlockThenGridAtomicHostInit_Data& operator=(
      MultiReduceBlockThenGridAtomicHostInit_Data&&) = delete;
  ~MultiReduceBlockThenGridAtomicHostInit_Data()     = default;


  //! defer to tally data for some functions
  using TallyData::get;
  using TallyData::identity;
  using TallyData::num_bins;
  using TallyData::reset_permanent;
  using TallyData::teardown_permanent;

  //! setup per launch, setup shared memory parameters
  void setup_launch(size_t block_size)
  {
    if (m_num_bins == size_t(0))
    {
      m_shared_offset = s_shared_offset_invalid;
      return;
    }

    size_t shared_replication = 0;
    const size_t shared_offset =
        allocateDynamicShmem<T>([&](size_t max_shmem_size) {
          struct
          {
            size_t func_threads_per_block;
            size_t func_max_shared_replication_per_block;
          } func_data {block_size, max_shmem_size / m_num_bins};

          shared_replication =
              SharedAtomicReplicationConcretizer {}
                  .template get_shared_replication<size_t>(func_data);
          return m_num_bins * shared_replication;
        });

    if (shared_offset != dynamic_smem_allocation_failure)
    {
      m_shared_replication = static_cast<int>(shared_replication);
      m_shared_offset      = static_cast<int>(shared_offset);
    }
    else
    {
      m_shared_offset = s_shared_offset_invalid;
    }
  }

  //! teardown per launch, unset shared memory parameters
  void teardown_launch()
  {
    m_shared_replication = 0;
    m_shared_offset      = s_shared_offset_unknown;
  }

  //! setup on device, initialize shared memory
  RAJA_DEVICE
  void setup_device()
  {
    T* shared_mem = get_shared_mem();
    if (shared_mem != nullptr)
    {
      impl::block_multi_reduce_init_shmem(m_num_bins, m_identity, shared_mem,
                                          m_shared_replication);
    }
  }

  //! finalize on device, combine values in shared memory into the tally
  RAJA_DEVICE
  void finalize_device()
  {
    T* shared_mem = get_shared_mem();
    if (shared_mem != nullptr)
    {
      impl::grid_multi_reduce_shmem_to_global_atomic<Combiner>(
          m_num_bins, m_identity, shared_mem, GetSharedOffset {},
          m_shared_replication, m_tally_mem, GetTallyOffset {},
          m_tally_replication, m_tally_bins);
    }
  }

  //! combine value on device, combine a value into shared memory
  RAJA_DEVICE
  void combine_device(int bin, T value)
  {
    T* shared_mem = get_shared_mem();
    if (shared_mem != nullptr)
    {
      impl::block_multi_reduce_combine_shmem_atomic<Combiner, GetSharedIndex>(
          m_num_bins, m_identity, bin, value, shared_mem, GetSharedOffset {},
          m_shared_replication);
    }
    else
    {
      impl::block_multi_reduce_combine_global_atomic<Combiner, GetTallyIndex>(
          m_num_bins, m_identity, bin, value, m_tally_mem, GetTallyOffset {},
          m_tally_replication, m_tally_bins);
    }
  }

  //! combine value on host, combine a value into the tally
  void combine_host(int bin, T value)
  {
    int tally_rep = 0;
#if defined(RAJA_ENABLE_OPENMP)
    tally_rep = omp_get_thread_num();
#endif
    int tally_offset =
        GetTallyOffset {}(bin, m_tally_bins, tally_rep, m_tally_replication);
    Combiner {}(m_tally_mem[tally_offset], value);
  }

private:
  using shared_tuning = typename tuning::SharedAtomicReplicationTuning;
  using SharedAtomicReplicationConcretizer =
      typename shared_tuning::AtomicReplicationConcretizer;
  using GetSharedIndex         = typename shared_tuning::ReplicationIndexer;
  using GetSharedOffset_rebind = typename shared_tuning::OffsetCalculator;
  using GetSharedOffset = typename GetSharedOffset_rebind::template rebind<int>;

  using typename TallyData::GetTallyIndex;
  using typename TallyData::GetTallyOffset;


  static constexpr int s_shared_offset_unknown =
      std::numeric_limits<int>::max();
  static constexpr int s_shared_offset_invalid =
      std::numeric_limits<int>::max() - 1;


  using TallyData::m_identity;
  using TallyData::m_num_bins;
  using TallyData::m_tally_bins;
  using TallyData::m_tally_mem;
  using TallyData::m_tally_replication;

  int m_shared_offset;       // in bytes
  int m_shared_replication;  // power of 2

  RAJA_DEVICE
  T* get_shared_mem() const
  {
    if (m_shared_offset == s_shared_offset_invalid)
    {
      return nullptr;
    }
    extern __shared__ char shared_mem[];
    return reinterpret_cast<T*>(&shared_mem[m_shared_offset]);
  }
};

/*!
 **************************************************************************
 *
 * \brief  Hip multi-reduce data class template.
 *
 * This class manages synchronization, data lifetimes, and interaction with
 * the runtime kernel launch info passing facilities.
 *
 * This class manages the lifetime of underlying reduce_data_type using
 * calls to setup and teardown methods. This includes storage durations:
 * - permanent, the lifetime of the parent object
 * - launch, setup before a launch using the launch parameters and
 *           teardown after the launch
 * - device, setup all device threads in a kernel before any block work and
 *           teardown all device threads after all block work is finished
 *
 **************************************************************************
 */
template<typename T, typename t_MultiReduceOp, typename tuning>
struct MultiReduceDataHip
{
  static constexpr bool atomic_available =
      RAJA::reduce::hip::hip_atomic_available<T>::value;

  //! hip reduction data storage class and folding algorithm
  using reduce_data_type = std::conditional_t<
      (atomic_available),
      std::conditional_t<
          (tuning::algorithm ==
           multi_reduce_algorithm::
               init_host_combine_block_atomic_then_grid_atomic),
          hip::MultiReduceBlockThenGridAtomicHostInit_Data<t_MultiReduceOp,
                                                           T,
                                                           tuning>,
          std::conditional_t<
              (tuning::algorithm ==
               multi_reduce_algorithm::init_host_combine_global_atomic),
              hip::MultiReduceGridAtomicHostInit_Data<t_MultiReduceOp,
                                                      T,
                                                      tuning>,
              void>>,
      void>;


  using SyncList = std::vector<resources::Hip>;

public:
  using value_type    = T;
  using MultiReduceOp = t_MultiReduceOp;

  MultiReduceDataHip() = delete;

  template<typename Container,
           std::enable_if_t<
               !std::is_same<Container, MultiReduceDataHip>::value>* = nullptr>
  MultiReduceDataHip(Container const& container, T identity)
      : m_parent(this),
        m_sync_list(new SyncList),
        m_data(container, identity),
        m_own_launch_data(false)
  {}

  //! copy and on host attempt to setup for device
  //  init val_ptr to avoid uninitialized read caused by host copy of
  //  reducer in host device lambda not being used on device.
  RAJA_HOST_DEVICE
  MultiReduceDataHip(MultiReduceDataHip const& other)
#if !defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
      : m_parent(other.m_parent)
#else
      : m_parent(&other)
#endif
        ,
        m_sync_list(other.m_sync_list),
        m_data(other.m_data),
        m_own_launch_data(false)
  {
#if !defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
    if (m_parent)
    {
      if (setupReducers())
      {
        // the copy made in make_launch_body does this setup
        add_resource_to_synchronization_list(currentResource());
        m_data.setup_launch(currentBlockSize());
        m_own_launch_data = true;
        m_parent          = nullptr;
      }
    }
#else
    if (!m_parent->m_parent)
    {
      // the first copy on device enters this branch
      m_data.setup_device();
    }
#endif
  }

  MultiReduceDataHip(MultiReduceDataHip&&)                 = delete;
  MultiReduceDataHip& operator=(MultiReduceDataHip const&) = delete;
  MultiReduceDataHip& operator=(MultiReduceDataHip&&)      = delete;

  //! cleanup resources owned by this copy
  //  on device store in pinned buffer on host
  RAJA_HOST_DEVICE
  ~MultiReduceDataHip()
  {
#if !defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
    if (m_parent == this)
    {
      // the original object, owns permanent storage
      synchronize_resources_and_clear_list();
      delete m_sync_list;
      m_sync_list = nullptr;
      m_data.teardown_permanent();
    }
    else if (m_parent)
    {
      // do nothing
    }
    else
    {
      if (m_own_launch_data)
      {
        // the copy made in make_launch_body, owns launch data
        m_data.teardown_launch();
        m_own_launch_data = false;
      }
    }
#else
    if (!m_parent->m_parent)
    {
      // the first copy on device, does finalization on the device
      m_data.finalize_device();
    }
#endif
  }

  template<typename Container>
  void reset(Container const& container, T identity)
  {
    synchronize_resources_and_clear_list();
    m_data.reset_permanent(container, identity);
  }

  //! apply reduction (const version) -- still combines internal values
  RAJA_HOST_DEVICE
  void combine(int bin, T const& value)
  {
#if !defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
    m_data.combine_host(bin, value);
#else
    m_data.combine_device(bin, value);
#endif
  }

  //! map result value back to host if not done already; return aggregate value
  T get(int bin)
  {
    synchronize_resources_and_clear_list();
    return m_data.get(bin);
  }

  size_t num_bins() const { return m_data.num_bins(); }

  T identity() const { return m_data.identity(); }


private:
  MultiReduceDataHip const* m_parent;
  SyncList* m_sync_list;
  reduce_data_type m_data;
  bool m_own_launch_data;

  void add_resource_to_synchronization_list(resources::Hip res)
  {
    for (resources::Hip& list_res : *m_sync_list)
    {
      if (list_res.get_stream() == res.get_stream())
      {
        return;
      }
    }
    m_sync_list->emplace_back(res);
  }

  void synchronize_resources_and_clear_list()
  {
    for (resources::Hip& list_res : *m_sync_list)
    {
      ::RAJA::hip::synchronize(list_res);
    }
    m_sync_list->clear();
  }
};

}  // end namespace hip

RAJA_DECLARE_ALL_MULTI_REDUCERS(policy::hip::hip_multi_reduce_policy,
                                hip::MultiReduceDataHip)

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
