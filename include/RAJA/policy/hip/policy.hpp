/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA HIP policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_hip_HPP
#define RAJA_policy_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_HIP_ACTIVE)

#include <utility>
#include "hip/hip_runtime.h"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/policy/sequential/policy.hpp"
#include "RAJA/policy/hip/intrinsics.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/OffsetOperators.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/util/math.hpp"

namespace RAJA
{

using hip_dim_t = RAJA_HIP_DIM_T;

using hip_dim_member_t = camp::decay<decltype(std::declval<hip_dim_t>().x)>;

//
/////////////////////////////////////////////////////////////////////
//
// Execution policies
//
/////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///

namespace detail
{
template<bool Async>
struct get_launch
{
  static constexpr RAJA::Launch value = RAJA::Launch::async;
};

template<>
struct get_launch<false>
{
  static constexpr RAJA::Launch value = RAJA::Launch::sync;
};
}  // end namespace detail

namespace hip
{

/// Type representing thread and block indexing within a grid
template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE>
struct IndexGlobal;

template<typename... indexers>
struct IndexFlatten;

template<size_t divisor, typename index>
struct IndexDivide;

template<size_t divisor, typename index>
struct IndexModulo;

/*!
 * Use the max occupancy of a kernel on the current device when launch
 * parameters are not fully determined.
 * Note that the maximum occupancy of the kernel may be less than the maximum
 * occupancy of the device in terms of total threads.
 */
struct MaxOccupancyConcretizer
{
  template<typename IdxT, typename Data>
  static IdxT get_max_grid_size(Data const& data)
  {
    IdxT device_sm_per_device   = data.device_sm_per_device;
    IdxT func_max_blocks_per_sm = data.func_max_blocks_per_sm;

    IdxT func_max_blocks_per_device =
        func_max_blocks_per_sm * device_sm_per_device;

    return func_max_blocks_per_device;
  }
};

/*!
 * Use a fraction and an offset of the max occupancy of a kernel on the current
 * device when launch parameters are not fully determined.
 * The following formula is used, with care to avoid zero, to determine the
 * maximum grid size:
 * (Fraction * kernel_max_blocks_per_sm + BLOCKS_PER_SM_OFFSET) * device_sm
 */
template<typename t_Fraction, std::ptrdiff_t BLOCKS_PER_SM_OFFSET>
struct FractionOffsetOccupancyConcretizer
{
  template<typename IdxT, typename Data>
  static IdxT get_max_grid_size(Data const& data)
  {
    using Fraction = typename t_Fraction::template rebind<IdxT>;

    IdxT device_sm_per_device   = data.device_sm_per_device;
    IdxT func_max_blocks_per_sm = data.func_max_blocks_per_sm;

    if (Fraction::multiply(func_max_blocks_per_sm) > IdxT(0))
    {
      func_max_blocks_per_sm = Fraction::multiply(func_max_blocks_per_sm);
    }

    if (IdxT(std::ptrdiff_t(func_max_blocks_per_sm) + BLOCKS_PER_SM_OFFSET) >
        IdxT(0))
    {
      func_max_blocks_per_sm =
          IdxT(std::ptrdiff_t(func_max_blocks_per_sm) + BLOCKS_PER_SM_OFFSET);
    }

    IdxT func_max_blocks_per_device =
        func_max_blocks_per_sm * device_sm_per_device;

    return func_max_blocks_per_device;
  }
};

/*!
 * Use an occupancy that is less than the max occupancy of the device when
 * launch parameters are not fully determined.
 * Use the MaxOccupancyConcretizer if the maximum occupancy of the kernel is
 * below the maximum occupancy of the device.
 * Otherwise use the given AvoidMaxOccupancyCalculator to determine the
 * maximum grid size.
 */
template<typename AvoidMaxOccupancyConcretizer>
struct AvoidDeviceMaxThreadOccupancyConcretizer
{
  template<typename IdxT, typename Data>
  static IdxT get_max_grid_size(Data const& data)
  {
    IdxT device_max_threads_per_sm = data.device_max_threads_per_sm;
    IdxT func_max_blocks_per_sm    = data.func_max_blocks_per_sm;
    IdxT func_threads_per_block    = data.func_threads_per_block;

    IdxT func_max_threads_per_sm =
        func_threads_per_block * func_max_blocks_per_sm;

    if (func_max_threads_per_sm < device_max_threads_per_sm)
    {
      return MaxOccupancyConcretizer::template get_max_grid_size<IdxT>(data);
    }
    else
    {
      return AvoidMaxOccupancyConcretizer::template get_max_grid_size<IdxT>(
          data);
    }
  }
};

/*!
 * Get an amount of replication that is preferred_replication.
 */
template<size_t preferred_replication>
struct ConstantPreferredReplicationConcretizer
{
  template<typename IdxT, typename Data>
  static IdxT get_preferred_replication(Data const& RAJA_UNUSED_ARG(data))
  {
    return IdxT(preferred_replication);
  }
};

/*!
 * Get an amount of replication that is preferred_replication_before_cutoff if
 * data.func_threads_per_block is less than t_cutoff or
 * preferred_replication_after_cutoff otherwise.
 */
template<size_t t_cutoff,
         size_t preferred_replication_before_cutoff,
         size_t preferred_replication_after_cutoff>
struct ThreadsPerBlockCutoffPreferredReplicationConcretizer
{
  template<typename IdxT, typename Data>
  static IdxT get_preferred_replication(Data const& data)
  {
    IdxT cutoff                 = t_cutoff;
    IdxT func_threads_per_block = data.func_threads_per_block;

    if (func_threads_per_block < cutoff)
    {
      return IdxT(preferred_replication_before_cutoff);
    }
    else
    {
      return IdxT(preferred_replication_after_cutoff);
    }
  }
};

/*!
 * Get an amount of shared atomic replication that is a power of 2 that is at
 * most the amount given by data.func_max_shared_replication_per_block or the
 * amount given by GetPreferredReplication.
 */
template<typename GetPreferredReplication>
struct SharedAtomicReplicationMaxPow2Concretizer
{
  template<typename IdxT, typename Data>
  static IdxT get_shared_replication(Data const& data)
  {
    IdxT func_max_shared_replication_per_block =
        data.func_max_shared_replication_per_block;

    IdxT preferred_replication =
        GetPreferredReplication {}.template get_preferred_replication<IdxT>(
            data);

    return prev_pow2(
        std::min(preferred_replication, func_max_shared_replication_per_block));
  }
};

/*!
 * Get an amount of global atomic replication that is a power of 2 that is at
 * least the amount given by data.func_min_global_replication or the
 * amount given by GetPreferredReplication.
 */
template<typename GetPreferredReplication>
struct GlobalAtomicReplicationMinPow2Concretizer
{
  template<typename IdxT, typename Data>
  static IdxT get_global_replication(Data const& data)
  {
    IdxT func_min_global_replication = data.func_min_global_replication;

    IdxT preferred_replication =
        GetPreferredReplication {}.template get_preferred_replication<IdxT>(
            data);

    return next_pow2(
        std::max(preferred_replication, func_min_global_replication));
  }
};


enum struct reduce_algorithm : int
{
  combine_last_block,
  init_device_combine_atomic_block,
  init_host_combine_atomic_block
};

enum struct block_communication_mode : int
{
  device_fence,
  block_fence
};

template<reduce_algorithm t_algorithm,
         block_communication_mode t_comm_mode,
         size_t t_replication,
         size_t t_atomic_stride>
struct ReduceTuning
{
  static constexpr reduce_algorithm algorithm         = t_algorithm;
  static constexpr block_communication_mode comm_mode = t_comm_mode;
  static constexpr size_t replication                 = t_replication;
  static constexpr size_t atomic_stride               = t_atomic_stride;
  static constexpr bool consistent =
      (algorithm == reduce_algorithm::combine_last_block);
};


enum struct multi_reduce_algorithm : int
{
  init_host_combine_block_atomic_then_grid_atomic,
  init_host_combine_global_atomic
};

template<typename t_AtomicReplicationConcretizer,
         typename t_ReplicationIndexer,
         typename t_OffsetCalculator>
struct AtomicReplicationTuning
{
  using AtomicReplicationConcretizer = t_AtomicReplicationConcretizer;
  using ReplicationIndexer           = t_ReplicationIndexer;
  using OffsetCalculator             = t_OffsetCalculator;
};

template<multi_reduce_algorithm t_algorithm,
         typename t_SharedAtomicReplicationTuning,
         typename t_GlobalAtomicReplicationTuning>
struct MultiReduceTuning
{
  static constexpr multi_reduce_algorithm algorithm = t_algorithm;
  using SharedAtomicReplicationTuning = t_SharedAtomicReplicationTuning;
  using GlobalAtomicReplicationTuning = t_GlobalAtomicReplicationTuning;
  static constexpr bool consistent    = false;
};

}  // namespace hip

namespace policy
{
namespace hip
{

template<typename _IterationMapping,
         kernel_sync_requirement sync,
         typename... _IterationGetters>
struct hip_indexer
{};

template<typename _IterationMapping,
         kernel_sync_requirement sync,
         typename... _IterationGetters>
struct hip_flatten_indexer : public RAJA::make_policy_pattern_launch_platform_t<
                                 RAJA::Policy::hip,
                                 RAJA::Pattern::region,
                                 detail::get_launch<true /*async */>::value,
                                 RAJA::Platform::hip>
{
  using IterationGetter = RAJA::hip::IndexFlatten<_IterationGetters...>;
};

template<typename _IterationMapping,
         typename _IterationGetter,
         typename _LaunchConcretizer,
         bool Async = false>
struct hip_exec : public RAJA::make_policy_pattern_launch_platform_t<
                      RAJA::Policy::hip,
                      RAJA::Pattern::forall,
                      detail::get_launch<Async>::value,
                      RAJA::Platform::hip>
{
  using IterationMapping  = _IterationMapping;
  using IterationGetter   = _IterationGetter;
  using LaunchConcretizer = _LaunchConcretizer;
};

template<bool Async, int num_threads = named_usage::unspecified>
struct hip_launch_t : public RAJA::make_policy_pattern_launch_platform_t<
                          RAJA::Policy::hip,
                          RAJA::Pattern::region,
                          detail::get_launch<Async>::value,
                          RAJA::Platform::hip>
{};

//
// NOTE: There is no Index set segment iteration policy for HIP
//

///
/// WorkGroup execution policies
///
template<size_t BLOCK_SIZE, bool Async = false>
struct hip_work : public RAJA::make_policy_pattern_launch_platform_t<
                      RAJA::Policy::hip,
                      RAJA::Pattern::workgroup_exec,
                      detail::get_launch<Async>::value,
                      RAJA::Platform::hip>
{};

/// execute the enqueued loops in an unordered fashion by mapping loops to
/// blocks in the y direction and loop iterations to threads in the x direction
/// with the size of the x direction being the average of the iteration counts
/// of all the loops
struct unordered_hip_loop_y_block_iter_x_threadblock_average
    : public RAJA::make_policy_pattern_platform_t<
          RAJA::Policy::hip,
          RAJA::Pattern::workgroup_order,
          RAJA::Platform::hip>
{};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction reduction policies
///
///////////////////////////////////////////////////////////////////////
///

template<typename tuning>
struct hip_reduce_policy : public RAJA::make_policy_pattern_launch_platform_t<
                               RAJA::Policy::hip,
                               RAJA::Pattern::reduce,
                               detail::get_launch<false>::value,
                               RAJA::Platform::hip,
                               std::conditional_t<tuning::consistent,
                                                  reduce::ordered,
                                                  reduce::unordered>>
{};

template<typename tuning>
struct hip_multi_reduce_policy
    : public RAJA::make_policy_pattern_launch_platform_t<
          RAJA::Policy::hip,
          RAJA::Pattern::multi_reduce,
          detail::get_launch<false>::value,
          RAJA::Platform::hip,
          std::conditional_t<tuning::consistent,
                             reduce::ordered,
                             reduce::unordered>>
{};

/*!
 * Hip atomic policy for using hip atomics on the device and
 * the provided policy on the host
 */
template<typename host_policy>
struct hip_atomic_explicit
{};

/*!
 * Default hip atomic policy uses hip atomics on the device and non-atomics
 * on the host
 */
using hip_atomic = hip_atomic_explicit<seq_atomic>;

// Policy for RAJA::statement::Reduce that reduces threads in a block
// down to threadIdx 0
struct hip_block_reduce
{};

// Policy for RAJA::statement::Reduce that reduces threads in a warp
// down to the first lane of the warp
struct hip_warp_reduce
{};

// Policy to map work directly to threads within a warp
// Maximum iteration count is WARP_SIZE
// Cannot be used in conjunction with hip_thread_x_*
// Multiple warps have to be created by using hip_thread_{yz}_*
// struct hip_warp_direct{};

// Policy to map work to threads within a warp using a warp-stride loop
// Cannot be used in conjunction with hip_thread_x_*
// Multiple warps have to be created by using hip_thread_{yz}_*
// struct hip_warp_loop{};


// Policy to map work to threads within a warp using a bit mask
// Cannot be used in conjunction with hip_thread_x_*
// Multiple warps have to be created by using hip_thread_{yz}_*
// Since we are masking specific threads, multiple nested
// hip_warp_masked
// can be used to create complex thread interleaving patterns
template<typename Mask>
struct hip_warp_masked_direct
{};

// Policy to map work to threads within a warp using a bit mask
// Cannot be used in conjunction with hip_thread_x_*
// Multiple warps have to be created by using hip_thread_{yz}_*
// Since we are masking specific threads, multiple nested
// hip_warp_masked
// can be used to create complex thread interleaving patterns
template<typename Mask>
struct hip_warp_masked_loop
{};

template<typename Mask>
struct hip_thread_masked_direct
{};

template<typename Mask>
struct hip_thread_masked_loop
{};

struct hip_synchronize : make_policy_pattern_launch_t<Policy::hip,
                                                      Pattern::synchronize,
                                                      Launch::sync>
{};

}  // end namespace hip
}  // end namespace policy

namespace internal
{

RAJA_INLINE
int get_size(hip_dim_t dims)
{
  if (dims.x == 0 && dims.y == 0 && dims.z == 0)
  {
    return 0;
  }
  return (dims.x ? dims.x : 1) * (dims.y ? dims.y : 1) * (dims.z ? dims.z : 1);
}

struct HipDims
{

  hip_dim_t blocks {0, 0, 0};
  hip_dim_t threads {0, 0, 0};

  HipDims()                          = default;
  HipDims(HipDims const&)            = default;
  HipDims& operator=(HipDims const&) = default;

  RAJA_INLINE
  HipDims(hip_dim_member_t default_val)
      : blocks {default_val, default_val, default_val},
        threads {default_val, default_val, default_val}
  {}

  RAJA_INLINE
  int num_blocks() const { return get_size(blocks); }

  RAJA_INLINE
  int num_threads() const { return get_size(threads); }

  RAJA_INLINE
  hip_dim_t get_blocks() const
  {
    if (num_blocks() != 0)
    {
      return {(blocks.x ? blocks.x : 1), (blocks.y ? blocks.y : 1),
              (blocks.z ? blocks.z : 1)};
    }
    else
    {
      return blocks;
    }
  }

  RAJA_INLINE
  hip_dim_t get_threads() const
  {
    if (num_threads() != 0)
    {
      return {(threads.x ? threads.x : 1), (threads.y ? threads.y : 1),
              (threads.z ? threads.z : 1)};
    }
    else
    {
      return threads;
    }
  }
};

template<named_dim dim>
struct HipDimHelper;

template<>
struct HipDimHelper<named_dim::x>
{

  template<typename dim_t>
  RAJA_HOST_DEVICE inline static constexpr hip_dim_member_t get(dim_t const& d)
  {
    return d.x;
  }

  template<typename dim_t>
  RAJA_HOST_DEVICE inline static void set(dim_t& d, hip_dim_member_t value)
  {
    d.x = value;
  }
};

template<>
struct HipDimHelper<named_dim::y>
{

  template<typename dim_t>
  RAJA_HOST_DEVICE inline static constexpr hip_dim_member_t get(dim_t const& d)
  {
    return d.y;
  }

  template<typename dim_t>
  RAJA_HOST_DEVICE inline static void set(dim_t& d, hip_dim_member_t value)
  {
    d.y = value;
  }
};

template<>
struct HipDimHelper<named_dim::z>
{

  template<typename dim_t>
  RAJA_HOST_DEVICE inline static constexpr hip_dim_member_t get(dim_t const& d)
  {
    return d.z;
  }

  template<typename dim_t>
  RAJA_HOST_DEVICE inline static void set(dim_t& d, hip_dim_member_t value)
  {
    d.z = value;
  }
};

template<named_dim dim, typename dim_t>
RAJA_HOST_DEVICE constexpr hip_dim_member_t get_hip_dim(dim_t const& d)
{
  return HipDimHelper<dim>::get(d);
}

template<named_dim dim, typename dim_t>
RAJA_HOST_DEVICE void set_hip_dim(dim_t& d, hip_dim_member_t value)
{
  return HipDimHelper<dim>::set(d, value);
}

}  // namespace internal

namespace hip
{

/// specify block size and grid size for one dimension at runtime
struct IndexSize
{
  hip_dim_member_t block_size = named_usage::unspecified;
  hip_dim_member_t grid_size  = named_usage::unspecified;

  RAJA_HOST_DEVICE constexpr IndexSize(
      hip_dim_member_t _block_size = named_usage::unspecified,
      hip_dim_member_t _grid_size  = named_usage::unspecified)
      : block_size(_block_size),
        grid_size(_grid_size)
  {}
};

/// Type representing thread indexing within a grid
/// It has various specializations that optimize specific patterns

/// useful for global indexing
/// with fixed block size and fixed grid size
template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE>
struct IndexGlobal
{
  static_assert(BLOCK_SIZE > 0, "block size must not be negative");
  static_assert(GRID_SIZE > 0, "grid size must not be negative");

  static constexpr int block_size = BLOCK_SIZE;
  static constexpr int grid_size  = GRID_SIZE;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(
               ::RAJA::internal::HipDimHelper<dim>::get(threadIdx)) +
           static_cast<IdxT>(block_size) *
               static_cast<IdxT>(
                   ::RAJA::internal::HipDimHelper<dim>::get(blockIdx));
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static constexpr IdxT size()
  {
    return static_cast<IdxT>(block_size) * static_cast<IdxT>(grid_size);
  }
};

/// with fixed block size of 1 and fixed grid size
template<named_dim dim, int GRID_SIZE>
struct IndexGlobal<dim, 1, GRID_SIZE>
{
  static_assert(GRID_SIZE > 0, "grid size must not be negative");

  static constexpr int block_size = 1;
  static constexpr int grid_size  = GRID_SIZE;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(
        ::RAJA::internal::HipDimHelper<dim>::get(blockIdx));
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static constexpr IdxT size()
  {
    return static_cast<IdxT>(grid_size);
  }
};

/// with fixed block size and fixed grid size of 1
template<named_dim dim, int BLOCK_SIZE>
struct IndexGlobal<dim, BLOCK_SIZE, 1>
{
  static_assert(BLOCK_SIZE > 0, "block size must not be negative");

  static constexpr int block_size = BLOCK_SIZE;
  static constexpr int grid_size  = 1;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(
        ::RAJA::internal::HipDimHelper<dim>::get(threadIdx));
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static constexpr IdxT size()
  {
    return static_cast<IdxT>(block_size);
  }
};

/// with fixed block size and fixed grid size of 1
template<named_dim dim>
struct IndexGlobal<dim, 1, 1>
{
  static constexpr int block_size = 1;
  static constexpr int grid_size  = 1;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(0);
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(1);
  }
};

/// with dynamic block size and fixed grid size
template<named_dim dim, int GRID_SIZE>
struct IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>
{
  static_assert(GRID_SIZE > 0, "grid size must not be negative");

  static constexpr int block_size = named_usage::unspecified;
  static constexpr int grid_size  = GRID_SIZE;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(
               ::RAJA::internal::HipDimHelper<dim>::get(threadIdx)) +
           static_cast<IdxT>(
               ::RAJA::internal::HipDimHelper<dim>::get(blockDim)) *
               static_cast<IdxT>(
                   ::RAJA::internal::HipDimHelper<dim>::get(blockIdx));
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(
               ::RAJA::internal::HipDimHelper<dim>::get(blockDim)) *
           static_cast<IdxT>(grid_size);
  }
};

/// with dynamic block size and fixed grid size of 1
template<named_dim dim>
struct IndexGlobal<dim, named_usage::unspecified, 1>
{
  static constexpr int block_size = named_usage::unspecified;
  static constexpr int grid_size  = 1;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(
        ::RAJA::internal::HipDimHelper<dim>::get(threadIdx));
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(
        ::RAJA::internal::HipDimHelper<dim>::get(blockDim));
  }
};

/// with fixed block size and dynamic grid size
template<named_dim dim, int BLOCK_SIZE>
struct IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>
{
  static_assert(BLOCK_SIZE > 0, "block size must not be negative");

  static constexpr int block_size = BLOCK_SIZE;
  static constexpr int grid_size  = named_usage::unspecified;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(
               ::RAJA::internal::HipDimHelper<dim>::get(threadIdx)) +
           static_cast<IdxT>(block_size) *
               static_cast<IdxT>(
                   ::RAJA::internal::HipDimHelper<dim>::get(blockIdx));
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(block_size) *
           static_cast<IdxT>(::RAJA::internal::HipDimHelper<dim>::get(gridDim));
  }
};

/// with fixed block size of 1 and dynamic grid size
template<named_dim dim>
struct IndexGlobal<dim, 1, named_usage::unspecified>
{
  static constexpr int block_size = 1;
  static constexpr int grid_size  = named_usage::unspecified;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(
        ::RAJA::internal::HipDimHelper<dim>::get(blockIdx));
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(::RAJA::internal::HipDimHelper<dim>::get(gridDim));
  }
};

/// with dynamic block size and dynamic grid size
template<named_dim dim>
struct IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>
{
  static constexpr int block_size = named_usage::unspecified;
  static constexpr int grid_size  = named_usage::unspecified;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(
               ::RAJA::internal::HipDimHelper<dim>::get(threadIdx)) +
           static_cast<IdxT>(
               ::RAJA::internal::HipDimHelper<dim>::get(blockDim)) *
               static_cast<IdxT>(
                   ::RAJA::internal::HipDimHelper<dim>::get(blockIdx));
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(
               ::RAJA::internal::HipDimHelper<dim>::get(blockDim)) *
           static_cast<IdxT>(::RAJA::internal::HipDimHelper<dim>::get(gridDim));
  }
};

/// useful for indexing blocks (ignores thread indices)
/// with fixed grid size
template<named_dim dim, int GRID_SIZE>
struct IndexGlobal<dim, named_usage::ignored, GRID_SIZE>
{
  static_assert(GRID_SIZE > 0, "grid size must not be negative");

  static constexpr int block_size = named_usage::ignored;
  static constexpr int grid_size  = GRID_SIZE;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(
        ::RAJA::internal::HipDimHelper<dim>::get(blockIdx));
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static constexpr IdxT size()
  {
    return static_cast<IdxT>(grid_size);
  }
};

/// with fixed grid sized of 1
template<named_dim dim>
struct IndexGlobal<dim, named_usage::ignored, 1>
{
  static constexpr int block_size = named_usage::ignored;
  static constexpr int grid_size  = 1;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(0);
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(1);
  }
};

/// with dynamic grid size
template<named_dim dim>
struct IndexGlobal<dim, named_usage::ignored, named_usage::unspecified>
{
  static constexpr int block_size = named_usage::ignored;
  static constexpr int grid_size  = named_usage::unspecified;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(
        ::RAJA::internal::HipDimHelper<dim>::get(blockIdx));
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(::RAJA::internal::HipDimHelper<dim>::get(gridDim));
  }
};

/// useful for indexing threads (ignores block indices)
/// with fixed block size
template<named_dim dim, int BLOCK_SIZE>
struct IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>
{
  static_assert(BLOCK_SIZE > 0, "block size must not be negative");

  static constexpr int block_size = BLOCK_SIZE;
  static constexpr int grid_size  = named_usage::ignored;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(
        ::RAJA::internal::HipDimHelper<dim>::get(threadIdx));
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static constexpr IdxT size()
  {
    return static_cast<IdxT>(block_size);
  }
};

/// with fixed block size of 1
template<named_dim dim>
struct IndexGlobal<dim, 1, named_usage::ignored>
{
  static constexpr int block_size = 1;
  static constexpr int grid_size  = named_usage::ignored;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(0);
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(1);
  }
};

/// with dynamic block size
template<named_dim dim>
struct IndexGlobal<dim, named_usage::unspecified, named_usage::ignored>
{
  static constexpr int block_size = named_usage::unspecified;
  static constexpr int grid_size  = named_usage::ignored;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(
        ::RAJA::internal::HipDimHelper<dim>::get(threadIdx));
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(
        ::RAJA::internal::HipDimHelper<dim>::get(blockDim));
  }
};

/// useful for doing single threaded sequential tasks
/// (ignores thread and block indices)
template<named_dim dim>
struct IndexGlobal<dim, named_usage::ignored, named_usage::ignored>
{
  static constexpr int block_size = named_usage::ignored;
  static constexpr int grid_size  = named_usage::ignored;

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(0);
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(1);
  }
};

// useful for flatten global index (includes x)
template<typename x_index>
struct IndexFlatten<x_index>
{

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {

    return x_index::template index<IdxT>();
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return x_index::template size<IdxT>();
  }
};

// useful for flatten global index (includes x,y)
template<typename x_index, typename y_index>
struct IndexFlatten<x_index, y_index>
{

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {

    return x_index::template index<IdxT>() +
           x_index::template size<IdxT>() * (y_index::template index<IdxT>());
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return x_index::template size<IdxT>() * y_index::template size<IdxT>();
  }
};

// useful for flatten global index (includes x,y,z)
template<typename x_index, typename y_index, typename z_index>
struct IndexFlatten<x_index, y_index, z_index>
{

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {

    return x_index::template index<IdxT>() +
           x_index::template size<IdxT>() *
               (y_index::template index<IdxT>() +
                y_index::template size<IdxT>() *
                    z_index::template index<IdxT>());
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return x_index::template size<IdxT>() * y_index::template size<IdxT>() *
           z_index::template size<IdxT>();
  }
};

template<size_t divisor, typename indexer>
struct IndexDivide
{
  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return indexer::template index<IdxT>() / static_cast<IdxT>(divisor);
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return RAJA_DIVIDE_CEILING_INT(indexer::template size<IdxT>(),
                                   static_cast<IdxT>(divisor));
  }
};

template<size_t divisor, typename indexer>
struct IndexModulo
{
  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT index()
  {
    return indexer::template index<IdxT>() % static_cast<IdxT>(divisor);
  }

  template<typename IdxT = hip_dim_member_t>
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(divisor);
  }
};


// helper to get just the thread indexing part of IndexGlobal
template<typename index_global>
struct get_index_thread;

///
template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE>
struct get_index_thread<IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>>
{
  using type = IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>;
};

///
template<typename x_index, typename y_index, typename z_index>
struct get_index_thread<IndexFlatten<x_index, y_index, z_index>>
{
  using type = IndexFlatten<typename get_index_thread<x_index>::type,
                            typename get_index_thread<y_index>::type,
                            typename get_index_thread<z_index>::type>;
};

// helper to get just the block indexing part of IndexGlobal
template<typename index_global>
struct get_index_block;

///
template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE>
struct get_index_block<IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>>
{
  using type = IndexGlobal<dim, named_usage::ignored, GRID_SIZE>;
};

///
template<typename x_index, typename y_index, typename z_index>
struct get_index_block<IndexFlatten<x_index, y_index, z_index>>
{
  using type = IndexFlatten<typename get_index_block<x_index>::type,
                            typename get_index_block<y_index>::type,
                            typename get_index_block<z_index>::type>;
};

template<size_t BLOCK_SIZE = named_usage::unspecified>
using thread_x = IndexGlobal<named_dim::x, BLOCK_SIZE, named_usage::ignored>;
template<size_t BLOCK_SIZE = named_usage::unspecified>
using thread_y = IndexGlobal<named_dim::y, BLOCK_SIZE, named_usage::ignored>;
template<size_t BLOCK_SIZE = named_usage::unspecified>
using thread_z = IndexGlobal<named_dim::z, BLOCK_SIZE, named_usage::ignored>;

template<size_t BLOCK_SIZE_X = named_usage::unspecified,
         size_t BLOCK_SIZE_Y = named_usage::unspecified,
         size_t BLOCK_SIZE_Z = named_usage::unspecified>
using thread_xyz = IndexFlatten<thread_x<BLOCK_SIZE_X>,
                                thread_y<BLOCK_SIZE_Y>,
                                thread_z<BLOCK_SIZE_Z>>;

template<size_t GRID_SIZE = named_usage::unspecified>
using block_x = IndexGlobal<named_dim::x, named_usage::ignored, GRID_SIZE>;
template<size_t GRID_SIZE = named_usage::unspecified>
using block_y = IndexGlobal<named_dim::y, named_usage::ignored, GRID_SIZE>;
template<size_t GRID_SIZE = named_usage::unspecified>
using block_z = IndexGlobal<named_dim::z, named_usage::ignored, GRID_SIZE>;

template<size_t GRID_SIZE_X = named_usage::unspecified,
         size_t GRID_SIZE_Y = named_usage::unspecified,
         size_t GRID_SIZE_Z = named_usage::unspecified>
using block_xyz = IndexFlatten<block_x<GRID_SIZE_X>,
                               block_y<GRID_SIZE_Y>,
                               block_z<GRID_SIZE_Z>>;

template<size_t BLOCK_SIZE, size_t GRID_SIZE = named_usage::unspecified>
using global_x = IndexGlobal<named_dim::x, BLOCK_SIZE, GRID_SIZE>;
template<size_t BLOCK_SIZE, size_t GRID_SIZE = named_usage::unspecified>
using global_y = IndexGlobal<named_dim::y, BLOCK_SIZE, GRID_SIZE>;
template<size_t BLOCK_SIZE, size_t GRID_SIZE = named_usage::unspecified>
using global_z = IndexGlobal<named_dim::z, BLOCK_SIZE, GRID_SIZE>;


template<size_t BLOCK_SIZE_X,
         size_t BLOCK_SIZE_Y,
         size_t BLOCK_SIZE_Z,
         size_t GRID_SIZE_X = named_usage::unspecified,
         size_t GRID_SIZE_Y = named_usage::unspecified,
         size_t GRID_SIZE_Z = named_usage::unspecified>
using global_xyz = IndexFlatten<global_x<BLOCK_SIZE_X, GRID_SIZE_X>,
                                global_y<BLOCK_SIZE_Y, GRID_SIZE_Y>,
                                global_z<BLOCK_SIZE_Z, GRID_SIZE_Z>>;


template<size_t WARP_SIZE    = RAJA::policy::hip::device_constants.WARP_SIZE,
         size_t BLOCK_SIZE_X = named_usage::unspecified,
         size_t BLOCK_SIZE_Y = named_usage::unspecified,
         size_t BLOCK_SIZE_Z = named_usage::unspecified>
using warp_xyz =
    IndexDivide<WARP_SIZE,
                thread_xyz<BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z>>;

template<size_t WARP_SIZE    = RAJA::policy::hip::device_constants.WARP_SIZE,
         size_t BLOCK_SIZE_X = named_usage::unspecified,
         size_t BLOCK_SIZE_Y = named_usage::unspecified,
         size_t BLOCK_SIZE_Z = named_usage::unspecified,
         size_t GRID_SIZE_X  = named_usage::unspecified,
         size_t GRID_SIZE_Y  = named_usage::unspecified,
         size_t GRID_SIZE_Z  = named_usage::unspecified>
using warp_global_xyz =
    IndexFlatten<warp_xyz<WARP_SIZE, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z>,
                 block_xyz<GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z>>;

}  // namespace hip

// contretizers used in forall, scan, and sort policies

using HipAvoidDeviceMaxThreadOccupancyConcretizer =
    hip::AvoidDeviceMaxThreadOccupancyConcretizer<
        hip::FractionOffsetOccupancyConcretizer<Fraction<size_t, 1, 1>, -1>>;

template<typename Fraction, std::ptrdiff_t BLOCKS_PER_SM_OFFSET>
using HipFractionOffsetOccupancyConcretizer =
    hip::FractionOffsetOccupancyConcretizer<Fraction, BLOCKS_PER_SM_OFFSET>;

using HipMaxOccupancyConcretizer = hip::MaxOccupancyConcretizer;

using HipReduceDefaultConcretizer =
    HipFractionOffsetOccupancyConcretizer<Fraction<size_t, 1, 2>, 0>;

using HipDefaultConcretizer = HipAvoidDeviceMaxThreadOccupancyConcretizer;

// policies usable with forall, scan, and sort

template<size_t BLOCK_SIZE, size_t GRID_SIZE, bool Async = false>
using hip_exec_grid = policy::hip::hip_exec<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    hip::global_x<BLOCK_SIZE, GRID_SIZE>,
    HipDefaultConcretizer,
    Async>;

template<size_t BLOCK_SIZE, size_t GRID_SIZE>
using hip_exec_grid_async = policy::hip::hip_exec<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    hip::global_x<BLOCK_SIZE, GRID_SIZE>,
    HipDefaultConcretizer,
    true>;

template<size_t BLOCK_SIZE, bool Async = false>
using hip_exec = policy::hip::hip_exec<iteration_mapping::Direct,
                                       hip::global_x<BLOCK_SIZE>,
                                       HipDefaultConcretizer,
                                       Async>;

template<size_t BLOCK_SIZE>
using hip_exec_async = policy::hip::hip_exec<iteration_mapping::Direct,
                                             hip::global_x<BLOCK_SIZE>,
                                             HipDefaultConcretizer,
                                             true>;

template<size_t BLOCK_SIZE, bool Async = false>
using hip_exec_occ_calc = policy::hip::hip_exec<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    hip::global_x<BLOCK_SIZE>,
    HipDefaultConcretizer,
    Async>;

template<size_t BLOCK_SIZE>
using hip_exec_occ_calc_async = policy::hip::hip_exec<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    hip::global_x<BLOCK_SIZE>,
    HipDefaultConcretizer,
    true>;

template<size_t BLOCK_SIZE, bool Async = false>
using hip_exec_occ_max = policy::hip::hip_exec<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    hip::global_x<BLOCK_SIZE>,
    HipMaxOccupancyConcretizer,
    Async>;

template<size_t BLOCK_SIZE>
using hip_exec_occ_max_async = policy::hip::hip_exec<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    hip::global_x<BLOCK_SIZE>,
    HipMaxOccupancyConcretizer,
    true>;

template<size_t BLOCK_SIZE, typename Fraction, bool Async = false>
using hip_exec_occ_fraction = policy::hip::hip_exec<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    hip::global_x<BLOCK_SIZE>,
    HipFractionOffsetOccupancyConcretizer<Fraction, 0>,
    Async>;

template<size_t BLOCK_SIZE, typename Fraction>
using hip_exec_occ_fraction_async = policy::hip::hip_exec<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    hip::global_x<BLOCK_SIZE>,
    HipFractionOffsetOccupancyConcretizer<Fraction, 0>,
    true>;

template<size_t BLOCK_SIZE, typename Concretizer, bool Async = false>
using hip_exec_occ_custom = policy::hip::hip_exec<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    hip::global_x<BLOCK_SIZE>,
    Concretizer,
    Async>;

template<size_t BLOCK_SIZE, typename Concretizer>
using hip_exec_occ_custom_async = policy::hip::hip_exec<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    hip::global_x<BLOCK_SIZE>,
    Concretizer,
    true>;

template<size_t BLOCK_SIZE, bool Async = false>
using hip_exec_with_reduce = policy::hip::hip_exec<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    hip::global_x<BLOCK_SIZE>,
    HipReduceDefaultConcretizer,
    Async>;

template<size_t BLOCK_SIZE>
using hip_exec_with_reduce_async = policy::hip::hip_exec<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    hip::global_x<BLOCK_SIZE>,
    HipReduceDefaultConcretizer,
    true>;

template<bool with_reduce, size_t BLOCK_SIZE, bool Async = false>
using hip_exec_base =
    std::conditional_t<with_reduce,
                       hip_exec_with_reduce<BLOCK_SIZE, Async>,
                       hip_exec<BLOCK_SIZE, Async>>;

template<bool with_reduce, size_t BLOCK_SIZE>
using hip_exec_base_async =
    std::conditional_t<with_reduce,
                       hip_exec_with_reduce_async<BLOCK_SIZE>,
                       hip_exec_async<BLOCK_SIZE>>;

// policies usable with WorkGroup
using policy::hip::hip_work;

template<size_t BLOCK_SIZE>
using hip_work_async = policy::hip::hip_work<BLOCK_SIZE, true>;

using policy::hip::unordered_hip_loop_y_block_iter_x_threadblock_average;

// policies usable with atomics
using policy::hip::hip_atomic;
using policy::hip::hip_atomic_explicit;


// policies usable with reducers
template<hip::reduce_algorithm algorithm,
         hip::block_communication_mode comm_mode,
         size_t replication   = named_usage::unspecified,
         size_t atomic_stride = named_usage::unspecified>
using hip_reduce_tuning = policy::hip::hip_reduce_policy<
    hip::ReduceTuning<algorithm, comm_mode, replication, atomic_stride>>;

// Policies for RAJA::Reduce* objects with specific behaviors.
// - non-atomic policies store partial results and combine them in the same
//   order every time, leading to consistent results for a loop run to run.
// - *atomic* policies may use atomics to combine partial results. The
//   use of atomics leads to order of operation differences which change the
//   results of floating point sum reductions for a loop run to run. Falls back
//   on a non-atomic implementation if atomics can't be used with the given
//   type. The memory used with atomics is initialized on the device using
//   atomics which adds overhead.
// - *atomic_host* policies are similar to the atomic policies above. However
//   the memory used with atomics is initialized on the host. This is faster
//   overall than other policies on HW with direct host access to device memory
//   such as the AMD MI300A El Capitan/Tuolumne systems.
// - *device_fence* policies use normal memory accesses with device scope fences
//                in the implementation. This works on all HW.
// - *block_fence* policies use special (atomic) memory accesses that use
//                 a cache shared by the whole device to avoid having to use
//                 device scope fences. This improves performance on some HW but
//                 is more difficult to code correctly.
using hip_reduce_device_fence =
    hip_reduce_tuning<hip::reduce_algorithm::combine_last_block,
                      hip::block_communication_mode::device_fence,
                      named_usage::unspecified,
                      named_usage::unspecified>;
///
using hip_reduce_block_fence =
    hip_reduce_tuning<hip::reduce_algorithm::combine_last_block,
                      hip::block_communication_mode::block_fence,
                      named_usage::unspecified,
                      named_usage::unspecified>;
///
using hip_reduce_atomic_device_init_device_fence =
    hip_reduce_tuning<hip::reduce_algorithm::init_device_combine_atomic_block,
                      hip::block_communication_mode::device_fence,
                      named_usage::unspecified,
                      named_usage::unspecified>;
///
using hip_reduce_atomic_device_init_block_fence =
    hip_reduce_tuning<hip::reduce_algorithm::init_device_combine_atomic_block,
                      hip::block_communication_mode::block_fence,
                      named_usage::unspecified,
                      named_usage::unspecified>;
///
using hip_reduce_atomic_host_init_device_fence =
    hip_reduce_tuning<hip::reduce_algorithm::init_host_combine_atomic_block,
                      hip::block_communication_mode::device_fence,
                      named_usage::unspecified,
                      named_usage::unspecified>;
///
using hip_reduce_atomic_host_init_block_fence =
    hip_reduce_tuning<hip::reduce_algorithm::init_host_combine_atomic_block,
                      hip::block_communication_mode::block_fence,
                      named_usage::unspecified,
                      named_usage::unspecified>;

// Policy for RAJA::Reduce* objects that gives the same answer every time when
// used in the same way
using hip_reduce = hip_reduce_block_fence;

// Policy for RAJA::Reduce* objects that may use atomics and may not give the
// same answer every time when used in the same way
using hip_reduce_atomic = hip_reduce_atomic_host_init_block_fence;

// Policy for RAJA::Reduce* objects that lets you select the default atomic or
// non-atomic policy with a bool
template<bool with_atomic>
using hip_reduce_base =
    std::conditional_t<with_atomic, hip_reduce_atomic, hip_reduce>;


// policies usable with multi_reducers
template<hip::multi_reduce_algorithm algorithm,
         typename SharedAtomicReplicationConcretizer,
         typename SharedAtomicReplicationIndexer,
         typename GlobalAtomicReplicationConcretizer,
         typename GlobalAtomicReplicationIndexer>
using hip_multi_reduce_tuning =
    policy::hip::hip_multi_reduce_policy<hip::MultiReduceTuning<
        algorithm,
        hip::AtomicReplicationTuning<SharedAtomicReplicationConcretizer,
                                     SharedAtomicReplicationIndexer,
                                     GetOffsetRight<int>>,
        hip::AtomicReplicationTuning<GlobalAtomicReplicationConcretizer,
                                     GlobalAtomicReplicationIndexer,
                                     GetOffsetLeft<int>>>>;

// Policies for RAJA::MultiReduce* objects with specific behaviors.
// - *atomic* policies may use atomics to combine partial results. The
//   use of atomics leads to order of operation differences which change the
//   results of floating point sum reductions for a loop run to run.
// - *no_replication* policies use the minimum amount of resources. The
//   lack of resources means they may perform poorly. These policies are
//   intended for use cases where low overhead is more important than high
//   performance such as error flags that are rarely set.
// - *host_init* policies initialize memory used with atomics on the host.
//   This is faster overall than other policies on HW with direct host access
//   to device memory such as the AMD MI300A El Capitan/Tuolumne systems.
using hip_multi_reduce_atomic_block_then_atomic_grid_host_init =
    hip_multi_reduce_tuning<
        hip::multi_reduce_algorithm::
            init_host_combine_block_atomic_then_grid_atomic,
        hip::SharedAtomicReplicationMaxPow2Concretizer<
            hip::ConstantPreferredReplicationConcretizer<4>>,
        hip::thread_xyz<>,
        hip::GlobalAtomicReplicationMinPow2Concretizer<
            hip::ConstantPreferredReplicationConcretizer<32>>,
        hip::warp_global_xyz<>>;
// special policy to test that multi-reducers work if there is not enough shmem
using hip_multi_reduce_atomic_block_then_atomic_grid_host_init_fallback_testing =
    hip_multi_reduce_tuning<
        hip::multi_reduce_algorithm::
            init_host_combine_block_atomic_then_grid_atomic,
        hip::SharedAtomicReplicationMaxPow2Concretizer<
            hip::ConstantPreferredReplicationConcretizer<0>>,
        hip::thread_xyz<>,
        hip::GlobalAtomicReplicationMinPow2Concretizer<
            hip::ConstantPreferredReplicationConcretizer<32>>,
        hip::warp_global_xyz<>>;
//
using hip_multi_reduce_atomic_global_host_init = hip_multi_reduce_tuning<
    hip::multi_reduce_algorithm::init_host_combine_global_atomic,
    void,  // unused with this algorithm
    void,  // unused with this algorithm
    hip::GlobalAtomicReplicationMinPow2Concretizer<
        hip::ConstantPreferredReplicationConcretizer<32>>,
    hip::warp_global_xyz<>>;
//
using hip_multi_reduce_atomic_global_no_replication_host_init =
    hip_multi_reduce_tuning<
        hip::multi_reduce_algorithm::init_host_combine_global_atomic,
        void,  // unused with this algorithm
        void,  // unused with this algorithm
        hip::GlobalAtomicReplicationMinPow2Concretizer<
            hip::ConstantPreferredReplicationConcretizer<1>>,
        hip::block_xyz<>>;

// Policy for RAJA::MultiReduce* objects that may use atomics and may not give
// the same answer every time when used in the same way
using hip_multi_reduce_atomic =
    hip_multi_reduce_atomic_block_then_atomic_grid_host_init;
// Similar to above but optimized for low overhead in cases where it is rarely
// used
using hip_multi_reduce_atomic_low_performance_low_overhead =
    hip_multi_reduce_atomic_global_no_replication_host_init;


// policies usable with kernel
using policy::hip::hip_block_reduce;
using policy::hip::hip_warp_reduce;

using hip_warp_direct_unchecked = RAJA::policy::hip::hip_indexer<
    iteration_mapping::DirectUnchecked,
    kernel_sync_requirement::none,
    hip::thread_x<RAJA::policy::hip::device_constants.WARP_SIZE>>;
using hip_warp_direct = RAJA::policy::hip::hip_indexer<
    iteration_mapping::Direct,
    kernel_sync_requirement::none,
    hip::thread_x<RAJA::policy::hip::device_constants.WARP_SIZE>>;
using hip_warp_loop = RAJA::policy::hip::hip_indexer<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    kernel_sync_requirement::none,
    hip::thread_x<RAJA::policy::hip::device_constants.WARP_SIZE>>;

using policy::hip::hip_warp_masked_direct;
using policy::hip::hip_warp_masked_loop;

using policy::hip::hip_thread_masked_direct;
using policy::hip::hip_thread_masked_loop;

// policies usable with synchronize
using policy::hip::hip_synchronize;

// policies usable with launch
using policy::hip::hip_launch_t;


// policies usable with kernel and launch
template<typename... indexers>
using hip_indexer_direct_unchecked =
    policy::hip::hip_indexer<iteration_mapping::DirectUnchecked,
                             kernel_sync_requirement::none,
                             indexers...>;

template<typename... indexers>
using hip_indexer_direct =
    policy::hip::hip_indexer<iteration_mapping::Direct,
                             kernel_sync_requirement::none,
                             indexers...>;

template<typename... indexers>
using hip_indexer_loop = policy::hip::hip_indexer<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    kernel_sync_requirement::none,
    indexers...>;

template<typename... indexers>
using hip_indexer_syncable_loop = policy::hip::hip_indexer<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    kernel_sync_requirement::sync,
    indexers...>;

template<typename... indexers>
using hip_flatten_indexer_direct_unchecked =
    policy::hip::hip_flatten_indexer<iteration_mapping::DirectUnchecked,
                                     kernel_sync_requirement::none,
                                     indexers...>;

template<typename... indexers>
using hip_flatten_indexer_direct =
    policy::hip::hip_flatten_indexer<iteration_mapping::Direct,
                                     kernel_sync_requirement::none,
                                     indexers...>;

template<typename... indexers>
using hip_flatten_indexer_loop = policy::hip::hip_flatten_indexer<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    kernel_sync_requirement::none,
    indexers...>;


/*!
 * Maps segment indices to HIP threads, blocks, or global threads.
 * This is the lowest overhead mapping, but requires that there are the same
 * number of physical threads, blocks, or global threads as map requests.
 * For example, a segment of size 1000 will only fit into 1000 threads, blocks,
 * or global threads, and triggers a runtime error in some cases.
 */
template<named_dim... dims>
using hip_thread_direct_unchecked = hip_indexer_direct_unchecked<
    hip::IndexGlobal<dims, named_usage::unspecified, named_usage::ignored>...>;
using hip_thread_x_direct_unchecked = hip_thread_direct_unchecked<named_dim::x>;
using hip_thread_y_direct_unchecked = hip_thread_direct_unchecked<named_dim::y>;
using hip_thread_z_direct_unchecked = hip_thread_direct_unchecked<named_dim::z>;
using hip_thread_xy_direct_unchecked =
    hip_thread_direct_unchecked<named_dim::x, named_dim::y>;
using hip_thread_xz_direct_unchecked =
    hip_thread_direct_unchecked<named_dim::x, named_dim::z>;
using hip_thread_yx_direct_unchecked =
    hip_thread_direct_unchecked<named_dim::y, named_dim::x>;
using hip_thread_yz_direct_unchecked =
    hip_thread_direct_unchecked<named_dim::y, named_dim::z>;
using hip_thread_zx_direct_unchecked =
    hip_thread_direct_unchecked<named_dim::z, named_dim::x>;
using hip_thread_zy_direct_unchecked =
    hip_thread_direct_unchecked<named_dim::z, named_dim::y>;
using hip_thread_xyz_direct_unchecked =
    hip_thread_direct_unchecked<named_dim::x, named_dim::y, named_dim::z>;
using hip_thread_xzy_direct_unchecked =
    hip_thread_direct_unchecked<named_dim::x, named_dim::z, named_dim::y>;
using hip_thread_yxz_direct_unchecked =
    hip_thread_direct_unchecked<named_dim::y, named_dim::x, named_dim::z>;
using hip_thread_yzx_direct_unchecked =
    hip_thread_direct_unchecked<named_dim::y, named_dim::z, named_dim::x>;
using hip_thread_zxy_direct_unchecked =
    hip_thread_direct_unchecked<named_dim::z, named_dim::x, named_dim::y>;
using hip_thread_zyx_direct_unchecked =
    hip_thread_direct_unchecked<named_dim::z, named_dim::y, named_dim::x>;

template<named_dim... dims>
using hip_block_direct_unchecked = hip_indexer_direct_unchecked<
    hip::IndexGlobal<dims, named_usage::ignored, named_usage::unspecified>...>;
using hip_block_x_direct_unchecked = hip_block_direct_unchecked<named_dim::x>;
using hip_block_y_direct_unchecked = hip_block_direct_unchecked<named_dim::y>;
using hip_block_z_direct_unchecked = hip_block_direct_unchecked<named_dim::z>;
using hip_block_xy_direct_unchecked =
    hip_block_direct_unchecked<named_dim::x, named_dim::y>;
using hip_block_xz_direct_unchecked =
    hip_block_direct_unchecked<named_dim::x, named_dim::z>;
using hip_block_yx_direct_unchecked =
    hip_block_direct_unchecked<named_dim::y, named_dim::x>;
using hip_block_yz_direct_unchecked =
    hip_block_direct_unchecked<named_dim::y, named_dim::z>;
using hip_block_zx_direct_unchecked =
    hip_block_direct_unchecked<named_dim::z, named_dim::x>;
using hip_block_zy_direct_unchecked =
    hip_block_direct_unchecked<named_dim::z, named_dim::y>;
using hip_block_xyz_direct_unchecked =
    hip_block_direct_unchecked<named_dim::x, named_dim::y, named_dim::z>;
using hip_block_xzy_direct_unchecked =
    hip_block_direct_unchecked<named_dim::x, named_dim::z, named_dim::y>;
using hip_block_yxz_direct_unchecked =
    hip_block_direct_unchecked<named_dim::y, named_dim::x, named_dim::z>;
using hip_block_yzx_direct_unchecked =
    hip_block_direct_unchecked<named_dim::y, named_dim::z, named_dim::x>;
using hip_block_zxy_direct_unchecked =
    hip_block_direct_unchecked<named_dim::z, named_dim::x, named_dim::y>;
using hip_block_zyx_direct_unchecked =
    hip_block_direct_unchecked<named_dim::z, named_dim::y, named_dim::x>;

template<named_dim... dims>
using hip_global_direct_unchecked =
    hip_indexer_direct_unchecked<hip::IndexGlobal<dims,
                                                  named_usage::unspecified,
                                                  named_usage::unspecified>...>;
using hip_global_x_direct_unchecked = hip_global_direct_unchecked<named_dim::x>;
using hip_global_y_direct_unchecked = hip_global_direct_unchecked<named_dim::y>;
using hip_global_z_direct_unchecked = hip_global_direct_unchecked<named_dim::z>;
using hip_global_xy_direct_unchecked =
    hip_global_direct_unchecked<named_dim::x, named_dim::y>;
using hip_global_xz_direct_unchecked =
    hip_global_direct_unchecked<named_dim::x, named_dim::z>;
using hip_global_yx_direct_unchecked =
    hip_global_direct_unchecked<named_dim::y, named_dim::x>;
using hip_global_yz_direct_unchecked =
    hip_global_direct_unchecked<named_dim::y, named_dim::z>;
using hip_global_zx_direct_unchecked =
    hip_global_direct_unchecked<named_dim::z, named_dim::x>;
using hip_global_zy_direct_unchecked =
    hip_global_direct_unchecked<named_dim::z, named_dim::y>;
using hip_global_xyz_direct_unchecked =
    hip_global_direct_unchecked<named_dim::x, named_dim::y, named_dim::z>;
using hip_global_xzy_direct_unchecked =
    hip_global_direct_unchecked<named_dim::x, named_dim::z, named_dim::y>;
using hip_global_yxz_direct_unchecked =
    hip_global_direct_unchecked<named_dim::y, named_dim::x, named_dim::z>;
using hip_global_yzx_direct_unchecked =
    hip_global_direct_unchecked<named_dim::y, named_dim::z, named_dim::x>;
using hip_global_zxy_direct_unchecked =
    hip_global_direct_unchecked<named_dim::z, named_dim::x, named_dim::y>;
using hip_global_zyx_direct_unchecked =
    hip_global_direct_unchecked<named_dim::z, named_dim::y, named_dim::x>;

/*!
 * Maps segment indices to HIP threads, blocks, or global threads.
 * This is a low overhead mapping, but requires that there are enough
 * physical threads, blocks, or global threads to fit all of the direct map
 * requests. For example, a segment of size 2000 will not fit into 1024 threads,
 * blocks, or global threads, and triggers a runtime error in some cases.
 */
template<named_dim... dims>
using hip_thread_direct = hip_indexer_direct<
    hip::IndexGlobal<dims, named_usage::unspecified, named_usage::ignored>...>;
using hip_thread_x_direct  = hip_thread_direct<named_dim::x>;
using hip_thread_y_direct  = hip_thread_direct<named_dim::y>;
using hip_thread_z_direct  = hip_thread_direct<named_dim::z>;
using hip_thread_xy_direct = hip_thread_direct<named_dim::x, named_dim::y>;
using hip_thread_xz_direct = hip_thread_direct<named_dim::x, named_dim::z>;
using hip_thread_yx_direct = hip_thread_direct<named_dim::y, named_dim::x>;
using hip_thread_yz_direct = hip_thread_direct<named_dim::y, named_dim::z>;
using hip_thread_zx_direct = hip_thread_direct<named_dim::z, named_dim::x>;
using hip_thread_zy_direct = hip_thread_direct<named_dim::z, named_dim::y>;
using hip_thread_xyz_direct =
    hip_thread_direct<named_dim::x, named_dim::y, named_dim::z>;
using hip_thread_xzy_direct =
    hip_thread_direct<named_dim::x, named_dim::z, named_dim::y>;
using hip_thread_yxz_direct =
    hip_thread_direct<named_dim::y, named_dim::x, named_dim::z>;
using hip_thread_yzx_direct =
    hip_thread_direct<named_dim::y, named_dim::z, named_dim::x>;
using hip_thread_zxy_direct =
    hip_thread_direct<named_dim::z, named_dim::x, named_dim::y>;
using hip_thread_zyx_direct =
    hip_thread_direct<named_dim::z, named_dim::y, named_dim::x>;

template<named_dim... dims>
using hip_block_direct = hip_indexer_direct<
    hip::IndexGlobal<dims, named_usage::ignored, named_usage::unspecified>...>;
using hip_block_x_direct  = hip_block_direct<named_dim::x>;
using hip_block_y_direct  = hip_block_direct<named_dim::y>;
using hip_block_z_direct  = hip_block_direct<named_dim::z>;
using hip_block_xy_direct = hip_block_direct<named_dim::x, named_dim::y>;
using hip_block_xz_direct = hip_block_direct<named_dim::x, named_dim::z>;
using hip_block_yx_direct = hip_block_direct<named_dim::y, named_dim::x>;
using hip_block_yz_direct = hip_block_direct<named_dim::y, named_dim::z>;
using hip_block_zx_direct = hip_block_direct<named_dim::z, named_dim::x>;
using hip_block_zy_direct = hip_block_direct<named_dim::z, named_dim::y>;
using hip_block_xyz_direct =
    hip_block_direct<named_dim::x, named_dim::y, named_dim::z>;
using hip_block_xzy_direct =
    hip_block_direct<named_dim::x, named_dim::z, named_dim::y>;
using hip_block_yxz_direct =
    hip_block_direct<named_dim::y, named_dim::x, named_dim::z>;
using hip_block_yzx_direct =
    hip_block_direct<named_dim::y, named_dim::z, named_dim::x>;
using hip_block_zxy_direct =
    hip_block_direct<named_dim::z, named_dim::x, named_dim::y>;
using hip_block_zyx_direct =
    hip_block_direct<named_dim::z, named_dim::y, named_dim::x>;

template<named_dim... dims>
using hip_global_direct =
    hip_indexer_direct<hip::IndexGlobal<dims,
                                        named_usage::unspecified,
                                        named_usage::unspecified>...>;
using hip_global_x_direct  = hip_global_direct<named_dim::x>;
using hip_global_y_direct  = hip_global_direct<named_dim::y>;
using hip_global_z_direct  = hip_global_direct<named_dim::z>;
using hip_global_xy_direct = hip_global_direct<named_dim::x, named_dim::y>;
using hip_global_xz_direct = hip_global_direct<named_dim::x, named_dim::z>;
using hip_global_yx_direct = hip_global_direct<named_dim::y, named_dim::x>;
using hip_global_yz_direct = hip_global_direct<named_dim::y, named_dim::z>;
using hip_global_zx_direct = hip_global_direct<named_dim::z, named_dim::x>;
using hip_global_zy_direct = hip_global_direct<named_dim::z, named_dim::y>;
using hip_global_xyz_direct =
    hip_global_direct<named_dim::x, named_dim::y, named_dim::z>;
using hip_global_xzy_direct =
    hip_global_direct<named_dim::x, named_dim::z, named_dim::y>;
using hip_global_yxz_direct =
    hip_global_direct<named_dim::y, named_dim::x, named_dim::z>;
using hip_global_yzx_direct =
    hip_global_direct<named_dim::y, named_dim::z, named_dim::x>;
using hip_global_zxy_direct =
    hip_global_direct<named_dim::z, named_dim::x, named_dim::y>;
using hip_global_zyx_direct =
    hip_global_direct<named_dim::z, named_dim::y, named_dim::x>;

/*!
 * Maps segment indices to HIP threads, blocks, or global threads.
 * Uses block-stride or grid-stride looping to exceed the maximum number of
 * physical threads, blocks, or global threads.
 */
template<named_dim... dims>
using hip_thread_loop = hip_indexer_loop<
    hip::IndexGlobal<dims, named_usage::unspecified, named_usage::ignored>...>;
using hip_thread_x_loop  = hip_thread_loop<named_dim::x>;
using hip_thread_y_loop  = hip_thread_loop<named_dim::y>;
using hip_thread_z_loop  = hip_thread_loop<named_dim::z>;
using hip_thread_xy_loop = hip_thread_loop<named_dim::x, named_dim::y>;
using hip_thread_xz_loop = hip_thread_loop<named_dim::x, named_dim::z>;
using hip_thread_yx_loop = hip_thread_loop<named_dim::y, named_dim::x>;
using hip_thread_yz_loop = hip_thread_loop<named_dim::y, named_dim::z>;
using hip_thread_zx_loop = hip_thread_loop<named_dim::z, named_dim::x>;
using hip_thread_zy_loop = hip_thread_loop<named_dim::z, named_dim::y>;
using hip_thread_xyz_loop =
    hip_thread_loop<named_dim::x, named_dim::y, named_dim::z>;
using hip_thread_xzy_loop =
    hip_thread_loop<named_dim::x, named_dim::z, named_dim::y>;
using hip_thread_yxz_loop =
    hip_thread_loop<named_dim::y, named_dim::x, named_dim::z>;
using hip_thread_yzx_loop =
    hip_thread_loop<named_dim::y, named_dim::z, named_dim::x>;
using hip_thread_zxy_loop =
    hip_thread_loop<named_dim::z, named_dim::x, named_dim::y>;
using hip_thread_zyx_loop =
    hip_thread_loop<named_dim::z, named_dim::y, named_dim::x>;

template<named_dim... dims>
using hip_block_loop = hip_indexer_loop<
    hip::IndexGlobal<dims, named_usage::ignored, named_usage::unspecified>...>;
using hip_block_x_loop  = hip_block_loop<named_dim::x>;
using hip_block_y_loop  = hip_block_loop<named_dim::y>;
using hip_block_z_loop  = hip_block_loop<named_dim::z>;
using hip_block_xy_loop = hip_block_loop<named_dim::x, named_dim::y>;
using hip_block_xz_loop = hip_block_loop<named_dim::x, named_dim::z>;
using hip_block_yx_loop = hip_block_loop<named_dim::y, named_dim::x>;
using hip_block_yz_loop = hip_block_loop<named_dim::y, named_dim::z>;
using hip_block_zx_loop = hip_block_loop<named_dim::z, named_dim::x>;
using hip_block_zy_loop = hip_block_loop<named_dim::z, named_dim::y>;
using hip_block_xyz_loop =
    hip_block_loop<named_dim::x, named_dim::y, named_dim::z>;
using hip_block_xzy_loop =
    hip_block_loop<named_dim::x, named_dim::z, named_dim::y>;
using hip_block_yxz_loop =
    hip_block_loop<named_dim::y, named_dim::x, named_dim::z>;
using hip_block_yzx_loop =
    hip_block_loop<named_dim::y, named_dim::z, named_dim::x>;
using hip_block_zxy_loop =
    hip_block_loop<named_dim::z, named_dim::x, named_dim::y>;
using hip_block_zyx_loop =
    hip_block_loop<named_dim::z, named_dim::y, named_dim::x>;

template<named_dim... dims>
using hip_global_loop =
    hip_indexer_loop<hip::IndexGlobal<dims,
                                      named_usage::unspecified,
                                      named_usage::unspecified>...>;
using hip_global_x_loop  = hip_global_loop<named_dim::x>;
using hip_global_y_loop  = hip_global_loop<named_dim::y>;
using hip_global_z_loop  = hip_global_loop<named_dim::z>;
using hip_global_xy_loop = hip_global_loop<named_dim::x, named_dim::y>;
using hip_global_xz_loop = hip_global_loop<named_dim::x, named_dim::z>;
using hip_global_yx_loop = hip_global_loop<named_dim::y, named_dim::x>;
using hip_global_yz_loop = hip_global_loop<named_dim::y, named_dim::z>;
using hip_global_zx_loop = hip_global_loop<named_dim::z, named_dim::x>;
using hip_global_zy_loop = hip_global_loop<named_dim::z, named_dim::y>;
using hip_global_xyz_loop =
    hip_global_loop<named_dim::x, named_dim::y, named_dim::z>;
using hip_global_xzy_loop =
    hip_global_loop<named_dim::x, named_dim::z, named_dim::y>;
using hip_global_yxz_loop =
    hip_global_loop<named_dim::y, named_dim::x, named_dim::z>;
using hip_global_yzx_loop =
    hip_global_loop<named_dim::y, named_dim::z, named_dim::x>;
using hip_global_zxy_loop =
    hip_global_loop<named_dim::z, named_dim::x, named_dim::y>;
using hip_global_zyx_loop =
    hip_global_loop<named_dim::z, named_dim::y, named_dim::x>;

/*!
 * Only used in the "kernel" abstraction.
 * Maps segment indices to HIP threads, blocks, or global threads.
 * Uses block-stride or grid-stride looping to exceed the maximum number of
 * physical threads, blocks, or global threads.
 * Allow synchronization in the loop, do not mask any threads out.
 */
template<named_dim... dims>
using hip_thread_syncable_loop = hip_indexer_syncable_loop<
    hip::IndexGlobal<dims, named_usage::unspecified, named_usage::ignored>...>;
using hip_thread_x_syncable_loop = hip_thread_syncable_loop<named_dim::x>;
using hip_thread_y_syncable_loop = hip_thread_syncable_loop<named_dim::y>;
using hip_thread_z_syncable_loop = hip_thread_syncable_loop<named_dim::z>;
using hip_thread_xy_syncable_loop =
    hip_thread_syncable_loop<named_dim::x, named_dim::y>;
using hip_thread_xz_syncable_loop =
    hip_thread_syncable_loop<named_dim::x, named_dim::z>;
using hip_thread_yx_syncable_loop =
    hip_thread_syncable_loop<named_dim::y, named_dim::x>;
using hip_thread_yz_syncable_loop =
    hip_thread_syncable_loop<named_dim::y, named_dim::z>;
using hip_thread_zx_syncable_loop =
    hip_thread_syncable_loop<named_dim::z, named_dim::x>;
using hip_thread_zy_syncable_loop =
    hip_thread_syncable_loop<named_dim::z, named_dim::y>;
using hip_thread_xyz_syncable_loop =
    hip_thread_syncable_loop<named_dim::x, named_dim::y, named_dim::z>;
using hip_thread_xzy_syncable_loop =
    hip_thread_syncable_loop<named_dim::x, named_dim::z, named_dim::y>;
using hip_thread_yxz_syncable_loop =
    hip_thread_syncable_loop<named_dim::y, named_dim::x, named_dim::z>;
using hip_thread_yzx_syncable_loop =
    hip_thread_syncable_loop<named_dim::y, named_dim::z, named_dim::x>;
using hip_thread_zxy_syncable_loop =
    hip_thread_syncable_loop<named_dim::z, named_dim::x, named_dim::y>;
using hip_thread_zyx_syncable_loop =
    hip_thread_syncable_loop<named_dim::z, named_dim::y, named_dim::x>;

template<named_dim... dims>
using hip_block_syncable_loop = hip_indexer_syncable_loop<
    hip::IndexGlobal<dims, named_usage::ignored, named_usage::unspecified>...>;
using hip_block_x_syncable_loop = hip_block_syncable_loop<named_dim::x>;
using hip_block_y_syncable_loop = hip_block_syncable_loop<named_dim::y>;
using hip_block_z_syncable_loop = hip_block_syncable_loop<named_dim::z>;
using hip_block_xy_syncable_loop =
    hip_block_syncable_loop<named_dim::x, named_dim::y>;
using hip_block_xz_syncable_loop =
    hip_block_syncable_loop<named_dim::x, named_dim::z>;
using hip_block_yx_syncable_loop =
    hip_block_syncable_loop<named_dim::y, named_dim::x>;
using hip_block_yz_syncable_loop =
    hip_block_syncable_loop<named_dim::y, named_dim::z>;
using hip_block_zx_syncable_loop =
    hip_block_syncable_loop<named_dim::z, named_dim::x>;
using hip_block_zy_syncable_loop =
    hip_block_syncable_loop<named_dim::z, named_dim::y>;
using hip_block_xyz_syncable_loop =
    hip_block_syncable_loop<named_dim::x, named_dim::y, named_dim::z>;
using hip_block_xzy_syncable_loop =
    hip_block_syncable_loop<named_dim::x, named_dim::z, named_dim::y>;
using hip_block_yxz_syncable_loop =
    hip_block_syncable_loop<named_dim::y, named_dim::x, named_dim::z>;
using hip_block_yzx_syncable_loop =
    hip_block_syncable_loop<named_dim::y, named_dim::z, named_dim::x>;
using hip_block_zxy_syncable_loop =
    hip_block_syncable_loop<named_dim::z, named_dim::x, named_dim::y>;
using hip_block_zyx_syncable_loop =
    hip_block_syncable_loop<named_dim::z, named_dim::y, named_dim::x>;

template<named_dim... dims>
using hip_global_syncable_loop =
    hip_indexer_syncable_loop<hip::IndexGlobal<dims,
                                               named_usage::unspecified,
                                               named_usage::unspecified>...>;
using hip_global_x_syncable_loop = hip_global_syncable_loop<named_dim::x>;
using hip_global_y_syncable_loop = hip_global_syncable_loop<named_dim::y>;
using hip_global_z_syncable_loop = hip_global_syncable_loop<named_dim::z>;
using hip_global_xy_syncable_loop =
    hip_global_syncable_loop<named_dim::x, named_dim::y>;
using hip_global_xz_syncable_loop =
    hip_global_syncable_loop<named_dim::x, named_dim::z>;
using hip_global_yx_syncable_loop =
    hip_global_syncable_loop<named_dim::y, named_dim::x>;
using hip_global_yz_syncable_loop =
    hip_global_syncable_loop<named_dim::y, named_dim::z>;
using hip_global_zx_syncable_loop =
    hip_global_syncable_loop<named_dim::z, named_dim::x>;
using hip_global_zy_syncable_loop =
    hip_global_syncable_loop<named_dim::z, named_dim::y>;
using hip_global_xyz_syncable_loop =
    hip_global_syncable_loop<named_dim::x, named_dim::y, named_dim::z>;
using hip_global_xzy_syncable_loop =
    hip_global_syncable_loop<named_dim::x, named_dim::z, named_dim::y>;
using hip_global_yxz_syncable_loop =
    hip_global_syncable_loop<named_dim::y, named_dim::x, named_dim::z>;
using hip_global_yzx_syncable_loop =
    hip_global_syncable_loop<named_dim::y, named_dim::z, named_dim::x>;
using hip_global_zxy_syncable_loop =
    hip_global_syncable_loop<named_dim::z, named_dim::x, named_dim::y>;
using hip_global_zyx_syncable_loop =
    hip_global_syncable_loop<named_dim::z, named_dim::y, named_dim::x>;

/*
 * Maps segment indices to flattened HIP threads, blocks, or global threads.
 * This is the lowest overhead mapping, but requires that there are the same
 * number of physical threads, blocks, or global threads as map requests.
 * Reshapes multiple physical threads, blocks, or global threads into a 1D
 * iteration space
 */
template<named_dim... dims>
using hip_flatten_thread_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::IndexGlobal<dims,
                         named_usage::unspecified,
                         named_usage::ignored>...>;
using hip_flatten_thread_x_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::x>;
using hip_flatten_thread_y_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::y>;
using hip_flatten_thread_z_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::z>;
using hip_flatten_thread_xy_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::x, named_dim::y>;
using hip_flatten_thread_xz_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::x, named_dim::z>;
using hip_flatten_thread_yx_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::y, named_dim::x>;
using hip_flatten_thread_yz_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::y, named_dim::z>;
using hip_flatten_thread_zx_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::z, named_dim::x>;
using hip_flatten_thread_zy_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::z, named_dim::y>;
using hip_flatten_thread_xyz_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::x,
                                        named_dim::y,
                                        named_dim::z>;
using hip_flatten_thread_xzy_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::x,
                                        named_dim::z,
                                        named_dim::y>;
using hip_flatten_thread_yxz_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::y,
                                        named_dim::x,
                                        named_dim::z>;
using hip_flatten_thread_yzx_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::y,
                                        named_dim::z,
                                        named_dim::x>;
using hip_flatten_thread_zxy_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::z,
                                        named_dim::x,
                                        named_dim::y>;
using hip_flatten_thread_zyx_direct_unchecked =
    hip_flatten_thread_direct_unchecked<named_dim::z,
                                        named_dim::y,
                                        named_dim::x>;

template<named_dim... dims>
using hip_flatten_block_direct_unchecked = hip_flatten_indexer_direct_unchecked<
    hip::IndexGlobal<dims, named_usage::ignored, named_usage::unspecified>...>;
using hip_flatten_block_x_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::x>;
using hip_flatten_block_y_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::y>;
using hip_flatten_block_z_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::z>;
using hip_flatten_block_xy_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::x, named_dim::y>;
using hip_flatten_block_xz_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::x, named_dim::z>;
using hip_flatten_block_yx_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::y, named_dim::x>;
using hip_flatten_block_yz_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::y, named_dim::z>;
using hip_flatten_block_zx_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::z, named_dim::x>;
using hip_flatten_block_zy_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::z, named_dim::y>;
using hip_flatten_block_xyz_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::x,
                                       named_dim::y,
                                       named_dim::z>;
using hip_flatten_block_xzy_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::x,
                                       named_dim::z,
                                       named_dim::y>;
using hip_flatten_block_yxz_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::y,
                                       named_dim::x,
                                       named_dim::z>;
using hip_flatten_block_yzx_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::y,
                                       named_dim::z,
                                       named_dim::x>;
using hip_flatten_block_zxy_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::z,
                                       named_dim::x,
                                       named_dim::y>;
using hip_flatten_block_zyx_direct_unchecked =
    hip_flatten_block_direct_unchecked<named_dim::z,
                                       named_dim::y,
                                       named_dim::x>;

template<named_dim... dims>
using hip_flatten_global_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::IndexGlobal<dims,
                         named_usage::unspecified,
                         named_usage::unspecified>...>;
using hip_flatten_global_x_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::x>;
using hip_flatten_global_y_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::y>;
using hip_flatten_global_z_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::z>;
using hip_flatten_global_xy_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::x, named_dim::y>;
using hip_flatten_global_xz_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::x, named_dim::z>;
using hip_flatten_global_yx_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::y, named_dim::x>;
using hip_flatten_global_yz_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::y, named_dim::z>;
using hip_flatten_global_zx_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::z, named_dim::x>;
using hip_flatten_global_zy_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::z, named_dim::y>;
using hip_flatten_global_xyz_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::x,
                                        named_dim::y,
                                        named_dim::z>;
using hip_flatten_global_xzy_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::x,
                                        named_dim::z,
                                        named_dim::y>;
using hip_flatten_global_yxz_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::y,
                                        named_dim::x,
                                        named_dim::z>;
using hip_flatten_global_yzx_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::y,
                                        named_dim::z,
                                        named_dim::x>;
using hip_flatten_global_zxy_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::z,
                                        named_dim::x,
                                        named_dim::y>;
using hip_flatten_global_zyx_direct_unchecked =
    hip_flatten_global_direct_unchecked<named_dim::z,
                                        named_dim::y,
                                        named_dim::x>;

/*
 * Maps segment indices to flattened HIP threads, blocks, or global threads.
 * This is a low overhead mapping, but requires that there are enough
 * physical threads, blocks, or global threads to fit all of the direct map
 * requests.
 * Reshapes multiple physical threads, blocks, or global threads into a 1D
 * iteration space
 */
template<named_dim... dims>
using hip_flatten_thread_direct = hip_flatten_indexer_direct<
    hip::IndexGlobal<dims, named_usage::unspecified, named_usage::ignored>...>;
using hip_flatten_thread_x_direct = hip_flatten_thread_direct<named_dim::x>;
using hip_flatten_thread_y_direct = hip_flatten_thread_direct<named_dim::y>;
using hip_flatten_thread_z_direct = hip_flatten_thread_direct<named_dim::z>;
using hip_flatten_thread_xy_direct =
    hip_flatten_thread_direct<named_dim::x, named_dim::y>;
using hip_flatten_thread_xz_direct =
    hip_flatten_thread_direct<named_dim::x, named_dim::z>;
using hip_flatten_thread_yx_direct =
    hip_flatten_thread_direct<named_dim::y, named_dim::x>;
using hip_flatten_thread_yz_direct =
    hip_flatten_thread_direct<named_dim::y, named_dim::z>;
using hip_flatten_thread_zx_direct =
    hip_flatten_thread_direct<named_dim::z, named_dim::x>;
using hip_flatten_thread_zy_direct =
    hip_flatten_thread_direct<named_dim::z, named_dim::y>;
using hip_flatten_thread_xyz_direct =
    hip_flatten_thread_direct<named_dim::x, named_dim::y, named_dim::z>;
using hip_flatten_thread_xzy_direct =
    hip_flatten_thread_direct<named_dim::x, named_dim::z, named_dim::y>;
using hip_flatten_thread_yxz_direct =
    hip_flatten_thread_direct<named_dim::y, named_dim::x, named_dim::z>;
using hip_flatten_thread_yzx_direct =
    hip_flatten_thread_direct<named_dim::y, named_dim::z, named_dim::x>;
using hip_flatten_thread_zxy_direct =
    hip_flatten_thread_direct<named_dim::z, named_dim::x, named_dim::y>;
using hip_flatten_thread_zyx_direct =
    hip_flatten_thread_direct<named_dim::z, named_dim::y, named_dim::x>;

template<named_dim... dims>
using hip_flatten_block_direct = hip_flatten_indexer_direct<
    hip::IndexGlobal<dims, named_usage::ignored, named_usage::unspecified>...>;
using hip_flatten_block_x_direct = hip_flatten_block_direct<named_dim::x>;
using hip_flatten_block_y_direct = hip_flatten_block_direct<named_dim::y>;
using hip_flatten_block_z_direct = hip_flatten_block_direct<named_dim::z>;
using hip_flatten_block_xy_direct =
    hip_flatten_block_direct<named_dim::x, named_dim::y>;
using hip_flatten_block_xz_direct =
    hip_flatten_block_direct<named_dim::x, named_dim::z>;
using hip_flatten_block_yx_direct =
    hip_flatten_block_direct<named_dim::y, named_dim::x>;
using hip_flatten_block_yz_direct =
    hip_flatten_block_direct<named_dim::y, named_dim::z>;
using hip_flatten_block_zx_direct =
    hip_flatten_block_direct<named_dim::z, named_dim::x>;
using hip_flatten_block_zy_direct =
    hip_flatten_block_direct<named_dim::z, named_dim::y>;
using hip_flatten_block_xyz_direct =
    hip_flatten_block_direct<named_dim::x, named_dim::y, named_dim::z>;
using hip_flatten_block_xzy_direct =
    hip_flatten_block_direct<named_dim::x, named_dim::z, named_dim::y>;
using hip_flatten_block_yxz_direct =
    hip_flatten_block_direct<named_dim::y, named_dim::x, named_dim::z>;
using hip_flatten_block_yzx_direct =
    hip_flatten_block_direct<named_dim::y, named_dim::z, named_dim::x>;
using hip_flatten_block_zxy_direct =
    hip_flatten_block_direct<named_dim::z, named_dim::x, named_dim::y>;
using hip_flatten_block_zyx_direct =
    hip_flatten_block_direct<named_dim::z, named_dim::y, named_dim::x>;

template<named_dim... dims>
using hip_flatten_global_direct =
    hip_flatten_indexer_direct<hip::IndexGlobal<dims,
                                                named_usage::unspecified,
                                                named_usage::unspecified>...>;
using hip_flatten_global_x_direct = hip_flatten_global_direct<named_dim::x>;
using hip_flatten_global_y_direct = hip_flatten_global_direct<named_dim::y>;
using hip_flatten_global_z_direct = hip_flatten_global_direct<named_dim::z>;
using hip_flatten_global_xy_direct =
    hip_flatten_global_direct<named_dim::x, named_dim::y>;
using hip_flatten_global_xz_direct =
    hip_flatten_global_direct<named_dim::x, named_dim::z>;
using hip_flatten_global_yx_direct =
    hip_flatten_global_direct<named_dim::y, named_dim::x>;
using hip_flatten_global_yz_direct =
    hip_flatten_global_direct<named_dim::y, named_dim::z>;
using hip_flatten_global_zx_direct =
    hip_flatten_global_direct<named_dim::z, named_dim::x>;
using hip_flatten_global_zy_direct =
    hip_flatten_global_direct<named_dim::z, named_dim::y>;
using hip_flatten_global_xyz_direct =
    hip_flatten_global_direct<named_dim::x, named_dim::y, named_dim::z>;
using hip_flatten_global_xzy_direct =
    hip_flatten_global_direct<named_dim::x, named_dim::z, named_dim::y>;
using hip_flatten_global_yxz_direct =
    hip_flatten_global_direct<named_dim::y, named_dim::x, named_dim::z>;
using hip_flatten_global_yzx_direct =
    hip_flatten_global_direct<named_dim::y, named_dim::z, named_dim::x>;
using hip_flatten_global_zxy_direct =
    hip_flatten_global_direct<named_dim::z, named_dim::x, named_dim::y>;
using hip_flatten_global_zyx_direct =
    hip_flatten_global_direct<named_dim::z, named_dim::y, named_dim::x>;

/*
 * Maps segment indices to flattened HIP threads, blocks, or global threads.
 * Reshapes multiple physical threads, blocks, or global threads into a 1D
 * iteration space
 * Uses block-stride or grid-stride looping to exceed the maximum number of
 * physical threads, blocks, or global threads
 */
template<named_dim... dims>
using hip_flatten_thread_loop = hip_flatten_indexer_loop<
    hip::IndexGlobal<dims, named_usage::unspecified, named_usage::ignored>...>;
using hip_flatten_thread_x_loop = hip_flatten_thread_loop<named_dim::x>;
using hip_flatten_thread_y_loop = hip_flatten_thread_loop<named_dim::y>;
using hip_flatten_thread_z_loop = hip_flatten_thread_loop<named_dim::z>;
using hip_flatten_thread_xy_loop =
    hip_flatten_thread_loop<named_dim::x, named_dim::y>;
using hip_flatten_thread_xz_loop =
    hip_flatten_thread_loop<named_dim::x, named_dim::z>;
using hip_flatten_thread_yx_loop =
    hip_flatten_thread_loop<named_dim::y, named_dim::x>;
using hip_flatten_thread_yz_loop =
    hip_flatten_thread_loop<named_dim::y, named_dim::z>;
using hip_flatten_thread_zx_loop =
    hip_flatten_thread_loop<named_dim::z, named_dim::x>;
using hip_flatten_thread_zy_loop =
    hip_flatten_thread_loop<named_dim::z, named_dim::y>;
using hip_flatten_thread_xyz_loop =
    hip_flatten_thread_loop<named_dim::x, named_dim::y, named_dim::z>;
using hip_flatten_thread_xzy_loop =
    hip_flatten_thread_loop<named_dim::x, named_dim::z, named_dim::y>;
using hip_flatten_thread_yxz_loop =
    hip_flatten_thread_loop<named_dim::y, named_dim::x, named_dim::z>;
using hip_flatten_thread_yzx_loop =
    hip_flatten_thread_loop<named_dim::y, named_dim::z, named_dim::x>;
using hip_flatten_thread_zxy_loop =
    hip_flatten_thread_loop<named_dim::z, named_dim::x, named_dim::y>;
using hip_flatten_thread_zyx_loop =
    hip_flatten_thread_loop<named_dim::z, named_dim::y, named_dim::x>;

template<named_dim... dims>
using hip_flatten_block_loop = hip_flatten_indexer_loop<
    hip::IndexGlobal<dims, named_usage::ignored, named_usage::unspecified>...>;
using hip_flatten_block_x_loop = hip_flatten_block_loop<named_dim::x>;
using hip_flatten_block_y_loop = hip_flatten_block_loop<named_dim::y>;
using hip_flatten_block_z_loop = hip_flatten_block_loop<named_dim::z>;
using hip_flatten_block_xy_loop =
    hip_flatten_block_loop<named_dim::x, named_dim::y>;
using hip_flatten_block_xz_loop =
    hip_flatten_block_loop<named_dim::x, named_dim::z>;
using hip_flatten_block_yx_loop =
    hip_flatten_block_loop<named_dim::y, named_dim::x>;
using hip_flatten_block_yz_loop =
    hip_flatten_block_loop<named_dim::y, named_dim::z>;
using hip_flatten_block_zx_loop =
    hip_flatten_block_loop<named_dim::z, named_dim::x>;
using hip_flatten_block_zy_loop =
    hip_flatten_block_loop<named_dim::z, named_dim::y>;
using hip_flatten_block_xyz_loop =
    hip_flatten_block_loop<named_dim::x, named_dim::y, named_dim::z>;
using hip_flatten_block_xzy_loop =
    hip_flatten_block_loop<named_dim::x, named_dim::z, named_dim::y>;
using hip_flatten_block_yxz_loop =
    hip_flatten_block_loop<named_dim::y, named_dim::x, named_dim::z>;
using hip_flatten_block_yzx_loop =
    hip_flatten_block_loop<named_dim::y, named_dim::z, named_dim::x>;
using hip_flatten_block_zxy_loop =
    hip_flatten_block_loop<named_dim::z, named_dim::x, named_dim::y>;
using hip_flatten_block_zyx_loop =
    hip_flatten_block_loop<named_dim::z, named_dim::y, named_dim::x>;

template<named_dim... dims>
using hip_flatten_global_loop =
    hip_flatten_indexer_loop<hip::IndexGlobal<dims,
                                              named_usage::unspecified,
                                              named_usage::unspecified>...>;
using hip_flatten_global_x_loop = hip_flatten_global_loop<named_dim::x>;
using hip_flatten_global_y_loop = hip_flatten_global_loop<named_dim::y>;
using hip_flatten_global_z_loop = hip_flatten_global_loop<named_dim::z>;
using hip_flatten_global_xy_loop =
    hip_flatten_global_loop<named_dim::x, named_dim::y>;
using hip_flatten_global_xz_loop =
    hip_flatten_global_loop<named_dim::x, named_dim::z>;
using hip_flatten_global_yx_loop =
    hip_flatten_global_loop<named_dim::y, named_dim::x>;
using hip_flatten_global_yz_loop =
    hip_flatten_global_loop<named_dim::y, named_dim::z>;
using hip_flatten_global_zx_loop =
    hip_flatten_global_loop<named_dim::z, named_dim::x>;
using hip_flatten_global_zy_loop =
    hip_flatten_global_loop<named_dim::z, named_dim::y>;
using hip_flatten_global_xyz_loop =
    hip_flatten_global_loop<named_dim::x, named_dim::y, named_dim::z>;
using hip_flatten_global_xzy_loop =
    hip_flatten_global_loop<named_dim::x, named_dim::z, named_dim::y>;
using hip_flatten_global_yxz_loop =
    hip_flatten_global_loop<named_dim::y, named_dim::x, named_dim::z>;
using hip_flatten_global_yzx_loop =
    hip_flatten_global_loop<named_dim::y, named_dim::z, named_dim::x>;
using hip_flatten_global_zxy_loop =
    hip_flatten_global_loop<named_dim::z, named_dim::x, named_dim::y>;
using hip_flatten_global_zyx_loop =
    hip_flatten_global_loop<named_dim::z, named_dim::y, named_dim::x>;

/*!
 * Maps segment indices to HIP threads, blocks, or global threads.
 * This is the lowest overhead mapping, but requires that there are the same
 * number of physical threads as the map requests.
 */
template<int X_SIZE>
using hip_thread_size_x_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_x<X_SIZE>>;
template<int Y_SIZE>
using hip_thread_size_y_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_y<Y_SIZE>>;
template<int Z_SIZE>
using hip_thread_size_z_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_z<Z_SIZE>>;
template<int X_SIZE, int Y_SIZE>
using hip_thread_size_xy_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_x<X_SIZE>, hip::thread_y<Y_SIZE>>;
template<int X_SIZE, int Z_SIZE>
using hip_thread_size_xz_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_x<X_SIZE>, hip::thread_z<Z_SIZE>>;
template<int Y_SIZE, int X_SIZE>
using hip_thread_size_yx_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_y<Y_SIZE>, hip::thread_x<X_SIZE>>;
template<int Y_SIZE, int Z_SIZE>
using hip_thread_size_yz_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_y<Y_SIZE>, hip::thread_z<Z_SIZE>>;
template<int Z_SIZE, int X_SIZE>
using hip_thread_size_zx_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_z<Z_SIZE>, hip::thread_x<X_SIZE>>;
template<int Z_SIZE, int Y_SIZE>
using hip_thread_size_zy_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_z<Z_SIZE>, hip::thread_y<Y_SIZE>>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using hip_thread_size_xyz_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_x<X_SIZE>,
                                 hip::thread_y<Y_SIZE>,
                                 hip::thread_z<Z_SIZE>>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using hip_thread_size_xzy_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_x<X_SIZE>,
                                 hip::thread_z<Z_SIZE>,
                                 hip::thread_y<Y_SIZE>>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using hip_thread_size_yxz_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_y<Y_SIZE>,
                                 hip::thread_x<X_SIZE>,
                                 hip::thread_z<Z_SIZE>>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using hip_thread_size_yzx_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_y<Y_SIZE>,
                                 hip::thread_z<Z_SIZE>,
                                 hip::thread_x<X_SIZE>>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using hip_thread_size_zxy_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_z<Z_SIZE>,
                                 hip::thread_x<X_SIZE>,
                                 hip::thread_y<Y_SIZE>>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using hip_thread_size_zyx_direct_unchecked =
    hip_indexer_direct_unchecked<hip::thread_z<Z_SIZE>,
                                 hip::thread_y<Y_SIZE>,
                                 hip::thread_x<X_SIZE>>;

template<int X_SIZE>
using hip_block_size_x_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_x<X_SIZE>>;
template<int Y_SIZE>
using hip_block_size_y_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_y<Y_SIZE>>;
template<int Z_SIZE>
using hip_block_size_z_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_z<Z_SIZE>>;
template<int X_SIZE, int Y_SIZE>
using hip_block_size_xy_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_x<X_SIZE>, hip::block_y<Y_SIZE>>;
template<int X_SIZE, int Z_SIZE>
using hip_block_size_xz_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_x<X_SIZE>, hip::block_z<Z_SIZE>>;
template<int Y_SIZE, int X_SIZE>
using hip_block_size_yx_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_y<Y_SIZE>, hip::block_x<X_SIZE>>;
template<int Y_SIZE, int Z_SIZE>
using hip_block_size_yz_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_y<Y_SIZE>, hip::block_z<Z_SIZE>>;
template<int Z_SIZE, int X_SIZE>
using hip_block_size_zx_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_z<Z_SIZE>, hip::block_x<X_SIZE>>;
template<int Z_SIZE, int Y_SIZE>
using hip_block_size_zy_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_z<Z_SIZE>, hip::block_y<Y_SIZE>>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using hip_block_size_xyz_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_x<X_SIZE>,
                                 hip::block_y<Y_SIZE>,
                                 hip::block_z<Z_SIZE>>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using hip_block_size_xzy_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_x<X_SIZE>,
                                 hip::block_z<Z_SIZE>,
                                 hip::block_y<Y_SIZE>>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using hip_block_size_yxz_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_y<Y_SIZE>,
                                 hip::block_x<X_SIZE>,
                                 hip::block_z<Z_SIZE>>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using hip_block_size_yzx_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_y<Y_SIZE>,
                                 hip::block_z<Z_SIZE>,
                                 hip::block_x<X_SIZE>>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using hip_block_size_zxy_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_z<Z_SIZE>,
                                 hip::block_x<X_SIZE>,
                                 hip::block_y<Y_SIZE>>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using hip_block_size_zyx_direct_unchecked =
    hip_indexer_direct_unchecked<hip::block_z<Z_SIZE>,
                                 hip::block_y<Y_SIZE>,
                                 hip::block_x<X_SIZE>>;

template<int X_BLOCK_SIZE, int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_x_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Y_BLOCK_SIZE, int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_y_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Z_BLOCK_SIZE, int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_z_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_xy_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                 hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_xz_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                 hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_yx_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                 hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_yz_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                 hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_zx_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                 hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_zy_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                 hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_xyz_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                 hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                 hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_xzy_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                 hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                 hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_yxz_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                 hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                 hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_yzx_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                 hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                 hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_zxy_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                 hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                 hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_zyx_direct_unchecked =
    hip_indexer_direct_unchecked<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                 hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                 hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;

/*!
 * Maps segment indices to HIP threads, blocks, or global threads.
 * This is a low overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 */
template<int X_SIZE>
using hip_thread_size_x_direct = hip_indexer_direct<hip::thread_x<X_SIZE>>;
template<int Y_SIZE>
using hip_thread_size_y_direct = hip_indexer_direct<hip::thread_y<Y_SIZE>>;
template<int Z_SIZE>
using hip_thread_size_z_direct = hip_indexer_direct<hip::thread_z<Z_SIZE>>;
template<int X_SIZE, int Y_SIZE>
using hip_thread_size_xy_direct =
    hip_indexer_direct<hip::thread_x<X_SIZE>, hip::thread_y<Y_SIZE>>;
template<int X_SIZE, int Z_SIZE>
using hip_thread_size_xz_direct =
    hip_indexer_direct<hip::thread_x<X_SIZE>, hip::thread_z<Z_SIZE>>;
template<int Y_SIZE, int X_SIZE>
using hip_thread_size_yx_direct =
    hip_indexer_direct<hip::thread_y<Y_SIZE>, hip::thread_x<X_SIZE>>;
template<int Y_SIZE, int Z_SIZE>
using hip_thread_size_yz_direct =
    hip_indexer_direct<hip::thread_y<Y_SIZE>, hip::thread_z<Z_SIZE>>;
template<int Z_SIZE, int X_SIZE>
using hip_thread_size_zx_direct =
    hip_indexer_direct<hip::thread_z<Z_SIZE>, hip::thread_x<X_SIZE>>;
template<int Z_SIZE, int Y_SIZE>
using hip_thread_size_zy_direct =
    hip_indexer_direct<hip::thread_z<Z_SIZE>, hip::thread_y<Y_SIZE>>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using hip_thread_size_xyz_direct = hip_indexer_direct<hip::thread_x<X_SIZE>,
                                                      hip::thread_y<Y_SIZE>,
                                                      hip::thread_z<Z_SIZE>>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using hip_thread_size_xzy_direct = hip_indexer_direct<hip::thread_x<X_SIZE>,
                                                      hip::thread_z<Z_SIZE>,
                                                      hip::thread_y<Y_SIZE>>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using hip_thread_size_yxz_direct = hip_indexer_direct<hip::thread_y<Y_SIZE>,
                                                      hip::thread_x<X_SIZE>,
                                                      hip::thread_z<Z_SIZE>>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using hip_thread_size_yzx_direct = hip_indexer_direct<hip::thread_y<Y_SIZE>,
                                                      hip::thread_z<Z_SIZE>,
                                                      hip::thread_x<X_SIZE>>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using hip_thread_size_zxy_direct = hip_indexer_direct<hip::thread_z<Z_SIZE>,
                                                      hip::thread_x<X_SIZE>,
                                                      hip::thread_y<Y_SIZE>>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using hip_thread_size_zyx_direct = hip_indexer_direct<hip::thread_z<Z_SIZE>,
                                                      hip::thread_y<Y_SIZE>,
                                                      hip::thread_x<X_SIZE>>;

template<int X_SIZE>
using hip_block_size_x_direct = hip_indexer_direct<hip::block_x<X_SIZE>>;
template<int Y_SIZE>
using hip_block_size_y_direct = hip_indexer_direct<hip::block_y<Y_SIZE>>;
template<int Z_SIZE>
using hip_block_size_z_direct = hip_indexer_direct<hip::block_z<Z_SIZE>>;
template<int X_SIZE, int Y_SIZE>
using hip_block_size_xy_direct =
    hip_indexer_direct<hip::block_x<X_SIZE>, hip::block_y<Y_SIZE>>;
template<int X_SIZE, int Z_SIZE>
using hip_block_size_xz_direct =
    hip_indexer_direct<hip::block_x<X_SIZE>, hip::block_z<Z_SIZE>>;
template<int Y_SIZE, int X_SIZE>
using hip_block_size_yx_direct =
    hip_indexer_direct<hip::block_y<Y_SIZE>, hip::block_x<X_SIZE>>;
template<int Y_SIZE, int Z_SIZE>
using hip_block_size_yz_direct =
    hip_indexer_direct<hip::block_y<Y_SIZE>, hip::block_z<Z_SIZE>>;
template<int Z_SIZE, int X_SIZE>
using hip_block_size_zx_direct =
    hip_indexer_direct<hip::block_z<Z_SIZE>, hip::block_x<X_SIZE>>;
template<int Z_SIZE, int Y_SIZE>
using hip_block_size_zy_direct =
    hip_indexer_direct<hip::block_z<Z_SIZE>, hip::block_y<Y_SIZE>>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using hip_block_size_xyz_direct = hip_indexer_direct<hip::block_x<X_SIZE>,
                                                     hip::block_y<Y_SIZE>,
                                                     hip::block_z<Z_SIZE>>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using hip_block_size_xzy_direct = hip_indexer_direct<hip::block_x<X_SIZE>,
                                                     hip::block_z<Z_SIZE>,
                                                     hip::block_y<Y_SIZE>>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using hip_block_size_yxz_direct = hip_indexer_direct<hip::block_y<Y_SIZE>,
                                                     hip::block_x<X_SIZE>,
                                                     hip::block_z<Z_SIZE>>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using hip_block_size_yzx_direct = hip_indexer_direct<hip::block_y<Y_SIZE>,
                                                     hip::block_z<Z_SIZE>,
                                                     hip::block_x<X_SIZE>>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using hip_block_size_zxy_direct = hip_indexer_direct<hip::block_z<Z_SIZE>,
                                                     hip::block_x<X_SIZE>,
                                                     hip::block_y<Y_SIZE>>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using hip_block_size_zyx_direct = hip_indexer_direct<hip::block_z<Z_SIZE>,
                                                     hip::block_y<Y_SIZE>,
                                                     hip::block_x<X_SIZE>>;

template<int X_BLOCK_SIZE, int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_x_direct =
    hip_indexer_direct<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Y_BLOCK_SIZE, int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_y_direct =
    hip_indexer_direct<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Z_BLOCK_SIZE, int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_z_direct =
    hip_indexer_direct<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_xy_direct =
    hip_indexer_direct<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                       hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_xz_direct =
    hip_indexer_direct<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                       hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_yx_direct =
    hip_indexer_direct<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                       hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_yz_direct =
    hip_indexer_direct<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                       hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_zx_direct =
    hip_indexer_direct<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                       hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_zy_direct =
    hip_indexer_direct<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                       hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_xyz_direct =
    hip_indexer_direct<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                       hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                       hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_xzy_direct =
    hip_indexer_direct<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                       hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                       hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_yxz_direct =
    hip_indexer_direct<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                       hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                       hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_yzx_direct =
    hip_indexer_direct<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                       hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                       hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_zxy_direct =
    hip_indexer_direct<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                       hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                       hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_zyx_direct =
    hip_indexer_direct<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                       hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                       hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;

/*!
 * Maps segment indices to HIP threads, blocks, or global threads.
 * Uses block-stride or grid-stride looping to exceed the maximum number of
 * threads, blocks, or global threads.
 */
template<int X_SIZE>
using hip_thread_size_x_loop = hip_indexer_loop<hip::thread_x<X_SIZE>>;
template<int Y_SIZE>
using hip_thread_size_y_loop = hip_indexer_loop<hip::thread_y<Y_SIZE>>;
template<int Z_SIZE>
using hip_thread_size_z_loop = hip_indexer_loop<hip::thread_z<Z_SIZE>>;
template<int X_SIZE, int Y_SIZE>
using hip_thread_size_xy_loop =
    hip_indexer_loop<hip::thread_x<X_SIZE>, hip::thread_y<Y_SIZE>>;
template<int X_SIZE, int Z_SIZE>
using hip_thread_size_xz_loop =
    hip_indexer_loop<hip::thread_x<X_SIZE>, hip::thread_z<Z_SIZE>>;
template<int Y_SIZE, int X_SIZE>
using hip_thread_size_yx_loop =
    hip_indexer_loop<hip::thread_y<Y_SIZE>, hip::thread_x<X_SIZE>>;
template<int Y_SIZE, int Z_SIZE>
using hip_thread_size_yz_loop =
    hip_indexer_loop<hip::thread_y<Y_SIZE>, hip::thread_z<Z_SIZE>>;
template<int Z_SIZE, int X_SIZE>
using hip_thread_size_zx_loop =
    hip_indexer_loop<hip::thread_z<Z_SIZE>, hip::thread_x<X_SIZE>>;
template<int Z_SIZE, int Y_SIZE>
using hip_thread_size_zy_loop =
    hip_indexer_loop<hip::thread_z<Z_SIZE>, hip::thread_y<Y_SIZE>>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using hip_thread_size_xyz_loop = hip_indexer_loop<hip::thread_x<X_SIZE>,
                                                  hip::thread_y<Y_SIZE>,
                                                  hip::thread_z<Z_SIZE>>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using hip_thread_size_xzy_loop = hip_indexer_loop<hip::thread_x<X_SIZE>,
                                                  hip::thread_z<Z_SIZE>,
                                                  hip::thread_y<Y_SIZE>>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using hip_thread_size_yxz_loop = hip_indexer_loop<hip::thread_y<Y_SIZE>,
                                                  hip::thread_x<X_SIZE>,
                                                  hip::thread_z<Z_SIZE>>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using hip_thread_size_yzx_loop = hip_indexer_loop<hip::thread_y<Y_SIZE>,
                                                  hip::thread_z<Z_SIZE>,
                                                  hip::thread_x<X_SIZE>>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using hip_thread_size_zxy_loop = hip_indexer_loop<hip::thread_z<Z_SIZE>,
                                                  hip::thread_x<X_SIZE>,
                                                  hip::thread_y<Y_SIZE>>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using hip_thread_size_zyx_loop = hip_indexer_loop<hip::thread_z<Z_SIZE>,
                                                  hip::thread_y<Y_SIZE>,
                                                  hip::thread_x<X_SIZE>>;

template<int X_SIZE>
using hip_block_size_x_loop = hip_indexer_loop<hip::block_x<X_SIZE>>;
template<int Y_SIZE>
using hip_block_size_y_loop = hip_indexer_loop<hip::block_y<Y_SIZE>>;
template<int Z_SIZE>
using hip_block_size_z_loop = hip_indexer_loop<hip::block_z<Z_SIZE>>;
template<int X_SIZE, int Y_SIZE>
using hip_block_size_xy_loop =
    hip_indexer_loop<hip::block_x<X_SIZE>, hip::block_y<Y_SIZE>>;
template<int X_SIZE, int Z_SIZE>
using hip_block_size_xz_loop =
    hip_indexer_loop<hip::block_x<X_SIZE>, hip::block_z<Z_SIZE>>;
template<int Y_SIZE, int X_SIZE>
using hip_block_size_yx_loop =
    hip_indexer_loop<hip::block_y<Y_SIZE>, hip::block_x<X_SIZE>>;
template<int Y_SIZE, int Z_SIZE>
using hip_block_size_yz_loop =
    hip_indexer_loop<hip::block_y<Y_SIZE>, hip::block_z<Z_SIZE>>;
template<int Z_SIZE, int X_SIZE>
using hip_block_size_zx_loop =
    hip_indexer_loop<hip::block_z<Z_SIZE>, hip::block_x<X_SIZE>>;
template<int Z_SIZE, int Y_SIZE>
using hip_block_size_zy_loop =
    hip_indexer_loop<hip::block_z<Z_SIZE>, hip::block_y<Y_SIZE>>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using hip_block_size_xyz_loop = hip_indexer_loop<hip::block_x<X_SIZE>,
                                                 hip::block_y<Y_SIZE>,
                                                 hip::block_z<Z_SIZE>>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using hip_block_size_xzy_loop = hip_indexer_loop<hip::block_x<X_SIZE>,
                                                 hip::block_z<Z_SIZE>,
                                                 hip::block_y<Y_SIZE>>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using hip_block_size_yxz_loop = hip_indexer_loop<hip::block_y<Y_SIZE>,
                                                 hip::block_x<X_SIZE>,
                                                 hip::block_z<Z_SIZE>>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using hip_block_size_yzx_loop = hip_indexer_loop<hip::block_y<Y_SIZE>,
                                                 hip::block_z<Z_SIZE>,
                                                 hip::block_x<X_SIZE>>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using hip_block_size_zxy_loop = hip_indexer_loop<hip::block_z<Z_SIZE>,
                                                 hip::block_x<X_SIZE>,
                                                 hip::block_y<Y_SIZE>>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using hip_block_size_zyx_loop = hip_indexer_loop<hip::block_z<Z_SIZE>,
                                                 hip::block_y<Y_SIZE>,
                                                 hip::block_x<X_SIZE>>;

template<int X_BLOCK_SIZE, int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_x_loop =
    hip_indexer_loop<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Y_BLOCK_SIZE, int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_y_loop =
    hip_indexer_loop<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Z_BLOCK_SIZE, int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_z_loop =
    hip_indexer_loop<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_xy_loop =
    hip_indexer_loop<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                     hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_xz_loop =
    hip_indexer_loop<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                     hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_yx_loop =
    hip_indexer_loop<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                     hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_yz_loop =
    hip_indexer_loop<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                     hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_zx_loop =
    hip_indexer_loop<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                     hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_zy_loop =
    hip_indexer_loop<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                     hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_xyz_loop =
    hip_indexer_loop<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                     hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                     hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_xzy_loop =
    hip_indexer_loop<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                     hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                     hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_global_size_yxz_loop =
    hip_indexer_loop<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                     hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                     hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_yzx_loop =
    hip_indexer_loop<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                     hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                     hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_global_size_zxy_loop =
    hip_indexer_loop<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                     hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                     hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_global_size_zyx_loop =
    hip_indexer_loop<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                     hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                     hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;

/*
 * Maps segment indices to flattened HIP threads, blocks, or global threads.
 * This is the lowest overhead mapping, but requires that there are the same
 * number of physical threads, blocks, or global threads as the map requests.
 * Reshapes multiple physical threads, blocks, or global threads into a 1D
 * iteration space.
 */
template<int X_SIZE>
using hip_flatten_thread_size_x_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_x<X_SIZE>>;
template<int Y_SIZE>
using hip_flatten_thread_size_y_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_y<Y_SIZE>>;
template<int Z_SIZE>
using hip_flatten_thread_size_z_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_z<Z_SIZE>>;
template<int X_SIZE, int Y_SIZE>
using hip_flatten_thread_size_xy_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_x<X_SIZE>,
                                         hip::thread_y<Y_SIZE>>;
template<int X_SIZE, int Z_SIZE>
using hip_flatten_thread_size_xz_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_x<X_SIZE>,
                                         hip::thread_z<Z_SIZE>>;
template<int Y_SIZE, int X_SIZE>
using hip_flatten_thread_size_yx_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_y<Y_SIZE>,
                                         hip::thread_x<X_SIZE>>;
template<int Y_SIZE, int Z_SIZE>
using hip_flatten_thread_size_yz_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_y<Y_SIZE>,
                                         hip::thread_z<Z_SIZE>>;
template<int Z_SIZE, int X_SIZE>
using hip_flatten_thread_size_zx_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_z<Z_SIZE>,
                                         hip::thread_x<X_SIZE>>;
template<int Z_SIZE, int Y_SIZE>
using hip_flatten_thread_size_zy_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_z<Z_SIZE>,
                                         hip::thread_y<Y_SIZE>>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using hip_flatten_thread_size_xyz_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_x<X_SIZE>,
                                         hip::thread_y<Y_SIZE>,
                                         hip::thread_z<Z_SIZE>>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using hip_flatten_thread_size_xzy_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_x<X_SIZE>,
                                         hip::thread_z<Z_SIZE>,
                                         hip::thread_y<Y_SIZE>>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using hip_flatten_thread_size_yxz_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_y<Y_SIZE>,
                                         hip::thread_x<X_SIZE>,
                                         hip::thread_z<Z_SIZE>>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using hip_flatten_thread_size_yzx_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_y<Y_SIZE>,
                                         hip::thread_z<Z_SIZE>,
                                         hip::thread_x<X_SIZE>>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using hip_flatten_thread_size_zxy_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_z<Z_SIZE>,
                                         hip::thread_x<X_SIZE>,
                                         hip::thread_y<Y_SIZE>>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using hip_flatten_thread_size_zyx_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::thread_z<Z_SIZE>,
                                         hip::thread_y<Y_SIZE>,
                                         hip::thread_x<X_SIZE>>;

template<int X_SIZE>
using hip_flatten_block_size_x_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_x<X_SIZE>>;
template<int Y_SIZE>
using hip_flatten_block_size_y_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_y<Y_SIZE>>;
template<int Z_SIZE>
using hip_flatten_block_size_z_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_z<Z_SIZE>>;
template<int X_SIZE, int Y_SIZE>
using hip_flatten_block_size_xy_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_x<X_SIZE>,
                                         hip::block_y<Y_SIZE>>;
template<int X_SIZE, int Z_SIZE>
using hip_flatten_block_size_xz_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_x<X_SIZE>,
                                         hip::block_z<Z_SIZE>>;
template<int Y_SIZE, int X_SIZE>
using hip_flatten_block_size_yx_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_y<Y_SIZE>,
                                         hip::block_x<X_SIZE>>;
template<int Y_SIZE, int Z_SIZE>
using hip_flatten_block_size_yz_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_y<Y_SIZE>,
                                         hip::block_z<Z_SIZE>>;
template<int Z_SIZE, int X_SIZE>
using hip_flatten_block_size_zx_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_z<Z_SIZE>,
                                         hip::block_x<X_SIZE>>;
template<int Z_SIZE, int Y_SIZE>
using hip_flatten_block_size_zy_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_z<Z_SIZE>,
                                         hip::block_y<Y_SIZE>>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using hip_flatten_block_size_xyz_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_x<X_SIZE>,
                                         hip::block_y<Y_SIZE>,
                                         hip::block_z<Z_SIZE>>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using hip_flatten_block_size_xzy_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_x<X_SIZE>,
                                         hip::block_z<Z_SIZE>,
                                         hip::block_y<Y_SIZE>>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using hip_flatten_block_size_yxz_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_y<Y_SIZE>,
                                         hip::block_x<X_SIZE>,
                                         hip::block_z<Z_SIZE>>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using hip_flatten_block_size_yzx_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_y<Y_SIZE>,
                                         hip::block_z<Z_SIZE>,
                                         hip::block_x<X_SIZE>>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using hip_flatten_block_size_zxy_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_z<Z_SIZE>,
                                         hip::block_x<X_SIZE>,
                                         hip::block_y<Y_SIZE>>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using hip_flatten_block_size_zyx_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<hip::block_z<Z_SIZE>,
                                         hip::block_y<Y_SIZE>,
                                         hip::block_x<X_SIZE>>;

template<int X_BLOCK_SIZE, int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_x_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Y_BLOCK_SIZE, int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_y_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Z_BLOCK_SIZE, int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_z_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_xy_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
        hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_xz_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
        hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_yx_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
        hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_yz_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
        hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_zx_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
        hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_zy_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
        hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_xyz_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
        hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
        hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_xzy_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
        hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
        hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_yxz_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
        hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
        hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_yzx_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
        hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
        hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_zxy_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
        hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
        hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_zyx_direct_unchecked =
    hip_flatten_indexer_direct_unchecked<
        hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
        hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
        hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;

/*
 * Maps segment indices to flattened HIP threads, blocks, or global threads.
 * This is a low overhead mapping, but requires that there are enough
 * physical threads, blocks, or global threads to fit all of the direct map
 * requests.
 * Reshapes multiple physical threads, blocks, or global threads into a 1D
 * iteration space.
 */
template<int X_SIZE>
using hip_flatten_thread_size_x_direct =
    hip_flatten_indexer_direct<hip::thread_x<X_SIZE>>;
template<int Y_SIZE>
using hip_flatten_thread_size_y_direct =
    hip_flatten_indexer_direct<hip::thread_y<Y_SIZE>>;
template<int Z_SIZE>
using hip_flatten_thread_size_z_direct =
    hip_flatten_indexer_direct<hip::thread_z<Z_SIZE>>;
template<int X_SIZE, int Y_SIZE>
using hip_flatten_thread_size_xy_direct =
    hip_flatten_indexer_direct<hip::thread_x<X_SIZE>, hip::thread_y<Y_SIZE>>;
template<int X_SIZE, int Z_SIZE>
using hip_flatten_thread_size_xz_direct =
    hip_flatten_indexer_direct<hip::thread_x<X_SIZE>, hip::thread_z<Z_SIZE>>;
template<int Y_SIZE, int X_SIZE>
using hip_flatten_thread_size_yx_direct =
    hip_flatten_indexer_direct<hip::thread_y<Y_SIZE>, hip::thread_x<X_SIZE>>;
template<int Y_SIZE, int Z_SIZE>
using hip_flatten_thread_size_yz_direct =
    hip_flatten_indexer_direct<hip::thread_y<Y_SIZE>, hip::thread_z<Z_SIZE>>;
template<int Z_SIZE, int X_SIZE>
using hip_flatten_thread_size_zx_direct =
    hip_flatten_indexer_direct<hip::thread_z<Z_SIZE>, hip::thread_x<X_SIZE>>;
template<int Z_SIZE, int Y_SIZE>
using hip_flatten_thread_size_zy_direct =
    hip_flatten_indexer_direct<hip::thread_z<Z_SIZE>, hip::thread_y<Y_SIZE>>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using hip_flatten_thread_size_xyz_direct =
    hip_flatten_indexer_direct<hip::thread_x<X_SIZE>,
                               hip::thread_y<Y_SIZE>,
                               hip::thread_z<Z_SIZE>>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using hip_flatten_thread_size_xzy_direct =
    hip_flatten_indexer_direct<hip::thread_x<X_SIZE>,
                               hip::thread_z<Z_SIZE>,
                               hip::thread_y<Y_SIZE>>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using hip_flatten_thread_size_yxz_direct =
    hip_flatten_indexer_direct<hip::thread_y<Y_SIZE>,
                               hip::thread_x<X_SIZE>,
                               hip::thread_z<Z_SIZE>>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using hip_flatten_thread_size_yzx_direct =
    hip_flatten_indexer_direct<hip::thread_y<Y_SIZE>,
                               hip::thread_z<Z_SIZE>,
                               hip::thread_x<X_SIZE>>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using hip_flatten_thread_size_zxy_direct =
    hip_flatten_indexer_direct<hip::thread_z<Z_SIZE>,
                               hip::thread_x<X_SIZE>,
                               hip::thread_y<Y_SIZE>>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using hip_flatten_thread_size_zyx_direct =
    hip_flatten_indexer_direct<hip::thread_z<Z_SIZE>,
                               hip::thread_y<Y_SIZE>,
                               hip::thread_x<X_SIZE>>;

template<int X_SIZE>
using hip_flatten_block_size_x_direct =
    hip_flatten_indexer_direct<hip::block_x<X_SIZE>>;
template<int Y_SIZE>
using hip_flatten_block_size_y_direct =
    hip_flatten_indexer_direct<hip::block_y<Y_SIZE>>;
template<int Z_SIZE>
using hip_flatten_block_size_z_direct =
    hip_flatten_indexer_direct<hip::block_z<Z_SIZE>>;
template<int X_SIZE, int Y_SIZE>
using hip_flatten_block_size_xy_direct =
    hip_flatten_indexer_direct<hip::block_x<X_SIZE>, hip::block_y<Y_SIZE>>;
template<int X_SIZE, int Z_SIZE>
using hip_flatten_block_size_xz_direct =
    hip_flatten_indexer_direct<hip::block_x<X_SIZE>, hip::block_z<Z_SIZE>>;
template<int Y_SIZE, int X_SIZE>
using hip_flatten_block_size_yx_direct =
    hip_flatten_indexer_direct<hip::block_y<Y_SIZE>, hip::block_x<X_SIZE>>;
template<int Y_SIZE, int Z_SIZE>
using hip_flatten_block_size_yz_direct =
    hip_flatten_indexer_direct<hip::block_y<Y_SIZE>, hip::block_z<Z_SIZE>>;
template<int Z_SIZE, int X_SIZE>
using hip_flatten_block_size_zx_direct =
    hip_flatten_indexer_direct<hip::block_z<Z_SIZE>, hip::block_x<X_SIZE>>;
template<int Z_SIZE, int Y_SIZE>
using hip_flatten_block_size_zy_direct =
    hip_flatten_indexer_direct<hip::block_z<Z_SIZE>, hip::block_y<Y_SIZE>>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using hip_flatten_block_size_xyz_direct =
    hip_flatten_indexer_direct<hip::block_x<X_SIZE>,
                               hip::block_y<Y_SIZE>,
                               hip::block_z<Z_SIZE>>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using hip_flatten_block_size_xzy_direct =
    hip_flatten_indexer_direct<hip::block_x<X_SIZE>,
                               hip::block_z<Z_SIZE>,
                               hip::block_y<Y_SIZE>>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using hip_flatten_block_size_yxz_direct =
    hip_flatten_indexer_direct<hip::block_y<Y_SIZE>,
                               hip::block_x<X_SIZE>,
                               hip::block_z<Z_SIZE>>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using hip_flatten_block_size_yzx_direct =
    hip_flatten_indexer_direct<hip::block_y<Y_SIZE>,
                               hip::block_z<Z_SIZE>,
                               hip::block_x<X_SIZE>>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using hip_flatten_block_size_zxy_direct =
    hip_flatten_indexer_direct<hip::block_z<Z_SIZE>,
                               hip::block_x<X_SIZE>,
                               hip::block_y<Y_SIZE>>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using hip_flatten_block_size_zyx_direct =
    hip_flatten_indexer_direct<hip::block_z<Z_SIZE>,
                               hip::block_y<Y_SIZE>,
                               hip::block_x<X_SIZE>>;

template<int X_BLOCK_SIZE, int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_x_direct =
    hip_flatten_indexer_direct<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Y_BLOCK_SIZE, int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_y_direct =
    hip_flatten_indexer_direct<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Z_BLOCK_SIZE, int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_z_direct =
    hip_flatten_indexer_direct<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_xy_direct =
    hip_flatten_indexer_direct<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                               hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_xz_direct =
    hip_flatten_indexer_direct<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                               hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_yx_direct =
    hip_flatten_indexer_direct<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                               hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_yz_direct =
    hip_flatten_indexer_direct<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                               hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_zx_direct =
    hip_flatten_indexer_direct<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                               hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_zy_direct =
    hip_flatten_indexer_direct<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                               hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_xyz_direct =
    hip_flatten_indexer_direct<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                               hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                               hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_xzy_direct =
    hip_flatten_indexer_direct<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                               hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                               hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_yxz_direct =
    hip_flatten_indexer_direct<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                               hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                               hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_yzx_direct =
    hip_flatten_indexer_direct<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                               hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                               hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_zxy_direct =
    hip_flatten_indexer_direct<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                               hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                               hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_zyx_direct =
    hip_flatten_indexer_direct<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                               hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                               hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;

/*
 * Maps segment indices to flattened HIP threads, blocks, or global threads.
 * Reshapes multiple physical threads, blocks, or global threads into a 1D
 * iteration space.
 * Uses block-stride or grid-stride looping to exceed the maximum number of
 * physical threads, blocks, or global threads.
 */
template<int X_SIZE>
using hip_flatten_thread_size_x_loop =
    hip_flatten_indexer_loop<hip::thread_x<X_SIZE>>;
template<int Y_SIZE>
using hip_flatten_thread_size_y_loop =
    hip_flatten_indexer_loop<hip::thread_y<Y_SIZE>>;
template<int Z_SIZE>
using hip_flatten_thread_size_z_loop =
    hip_flatten_indexer_loop<hip::thread_z<Z_SIZE>>;
template<int X_SIZE, int Y_SIZE>
using hip_flatten_thread_size_xy_loop =
    hip_flatten_indexer_loop<hip::thread_x<X_SIZE>, hip::thread_y<Y_SIZE>>;
template<int X_SIZE, int Z_SIZE>
using hip_flatten_thread_size_xz_loop =
    hip_flatten_indexer_loop<hip::thread_x<X_SIZE>, hip::thread_z<Z_SIZE>>;
template<int Y_SIZE, int X_SIZE>
using hip_flatten_thread_size_yx_loop =
    hip_flatten_indexer_loop<hip::thread_y<Y_SIZE>, hip::thread_x<X_SIZE>>;
template<int Y_SIZE, int Z_SIZE>
using hip_flatten_thread_size_yz_loop =
    hip_flatten_indexer_loop<hip::thread_y<Y_SIZE>, hip::thread_z<Z_SIZE>>;
template<int Z_SIZE, int X_SIZE>
using hip_flatten_thread_size_zx_loop =
    hip_flatten_indexer_loop<hip::thread_z<Z_SIZE>, hip::thread_x<X_SIZE>>;
template<int Z_SIZE, int Y_SIZE>
using hip_flatten_thread_size_zy_loop =
    hip_flatten_indexer_loop<hip::thread_z<Z_SIZE>, hip::thread_y<Y_SIZE>>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using hip_flatten_thread_size_xyz_loop =
    hip_flatten_indexer_loop<hip::thread_x<X_SIZE>,
                             hip::thread_y<Y_SIZE>,
                             hip::thread_z<Z_SIZE>>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using hip_flatten_thread_size_xzy_loop =
    hip_flatten_indexer_loop<hip::thread_x<X_SIZE>,
                             hip::thread_z<Z_SIZE>,
                             hip::thread_y<Y_SIZE>>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using hip_flatten_thread_size_yxz_loop =
    hip_flatten_indexer_loop<hip::thread_y<Y_SIZE>,
                             hip::thread_x<X_SIZE>,
                             hip::thread_z<Z_SIZE>>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using hip_flatten_thread_size_yzx_loop =
    hip_flatten_indexer_loop<hip::thread_y<Y_SIZE>,
                             hip::thread_z<Z_SIZE>,
                             hip::thread_x<X_SIZE>>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using hip_flatten_thread_size_zxy_loop =
    hip_flatten_indexer_loop<hip::thread_z<Z_SIZE>,
                             hip::thread_x<X_SIZE>,
                             hip::thread_y<Y_SIZE>>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using hip_flatten_thread_size_zyx_loop =
    hip_flatten_indexer_loop<hip::thread_z<Z_SIZE>,
                             hip::thread_y<Y_SIZE>,
                             hip::thread_x<X_SIZE>>;

template<int X_SIZE>
using hip_flatten_block_size_x_loop =
    hip_flatten_indexer_loop<hip::block_x<X_SIZE>>;
template<int Y_SIZE>
using hip_flatten_block_size_y_loop =
    hip_flatten_indexer_loop<hip::block_y<Y_SIZE>>;
template<int Z_SIZE>
using hip_flatten_block_size_z_loop =
    hip_flatten_indexer_loop<hip::block_z<Z_SIZE>>;
template<int X_SIZE, int Y_SIZE>
using hip_flatten_block_size_xy_loop =
    hip_flatten_indexer_loop<hip::block_x<X_SIZE>, hip::block_y<Y_SIZE>>;
template<int X_SIZE, int Z_SIZE>
using hip_flatten_block_size_xz_loop =
    hip_flatten_indexer_loop<hip::block_x<X_SIZE>, hip::block_z<Z_SIZE>>;
template<int Y_SIZE, int X_SIZE>
using hip_flatten_block_size_yx_loop =
    hip_flatten_indexer_loop<hip::block_y<Y_SIZE>, hip::block_x<X_SIZE>>;
template<int Y_SIZE, int Z_SIZE>
using hip_flatten_block_size_yz_loop =
    hip_flatten_indexer_loop<hip::block_y<Y_SIZE>, hip::block_z<Z_SIZE>>;
template<int Z_SIZE, int X_SIZE>
using hip_flatten_block_size_zx_loop =
    hip_flatten_indexer_loop<hip::block_z<Z_SIZE>, hip::block_x<X_SIZE>>;
template<int Z_SIZE, int Y_SIZE>
using hip_flatten_block_size_zy_loop =
    hip_flatten_indexer_loop<hip::block_z<Z_SIZE>, hip::block_y<Y_SIZE>>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using hip_flatten_block_size_xyz_loop =
    hip_flatten_indexer_loop<hip::block_x<X_SIZE>,
                             hip::block_y<Y_SIZE>,
                             hip::block_z<Z_SIZE>>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using hip_flatten_block_size_xzy_loop =
    hip_flatten_indexer_loop<hip::block_x<X_SIZE>,
                             hip::block_z<Z_SIZE>,
                             hip::block_y<Y_SIZE>>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using hip_flatten_block_size_yxz_loop =
    hip_flatten_indexer_loop<hip::block_y<Y_SIZE>,
                             hip::block_x<X_SIZE>,
                             hip::block_z<Z_SIZE>>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using hip_flatten_block_size_yzx_loop =
    hip_flatten_indexer_loop<hip::block_y<Y_SIZE>,
                             hip::block_z<Z_SIZE>,
                             hip::block_x<X_SIZE>>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using hip_flatten_block_size_zxy_loop =
    hip_flatten_indexer_loop<hip::block_z<Z_SIZE>,
                             hip::block_x<X_SIZE>,
                             hip::block_y<Y_SIZE>>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using hip_flatten_block_size_zyx_loop =
    hip_flatten_indexer_loop<hip::block_z<Z_SIZE>,
                             hip::block_y<Y_SIZE>,
                             hip::block_x<X_SIZE>>;

template<int X_BLOCK_SIZE, int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_x_loop =
    hip_flatten_indexer_loop<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Y_BLOCK_SIZE, int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_y_loop =
    hip_flatten_indexer_loop<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Z_BLOCK_SIZE, int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_z_loop =
    hip_flatten_indexer_loop<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_xy_loop =
    hip_flatten_indexer_loop<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                             hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_xz_loop =
    hip_flatten_indexer_loop<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                             hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_yx_loop =
    hip_flatten_indexer_loop<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                             hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_yz_loop =
    hip_flatten_indexer_loop<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                             hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_zx_loop =
    hip_flatten_indexer_loop<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                             hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_zy_loop =
    hip_flatten_indexer_loop<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                             hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_xyz_loop =
    hip_flatten_indexer_loop<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                             hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                             hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_xzy_loop =
    hip_flatten_indexer_loop<hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                             hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                             hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_yxz_loop =
    hip_flatten_indexer_loop<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                             hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                             hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = named_usage::unspecified,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_yzx_loop =
    hip_flatten_indexer_loop<hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                             hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                             hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_zxy_loop =
    hip_flatten_indexer_loop<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                             hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                             hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = named_usage::unspecified,
         int Y_GRID_SIZE = named_usage::unspecified,
         int X_GRID_SIZE = named_usage::unspecified>
using hip_flatten_global_size_zyx_loop =
    hip_flatten_indexer_loop<hip::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                             hip::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                             hip::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;


/*
 * Deprecated policies
 */
using hip_global_thread_x = hip_global_x_direct;
using hip_global_thread_y = hip_global_y_direct;
using hip_global_thread_z = hip_global_z_direct;

using hip_global_thread_xy = hip_global_xy_direct;
using hip_global_thread_xz = hip_global_xz_direct;
using hip_global_thread_yx = hip_global_yx_direct;
using hip_global_thread_yz = hip_global_yz_direct;
using hip_global_thread_zx = hip_global_zx_direct;
using hip_global_thread_zy = hip_global_zy_direct;

using hip_global_thread_xyz = hip_global_xyz_direct;
using hip_global_thread_xzy = hip_global_xzy_direct;
using hip_global_thread_yxz = hip_global_yxz_direct;
using hip_global_thread_yzx = hip_global_yzx_direct;
using hip_global_thread_zxy = hip_global_zxy_direct;
using hip_global_thread_zyx = hip_global_zyx_direct;

using hip_flatten_block_threads_xy_direct = hip_flatten_thread_xy_direct;
using hip_flatten_block_threads_xz_direct = hip_flatten_thread_xz_direct;
using hip_flatten_block_threads_yx_direct = hip_flatten_thread_yx_direct;
using hip_flatten_block_threads_yz_direct = hip_flatten_thread_yz_direct;
using hip_flatten_block_threads_zx_direct = hip_flatten_thread_zx_direct;
using hip_flatten_block_threads_zy_direct = hip_flatten_thread_zy_direct;

using hip_flatten_block_threads_xyz_direct = hip_flatten_thread_xyz_direct;
using hip_flatten_block_threads_xzy_direct = hip_flatten_thread_xzy_direct;
using hip_flatten_block_threads_yxz_direct = hip_flatten_thread_yxz_direct;
using hip_flatten_block_threads_yzx_direct = hip_flatten_thread_yzx_direct;
using hip_flatten_block_threads_zxy_direct = hip_flatten_thread_zxy_direct;
using hip_flatten_block_threads_zyx_direct = hip_flatten_thread_zyx_direct;

using hip_flatten_block_threads_xy_loop = hip_flatten_thread_xy_loop;
using hip_flatten_block_threads_xz_loop = hip_flatten_thread_xz_loop;
using hip_flatten_block_threads_yx_loop = hip_flatten_thread_yx_loop;
using hip_flatten_block_threads_yz_loop = hip_flatten_thread_yz_loop;
using hip_flatten_block_threads_zx_loop = hip_flatten_thread_zx_loop;
using hip_flatten_block_threads_zy_loop = hip_flatten_thread_zy_loop;

using hip_flatten_block_threads_xyz_loop = hip_flatten_thread_xyz_loop;
using hip_flatten_block_threads_xzy_loop = hip_flatten_thread_xzy_loop;
using hip_flatten_block_threads_yxz_loop = hip_flatten_thread_yxz_loop;
using hip_flatten_block_threads_yzx_loop = hip_flatten_thread_yzx_loop;
using hip_flatten_block_threads_zxy_loop = hip_flatten_thread_zxy_loop;
using hip_flatten_block_threads_zyx_loop = hip_flatten_thread_zyx_loop;

using hip_block_xy_nested_direct = hip_block_xy_direct;
using hip_block_xz_nested_direct = hip_block_xz_direct;
using hip_block_yx_nested_direct = hip_block_yx_direct;
using hip_block_yz_nested_direct = hip_block_yz_direct;
using hip_block_zx_nested_direct = hip_block_zx_direct;
using hip_block_zy_nested_direct = hip_block_zy_direct;

using hip_block_xyz_nested_direct = hip_block_xyz_direct;
using hip_block_xzy_nested_direct = hip_block_xzy_direct;
using hip_block_yxz_nested_direct = hip_block_yxz_direct;
using hip_block_yzx_nested_direct = hip_block_yzx_direct;
using hip_block_zxy_nested_direct = hip_block_zxy_direct;
using hip_block_zyx_nested_direct = hip_block_zyx_direct;

using hip_block_xy_nested_loop = hip_block_xy_loop;
using hip_block_xz_nested_loop = hip_block_xz_loop;
using hip_block_yx_nested_loop = hip_block_yx_loop;
using hip_block_yz_nested_loop = hip_block_yz_loop;
using hip_block_zx_nested_loop = hip_block_zx_loop;
using hip_block_zy_nested_loop = hip_block_zy_loop;

using hip_block_xyz_nested_loop = hip_block_xyz_loop;
using hip_block_xzy_nested_loop = hip_block_xzy_loop;
using hip_block_yxz_nested_loop = hip_block_yxz_loop;
using hip_block_yzx_nested_loop = hip_block_yzx_loop;
using hip_block_zxy_nested_loop = hip_block_zxy_loop;
using hip_block_zyx_nested_loop = hip_block_zyx_loop;

}  // namespace RAJA

#endif  // RAJA_ENABLE_HIP
#endif
