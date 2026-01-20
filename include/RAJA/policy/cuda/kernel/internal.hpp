/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *          traversals on GPU with CUDA.
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


#ifndef RAJA_policy_cuda_kernel_internal_HPP
#define RAJA_policy_cuda_kernel_internal_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/pattern/kernel.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"

namespace RAJA
{

namespace internal
{

struct LaunchDims
{

  CudaDims active {0};
  CudaDims dims {0};
  CudaDims min_dims {0};

  LaunchDims()                             = default;
  LaunchDims(LaunchDims const&)            = default;
  LaunchDims(LaunchDims&&)                 = default;
  LaunchDims& operator=(LaunchDims const&) = default;
  LaunchDims& operator=(LaunchDims&&)      = default;

  RAJA_INLINE
  LaunchDims(CudaDims _active, CudaDims _dims, CudaDims _min_dims)
      : active {_active},
        dims {_dims},
        min_dims {_min_dims}
  {}

  RAJA_INLINE
  LaunchDims max(LaunchDims const& c) const
  {
    LaunchDims result;

    result.active.blocks.x = std::max(c.active.blocks.x, active.blocks.x);
    result.active.blocks.y = std::max(c.active.blocks.y, active.blocks.y);
    result.active.blocks.z = std::max(c.active.blocks.z, active.blocks.z);

    result.dims.blocks.x = std::max(c.dims.blocks.x, dims.blocks.x);
    result.dims.blocks.y = std::max(c.dims.blocks.y, dims.blocks.y);
    result.dims.blocks.z = std::max(c.dims.blocks.z, dims.blocks.z);

    result.min_dims.blocks.x = std::max(c.min_dims.blocks.x, min_dims.blocks.x);
    result.min_dims.blocks.y = std::max(c.min_dims.blocks.y, min_dims.blocks.y);
    result.min_dims.blocks.z = std::max(c.min_dims.blocks.z, min_dims.blocks.z);

    result.active.threads.x = std::max(c.active.threads.x, active.threads.x);
    result.active.threads.y = std::max(c.active.threads.y, active.threads.y);
    result.active.threads.z = std::max(c.active.threads.z, active.threads.z);

    result.dims.threads.x = std::max(c.dims.threads.x, dims.threads.x);
    result.dims.threads.y = std::max(c.dims.threads.y, dims.threads.y);
    result.dims.threads.z = std::max(c.dims.threads.z, dims.threads.z);

    result.min_dims.threads.x =
        std::max(c.min_dims.threads.x, min_dims.threads.x);
    result.min_dims.threads.y =
        std::max(c.min_dims.threads.y, min_dims.threads.y);
    result.min_dims.threads.z =
        std::max(c.min_dims.threads.z, min_dims.threads.z);

    return result;
  }

  RAJA_INLINE
  int blocks_are_active() const
  {
    return active.blocks.x || active.blocks.y || active.blocks.z;
  }

  RAJA_INLINE
  int threads_are_active() const
  {
    return active.threads.x || active.threads.y || active.threads.z;
  }

  RAJA_INLINE
  int num_blocks() const
  {
    if (blocks_are_active())
    {
      return (active.blocks.x ? dims.blocks.x : 1) *
             (active.blocks.y ? dims.blocks.y : 1) *
             (active.blocks.z ? dims.blocks.z : 1);
    }
    else
    {
      return 0;
    }
  }

  RAJA_INLINE
  int num_threads() const
  {
    if (threads_are_active())
    {
      return (active.threads.x ? dims.threads.x : 1) *
             (active.threads.y ? dims.threads.y : 1) *
             (active.threads.z ? dims.threads.z : 1);
    }
    else
    {
      return 0;
    }
  }

  RAJA_INLINE
  void clamp_to_min_blocks()
  {
    dims.blocks.x = std::max(min_dims.blocks.x, dims.blocks.x);
    dims.blocks.y = std::max(min_dims.blocks.y, dims.blocks.y);
    dims.blocks.z = std::max(min_dims.blocks.z, dims.blocks.z);
  };

  RAJA_INLINE
  void clamp_to_min_threads()
  {
    dims.threads.x = std::max(min_dims.threads.x, dims.threads.x);
    dims.threads.y = std::max(min_dims.threads.y, dims.threads.y);
    dims.threads.z = std::max(min_dims.threads.z, dims.threads.z);
  };
};

RAJA_INLINE
LaunchDims combine(LaunchDims const& lhs, LaunchDims const& rhs)
{
  return lhs.max(rhs);
}

template<camp::idx_t cur_stmt, camp::idx_t num_stmts, typename StmtList>
struct CudaStatementListExecutorHelper
{

  using next_helper_t =
      CudaStatementListExecutorHelper<cur_stmt + 1, num_stmts, StmtList>;

  using cur_stmt_t = camp::at_v<StmtList, cur_stmt>;

  template<typename Data>
  inline static RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // Execute stmt
    cur_stmt_t::exec(data, thread_active);

    // Execute next stmt
    next_helper_t::exec(data, thread_active);
  }

  template<typename Data>
  inline static LaunchDims calculateDimensions(Data& data)
  {
    LaunchDims statement_dims = cur_stmt_t::calculateDimensions(data);

    LaunchDims next_dims = next_helper_t::calculateDimensions(data);

    return combine(statement_dims, next_dims);
  }
};

template<camp::idx_t num_stmts, typename StmtList>
struct CudaStatementListExecutorHelper<num_stmts, num_stmts, StmtList>
{

  template<typename Data>
  inline static RAJA_DEVICE void exec(Data&, bool)
  {
    // nop terminator
  }

  template<typename Data>
  inline static LaunchDims calculateDimensions(Data&)
  {
    return LaunchDims();
  }
};


template<typename Data, typename Policy, typename Types>
struct CudaStatementExecutor;

template<typename Data, typename StmtList, typename Types>
struct CudaStatementListExecutor;

template<typename Data, typename... Stmts, typename Types>
struct CudaStatementListExecutor<Data, StatementList<Stmts...>, Types>
{

  using enclosed_stmts_t =
      camp::list<CudaStatementExecutor<Data, Stmts, Types>...>;

  static constexpr size_t num_stmts = sizeof...(Stmts);

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // Execute statements in order with helper class
    CudaStatementListExecutorHelper<0, num_stmts, enclosed_stmts_t>::exec(
        data, thread_active);
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {
    // Compute this statements launch dimensions
    return CudaStatementListExecutorHelper<
        0, num_stmts, enclosed_stmts_t>::calculateDimensions(data);
  }
};

template<typename StmtList, typename Data, typename Types>
using cuda_statement_list_executor_t =
    CudaStatementListExecutor<Data, StmtList, Types>;


template<typename kernel_indexer>
struct KernelDimensionCalculator;

// specialization for direct unchecked sequential policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::DirectUnchecked,
    sync,
    cuda::IndexGlobal<dim, named_usage::ignored, named_usage::ignored>>>
{
  using IndexMapper =
      cuda::IndexGlobal<dim, named_usage::ignored, named_usage::ignored>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    if (len != static_cast<IdxT>(1))
    {
      RAJA_ABORT_OR_THROW("len does not match the size of the direct_unchecked "
                          "mapped index space");
    }

    return LaunchDims {};
  }
};

// specialization for direct unchecked thread policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::DirectUnchecked,
    sync,
    cuda::IndexGlobal<dim, named_usage::unspecified, named_usage::ignored>>>
{
  using IndexMapper =
      cuda::IndexGlobal<dim, named_usage::unspecified, named_usage::ignored>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    LaunchDims dims;

    // BEWARE: if calculated block_size is too high then the kernel launch will
    // fail
    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads, static_cast<cuda_dim_member_t>(len));
    set_cuda_dim<dim>(dims.min_dims.threads,
                      static_cast<cuda_dim_member_t>(len));

    return dims;
  }
};

///
template<named_dim dim, int BLOCK_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::DirectUnchecked,
    sync,
    cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>>>
{
  static_assert(BLOCK_SIZE > 0,
                "block size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper = cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    if (len != static_cast<IdxT>(IndexMapper::block_size))
    {
      RAJA_ABORT_OR_THROW("len does not match the size of the direct_unchecked "
                          "mapped index space");
    }

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads,
                      static_cast<cuda_dim_member_t>(IndexMapper::block_size));
    set_cuda_dim<dim>(dims.min_dims.threads,
                      static_cast<cuda_dim_member_t>(IndexMapper::block_size));

    return dims;
  }
};

// specialization for direct unchecked block policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::DirectUnchecked,
    sync,
    cuda::IndexGlobal<dim, named_usage::ignored, named_usage::unspecified>>>
{
  using IndexMapper =
      cuda::IndexGlobal<dim, named_usage::ignored, named_usage::unspecified>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks, static_cast<cuda_dim_member_t>(len));
    set_cuda_dim<dim>(dims.min_dims.blocks,
                      static_cast<cuda_dim_member_t>(len));

    return dims;
  }
};

///
template<named_dim dim, int GRID_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::DirectUnchecked,
    sync,
    cuda::IndexGlobal<dim, named_usage::ignored, GRID_SIZE>>>
{
  static_assert(GRID_SIZE > 0,
                "grid size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper = cuda::IndexGlobal<dim, named_usage::ignored, GRID_SIZE>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    if (len != static_cast<IdxT>(IndexMapper::grid_size))
    {
      RAJA_ABORT_OR_THROW("len does not match the size of the direct_unchecked "
                          "mapped index space");
    }

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks,
                      static_cast<cuda_dim_member_t>(IndexMapper::grid_size));
    set_cuda_dim<dim>(dims.min_dims.blocks,
                      static_cast<cuda_dim_member_t>(IndexMapper::grid_size));

    return dims;
  }
};

// specialization for direct unchecked global policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::DirectUnchecked,
    sync,
    cuda::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>>>
{
  using IndexMapper = cuda::
      IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    if (len != static_cast<IdxT>(0))
    {
      RAJA_ABORT_OR_THROW("must know one of block_size or grid_size");
    }

    return {};
  }
};

///
template<named_dim dim, int GRID_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::DirectUnchecked,
    sync,
    cuda::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>>>
{
  static_assert(GRID_SIZE > 0,
                "grid size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper =
      cuda::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    // BEWARE: if calculated block_size is too high then the kernel launch will
    // fail
    const IdxT block_size =
        RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexMapper::grid_size));
    if (len != (block_size * static_cast<IdxT>(IndexMapper::grid_size)))
    {
      RAJA_ABORT_OR_THROW("len does not match the size of the direct_unchecked "
                          "mapped index space");
    }

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads,
                      static_cast<cuda_dim_member_t>(block_size));
    set_cuda_dim<dim>(dims.min_dims.threads,
                      static_cast<cuda_dim_member_t>(block_size));

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks,
                      static_cast<cuda_dim_member_t>(IndexMapper::grid_size));
    set_cuda_dim<dim>(dims.min_dims.blocks,
                      static_cast<cuda_dim_member_t>(IndexMapper::grid_size));

    return dims;
  }
};

///
template<named_dim dim, int BLOCK_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::DirectUnchecked,
    sync,
    cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>>>
{
  static_assert(BLOCK_SIZE > 0,
                "block size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper =
      cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    const IdxT grid_size = RAJA_DIVIDE_CEILING_INT(
        len, static_cast<IdxT>(IndexMapper::block_size));
    if (len != (static_cast<IdxT>(IndexMapper::block_size) * grid_size))
    {
      RAJA_ABORT_OR_THROW("len does not match the size of the direct_unchecked "
                          "mapped index space");
    }

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads,
                      static_cast<cuda_dim_member_t>(IndexMapper::block_size));
    set_cuda_dim<dim>(dims.min_dims.threads,
                      static_cast<cuda_dim_member_t>(IndexMapper::block_size));

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks,
                      static_cast<cuda_dim_member_t>(grid_size));
    set_cuda_dim<dim>(dims.min_dims.blocks,
                      static_cast<cuda_dim_member_t>(grid_size));

    return dims;
  }
};

///
template<named_dim dim,
         int BLOCK_SIZE,
         int GRID_SIZE,
         kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::DirectUnchecked,
    sync,
    cuda::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>>>
{
  static_assert(BLOCK_SIZE > 0,
                "block size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");
  static_assert(GRID_SIZE > 0,
                "grid size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper = cuda::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    if (len != (static_cast<IdxT>(IndexMapper::block_size) *
                static_cast<IdxT>(IndexMapper::grid_size)))
    {
      RAJA_ABORT_OR_THROW("len does not match the size of the direct_unchecked "
                          "mapped index space");
    }

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads,
                      static_cast<cuda_dim_member_t>(IndexMapper::block_size));
    set_cuda_dim<dim>(dims.min_dims.threads,
                      static_cast<cuda_dim_member_t>(IndexMapper::block_size));

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks,
                      static_cast<cuda_dim_member_t>(IndexMapper::grid_size));
    set_cuda_dim<dim>(dims.min_dims.blocks,
                      static_cast<cuda_dim_member_t>(IndexMapper::grid_size));

    return dims;
  }
};

// specialization for direct sequential policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::Direct,
    sync,
    cuda::IndexGlobal<dim, named_usage::ignored, named_usage::ignored>>>
{
  using IndexMapper =
      cuda::IndexGlobal<dim, named_usage::ignored, named_usage::ignored>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    if (len > static_cast<IdxT>(1))
    {
      RAJA_ABORT_OR_THROW(
          "len exceeds the size of the directly mapped index space");
    }

    return LaunchDims {};
  }
};

// specialization for direct thread policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::Direct,
    sync,
    cuda::IndexGlobal<dim, named_usage::unspecified, named_usage::ignored>>>
{
  using IndexMapper =
      cuda::IndexGlobal<dim, named_usage::unspecified, named_usage::ignored>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    LaunchDims dims;

    // BEWARE: if calculated block_size is too high then the kernel launch will
    // fail
    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads, static_cast<cuda_dim_member_t>(len));
    set_cuda_dim<dim>(dims.min_dims.threads,
                      static_cast<cuda_dim_member_t>(len));

    return dims;
  }
};

///
template<named_dim dim, int BLOCK_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::Direct,
    sync,
    cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>>>
{
  static_assert(BLOCK_SIZE > 0,
                "block size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper = cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    constexpr auto zero = static_cast<IdxT>(0);

    if (len > static_cast<IdxT>(IndexMapper::block_size))
    {
      RAJA_ABORT_OR_THROW(
          "len exceeds the size of the directly mapped index space");
    }

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads,
                      static_cast<cuda_dim_member_t>(
                          (len > zero) ? IndexMapper::block_size : 0));
    set_cuda_dim<dim>(dims.min_dims.threads,
                      static_cast<cuda_dim_member_t>(IndexMapper::block_size));

    return dims;
  }
};

// specialization for direct block policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::Direct,
    sync,
    cuda::IndexGlobal<dim, named_usage::ignored, named_usage::unspecified>>>
{
  using IndexMapper =
      cuda::IndexGlobal<dim, named_usage::ignored, named_usage::unspecified>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks, static_cast<cuda_dim_member_t>(len));
    set_cuda_dim<dim>(dims.min_dims.blocks,
                      static_cast<cuda_dim_member_t>(len));

    return dims;
  }
};

///
template<named_dim dim, int GRID_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::Direct,
    sync,
    cuda::IndexGlobal<dim, named_usage::ignored, GRID_SIZE>>>
{
  static_assert(GRID_SIZE > 0,
                "grid size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper = cuda::IndexGlobal<dim, named_usage::ignored, GRID_SIZE>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    constexpr auto zero = static_cast<IdxT>(0);

    if (len > static_cast<IdxT>(IndexMapper::grid_size))
    {
      RAJA_ABORT_OR_THROW(
          "len exceeds the size of the directly mapped index space");
    }

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks,
                      static_cast<cuda_dim_member_t>(
                          (len > zero) ? IndexMapper::grid_size : 0));
    set_cuda_dim<dim>(dims.min_dims.blocks,
                      static_cast<cuda_dim_member_t>(IndexMapper::grid_size));

    return dims;
  }
};

// specialization for direct global policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::Direct,
    sync,
    cuda::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>>>
{
  using IndexMapper = cuda::
      IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    if (len > static_cast<IdxT>(0))
    {
      RAJA_ABORT_OR_THROW("must know one of block_size or grid_size");
    }

    return LaunchDims {};
  }
};

///
template<named_dim dim, int GRID_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::Direct,
    sync,
    cuda::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>>>
{
  static_assert(GRID_SIZE > 0,
                "grid size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper =
      cuda::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    constexpr auto zero = static_cast<IdxT>(0);

    // BEWARE: if calculated block_size is too high then the kernel launch will
    // fail
    const IdxT block_size =
        RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexMapper::grid_size));

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads,
                      static_cast<cuda_dim_member_t>(block_size));
    set_cuda_dim<dim>(dims.min_dims.threads,
                      static_cast<cuda_dim_member_t>(block_size));

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks,
                      static_cast<cuda_dim_member_t>(
                          (len > zero) ? IndexMapper::grid_size : 0));
    set_cuda_dim<dim>(dims.min_dims.blocks,
                      static_cast<cuda_dim_member_t>(IndexMapper::grid_size));

    return dims;
  }
};

///
template<named_dim dim, int BLOCK_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::Direct,
    sync,
    cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>>>
{
  static_assert(BLOCK_SIZE > 0,
                "block size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper =
      cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    constexpr auto zero = static_cast<IdxT>(0);

    const IdxT grid_size = RAJA_DIVIDE_CEILING_INT(
        len, static_cast<IdxT>(IndexMapper::block_size));

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads,
                      static_cast<cuda_dim_member_t>(
                          (len > zero) ? IndexMapper::block_size : 0));
    set_cuda_dim<dim>(dims.min_dims.threads,
                      static_cast<cuda_dim_member_t>(IndexMapper::block_size));

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks,
                      static_cast<cuda_dim_member_t>(grid_size));
    set_cuda_dim<dim>(dims.min_dims.blocks,
                      static_cast<cuda_dim_member_t>(grid_size));

    return dims;
  }
};

///
template<named_dim dim,
         int BLOCK_SIZE,
         int GRID_SIZE,
         kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::Direct,
    sync,
    cuda::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>>>
{
  static_assert(BLOCK_SIZE > 0,
                "block size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");
  static_assert(GRID_SIZE > 0,
                "grid size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper = cuda::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    constexpr auto zero = static_cast<IdxT>(0);

    if (len > (static_cast<IdxT>(IndexMapper::block_size) *
               static_cast<IdxT>(IndexMapper::grid_size)))
    {
      RAJA_ABORT_OR_THROW(
          "len exceeds the size of the directly mapped index space");
    }

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads,
                      static_cast<cuda_dim_member_t>(
                          (len > zero) ? IndexMapper::block_size : 0));
    set_cuda_dim<dim>(dims.min_dims.threads,
                      static_cast<cuda_dim_member_t>(IndexMapper::block_size));

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks,
                      static_cast<cuda_dim_member_t>(
                          (len > zero) ? IndexMapper::grid_size : 0));
    set_cuda_dim<dim>(dims.min_dims.blocks,
                      static_cast<cuda_dim_member_t>(IndexMapper::grid_size));

    return dims;
  }
};

// specialization for strided loop sequential policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    sync,
    cuda::IndexGlobal<dim, named_usage::ignored, named_usage::ignored>>>
{
  using IndexMapper =
      cuda::IndexGlobal<dim, named_usage::ignored, named_usage::ignored>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT RAJA_UNUSED_ARG(len))
  {
    return LaunchDims {};
  }
};

// specialization for strided loop thread policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    sync,
    cuda::IndexGlobal<dim, named_usage::unspecified, named_usage::ignored>>>
{
  using IndexMapper =
      cuda::IndexGlobal<dim, named_usage::unspecified, named_usage::ignored>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    LaunchDims dims;

    // BEWARE: if calculated block_size is too high then the kernel launch will
    // fail
    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads, static_cast<cuda_dim_member_t>(len));
    set_cuda_dim<dim>(dims.min_dims.threads, static_cast<cuda_dim_member_t>(1));

    return dims;
  }
};

///
template<named_dim dim, int BLOCK_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    sync,
    cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>>>
{
  static_assert(BLOCK_SIZE > 0,
                "block size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper = cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    constexpr auto zero = static_cast<IdxT>(0);

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads,
                      static_cast<cuda_dim_member_t>(
                          (len > zero) ? IndexMapper::block_size : 0));
    set_cuda_dim<dim>(dims.min_dims.threads,
                      static_cast<cuda_dim_member_t>(IndexMapper::block_size));

    return dims;
  }
};

// specialization for strided loop block policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    sync,
    cuda::IndexGlobal<dim, named_usage::ignored, named_usage::unspecified>>>
{
  using IndexMapper =
      cuda::IndexGlobal<dim, named_usage::ignored, named_usage::unspecified>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks, static_cast<cuda_dim_member_t>(len));
    set_cuda_dim<dim>(dims.min_dims.blocks, static_cast<cuda_dim_member_t>(1));

    return dims;
  }
};

///
template<named_dim dim, int GRID_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    sync,
    cuda::IndexGlobal<dim, named_usage::ignored, GRID_SIZE>>>
{
  static_assert(GRID_SIZE > 0,
                "grid size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper = cuda::IndexGlobal<dim, named_usage::ignored, GRID_SIZE>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    constexpr auto zero = static_cast<IdxT>(0);

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks,
                      static_cast<cuda_dim_member_t>(
                          (len > zero) ? IndexMapper::grid_size : 0));
    set_cuda_dim<dim>(dims.min_dims.blocks,
                      static_cast<cuda_dim_member_t>(IndexMapper::grid_size));

    return dims;
  }
};

// specialization for strided loop global policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    sync,
    cuda::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>>>
{
  using IndexMapper = cuda::
      IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    constexpr auto zero = static_cast<IdxT>(0);

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads,
                      static_cast<cuda_dim_member_t>((len > zero) ? 1 : 0));
    set_cuda_dim<dim>(dims.min_dims.threads, static_cast<cuda_dim_member_t>(1));

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks,
                      static_cast<cuda_dim_member_t>((len > zero) ? 1 : 0));
    set_cuda_dim<dim>(dims.min_dims.blocks, static_cast<cuda_dim_member_t>(1));

    return dims;
  }
};

///
template<named_dim dim, int GRID_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    sync,
    cuda::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>>>
{
  static_assert(GRID_SIZE > 0,
                "grid size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper =
      cuda::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    constexpr auto zero = static_cast<IdxT>(0);

    // BEWARE: if calculated block_size is too high then the kernel launch will
    // fail
    const IdxT block_size =
        RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexMapper::grid_size));

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads,
                      static_cast<cuda_dim_member_t>(block_size));
    set_cuda_dim<dim>(dims.min_dims.threads, static_cast<cuda_dim_member_t>(1));

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks,
                      static_cast<cuda_dim_member_t>(
                          (len > zero) ? IndexMapper::grid_size : 0));
    set_cuda_dim<dim>(dims.min_dims.blocks,
                      static_cast<cuda_dim_member_t>(IndexMapper::grid_size));

    return dims;
  }
};

///
template<named_dim dim, int BLOCK_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    sync,
    cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>>>
{
  static_assert(BLOCK_SIZE > 0,
                "block size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper =
      cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    constexpr auto zero = static_cast<IdxT>(0);

    const IdxT grid_size = RAJA_DIVIDE_CEILING_INT(
        len, static_cast<IdxT>(IndexMapper::block_size));

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads,
                      static_cast<cuda_dim_member_t>(
                          (len > zero) ? IndexMapper::block_size : 0));
    set_cuda_dim<dim>(dims.min_dims.threads,
                      static_cast<cuda_dim_member_t>(IndexMapper::block_size));

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks,
                      static_cast<cuda_dim_member_t>(grid_size));
    set_cuda_dim<dim>(dims.min_dims.blocks, static_cast<cuda_dim_member_t>(1));

    return dims;
  }
};

///
template<named_dim dim,
         int BLOCK_SIZE,
         int GRID_SIZE,
         kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::StridedLoop<named_usage::unspecified>,
    sync,
    cuda::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>>>
{
  static_assert(BLOCK_SIZE > 0,
                "block size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");
  static_assert(GRID_SIZE > 0,
                "grid size must be > 0, named_usage::unspecified, or "
                "named_usage::ignored with kernel");

  using IndexMapper = cuda::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>;

  template<typename IdxT>
  static LaunchDims get_dimensions(IdxT len)
  {
    constexpr auto zero = static_cast<IdxT>(0);

    LaunchDims dims;

    set_cuda_dim<dim>(dims.active.threads,
                      static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.threads,
                      static_cast<cuda_dim_member_t>(
                          (len > zero) ? IndexMapper::block_size : 0));
    set_cuda_dim<dim>(dims.min_dims.threads,
                      static_cast<cuda_dim_member_t>(IndexMapper::block_size));

    set_cuda_dim<dim>(dims.active.blocks, static_cast<cuda_dim_member_t>(true));
    set_cuda_dim<dim>(dims.dims.blocks,
                      static_cast<cuda_dim_member_t>(
                          (len > zero) ? IndexMapper::grid_size : 0));
    set_cuda_dim<dim>(dims.min_dims.blocks,
                      static_cast<cuda_dim_member_t>(IndexMapper::grid_size));

    return dims;
  }
};

}  // namespace internal

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
