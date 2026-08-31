/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for tile wrapper and iterator.
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

#ifndef RAJA_pattern_kernel_Tile_HPP
#define RAJA_pattern_kernel_Tile_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include "RAJA/index/IndexValue.hpp"
#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

struct TileSize
{
  const camp::idx_t size;

  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr TileSize(camp::idx_t size_) : size {size_} {}
};



namespace statement
{


/*!
 * A RAJA::kernel statement that implements a tiling (or blocking) loop.
 *
 */
template<camp::idx_t ArgumentId,
         typename TilePolicy,
         typename ExecPolicy,
         typename... EnclosedStmts>
struct Tile : public internal::Statement<ExecPolicy, EnclosedStmts...>
{
  using tile_policy_t = TilePolicy;
  using exec_policy_t = ExecPolicy;
};

}  // end namespace statement

///! tag for a tiling loop
template<camp::idx_t chunk_size_>
struct tile_fixed
{
  static constexpr camp::idx_t chunk_size = chunk_size_;
};

template<camp::idx_t ArgumentId>
struct tile_dynamic
{
  static constexpr camp::idx_t id = ArgumentId;
};

namespace internal
{
// template<typename T>
// struct is_instance_of_tile_size : std::false_type {};

// template<typename SizeT>
// struct is_instance_of_tile_size<TileSize<SizeT>> : std::true_type {};


/*!
 * A generic RAJA::kernel forall_impl tile wrapper for statement::For
 * Assigns the tile segment to segment ArgumentId
 *
 */
template<camp::idx_t ArgumentId,
         typename Data,
         typename Types,
         typename... EnclosedStmts>
struct TileWrapper : public GenericWrapper<Data, Types, EnclosedStmts...>
{

  using Base = GenericWrapper<Data, Types, EnclosedStmts...>;
  using Base::Base;
  using privatizer = NestedPrivatizer<TileWrapper>;

  template<typename InSegmentIndexType>
  RAJA_INLINE void operator()(InSegmentIndexType si)
  {
    // Assign the tile's segment to the tuple
    camp::get<ArgumentId>(Base::data.segment_tuple) = si.s;

    // Execute enclosed statements
    Base::exec();
  }
};

template<typename Iterable, typename BlockSizeT>
struct IterableTiler
{
  using value_type = camp::decay<Iterable>;
  using slice_type = typename value_type::size_type;
  using block_type = RAJA::strip_index_type_t<slice_type>;

  struct iterate
  {
    value_type s;
    block_type i;
  };

  class iterator
  {
    // NOTE: this must be held by value for NVCC support, *even on the host*
    const IterableTiler itiler;
    const block_type block_id;

  public:
    using value_type        = iterate;
    using difference_type   = camp::idx_t;
    using pointer           = value_type*;
    using reference         = value_type&;
    using iterator_category = std::random_access_iterator_tag;

    RAJA_HOST_DEVICE

    RAJA_INLINE
    constexpr iterator(IterableTiler const& itiler_, block_type block_id_)
        : itiler {itiler_},
          block_id {block_id_}
    {}

    RAJA_HOST_DEVICE

    RAJA_INLINE
    value_type operator*()
    {
      auto start = slice_type {
          block_id *
          static_cast<block_type>(RAJA::stripIndexType(itiler.block_size))};
      return iterate {itiler.it.slice(start, itiler.block_size), block_id};
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE difference_type operator-(const iterator& rhs) const
    {
      return static_cast<difference_type>(block_id) -
             static_cast<difference_type>(rhs.block_id);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE iterator operator-(const difference_type& rhs) const
    {
      return iterator(
          itiler,
          static_cast<block_type>(static_cast<difference_type>(block_id) - rhs));
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE iterator operator+(const difference_type& rhs) const
    {
      const difference_type next =
          static_cast<difference_type>(block_id) + rhs;
      return iterator(itiler,
                      next >= static_cast<difference_type>(itiler.num_blocks)
                          ? itiler.num_blocks
                          : static_cast<block_type>(next));
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE value_type operator[](difference_type rhs) const
    {
      return *((*this) + rhs);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator!=(const iterator& rhs) const
    {
      return block_id != rhs.block_id;
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE bool operator<(const iterator& rhs) const
    {
      return block_id < rhs.block_id;
    }
  };

  RAJA_HOST_DEVICE RAJA_INLINE
  IterableTiler(const Iterable& it_, BlockSizeT block_size_)
      : it {it_},
        block_size {block_size_}
  {
    using std::begin;
    using std::distance;
    using std::end;
    const block_type stripped_block_size =
        static_cast<block_type>(RAJA::stripIndexType(block_size));
    dist       = static_cast<block_type>(it.end() - it.begin());
    num_blocks = dist / stripped_block_size;
    // if (dist % block_size) num_blocks += 1;
    if (dist - num_blocks * stripped_block_size > block_type {0})
    {
      num_blocks += 1;
    }
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  iterator begin() const { return iterator(*this, block_type {0}); }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  iterator end() const { return iterator(*this, num_blocks); }

  value_type it;
  BlockSizeT block_size;
  block_type num_blocks;
  block_type dist;
};

/*!
 * A generic RAJA::kernel forall_impl executor for statement::Tile
 *
 *
 */
template<camp::idx_t ArgumentId,
         camp::idx_t ChunkSize,
         typename EPol,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<
    statement::Tile<ArgumentId, tile_fixed<ChunkSize>, EPol, EnclosedStmts...>,
    Types>
{

  template<typename Data>
  static RAJA_INLINE void exec(Data& data)
  {
    // Get the segment we are going to tile
    auto const& segment = camp::get<ArgumentId>(data.segment_tuple);

    // Get the tiling policies chunk size
    constexpr auto chunk_size = tile_fixed<ChunkSize>::chunk_size;
    using segment_t = decltype(segment);
    using slice_t = typename std::decay_t<segment_t>::size_type;
    using slice_value_t = RAJA::strip_index_type_t<slice_t>;

    // Create a tile iterator, needs to survive until the forall is
    // done executing.
    IterableTiler<segment_t, slice_t> tiled_iterable(
        segment, slice_t {static_cast<slice_value_t>(chunk_size)});

    // Wrap in case forall_impl needs to thread_privatize
    TileWrapper<ArgumentId, Data, Types, EnclosedStmts...> tile_wrapper(data);

    // Loop over tiles, executing enclosed statement list
    auto r = resources::get_resource<EPol>::type::get_default();
    forall_impl(r, EPol {}, tiled_iterable, tile_wrapper,
                RAJA::expt::get_empty_forall_param_pack());

    // Set range back to original values
    camp::get<ArgumentId>(data.segment_tuple) = tiled_iterable.it;
  }
};

template<camp::idx_t ArgumentId,
         typename EPol,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<
    statement::
        Tile<ArgumentId, tile_dynamic<ArgumentId>, EPol, EnclosedStmts...>,
    Types>
{

  template<typename Data>
  static RAJA_INLINE void exec(Data& data)
  {
    // Get the segment we are going to tile
    auto const& segment = camp::get<ArgumentId>(data.segment_tuple);

    using segment_t = decltype(segment);
    using slice_t = typename std::decay_t<segment_t>::size_type;
    using slice_value_t = RAJA::strip_index_type_t<slice_t>;
    // Get the tiling policies chunk size
    auto chunk_size = camp::get<ArgumentId>(data.param_tuple);
    static_assert(
        camp::concepts::metalib::is_same<TileSize, decltype(chunk_size)>::value,
        // is_instance_of_tile_size<decltype(chunk_size)>::value,
        "Extracted parameter must be of type TileSize.");

    // Create a tile iterator
    IterableTiler<segment_t, slice_t> tiled_iterable(
        segment,
        slice_t {static_cast<slice_value_t>(RAJA::stripIndexType(chunk_size.size))});

    // Wrap in case forall_impl needs to thread_privatize
    TileWrapper<ArgumentId, Data, Types, EnclosedStmts...> tile_wrapper(data);

    // Loop over tiles, executing enclosed statement list
    auto r = resources::get_resource<EPol>::type::get_default();
    forall_impl(r, EPol {}, tiled_iterable, tile_wrapper,
                RAJA::expt::get_empty_forall_param_pack());

    // Set range back to original values
    camp::get<ArgumentId>(data.segment_tuple) = tiled_iterable.it;
  }
};

}  // end namespace internal
}  // end namespace RAJA

#endif /* RAJA_pattern_kernel_HPP */
