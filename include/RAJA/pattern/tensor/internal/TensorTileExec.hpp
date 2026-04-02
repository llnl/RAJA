/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining SIMD/SIMT register operations.
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

#ifndef RAJA_pattern_tensor_TensorTileExec_HPP
#define RAJA_pattern_tensor_TensorTileExec_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/tensor/internal/TensorRef.hpp"
#include "RAJA/pattern/tensor/stats.hpp"

namespace RAJA
{
namespace internal
{
namespace expt
{


template<typename STORAGE, typename DIM_SEQ, typename IDX_SEQ>
struct StaticTensorTileExec;

template<typename STORAGE, typename DIM_SEQ>
struct TensorTileExec;

/**
 * Implement a dimension tiling loop
 */
template<typename STORAGE, camp::idx_t DIM0, camp::idx_t... DIM_REST>
struct TensorTileExec<STORAGE, camp::idx_seq<DIM0, DIM_REST...>>
{

  using inner_t = TensorTileExec<STORAGE, camp::idx_seq<DIM_REST...>>;

  template<typename OTILE, typename TTYPE, typename BODY>
  RAJA_HOST_DEVICE RAJA_INLINE static void exec(OTILE const& otile,
                                                TTYPE& tile,
                                                BODY&& body)
  {

    auto const orig_begin = otile.m_begin[DIM0];
    auto const orig_size  = otile.m_size[DIM0];

    // Do the full tile sizes
    for (tile.m_begin[DIM0] = orig_begin;

         tile.m_begin[DIM0] + STORAGE::s_dim_elem(DIM0) <=
         orig_begin + orig_size;

         tile.m_begin[DIM0] += STORAGE::s_dim_elem(DIM0))
    {

      // Do the next inner tiling loop
      inner_t::exec(otile, tile, body);
    }

    // Postamble if needed
    if (tile.m_begin[DIM0] < orig_begin + orig_size)
    {

      // convert tile to a partial tile
      auto& part_tile = make_tensor_tile_partial(tile);

      // store original size
      auto tmp_size = part_tile.m_size[DIM0];

      // set tile size to the remainder
      part_tile.m_size[DIM0] = orig_begin + orig_size - tile.m_begin[DIM0];

      // Do the next inner tiling loop
      inner_t::exec(otile, part_tile, body);

      // restore size
      part_tile.m_size[DIM0] = tmp_size;
    }

    // reset tile dimension
    tile.m_begin[DIM0] = orig_begin;
  }

  template<typename OTILE, typename TTYPE, typename BODY>
  RAJA_HOST_DEVICE RAJA_INLINE static void static_exec(OTILE const& otile,
                                                       TTYPE const& tile,
                                                       BODY&& body)
  {


    auto constexpr orig_begin = OTILE::begin_type::value_at(DIM0);
    auto constexpr orig_size  = OTILE::size_type::value_at(DIM0);

    auto constexpr tile_begin = TTYPE::begin_type::value_at(DIM0);

    auto constexpr step_size = STORAGE::s_dim_elem(DIM0);

    auto constexpr iter_count =
        (tile_begin >= orig_begin) && (tile_begin < (orig_begin + orig_size))
            ? ((orig_begin + orig_size) - tile_begin + step_size - 1) /
                  step_size
            : 0;


    using IterCount =
        camp::integral_constant<typename TTYPE::index_type, iter_count>;
    using DimSeq = camp::idx_seq<DIM0, DIM_REST...>;
    using IdxSeq = typename camp::detail::gen_seq<typename TTYPE::index_type,
                                                  IterCount>::type;

    StaticTensorTileExec<STORAGE, DimSeq, IdxSeq>::exec(otile, tile, body);
  }
};

/**
 * Termination of nested loop:  execute evaluation of ET
 */
template<typename STORAGE>
struct TensorTileExec<STORAGE, camp::idx_seq<>>
{

  template<typename OTILE, typename TTYPE, typename BODY>
  RAJA_HOST_DEVICE RAJA_INLINE static void exec(OTILE&,
                                                TTYPE const& tile,
                                                BODY&& body)
  {

    // execute body, passing in the current tile
    body(tile);
  }

  template<typename OTILE, typename TTYPE, typename BODY>
  RAJA_HOST_DEVICE RAJA_INLINE static void static_exec(OTILE const&,
                                                       TTYPE const& tile,
                                                       BODY&& body)
  {

    // execute body, passing in the current tile
    body(tile);
  }
};

template<typename STORAGE,
         typename TILE_TYPE,
         typename BODY,
         camp::idx_t... IDX_SEQ,
         camp::idx_t... DIM_SEQ>
RAJA_INLINE RAJA_HOST_DEVICE void tensorTileExec_expanded(
    TILE_TYPE const& orig_tile,
    BODY&& body,
    camp::idx_seq<IDX_SEQ...> const&,
    camp::idx_seq<DIM_SEQ...> const&)
{

  // tile over full rows and columns
  // tile_type tile{{0,0},{row_tile_size, col_tile_size}};
  TILE_TYPE tile {
      {orig_tile.m_begin[IDX_SEQ]...},
      {STORAGE::s_dim_elem(IDX_SEQ)...},
  };


  // Promote the tile type to a "full-tile" so that the full-element
  // register operations are used.
  // Any of the tiling loops can demote this to a partial-tile when
  // they do postamble execution
  auto& full_tile = make_tensor_tile_full(tile);

  // Do all of the tiling loops in layout order, this may improve
  // cache performance
  using layout_order       = typename STORAGE::layout_type::seq_t;
  using tensor_tile_exec_t = TensorTileExec<STORAGE, layout_order>;


  tensor_tile_exec_t::exec(orig_tile, full_tile, body);
}


template<typename STORAGE, typename DIM_SEQ, typename IDX_SEQ>
struct StaticTensorTileExec;

/**
 * Implement a dimension tiling loop
 */

template<typename STORAGE,
         camp::idx_t DIM0,
         camp::idx_t... DIM_REST,
         camp::idx_t IDX,
         camp::idx_t... IDX_REST>
struct StaticTensorTileExec<STORAGE,
                            camp::idx_seq<DIM0, DIM_REST...>,
                            camp::idx_seq<IDX, IDX_REST...>>
{

  using DimList = camp::idx_seq<DIM0, DIM_REST...>;
  using DimTail = camp::idx_seq<DIM_REST...>;
  using IdxList = camp::idx_seq<IDX, IDX_REST...>;
  using IdxTail = camp::idx_seq<IDX_REST...>;

  using DownExec = TensorTileExec<STORAGE, camp::idx_seq<DIM_REST...>>;
  using NextExec = StaticTensorTileExec<STORAGE,
                                        camp::idx_seq<DIM0, DIM_REST...>,
                                        camp::idx_seq<IDX_REST...>>;

  static auto const step_size = STORAGE::s_dim_elem(DIM0);

  template<typename OTILE, typename TTYPE, typename BODY>
  RAJA_HOST_DEVICE RAJA_INLINE static void exec(OTILE const& otile,
                                                TTYPE const& tile,
                                                BODY&& body)
  {

    auto constexpr orig_begin = OTILE::begin_type::value_at(DIM0);
    auto constexpr orig_size  = OTILE::size_type::value_at(DIM0);

    auto constexpr tile_begin = TTYPE::begin_type::value_at(DIM0);

    using NextBegin =
        camp::integral_constant<typename TTYPE::index_type,
                                tile_begin + STORAGE::s_dim_elem(DIM0)>;
    using TailSize =
        camp::integral_constant<typename TTYPE::index_type,
                                (orig_begin + orig_size) - tile_begin>;

    using NextTile =
        typename expt::SetStaticTensorTileBegin<TTYPE, NextBegin,
                                                (size_t)DIM0>::Type;

    using TailTile = typename expt::SetStaticTensorTileSize<TTYPE, TailSize,
                                                            (size_t)DIM0>::Type;
    using PartTile = typename TailTile::Partial;


    static_assert((tile_begin + STORAGE::s_dim_elem(DIM0)) <=
                      (orig_begin + orig_size + STORAGE::s_dim_elem(DIM0)),
                  "OOB StaticTensorTileExec DOWN");

    if ((tile_begin + STORAGE::s_dim_elem(DIM0)) <= (orig_begin + orig_size))
    {
      DownExec::static_exec(otile, tile, body);
      NextTile next_tile;
      NextExec::exec(otile, next_tile, body);
    }
    else if (tile_begin < (orig_begin + orig_size))
    {
      PartTile part_tile;
      DownExec::static_exec(otile, part_tile, body);
    }
  }
};

template<typename STORAGE,
         camp::idx_t DIM0,
         camp::idx_t IDX,
         camp::idx_t... IDX_REST>
struct StaticTensorTileExec<STORAGE,
                            camp::idx_seq<DIM0>,
                            camp::idx_seq<IDX, IDX_REST...>>
{
  using NextExec = StaticTensorTileExec<STORAGE,
                                        camp::idx_seq<DIM0>,
                                        camp::idx_seq<IDX_REST...>>;

  template<typename OTILE, typename TTYPE, typename BODY>
  RAJA_HOST_DEVICE RAJA_INLINE static void exec(OTILE const& otile,
                                                TTYPE const& tile,
                                                BODY&& body)
  {
    auto constexpr orig_begin = OTILE::begin_type::value_at(DIM0);
    auto constexpr orig_size  = OTILE::size_type::value_at(DIM0);

    auto constexpr tile_begin = TTYPE::begin_type::value_at(DIM0);

    using NextBegin =
        camp::integral_constant<typename TTYPE::index_type,
                                tile_begin + STORAGE::s_dim_elem(DIM0)>;
    using TailSize =
        camp::integral_constant<typename TTYPE::index_type,
                                (orig_begin + orig_size) - tile_begin>;

    using NextTile =
        typename expt::SetStaticTensorTileBegin<TTYPE, NextBegin,
                                                (size_t)DIM0>::Type;

    using TailTile = typename expt::SetStaticTensorTileSize<TTYPE, TailSize,
                                                            (size_t)DIM0>::Type;
    using PartTile = typename TailTile::Partial;


    static_assert((tile_begin + STORAGE::s_dim_elem(DIM0)) <=
                      (orig_begin + orig_size + STORAGE::s_dim_elem(DIM0)),
                  "OOB StaticTensorTileExec ACROSS");

    if ((tile_begin + STORAGE::s_dim_elem(DIM0)) <= (orig_begin + orig_size))
    {
      body(tile);
      NextTile next_tile;
      NextExec::exec(otile, next_tile, body);
    }
    else if (tile_begin < (orig_begin + orig_size))
    {
      PartTile part_tile;
      body(part_tile);
    }
  }
};

template<typename STORAGE, camp::idx_t... DIM_REST>
struct StaticTensorTileExec<STORAGE,
                            camp::idx_seq<DIM_REST...>,
                            camp::idx_seq<>>
{

  template<typename OTILE, typename TTYPE, typename BODY>
  RAJA_HOST_DEVICE RAJA_INLINE static void exec(OTILE const&,
                                                TTYPE const&,
                                                BODY&&)
  {}
};

template<typename STORAGE,
         typename INDEX_TYPE,
         TensorTileSize TENSOR_SIZE,
         typename TBEGIN,
         typename TSIZE,
         typename BODY,
         camp::idx_t... IDX_SEQ,
         camp::idx_t... DIM_SEQ>
RAJA_INLINE RAJA_HOST_DEVICE void tensorTileExec_expanded(
    StaticTensorTile<INDEX_TYPE, TENSOR_SIZE, TBEGIN, TSIZE> const& orig_tile,
    BODY&& body,
    camp::idx_seq<IDX_SEQ...> const&,
    camp::idx_seq<DIM_SEQ...> const&)
{

  using InputType = StaticTensorTile<INDEX_TYPE, TENSOR_SIZE, TBEGIN, TSIZE>;

  using InputBegin = typename InputType::begin_type;

  using Type = StaticTensorTile<
      INDEX_TYPE, TENSOR_FULL,
      camp::int_seq<INDEX_TYPE, InputBegin::value_at(IDX_SEQ)...>,
      camp::int_seq<INDEX_TYPE, STORAGE::s_dim_elem(IDX_SEQ)...>>;

  Type full_tile;

  // Do all of the tiling loops in layout order, this may improve
  // cache performance
  using layout_order       = typename STORAGE::layout_type::seq_t;
  using tensor_tile_exec_t = TensorTileExec<STORAGE, layout_order>;


  tensor_tile_exec_t::static_exec(orig_tile, full_tile, body);
}

template<typename STORAGE, typename TILE_TYPE, typename BODY>
RAJA_INLINE RAJA_HOST_DEVICE void tensorTileExec(TILE_TYPE const& tile,
                                                 BODY&& body)
{
  using layout_type = typename STORAGE::layout_type;
  tensorTileExec_expanded<STORAGE>(
      tile, body, camp::make_idx_seq_t<STORAGE::s_num_dims> {}, layout_type {});
}

}  // namespace expt
}  // namespace internal

}  // namespace RAJA


#endif
