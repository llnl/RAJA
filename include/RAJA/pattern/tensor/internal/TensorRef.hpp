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

#ifndef RAJA_pattern_tensor_tensorref_HPP
#define RAJA_pattern_tensor_tensorref_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

namespace RAJA
{
namespace internal
{
namespace expt
{

template<typename INT_SEQ>
struct StaticIndexArray;

template<typename INDEX_TYPE, INDEX_TYPE NEW_HEAD, typename ARRAY>
struct PrependStaticIndexArray;

template<typename INDEX_TYPE, size_t IDX, INDEX_TYPE DELTA, typename ARRAY>
struct AddStaticIndexArray;

template<typename INDEX_TYPE, size_t IDX, INDEX_TYPE DELTA, typename ARRAY>
struct SetStaticIndexArray;

template<typename INDEX_TYPE, INDEX_TYPE HEAD, INDEX_TYPE... TAIL>
struct StaticIndexArray<camp::int_seq<INDEX_TYPE, HEAD, TAIL...>>
{

  using seq_type = camp::int_seq<INDEX_TYPE, HEAD, TAIL...>;
  using Self     = StaticIndexArray<camp::int_seq<INDEX_TYPE, HEAD, TAIL...>>;
  using Tail     = StaticIndexArray<camp::int_seq<INDEX_TYPE, TAIL...>>;

  Tail tail;

  RAJA_INLINE
  StaticIndexArray<camp::int_seq<INDEX_TYPE, HEAD, TAIL...>>() = default;


  RAJA_HOST_DEVICE

  RAJA_INLINE
  static constexpr INDEX_TYPE value_at(size_t index)
  {
    if (index == 0)
    {
      return HEAD;
    }
    else
    {
      return Tail::value_at(index - 1);
    }
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr INDEX_TYPE operator[](size_t index) const
  {
    if (index == 0)
    {
      return HEAD;
    }
    else
    {
      return tail[index - 1];
    }
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  void print_values() const
  {
    printf("%ld ", (long)HEAD);
    tail.print_values();
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  void print() const
  {
    printf("[");
    print_values();
    printf("]");
  }
};

template<typename INDEX_TYPE>
struct StaticIndexArray<camp::int_seq<INDEX_TYPE>>
{

  using seq_type = camp::int_seq<INDEX_TYPE>;

  RAJA_INLINE
  StaticIndexArray<camp::int_seq<INDEX_TYPE>>() = default;


  RAJA_HOST_DEVICE

  RAJA_INLINE
  static constexpr INDEX_TYPE value_at(size_t) { return 0; }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr INDEX_TYPE operator[](size_t) const { return 0; }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  void print_values() const {}

  RAJA_HOST_DEVICE

  RAJA_INLINE
  void print() const { print("[]"); }
};

template<typename INDEX_TYPE, INDEX_TYPE NEW_HEAD, INDEX_TYPE... ORIG_INTS>
struct PrependStaticIndexArray<
    INDEX_TYPE,
    NEW_HEAD,
    StaticIndexArray<camp::int_seq<INDEX_TYPE, ORIG_INTS...>>>
{
  using Type =
      StaticIndexArray<camp::int_seq<INDEX_TYPE, NEW_HEAD, ORIG_INTS...>>;
  using Seq = camp::int_seq<INDEX_TYPE, NEW_HEAD, ORIG_INTS...>;
};

template<typename INDEX_TYPE,
         size_t IDX,
         INDEX_TYPE DELTA,
         INDEX_TYPE HEAD,
         INDEX_TYPE... TAIL>
struct AddStaticIndexArray<
    INDEX_TYPE,
    IDX,
    DELTA,
    StaticIndexArray<camp::int_seq<INDEX_TYPE, HEAD, TAIL...>>>
{
  using Orig    = StaticIndexArray<camp::int_seq<INDEX_TYPE, HEAD, TAIL...>>;
  using AddTail = typename AddStaticIndexArray<INDEX_TYPE,
                                               IDX - 1,
                                               DELTA,
                                               typename Orig::Tail>::Type;
  using Type =
      typename PrependStaticIndexArray<INDEX_TYPE, HEAD, AddTail>::Type;
  using Seq = typename PrependStaticIndexArray<INDEX_TYPE, HEAD, AddTail>::Seq;
};

template<typename INDEX_TYPE,
         INDEX_TYPE DELTA,
         INDEX_TYPE HEAD,
         INDEX_TYPE... TAIL>
struct AddStaticIndexArray<
    INDEX_TYPE,
    0,
    DELTA,
    StaticIndexArray<camp::int_seq<INDEX_TYPE, HEAD, TAIL...>>>
{

  using Orig = StaticIndexArray<camp::int_seq<INDEX_TYPE, HEAD, TAIL...>>;
  using Type = typename PrependStaticIndexArray<INDEX_TYPE,
                                                HEAD + DELTA,
                                                typename Orig::Tail>::Type;
  using Seq  = typename PrependStaticIndexArray<INDEX_TYPE,
                                               HEAD + DELTA,
                                               typename Orig::Tail>::Seq;
};

template<typename INDEX_TYPE,
         size_t IDX,
         INDEX_TYPE VALUE,
         INDEX_TYPE HEAD,
         INDEX_TYPE... TAIL>
struct SetStaticIndexArray<
    INDEX_TYPE,
    IDX,
    VALUE,
    StaticIndexArray<camp::int_seq<INDEX_TYPE, HEAD, TAIL...>>>
{
  using Orig    = StaticIndexArray<camp::int_seq<INDEX_TYPE, HEAD, TAIL...>>;
  using SetTail = typename SetStaticIndexArray<INDEX_TYPE,
                                               IDX - 1,
                                               VALUE,
                                               typename Orig::Tail>::Type;
  using Type =
      typename PrependStaticIndexArray<INDEX_TYPE, HEAD, SetTail>::Type;
  using Seq = typename PrependStaticIndexArray<INDEX_TYPE, HEAD, SetTail>::Seq;
};

template<typename INDEX_TYPE,
         INDEX_TYPE VALUE,
         INDEX_TYPE HEAD,
         INDEX_TYPE... TAIL>
struct SetStaticIndexArray<
    INDEX_TYPE,
    0,
    VALUE,
    StaticIndexArray<camp::int_seq<INDEX_TYPE, HEAD, TAIL...>>>
{
  using Orig = StaticIndexArray<camp::int_seq<INDEX_TYPE, HEAD, TAIL...>>;
  using Type = typename PrependStaticIndexArray<INDEX_TYPE,
                                                VALUE,
                                                typename Orig::Tail>::Type;
  using Seq  = typename PrependStaticIndexArray<INDEX_TYPE,
                                               VALUE,
                                               typename Orig::Tail>::Seq;
};

enum TensorTileSize
{
  TENSOR_PARTIAL,  // the tile is a full TensorRegister
  TENSOR_FULL,     // the tile is a partial TensorRegister
  TENSOR_MULTIPLE  // the tile is multiple TennsorRegisters
};

template<typename INDEX_TYPE, TensorTileSize TENSOR_SIZE, camp::idx_t NUM_DIMS>
struct TensorTile
{
  using self_type           = TensorTile<INDEX_TYPE, TENSOR_SIZE, NUM_DIMS>;
  using nonstatic_self_type = self_type;
  using index_type          = INDEX_TYPE;
  index_type m_begin[NUM_DIMS];
  index_type m_size[NUM_DIMS];

  static constexpr camp::idx_t s_num_dims       = NUM_DIMS;
  static constexpr TensorTileSize s_tensor_size = TENSOR_SIZE;

  template<typename I, TensorTileSize S>
  void copy(TensorTile<I, S, NUM_DIMS> const& c)
  {
    for (camp::idx_t i = 0; i < NUM_DIMS; ++i)
    {
      m_begin[i] = c.m_begin[i];
      m_size[i]  = c.m_size[i];
    }
  }

  /*!
   * Subtract begin offsets of two tiles.
   *
   * The resulting tile has the sizes of the left operand, but has
   * m_begin[i] = left.m_begin[i] - right.m_begin[i]
   *
   */
  template<typename INDEX_TYPE2, TensorTileSize TENSOR_SIZE2>
  RAJA_HOST_DEVICE RAJA_INLINE self_type
  operator-(TensorTile<INDEX_TYPE2, TENSOR_SIZE2, NUM_DIMS> const& sub) const
  {
    self_type result(*this);
    for (camp::idx_t i = 0; i < s_num_dims; ++i)
    {
      result.m_begin[i] -= sub.m_begin[i];
    }
    return result;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  void print() const
  {
    printf("TensorTile: dims=%d, m_begin=[", (int)NUM_DIMS);

    for (camp::idx_t i = 0; i < NUM_DIMS; ++i)
    {
      printf("%ld ", (long)m_begin[i]);
    }

    printf("], m_size=[");

    for (camp::idx_t i = 0; i < NUM_DIMS; ++i)
    {
      printf("%ld ", (long)m_size[i]);
    }

    printf("]\n");
  }
};


template<typename INDEX_TYPE,
         TensorTileSize TENSOR_SIZE,
         typename TBEGIN,
         typename TSIZE>
struct StaticTensorTile;

template<typename INDEX_TYPE,
         TensorTileSize TENSOR_SIZE,
         INDEX_TYPE... BeginInts,
         INDEX_TYPE... SizeInts>
struct StaticTensorTile<INDEX_TYPE,
                        TENSOR_SIZE,
                        camp::int_seq<INDEX_TYPE, BeginInts...>,
                        camp::int_seq<INDEX_TYPE, SizeInts...>>
{


  using begin_seq  = camp::int_seq<INDEX_TYPE, BeginInts...>;
  using size_seq   = camp::int_seq<INDEX_TYPE, SizeInts...>;
  using begin_type = StaticIndexArray<begin_seq>;
  using size_type  = StaticIndexArray<size_seq>;
  using self_type =
      StaticTensorTile<INDEX_TYPE, TENSOR_SIZE, begin_seq, size_seq>;
  using index_type = INDEX_TYPE;

  using nonstatic_self_type =
      TensorTile<INDEX_TYPE, TENSOR_SIZE, sizeof...(BeginInts)>;

  using Partial =
      StaticTensorTile<INDEX_TYPE, TENSOR_PARTIAL, begin_seq, size_seq>;
  using Full = StaticTensorTile<INDEX_TYPE, TENSOR_FULL, begin_seq, size_seq>;

  begin_type m_begin;
  size_type m_size;

  static_assert(sizeof...(BeginInts) == sizeof...(SizeInts),
                "Mismatch between number of elements in Begin and Size series "
                "of StaticTensorTile");

  static constexpr camp::idx_t s_num_dims       = sizeof...(BeginInts);
  static constexpr TensorTileSize s_tensor_size = TENSOR_SIZE;

  constexpr operator nonstatic_self_type() const
  {
    return nonstatic_self_type {{BeginInts...}, {SizeInts...}};
  }

  constexpr nonstatic_self_type nonstatic() const { return *this; }

  template<TensorTileSize S>
  constexpr void copy(StaticTensorTile<INDEX_TYPE, S, begin_seq, size_seq> const
                          RAJA_UNUSED_ARG(&c)) const
  {}

  RAJA_HOST_DEVICE

  RAJA_INLINE
  void print() const
  {
    printf("StaticTensorTile: dims=%d, m_begin=", (int)s_num_dims);

    m_begin.print();

    printf(", m_size=");

    m_size.print();

    printf("\n");
  }
};

template<typename TILE, typename VALUE, size_t IDX>
struct SetStaticTensorTileBegin;

template<typename INDEX_TYPE,
         TensorTileSize TENSOR_SIZE,
         typename TBEGIN,
         typename TSIZE,
         INDEX_TYPE VALUE,
         size_t IDX>
struct SetStaticTensorTileBegin<
    StaticTensorTile<INDEX_TYPE, TENSOR_SIZE, TBEGIN, TSIZE>,
    camp::integral_constant<INDEX_TYPE, VALUE>,
    IDX>
{
  using BeginType = StaticIndexArray<TBEGIN>;
  using Type      = StaticTensorTile<
      INDEX_TYPE,
      TENSOR_SIZE,
      typename SetStaticIndexArray<INDEX_TYPE, IDX, VALUE, BeginType>::Seq,
      TSIZE>;
};

template<typename TILE, typename VALUE, size_t IDX>
struct SetStaticTensorTileSize;

template<typename INDEX_TYPE,
         TensorTileSize TENSOR_SIZE,
         typename TBEGIN,
         typename TSIZE,
         INDEX_TYPE VALUE,
         size_t IDX>
struct SetStaticTensorTileSize<
    StaticTensorTile<INDEX_TYPE, TENSOR_SIZE, TBEGIN, TSIZE>,
    camp::integral_constant<INDEX_TYPE, VALUE>,
    IDX>
{
  using SizeType = StaticIndexArray<TSIZE>;
  using Type     = StaticTensorTile<
      INDEX_TYPE,
      TENSOR_SIZE,
      TBEGIN,
      typename SetStaticIndexArray<INDEX_TYPE, IDX, VALUE, SizeType>::Seq>;
};

template<typename POINTER_TYPE,
         typename INDEX_TYPE,
         TensorTileSize TENSOR_SIZE,
         camp::idx_t NUM_DIMS,
         camp::idx_t STRIDE_ONE_DIM = -1>
struct TensorRef
{
  static constexpr camp::idx_t s_stride_one_dim = STRIDE_ONE_DIM;
  static constexpr camp::idx_t s_num_dims       = NUM_DIMS;
  static constexpr TensorTileSize s_tensor_size = TENSOR_SIZE;

  using self_type    = TensorRef<POINTER_TYPE,
                              INDEX_TYPE,
                              TENSOR_SIZE,
                              NUM_DIMS,
                              STRIDE_ONE_DIM>;
  using tile_type    = TensorTile<INDEX_TYPE, TENSOR_SIZE, NUM_DIMS>;
  using pointer_type = POINTER_TYPE;
  using index_type   = INDEX_TYPE;


  pointer_type m_pointer;
  index_type m_stride[NUM_DIMS];
  tile_type m_tile;

  RAJA_HOST_DEVICE

  RAJA_INLINE
  void print() const
  {
    printf("TensorRef: dims=%d, m_pointer=%p, m_stride=[", (int)NUM_DIMS,
           m_pointer);

    for (camp::idx_t i = 0; i < NUM_DIMS; ++i)
    {
      printf("%ld ", (long)m_stride[i]);
    }

    printf("], stride_one_dim=%d\n", (int)STRIDE_ONE_DIM);

    m_tile.print();
  }
};


template<typename POINTER_TYPE,
         typename INDEX_TYPE,
         TensorTileSize TENSOR_SIZE,
         typename STRIDE_TYPE,
         typename BEGIN_TYPE,
         typename SIZE_TYPE,
         camp::idx_t STRIDE_ONE_DIM = -1>
struct StaticTensorRef;

template<typename POINTER_TYPE,
         typename INDEX_TYPE,
         TensorTileSize TENSOR_SIZE,
         INDEX_TYPE... StrideInts,
         INDEX_TYPE... BeginInts,
         INDEX_TYPE... SizeInts,
         camp::idx_t STRIDE_ONE_DIM>
struct StaticTensorRef<POINTER_TYPE,
                       INDEX_TYPE,
                       TENSOR_SIZE,
                       camp::int_seq<INDEX_TYPE, StrideInts...>,
                       camp::int_seq<INDEX_TYPE, BeginInts...>,
                       camp::int_seq<INDEX_TYPE, SizeInts...>,
                       STRIDE_ONE_DIM>
{

  static constexpr camp::idx_t s_num_dims           = sizeof...(BeginInts);
  static constexpr camp::idx_t s_stride_one_dim     = STRIDE_ONE_DIM;
  static constexpr TensorTileSize s_ref_tensor_size = TENSOR_SIZE;
  using pointer_type                                = POINTER_TYPE;
  using index_type                                  = INDEX_TYPE;

  using stride_seq = camp::int_seq<INDEX_TYPE, StrideInts...>;
  using begin_seq  = camp::int_seq<INDEX_TYPE, BeginInts...>;
  using size_seq   = camp::int_seq<INDEX_TYPE, SizeInts...>;

  using stride_type = StaticIndexArray<stride_seq>;

  static_assert((sizeof...(BeginInts) == sizeof...(SizeInts)) &&
                    (sizeof...(SizeInts) == sizeof...(StrideInts)),
                "Mismatch between number of elements in Begin and Size series "
                "of StaticTensorRef");


  using self_type = StaticTensorRef<POINTER_TYPE,
                                    INDEX_TYPE,
                                    TENSOR_SIZE,
                                    stride_seq,
                                    begin_seq,
                                    size_seq>;
  using tile_type =
      StaticTensorTile<INDEX_TYPE, TENSOR_SIZE, begin_seq, size_seq>;


  pointer_type m_pointer;
  stride_type m_stride;
  tile_type m_tile;

  RAJA_HOST_DEVICE

  RAJA_INLINE
  void print() const
  {
    printf("StaticTensorRef: dims=%d, m_pointer=%p, m_stride=", (int)s_num_dims,
           m_pointer);

    m_stride.print();

    printf(", stride_one_dim=%d\n", (int)STRIDE_ONE_DIM);

    m_tile.print();
  }
};


template<typename REF_TYPE, typename TILE_TYPE, typename DIM_SEQ>
struct MergeRefTile;

template<typename REF_TYPE, typename TILE_TYPE, camp::idx_t... DIM_SEQ>
struct MergeRefTile<REF_TYPE, TILE_TYPE, camp::idx_seq<DIM_SEQ...>>
{

  static_assert(
      REF_TYPE::s_num_dims == TILE_TYPE::s_num_dims,
      "Merging a ref with a tile requires an equivalent number of dimensions.");

  static constexpr camp::idx_t s_num_dims       = REF_TYPE::s_num_dims;
  static constexpr camp::idx_t s_stride_one_dim = REF_TYPE::s_stride_one_dim;
  static constexpr TensorTileSize s_ref_tensor_size = TILE_TYPE::s_tensor_size;
  using pointer_type   = typename REF_TYPE::pointer_type;
  using ref_index_type = typename REF_TYPE::index_type;

  static constexpr TensorTileSize s_tile_tensor_size = TILE_TYPE::s_tensor_size;
  using tile_index_type = typename TILE_TYPE::index_type;

  using merge_type = TensorRef<pointer_type,
                               tile_index_type,
                               s_tile_tensor_size,
                               s_num_dims,
                               s_stride_one_dim>;
  using shift_type = merge_type;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr merge_type merge(REF_TYPE const& ref, TILE_TYPE const& tile)
  {
    return merge_type {
        ref.m_pointer, {tile_index_type(ref.m_stride[DIM_SEQ])...}, tile};
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr shift_type shift_origin(REF_TYPE const& ref,
                                           TILE_TYPE const& tile_origin)
  {
    return shift_type {
        ref.m_pointer - RAJA::sum<camp::idx_t>((tile_origin.m_begin[DIM_SEQ] *
                                                ref.m_stride[DIM_SEQ])...),
        {tile_index_type(ref.m_stride[DIM_SEQ])...},
        ref.m_tile};
  }
};

template<typename POINTER_TYPE,
         typename INDEX_TYPE1,
         TensorTileSize RTENSOR_SIZE,
         typename STRIDE,
         INDEX_TYPE1... BEGIN1,
         INDEX_TYPE1... SIZE1,
         camp::idx_t STRIDE_ONE_DIM,
         typename INDEX_TYPE2,
         TensorTileSize TENSOR_SIZE,
         typename BEGIN2,
         typename SIZE2,
         camp::idx_t... DIM_SEQ>
struct MergeRefTile<StaticTensorRef<POINTER_TYPE,
                                    INDEX_TYPE1,
                                    RTENSOR_SIZE,
                                    STRIDE,
                                    camp::int_seq<INDEX_TYPE1, BEGIN1...>,
                                    camp::int_seq<INDEX_TYPE1, SIZE1...>,
                                    STRIDE_ONE_DIM>,
                    StaticTensorTile<INDEX_TYPE2, TENSOR_SIZE, BEGIN2, SIZE2>,
                    camp::idx_seq<DIM_SEQ...>>
{

  using ref_tile_type = StaticTensorTile<INDEX_TYPE1,
                                         RTENSOR_SIZE,
                                         camp::int_seq<INDEX_TYPE1, BEGIN1...>,
                                         camp::int_seq<INDEX_TYPE1, SIZE1...>>;

  using ref_type = StaticTensorRef<POINTER_TYPE,
                                   INDEX_TYPE1,
                                   RTENSOR_SIZE,
                                   STRIDE,
                                   camp::int_seq<INDEX_TYPE1, BEGIN1...>,
                                   camp::int_seq<INDEX_TYPE1, SIZE1...>,
                                   STRIDE_ONE_DIM>;

  using tile_type = StaticTensorTile<INDEX_TYPE2, TENSOR_SIZE, BEGIN2, SIZE2>;

  using ref_stride_type = typename ref_type ::stride_type;

  using new_stride_seq =
      camp::int_seq<INDEX_TYPE2,
                    INDEX_TYPE2(ref_stride_type::value_at(DIM_SEQ))...>;

  using shift_begin_seq = camp::int_seq<INDEX_TYPE2, INDEX_TYPE2(BEGIN1)...>;
  using shift_size_seq  = camp::int_seq<INDEX_TYPE2, INDEX_TYPE2(SIZE1)...>;

  using shift_tile_type = StaticTensorTile<INDEX_TYPE2,
                                           TENSOR_SIZE,
                                           shift_begin_seq,
                                           shift_size_seq>;

  using new_stride_type = StaticIndexArray<new_stride_seq>;

  using merge_type = StaticTensorRef<POINTER_TYPE,
                                     INDEX_TYPE2,
                                     TENSOR_SIZE,
                                     new_stride_seq,
                                     BEGIN2,
                                     SIZE2,
                                     STRIDE_ONE_DIM>;

  using shift_type = StaticTensorRef<POINTER_TYPE,
                                     INDEX_TYPE2,
                                     TENSOR_SIZE,
                                     new_stride_seq,
                                     shift_begin_seq,
                                     shift_size_seq,
                                     STRIDE_ONE_DIM>;


  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr merge_type merge(ref_type const& ref, tile_type const& tile)
  {
    return merge_type {ref.m_pointer, new_stride_type(), tile};
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr shift_type shift_origin(ref_type const& ref,
                                           tile_type const& tile_origin)
  {
    return shift_type {
        ref.m_pointer - RAJA::sum<camp::idx_t>((tile_origin.m_begin[DIM_SEQ] *
                                                ref.m_stride[DIM_SEQ])...),
        new_stride_type(), shift_tile_type()};
  }
};

template<typename REF_TYPE, typename TILE_TYPE>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto merge_ref_tile(
    REF_TYPE const& ref,
    TILE_TYPE const& tile) ->
    typename MergeRefTile<
        REF_TYPE,
        TILE_TYPE,
        camp::make_idx_seq_t<TILE_TYPE::s_num_dims>>::merge_type
{
  return MergeRefTile<REF_TYPE, TILE_TYPE,
                      camp::make_idx_seq_t<TILE_TYPE::s_num_dims>>::merge(ref,
                                                                          tile);
}

/*!
 * Modifies a ref's pointer so that the supplied tile_origin will resolve
 * to the original pointer.
 */
template<typename REF_TYPE, typename TILE_TYPE>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto shift_tile_origin(
    REF_TYPE const& ref,
    TILE_TYPE const& tile_origin) ->
    typename MergeRefTile<
        REF_TYPE,
        TILE_TYPE,
        camp::make_idx_seq_t<TILE_TYPE::s_num_dims>>::shift_type
{
  return MergeRefTile<
      REF_TYPE, TILE_TYPE,
      camp::make_idx_seq_t<TILE_TYPE::s_num_dims>>::shift_origin(ref,
                                                                 tile_origin);
}

/*!
 * Changes TensorTile size type to FULL
 */
template<typename INDEX_TYPE, TensorTileSize RTENSOR_SIZE, camp::idx_t NUM_DIMS>
RAJA_INLINE RAJA_HOST_DEVICE constexpr TensorTile<INDEX_TYPE,
                                                  TENSOR_FULL,
                                                  NUM_DIMS>&
make_tensor_tile_full(TensorTile<INDEX_TYPE, RTENSOR_SIZE, NUM_DIMS>& tile)
{
  return reinterpret_cast<TensorTile<INDEX_TYPE, TENSOR_FULL, NUM_DIMS>&>(tile);
}

/*!
 * Changes TensorTile size type to PARTIAL
 */
template<typename INDEX_TYPE, TensorTileSize RTENSOR_SIZE, camp::idx_t NUM_DIMS>
RAJA_INLINE RAJA_HOST_DEVICE constexpr TensorTile<INDEX_TYPE,
                                                  TENSOR_PARTIAL,
                                                  NUM_DIMS>&
make_tensor_tile_partial(TensorTile<INDEX_TYPE, RTENSOR_SIZE, NUM_DIMS>& tile)
{
  return reinterpret_cast<TensorTile<INDEX_TYPE, TENSOR_PARTIAL, NUM_DIMS>&>(
      tile);
}

/*!
 * Changes StaticTensorTile size type to FULL
 */
template<typename INDEX_TYPE,
         TensorTileSize RTENSOR_SIZE,
         typename TBEGIN,
         typename TSIZE>
RAJA_INLINE RAJA_HOST_DEVICE constexpr StaticTensorTile<INDEX_TYPE,
                                                        TENSOR_FULL,
                                                        TBEGIN,
                                                        TSIZE>&
make_tensor_tile_full(
    StaticTensorTile<INDEX_TYPE, RTENSOR_SIZE, TBEGIN, TSIZE>& tile)
{
  return reinterpret_cast<
      StaticTensorTile<INDEX_TYPE, TENSOR_FULL, TBEGIN, TSIZE>&>(tile);
}

/*!
 * Changes StaticTensorTile size type to PARTIAL
 */
template<typename INDEX_TYPE,
         TensorTileSize RTENSOR_SIZE,
         typename TBEGIN,
         typename TSIZE>
RAJA_INLINE RAJA_HOST_DEVICE constexpr StaticTensorTile<INDEX_TYPE,
                                                        TENSOR_PARTIAL,
                                                        TBEGIN,
                                                        TSIZE>&
make_tensor_tile_partial(
    StaticTensorTile<INDEX_TYPE, RTENSOR_SIZE, TBEGIN, TSIZE>& tile)
{
  return reinterpret_cast<
      StaticTensorTile<INDEX_TYPE, TENSOR_PARTIAL, TBEGIN, TSIZE>&>(tile);
}


}  // namespace expt
}  // namespace internal

}  // namespace RAJA


#endif
