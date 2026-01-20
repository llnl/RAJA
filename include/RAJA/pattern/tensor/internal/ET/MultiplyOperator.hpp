/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header defining expression template behavior for operator*
 *
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

#ifndef RAJA_pattern_tensor_ET_MultiplyOperator_HPP
#define RAJA_pattern_tensor_ET_MultiplyOperator_HPP

namespace RAJA
{
namespace internal
{
namespace expt
{
// forward
class TensorBlockConcreteBase;

namespace ET
{


/*!
 * Provides default multiply, multiply add, and multiply subtract
 * operations.
 *
 * If the operands are both matrices, we perform a matrix-matrix multiply.
 * Otherwise, we perform element-wise operations.
 */
template<typename LEFT_OPERAND_TYPE,
         typename RIGHT_OPERAND_TYPE,
         class ENABLE = void>
struct MultiplyOperator
{

  using result_type = typename LEFT_OPERAND_TYPE::result_type;
  static constexpr camp::idx_t s_num_dims = LEFT_OPERAND_TYPE::s_num_dims;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static void print_ast()
  {
    printf("Elemental(%d,%d)", (int)s_num_dims,
           (int)RIGHT_OPERAND_TYPE::s_num_dims);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim,
                        LEFT_OPERAND_TYPE const& left,
                        RIGHT_OPERAND_TYPE const& right)
  {
    return dim == 0 ? left.getDimSize(0) : right.getDimSize(1);
  }

  /*!
   * Evaluate operands and perform element-wise multiply
   */
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static auto multiply(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right)
      -> decltype(left.eval(tile) * right.eval(tile))
  {
    return left.eval(tile) * right.eval(tile);
  }

  /*!
   * Evaluate operands and perform element-wise multiply add
   */
  template<typename TILE_TYPE, typename ADD_OPERAND_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static auto multiply_add(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right,
      ADD_OPERAND_TYPE const& add)
      -> decltype(left.eval(tile).multiply_add(right.eval(tile),
                                               add.eval(tile)))
  {
    return left.eval(tile).multiply_add(right.eval(tile), add.eval(tile));
  }

  /*!
   * Evaluate operands and perform element-wise multiply subtract
   */
  template<typename TILE_TYPE, typename SUBTRACT_OPERAND_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static auto multiply_subtract(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right,
      SUBTRACT_OPERAND_TYPE const& subtract)
      -> decltype(left.eval(tile).multiply_subtract(right.eval(tile),
                                                    subtract.eval(tile)))
  {
    return left.eval(tile).multiply_subtract(right.eval(tile),
                                             subtract.eval(tile));
  }
};

/*!
 * Specialization that provides multiplying a scalar * tensor
 */
template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
struct MultiplyOperator<
    LEFT_OPERAND_TYPE,
    RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 0>::type>
{

  using result_type = typename RIGHT_OPERAND_TYPE::result_type;
  static constexpr camp::idx_t s_num_dims = RIGHT_OPERAND_TYPE::s_num_dims;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static void print_ast() { printf("Scale"); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim,
                        LEFT_OPERAND_TYPE const&,
                        RIGHT_OPERAND_TYPE const& right)
  {
    return right.getDimSize(dim);
  }

  /*!
   * Evaluate operands and perform scaling operation
   */
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static auto multiply(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right)
      -> decltype(right.eval(tile).scale(left.eval(tile)))
  {
    return right.eval(tile).scale(left.eval(tile));
  }

  /*!
   * Evaluate operands and perform element-wise multiply add
   */
  template<typename TILE_TYPE, typename ADD_OPERAND_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static auto multiply_add(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right,
      ADD_OPERAND_TYPE const& add)
      -> decltype(right.eval(tile).scale(left.eval(tile)) + add.eval(tile))
  {
    return right.eval(tile).scale(left.eval(tile)) + add.eval(tile);
  }

  /*!
   * Evaluate operands and perform element-wise multiply subtract
   */
  template<typename TILE_TYPE, typename SUBTRACT_OPERAND_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static auto multiply_subtract(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right,
      SUBTRACT_OPERAND_TYPE const& subtract)
      -> decltype(right.eval(tile).scale(left.eval(tile)) - subtract.eval(tile))
  {
    return right.eval(tile).scale(left.eval(tile)) - subtract.eval(tile);
  }
};

/*!
 * Specialization that provides multiplying a tensor*scalar
 */
template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
struct MultiplyOperator<
    LEFT_OPERAND_TYPE,
    RIGHT_OPERAND_TYPE,
    typename std::enable_if<RIGHT_OPERAND_TYPE::s_num_dims == 0>::type>
{

  using result_type = typename LEFT_OPERAND_TYPE::result_type;
  static constexpr camp::idx_t s_num_dims = LEFT_OPERAND_TYPE::s_num_dims;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static void print_ast() { printf("Scale"); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim,
                        LEFT_OPERAND_TYPE const& left,
                        RIGHT_OPERAND_TYPE const&)
  {
    return left.getDimSize(dim);
  }

  /*!
   * Evaluate operands and perform scaling operation
   */
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static auto multiply(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right)
      -> decltype(left.eval(tile).scale(right.eval(tile)))
  {
    return left.eval(tile).scale(right.eval(tile));
  }

  /*!
   * Evaluate operands and perform element-wise multiply add
   */
  template<typename TILE_TYPE, typename ADD_OPERAND_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static auto multiply_add(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right,
      ADD_OPERAND_TYPE const& add)
      -> decltype(left.eval(tile).scale(right.eval(tile)) + add.eval(tile))
  {
    return left.eval(tile).scale(right.eval(tile)) + add.eval(tile);
  }

  /*!
   * Evaluate operands and perform element-wise multiply subtract
   */
  template<typename TILE_TYPE, typename SUBTRACT_OPERAND_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static auto multiply_subtract(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right,
      SUBTRACT_OPERAND_TYPE const& subtract)
      -> decltype(left.eval(tile).scale(right.eval(tile)) - subtract.eval(tile))
  {
    return left.eval(tile).scale(right.eval(tile)) - subtract.eval(tile);
  }
};

/*!
 * Specialization for matrix-vector right multiplication.
 *
 * By default the A*x operator for two matrices produces a matrix-vector
 * multiplication.
 *
 * The right hand side vector is always treated as a column vector.
 *
 * The resulting vector type is inherited from the RHS
 *
 *
 */
template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
struct MultiplyOperator<
    LEFT_OPERAND_TYPE,
    RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 2 &&
                            RIGHT_OPERAND_TYPE::s_num_dims == 1>::type>
{

  using left_type  = LEFT_OPERAND_TYPE;
  using right_type = RIGHT_OPERAND_TYPE;
  using result_type =
      typename LEFT_OPERAND_TYPE::result_type::column_vector_type;
  static constexpr camp::idx_t s_num_dims = 1;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static void print_ast() { printf("Matrx*Vector"); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim,
                        LEFT_OPERAND_TYPE const&,
                        RIGHT_OPERAND_TYPE const& right)
  {
    return dim == 0 ? right.getDimSize(0) : 0;
  }

  /*!
   * Evaluate operands and perform element-wise multiply
   */
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static result_type multiply(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right)
  {

    // clear result
    result_type result(0);

    // multiply left and right into result
    multiply_into_result(result, tile, left, right);

    return result;
  }

  template<typename TILE_TYPE, typename ADD_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static result_type multiply_add(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right,
      ADD_TYPE const& add)
  {

    // evaluate add into result
    result_type result = add.eval(tile);

    // multiply left and right into result
    multiply_into_result(result, tile, left, right);

    return result;
  }

private:
  template<typename STORAGE, typename TILE_TYPE, typename INDEX = void>
  struct MultiplyBridge;

  template<typename STORAGE, typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static void multiply_into_result(
      STORAGE& result,
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& et_left,
      RIGHT_OPERAND_TYPE const& et_right)
  {
    // using LHS_STORAGE = typename LEFT_OPERAND_TYPE::result_type;

    // get tile size from matrix type
    auto tile_size = left_type::result_type::s_dim_elem(1);
    auto k_size    = et_left.getDimSize(1);
    // TODO: check that left and right are compatible
    // m_left.getDimSize(1) == m_right.getDimSize(0)
    // how do we provide checking for this kind of error?

    // tile over row of left and column of right
    auto left_tile =
        LEFT_OPERAND_TYPE::result_type::s_get_default_tile().nonstatic();
    left_tile.m_begin[0] = tile.m_begin[0];
    left_tile.m_size[0]  = tile.m_size[0];
    left_tile.m_size[1]  = tile_size;

    using RightType = typename TILE_TYPE::nonstatic_self_type;

    RightType right_tile = tile;
    right_tile.m_size[0] = tile_size;

    // Do full tiles in k
    decltype(k_size) k = 0;
    for (; k + tile_size <= k_size; k += tile_size)
    {

      // evaluate both sides of operator
      left_tile.m_begin[1] = k;
      auto left            = et_left.eval(left_tile);

      right_tile.m_begin[0] = k;
      auto right            = et_right.eval(right_tile);

      // accumulate product
      result = left.right_multiply_vector_accumulate(right, result);
    }
    // remainder tile in k
    if (k < k_size)
    {
      auto& left_part_tile      = make_tensor_tile_partial(left_tile);
      left_part_tile.m_begin[1] = k;
      left_part_tile.m_size[1]  = k_size - k;
      auto left                 = et_left.eval(left_part_tile);

      auto& right_part_tile      = make_tensor_tile_partial(right_tile);
      right_part_tile.m_begin[0] = k;
      right_part_tile.m_size[0]  = k_size - k;
      auto right                 = et_right.eval(right_part_tile);

      // accumulate product of partial tile
      result = left.right_multiply_vector_accumulate(right, result);
    }
  }

  template<typename T>
  struct Diag
  {
    static_assert(!std::is_same<T, void>::value, "diag");
  };

  template<typename I, TensorTileSize TTS, typename B, typename S>
  struct Diag<StaticTensorTile<I, TTS, B, S>>
  {
    static_assert(std::is_same<I, void>::value, "diag");
  };

  template<typename STORAGE, typename TILE_TYPE, typename INDEX>
  struct MultiplyBridge
  {

    Diag<TILE_TYPE> diag;

    RAJA_INLINE

    RAJA_HOST_DEVICE
    static void multiply_into_result(STORAGE& result,
                                     TILE_TYPE const& tile,
                                     LEFT_OPERAND_TYPE const& et_left,
                                     RIGHT_OPERAND_TYPE const& et_right)
    {
      // using LHS_STORAGE = typename LEFT_OPERAND_TYPE::result_type;

      // get tile size from matrix type
      auto tile_size = left_type::result_type::s_dim_elem(1);
      auto k_size    = et_left.getDimSize(1);
      // TODO: check that left and right are compatible
      // m_left.getDimSize(1) == m_right.getDimSize(0)
      // how do we provide checking for this kind of error?

      // tile over row of left and column of right
      auto left_tile =
          LEFT_OPERAND_TYPE::result_type::s_get_default_tile().nonstatic();
      left_tile.m_begin[0] = tile.m_begin[0];
      left_tile.m_size[0]  = tile.m_size[0];
      left_tile.m_size[1]  = tile_size;

      using RightType = typename TILE_TYPE::nonstatic_self_type;

      RightType right_tile = tile;
      right_tile.m_size[0] = tile_size;

      // Do full tiles in k
      decltype(k_size) k = 0;
      for (; k + tile_size <= k_size; k += tile_size)
      {

        // evaluate both sides of operator
        left_tile.m_begin[1] = k;
        auto left            = et_left.eval(left_tile);

        right_tile.m_begin[0] = k;
        auto right            = et_right.eval(right_tile);

        // accumulate product
        result = left.right_multiply_vector_accumulate(right, result);
      }
      // remainder tile in k
      if (k < k_size)
      {
        auto& left_part_tile      = make_tensor_tile_partial(left_tile);
        left_part_tile.m_begin[1] = k;
        left_part_tile.m_size[1]  = k_size - k;
        auto left                 = et_left.eval(left_part_tile);

        auto& right_part_tile      = make_tensor_tile_partial(right_tile);
        right_part_tile.m_begin[0] = k;
        right_part_tile.m_size[0]  = k_size - k;
        auto right                 = et_right.eval(right_part_tile);

        // accumulate product of partial tile
        result = left.right_multiply_vector_accumulate(right, result);
      }
    }
  };

  template<size_t INDEX,
           typename STORAGE,
           typename INDEX_TYPE,
           TensorTileSize TENSOR_SIZE,
           INDEX_TYPE Begin0,
           INDEX_TYPE... BeginTail,
           INDEX_TYPE Size0,
           INDEX_TYPE... SizeTail>
  struct MultiplyBridge<
      STORAGE,
      StaticTensorTile<INDEX_TYPE,
                       TENSOR_SIZE,
                       camp::int_seq<INDEX_TYPE, Begin0, BeginTail...>,
                       camp::int_seq<INDEX_TYPE, Size0, SizeTail...>>,
      camp::integral_constant<size_t, INDEX>>
  {

    using TileType =
        StaticTensorTile<INDEX_TYPE,
                         TENSOR_SIZE,
                         camp::int_seq<INDEX_TYPE, Begin0, BeginTail...>,
                         camp::int_seq<INDEX_TYPE, Size0, SizeTail...>>;

    RAJA_INLINE

    RAJA_HOST_DEVICE
    static void multiply_into_result(STORAGE& result,
                                     TileType const& tile,
                                     LEFT_OPERAND_TYPE const& et_left,
                                     RIGHT_OPERAND_TYPE const& et_right)
    {

      // get tile size from matrix type
      const auto tile_size = left_type::result_type::s_dim_elem(1);
      const auto k_size    = et_left.getDimSize(1);

      auto const offset = INDEX * tile_size;

      if ((offset + tile_size) <= k_size)
      {

        using LeftType =
            StaticTensorTile<INDEX_TYPE, TENSOR_SIZE,
                             camp::int_seq<INDEX_TYPE, Begin0, offset>,
                             camp::int_seq<INDEX_TYPE, Size0, tile_size>>;
        // evaluate both sides of operator
        auto left = et_left.eval(LeftType());

        using RightType =
            StaticTensorTile<INDEX_TYPE, TENSOR_SIZE,
                             camp::int_seq<INDEX_TYPE, offset>,
                             camp::int_seq<INDEX_TYPE, tile_size>>;

        auto right = et_right.eval(RightType());

        // accumulate product
        auto temp = left.right_multiply_vector_accumulate(right, result);
        MultiplyBridge<STORAGE, TileType,
                       camp::integral_constant<size_t, INDEX - 1>>::
            multiply_into_result(result, tile, et_left, et_right);
        result += temp;
      }
      else
      {

        using LeftType =
            StaticTensorTile<INDEX_TYPE, TENSOR_PARTIAL,
                             camp::int_seq<INDEX_TYPE, Begin0, offset>,
                             camp::int_seq<INDEX_TYPE, Size0, k_size - offset>>;
        auto left = et_left.eval(LeftType());

        using RightType =
            StaticTensorTile<INDEX_TYPE, TENSOR_PARTIAL,
                             camp::int_seq<INDEX_TYPE, offset>,
                             camp::int_seq<INDEX_TYPE, k_size - offset>>;
        auto right = et_right.eval(RightType());

        // accumulate product of partial tile
        result = left.right_multiply_vector_accumulate(right, result);
      }
    }
  };

  template<typename STORAGE,
           typename INDEX_TYPE,
           TensorTileSize TENSOR_SIZE,
           INDEX_TYPE Begin0,
           INDEX_TYPE... BeginTail,
           INDEX_TYPE Size0,
           INDEX_TYPE... SizeTail>
  struct MultiplyBridge<
      STORAGE,
      StaticTensorTile<INDEX_TYPE,
                       TENSOR_SIZE,
                       camp::int_seq<INDEX_TYPE, Begin0, BeginTail...>,
                       camp::int_seq<INDEX_TYPE, Size0, SizeTail...>>,
      camp::integral_constant<size_t, 0>>
  {

    using TileType =
        StaticTensorTile<INDEX_TYPE,
                         TENSOR_SIZE,
                         camp::int_seq<INDEX_TYPE, Begin0, BeginTail...>,
                         camp::int_seq<INDEX_TYPE, Size0, SizeTail...>>;

    RAJA_INLINE

    RAJA_HOST_DEVICE
    static void multiply_into_result(STORAGE& result,
                                     TileType const&,
                                     LEFT_OPERAND_TYPE const& et_left,
                                     RIGHT_OPERAND_TYPE const& et_right)
    {

      // get tile size from matrix type
      const auto tile_size = left_type::result_type::s_dim_elem(1);
      const auto k_size    = et_left.getDimSize(1);

      auto const offset = 0;

      if ((offset + tile_size) <= k_size)
      {

        using LeftType =
            StaticTensorTile<INDEX_TYPE, TENSOR_SIZE,
                             camp::int_seq<INDEX_TYPE, Begin0, offset>,
                             camp::int_seq<INDEX_TYPE, Size0, tile_size>>;
        // evaluate both sides of operator
        auto left = et_left.eval(LeftType());

        using RightType =
            StaticTensorTile<INDEX_TYPE, TENSOR_SIZE,
                             camp::int_seq<INDEX_TYPE, offset>,
                             camp::int_seq<INDEX_TYPE, tile_size>>;

        auto right = et_right.eval(RightType());

        // accumulate product
        auto temp = left.right_multiply_vector_accumulate(right, result);
        result += temp;
      }
      else
      {

        using LeftType =
            StaticTensorTile<INDEX_TYPE, TENSOR_PARTIAL,
                             camp::int_seq<INDEX_TYPE, Begin0, offset>,
                             camp::int_seq<INDEX_TYPE, Size0, k_size - offset>>;
        auto left = et_left.eval(LeftType());

        using RightType =
            StaticTensorTile<INDEX_TYPE, TENSOR_PARTIAL,
                             camp::int_seq<INDEX_TYPE, offset>,
                             camp::int_seq<INDEX_TYPE, k_size - offset>>;
        auto right = et_right.eval(RightType());

        // accumulate product of partial tile
        result = left.right_multiply_vector_accumulate(right, result);
      }
    }
  };

  template<typename STORAGE,
           typename INDEX_TYPE,
           TensorTileSize TENSOR_SIZE,
           INDEX_TYPE Begin0,
           INDEX_TYPE... BeginTail,
           INDEX_TYPE Size0,
           INDEX_TYPE... SizeTail>
  struct MultiplyBridge<
      STORAGE,
      StaticTensorTile<INDEX_TYPE,
                       TENSOR_SIZE,
                       camp::int_seq<INDEX_TYPE, Begin0, BeginTail...>,
                       camp::int_seq<INDEX_TYPE, Size0, SizeTail...>>,
      void>
  {

    using TileType =
        StaticTensorTile<INDEX_TYPE,
                         TENSOR_SIZE,
                         camp::int_seq<INDEX_TYPE, Begin0, BeginTail...>,
                         camp::int_seq<INDEX_TYPE, Size0, SizeTail...>>;

    RAJA_INLINE

    RAJA_HOST_DEVICE
    static void multiply_into_result(STORAGE& result,
                                     TileType const& tile,
                                     LEFT_OPERAND_TYPE const& et_left,
                                     RIGHT_OPERAND_TYPE const& et_right)
    {

      const auto tile_size = left_type::result_type::s_dim_elem(1);
      const auto k_size    = et_left.getDimSize(1);
      const size_t iter_count =
          (k_size / tile_size) + ((k_size % tile_size != 0) ? 1 : 0);

      MultiplyBridge<STORAGE, TileType,
                     camp::integral_constant<size_t, iter_count>>::
          multiply_into_result(result, tile, et_left, et_right);
    }
  };
};


template<typename LEFT_OPERAND_TYPE,
         typename RIGHT_OPERAND_TYPE,
         typename ADD_OPERAND_TYPE>
class TensorMultiplyAdd;

/*!
 * Specialization for vector*matrix left multiplication.
 *
 * By default the x'*A operator for two matrices produces a vector-matrix
 * multiplication.
 *
 * The left hand side vector is always treated as a row vector.
 *
 * The resulting vector type is inherited from the LHS
 *
 *
 */
template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
struct MultiplyOperator<
    LEFT_OPERAND_TYPE,
    RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 1 &&
                            RIGHT_OPERAND_TYPE::s_num_dims == 2>::type>
{

  using left_type   = LEFT_OPERAND_TYPE;
  using right_type  = RIGHT_OPERAND_TYPE;
  using result_type = typename RIGHT_OPERAND_TYPE::result_type::row_vector_type;
  static constexpr camp::idx_t s_num_dims = 1;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static void print_ast() { printf("Vector*Matrix"); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim,
                        LEFT_OPERAND_TYPE const& left,
                        RIGHT_OPERAND_TYPE const&)
  {
    return dim == 0 ? left.getDimSize(0) : 0;
  }

  /*!
   * Evaluate operands and perform element-wise multiply
   */
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static result_type multiply(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right)
  {
    // clear result
    result_type result(0);

    // multiply left and right into result
    multiply_into_result(result, tile, left, right);

    return result;
  }

  template<typename TILE_TYPE, typename ADD_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static result_type multiply_add(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right,
      ADD_TYPE const& add)
  {
    // evaluate add into result
    result_type result = add.eval(tile);

    // multiply left and right into result
    multiply_into_result(result, tile, left, right);

    return result;
  }

private:
  template<typename STORAGE, typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static void multiply_into_result(
      STORAGE& result,
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& et_left,
      RIGHT_OPERAND_TYPE const& et_right)
  {
    // get tile size from matrix type
    auto tile_size = right_type::result_type::s_dim_elem(0);
    auto k_size    = et_right.getDimSize(0);


    // TODO: check that left and right are compatible
    // m_left.getDimSize(1) == m_right.getDimSize(0)
    // how do we provide checking for this kind of error?

    // tile over row of left and column of right
    auto right_tile =
        RIGHT_OPERAND_TYPE::result_type::s_get_default_tile().nonstatic();
    right_tile.m_begin[1] = tile.m_begin[0];
    right_tile.m_size[1]  = tile.m_size[0];
    right_tile.m_size[0]  = tile_size;

    TILE_TYPE left_tile = tile;
    left_tile.m_size[0] = tile_size;


    // Do full tiles in k
    decltype(k_size) k = 0;
    for (; k + tile_size <= k_size; k += tile_size)
    {

      // evaluate both sides of operator
      right_tile.m_begin[0] = k;
      auto right            = et_right.eval(right_tile);

      left_tile.m_begin[0] = k;
      auto left            = et_left.eval(left_tile);

      // accumulate product
      result = right.left_multiply_vector_accumulate(left, result);
    }
    // remainder tile in k
    if (k < k_size)
    {
      auto& right_part_tile      = make_tensor_tile_partial(right_tile);
      right_part_tile.m_begin[0] = k;
      right_part_tile.m_size[0]  = k_size - k;
      auto right                 = et_right.eval(right_part_tile);

      auto& left_part_tile      = make_tensor_tile_partial(left_tile);
      left_part_tile.m_begin[0] = k;
      left_part_tile.m_size[0]  = k_size - k;
      auto left                 = et_left.eval(left_part_tile);

      // compute product into x of partial tile
      result = right.left_multiply_vector_accumulate(left, result);
    }
  }
};

/*!
 * Specialization for matrix-matrix multiplication for TensorRegisters
 *
 * By default the A*B operator for two matrices produces a matrix-matrix
 * multiplication.
 *
 */
template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
struct MultiplyOperator<
    LEFT_OPERAND_TYPE,
    RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 2 &&
                            RIGHT_OPERAND_TYPE::s_num_dims == 2>::type>
{

  using left_type   = LEFT_OPERAND_TYPE;
  using right_type  = RIGHT_OPERAND_TYPE;
  using result_type = typename LEFT_OPERAND_TYPE::result_type::product_type;
  static constexpr camp::idx_t s_num_dims = 2;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static void print_ast() { printf("Matrx*Matrix"); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim,
                        LEFT_OPERAND_TYPE const& left,
                        RIGHT_OPERAND_TYPE const& right)
  {
    return dim == 0 ? left.getDimSize(0) : right.getDimSize(1);
  }

  /*!
   * Evaluate operands and perform element-wise multiply
   */
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static result_type multiply(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right)
  {

    /*
     *
     * For TensorRegister:
     *
     *   Return's a register containing product of left and right operands
     *
     * For TensorBlock:
     *
     *  Return's an ET TensorLiteral containing the left and right operrands
     *
     *  OR
     *
     *  Returns an ET multiply
     *
     */
    // create zeroed temporary
    result_type result;
    result.broadcast(0);

    // multiply left and right operands into temporary
    multiply_into_result(result, tile, left, right);

    return result;
  }

  template<typename TILE_TYPE, typename ADD_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static result_type multiply_add(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right,
      ADD_TYPE const& add)
  {

    // start accumulator with addition term
    result_type result = add.eval(tile);

    multiply_into_result(result, tile, left, right);

    return result;
  }

private:
  template<typename STORAGE, typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static void multiply_into_result(
      STORAGE& result,
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& et_left,
      RIGHT_OPERAND_TYPE const& et_right)
  {
    // get tile size from matrix type
    using right_tensor_type = typename right_type::result_type;
    auto tile_size          = right_tensor_type::s_dim_elem(0);
    auto k_size             = et_left.getDimSize(1);

    // TODO: check that left and right are compatible
    // m_left.getDimSize(1) == m_right.getDimSize(0)
    // how do we provide checking for this kind of error?

    // tile over row of left and column of right
    TILE_TYPE left_tile = tile;
    left_tile.m_size[1] = tile_size;
    auto left_begin     = et_left.getDimBegin(1);

    TILE_TYPE right_tile = tile;
    right_tile.m_size[0] = tile_size;
    auto right_begin     = et_right.getDimBegin(0);


    // Do full tiles in k
    decltype(k_size) k = 0;
    for (; k + tile_size <= k_size; k += tile_size)
    {

      // evaluate both sides of operator
      left_tile.m_begin[1] = k + left_begin;
      auto left            = et_left.eval(left_tile);

      right_tile.m_begin[0] = k + right_begin;
      auto right            = et_right.eval(right_tile);

      // accumulate product
      left.matrix_multiply_accumulate(result, right);
    }
    // remainder tile in k
    if (k < k_size)
    {

      auto& left_part_tile      = make_tensor_tile_partial(left_tile);
      left_part_tile.m_begin[1] = k + left_begin;
      left_part_tile.m_size[1]  = k_size - k;
      auto left                 = et_left.eval(left_part_tile);

      auto& right_part_tile      = make_tensor_tile_partial(right_tile);
      right_part_tile.m_begin[0] = k + right_begin;
      right_part_tile.m_size[0]  = k_size - k;
      auto right                 = et_right.eval(right_part_tile);

      // accumulate product
      left.matrix_multiply_accumulate(result, right);
    }
  }
};

template<typename OPERAND_TYPE, typename TILE_TYPE>
class RestrictExtents
    : public TensorExpressionBase<RestrictExtents<OPERAND_TYPE, TILE_TYPE>>
{
public:
  using self_type    = RestrictExtents<OPERAND_TYPE, TILE_TYPE>;
  using operand_type = OPERAND_TYPE;
  using result_type  = typename OPERAND_TYPE::result_type;
  using index_type   = typename TILE_TYPE::index_type;
  using tile_type    = TILE_TYPE;
  static constexpr camp::idx_t s_num_dims = OPERAND_TYPE::s_num_dims;

private:
  operand_type m_operand;
  tile_type m_tile;

public:
  RAJA_INLINE

  RAJA_HOST_DEVICE
  RestrictExtents(operand_type const& operand, tile_type const& tile)
      : m_operand {operand},
        m_tile {tile}
  {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr index_type getDimSize(index_type dim) const
  {
    return m_tile.m_size[dim];
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  constexpr index_type getDimBegin(camp::idx_t dim) const
  {
    return m_tile.m_begin[dim];
  }

  template<typename TILE_TYPE2>
  RAJA_INLINE RAJA_HOST_DEVICE auto eval(TILE_TYPE2 const& tile) const
      -> decltype(m_operand.eval(tile))
  {
    return m_operand.eval(tile);
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  void print_ast() const
  {
    printf("RestrictExtents(");
    m_operand.print_ast();
    printf(")");
  }
};

template<typename OPERAND, typename TILE>
RestrictExtents<OPERAND, TILE> restrictExtents(OPERAND const& operand,
                                               TILE const& tile)
{
  using tile_type = typename OPERAND::tile_type;
  tile_type new_tile;
  new_tile.copy(tile);
  return RestrictExtents<OPERAND, TILE>(operand, new_tile);
}

/*!
 * Specialization for matrix-matrix multiplication for TensorBlocks
 *
 * By default the A*B operator for two matrices produces a matrix-matrix
 * multiplication.
 *
 */

template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
struct MultiplyOperator<
    LEFT_OPERAND_TYPE,
    RIGHT_OPERAND_TYPE,
    typename std::enable_if<
        std::is_base_of<TensorBlockConcreteBase,
                        typename RIGHT_OPERAND_TYPE::tensor_type>::value &&
        LEFT_OPERAND_TYPE::s_num_dims == 2 &&
        RIGHT_OPERAND_TYPE::s_num_dims == 2>::type>
{
  using left_type   = LEFT_OPERAND_TYPE;
  using right_type  = RIGHT_OPERAND_TYPE;
  using result_type = typename LEFT_OPERAND_TYPE::result_type::product_type;
  static constexpr camp::idx_t s_num_dims = 2;

  //      static_assert(LEFT_OPERAND_TYPE::s_num_dims == 1, "WHAOO");
  //      static_assert(! std::is_base_of<TensorBlockConcreteBase, typename
  //      RIGHT_OPERAND_TYPE::tensor_type>::value, "MATCH");


  // This tensor type is a TensorBlock of some kind
  using tensor_type = typename RIGHT_OPERAND_TYPE::tensor_type;

  // Get the storage type from the TensorBlock
  using storage_type = typename tensor_type::storage_type;

  // Create a BlockLiteral that uses the TensorBlock's indicated storage
  // and has an eval() that produces the TensorBlock's register type
  using block_literal =
      BlockLiteral<storage_type, typename tensor_type::register_type>;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static void print_ast() { printf("Matrx*Matrix"); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static int getDimSize(int dim,
                        LEFT_OPERAND_TYPE const& left,
                        RIGHT_OPERAND_TYPE const& right)
  {
    return dim == 0 ? left.getDimSize(0) : right.getDimSize(1);
  }

  /*!
   * Evaluate operands and perform element-wise multiply
   */
  template<typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static block_literal multiply(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const&,
      RIGHT_OPERAND_TYPE const&)  //->
                                  /// decltype(TensorMultiply<decltype(left.eval(tile)),
                                  /// decltype(right.eval(tile))>(left.eval(tile),
                                  /// right.eval(tile)))
  {

    /*
     * First pass:  just return a Multiply ET that evaluates the block
     * with underlying TensorRegisters
     *
     *
     * Second pass: we want to return a TensorLiteral ET node with the
     * matrix product already evaluated.?
     *
     * What we really care about is improving the data reuse: so perhaps
     * returning a Multiply ET node with TensorLiteral nodes for each
     * of the operands
     *
     */
    // create a BlockLiteral
    block_literal result(tile);

    // evaluate the block-wise product into result

    // return TensorMultiply<decltype(left.eval(tile)),
    // decltype(right.eval(tile))>(left.eval(tile), right.eval(tile));

    // return the BlockLiterat ET
    return result;
  }

  template<typename TILE_TYPE, typename ADD_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static block_literal multiply_add(
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& left,
      RIGHT_OPERAND_TYPE const& right,
      ADD_TYPE const&
          add)  //->
                // decltype(TensorMultiplyAdd<decltype(left.eval(tile)),
                // decltype(right.eval(tile)),
                // decltype(add.eval(tile))>(left.eval(tile),
                // right.eval(tile), add.eval(tile)))
  {
    /*
     * First pass:  we want to return a BlockLiteral ET node with the
     * matrix product already evaluated.  We do this by creating
     * a LoadStore node wrapping the BlockLiteral, and evaluating it as
     * a sub-expression.
     *
     * What we really care about is improving the data reuse: so perhaps
     * returning a Multiply ET node with TensorLiteral nodes for each
     * of the operands
     *
     */

    // create a BlockLiteral
    using block_tile_type = typename block_literal::tile_type;
    block_tile_type block_tile;
    block_tile.copy(tile);
    block_literal result(block_tile);

    using ref_type        = typename block_literal::ref_type;
    using load_store_type = TensorLoadStore<tensor_type, ref_type>;

    // initialize the result with our addition term
    auto result_et = load_store_type(result.get_ref()).eval(tile);
    result_et      = add.eval(tile);

    // return TensorMultiplyAdd<decltype(left.eval(tile)),
    // decltype(right.eval(tile)), decltype(add.eval(tile))>(left.eval(tile),
    // right.eval(tile), add.eval(tile));

    //          multiply_into_result(result_et, tile, restrictExtents(left,
    //          tile), restrictExtents(right, tile));
    multiply_into_result(result_et, tile, left, right);

    // return the BlockLiterat ET
    return result;
  }

private:
  template<typename STORAGE, typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE static void multiply_into_result(
      STORAGE& result,
      TILE_TYPE const& tile,
      LEFT_OPERAND_TYPE const& et_left,
      RIGHT_OPERAND_TYPE const& et_right)
  {

    // get tile size from matrix type
    auto tile_size = result_type::s_dim_elem(1);
    auto k_size    = et_left.getDimSize(1);

    // TODO: check that left and right are compatible
    // m_left.getDimSize(1) == m_right.getDimSize(0)
    // how do we provide checking for this kind of error?

    // tile over row of left and column of right
    TILE_TYPE left_tile = tile;
    left_tile.m_size[1] = tile_size;
    auto left_begin     = et_left.getDimBegin(1);

    TILE_TYPE right_tile = tile;
    right_tile.m_size[0] = tile_size;
    auto right_begin     = et_right.getDimBegin(0);


    // Do full tiles in k
    decltype(k_size) k = 0;
    for (; k + tile_size <= k_size; k += tile_size)
    {


      // evaluate both sides of operator
      left_tile.m_begin[1] = k + left_begin;
      auto left            = et_left.eval(left_tile);

      right_tile.m_begin[0] = k + right_begin;
      auto right            = et_right.eval(right_tile);

      // accumulate product
      // left.matrix_multiply_accumulate(result, right);
      result +=
          restrictExtents(left, left_tile) * restrictExtents(right, right_tile);
    }
    // remainder tile in k
    if (k < k_size)
    {

      auto& left_part_tile      = make_tensor_tile_partial(left_tile);
      left_part_tile.m_begin[1] = k + left_begin;
      left_part_tile.m_size[1]  = k_size - k;
      auto left                 = et_left.eval(left_part_tile);

      auto& right_part_tile      = make_tensor_tile_partial(right_tile);
      right_part_tile.m_begin[0] = k + right_begin;
      right_part_tile.m_size[0]  = k_size - k;
      auto right                 = et_right.eval(right_part_tile);

      // accumulate product
      // left.matrix_multiply_accumulate(result, right);
      result += restrictExtents(left, left_part_tile) *
                restrictExtents(right, right_part_tile);
    }
  }
};


}  // namespace ET

}  // namespace expt
}  // namespace internal

}  // namespace RAJA


#endif
