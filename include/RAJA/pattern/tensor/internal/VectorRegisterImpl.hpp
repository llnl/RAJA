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

#ifndef RAJA_pattern_tensor_VectorRegisterImpl_HPP
#define RAJA_pattern_tensor_VectorRegisterImpl_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "camp/camp.hpp"
#include "RAJA/pattern/tensor/internal/TensorRegisterBase.hpp"
#include "RAJA/pattern/tensor/stats.hpp"
#include "RAJA/util/BitMask.hpp"

namespace RAJA
{

namespace expt
{

/*!
 * This provides a Tensor specialization for vectors
 */
template<typename REGISTER_POLICY, typename T, camp::idx_t SIZE>
class TensorRegister<REGISTER_POLICY,
                     T,
                     RAJA::expt::VectorLayout,
                     camp::idx_seq<SIZE>>
    : public internal::expt::TensorRegisterBase<
          RAJA::expt::TensorRegister<REGISTER_POLICY,
                                     T,
                                     RAJA::expt::VectorLayout,
                                     camp::idx_seq<SIZE>>>
{
public:
  using self_type = TensorRegister<REGISTER_POLICY,
                                   T,
                                   RAJA::expt::VectorLayout,
                                   camp::idx_seq<SIZE>>;
  using base_type = internal::expt::TensorRegisterBase<
      RAJA::expt::TensorRegister<REGISTER_POLICY,
                                 T,
                                 RAJA::expt::VectorLayout,
                                 camp::idx_seq<SIZE>>>;
  using element_type  = camp::decay<T>;
  using layout_type   = TensorLayout<0>;
  using register_type = Register<T, REGISTER_POLICY>;

  static constexpr camp::idx_t s_num_elem = SIZE;

  using int_element_type =
      typename register_type::int_vector_type::element_type;
  using int_vector_type = TensorRegister<REGISTER_POLICY,
                                         int_element_type,
                                         RAJA::expt::VectorLayout,
                                         camp::idx_seq<SIZE>>;

private:
  static constexpr camp::idx_t s_register_num_elem = register_type::s_num_elem;

  static constexpr camp::idx_t s_num_full_registers =
      s_num_elem / s_register_num_elem;

  static constexpr camp::idx_t s_num_partial_lanes =
      s_num_elem % s_register_num_elem;

  static constexpr camp::idx_t s_num_registers = (s_num_partial_lanes > 0)
                                                     ? s_num_full_registers + 1
                                                     : s_num_full_registers;

  using log_base2_t = RAJA::LogBase2<s_register_num_elem>;

  static constexpr camp::idx_t s_shift_per_register = log_base2_t::value;

  static constexpr camp::idx_t s_mask_per_register =
      (1 << log_base2_t::value) - 1;

  // Offset of last regiser in m_registers
  static constexpr camp::idx_t s_final_register = s_num_partial_lanes == 0
                                                      ? s_num_full_registers - 1
                                                      : s_num_full_registers;

  template<typename IDX>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr static auto to_register(IDX i) -> IDX
  {
    return i >> IDX(s_shift_per_register);
  }

  template<typename IDX>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr static auto to_lane(IDX i) -> IDX
  {
    return i & IDX(s_mask_per_register);
  }

  using base_type::m_registers;

public:
  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr TensorRegister() {}

  RAJA_HOST_DEVICE

  RAJA_INLINE
  TensorRegister(element_type c) { this->broadcast(c); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  TensorRegister(self_type const& c) : base_type(c) {}

  /*
   * Overload for:    assignment of ET to a RAJA::expt::TensorRegister
   */
  template<typename RHS,
           typename std::enable_if<
               std::is_base_of<
                   RAJA::internal::expt::ET::TensorExpressionConcreteBase,
                   RHS>::value,
               bool>::type = true>
  RAJA_INLINE RAJA_HOST_DEVICE TensorRegister(RHS const& rhs)
  {
    // evaluate a single tile of the ET, storing in this
    // RAJA::expt::TensorRegister
    *this = rhs.eval(base_type::s_get_default_tile());
  }

  template<typename... REGS>
  explicit RAJA_HOST_DEVICE RAJA_INLINE TensorRegister(register_type reg0,
                                                       REGS const&... regs)
      : base_type(reg0, regs...)
  {}

  RAJA_HOST_DEVICE

  RAJA_INLINE
  static constexpr bool is_root() { return register_type::is_root(); }

  /*!
   * Returns true if the underlying data packed for a given tensor ref
   *
   * This is true if either:
   *   It's column major and the rows are stride one
   *   It's row major and the columns are stride one
   */
  template<camp::idx_t STRIDE_ONE_DIM>
  RAJA_HOST_DEVICE RAJA_INLINE static constexpr bool is_ref_packed()
  {
    return STRIDE_ONE_DIM == 0;
  }

  /*!
   * Gets the maximum size of matrix along specified dimension
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  static constexpr camp::idx_t s_dim_elem(camp::idx_t dim)
  {
    return dim == 0 ? s_num_elem : 0;
  }

  /*!
   * @brief Set entire vector to a single scalar value
   * @param value Value to set all vector elements to
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& operator=(element_type value)
  {
    this->broadcast(value);
    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& operator=(self_type const& c) { return this->copy(c); }

  /*!
   * Provide left vector-matrix multiply for operator* between
   * this vector and a matrix
   */
  template<typename T2, typename L, typename RP>
  self_type operator*(SquareMatrixRegister<T2, L, RP> const& y) const
  {
    return y.left_vector_multiply(*this);
  }


  template<typename REF_TYPE>
  struct RefBridge;

  template<typename REF_TYPE>
  RAJA_HOST_DEVICE RAJA_INLINE self_type& load_ref(REF_TYPE const& ref)
  {
    RefBridge<REF_TYPE>::load_ref(*this, ref);
    return *this;
  }

  template<typename REF_TYPE>
  RAJA_HOST_DEVICE RAJA_INLINE self_type const& store_ref(REF_TYPE& ref) const
  {
    RefBridge<REF_TYPE>::store_ref(*this, ref);
    return *this;
  }

  template<typename POINTER_TYPE,
           typename INDEX_TYPE,
           RAJA::internal::expt::TensorTileSize TENSOR_SIZE,
           camp::idx_t STRIDE_ONE_DIM>
  struct RefBridge<
      RAJA::internal::expt::
          TensorRef<POINTER_TYPE, INDEX_TYPE, TENSOR_SIZE, 1, STRIDE_ONE_DIM>>
  {

    using RefType = RAJA::internal::expt::
        TensorRef<POINTER_TYPE, INDEX_TYPE, TENSOR_SIZE, 1, STRIDE_ONE_DIM>;

    /*!
     * @brief Performs load specified by TensorRef object.
     */
    RAJA_HOST_DEVICE

    RAJA_INLINE
    static void load_ref(self_type& self, RefType const& ref)
    {

      auto ptr = ref.m_pointer + ref.m_tile.m_begin[0] * ref.m_stride[0];

      // check for packed data
      if (STRIDE_ONE_DIM == 0)
      {
        // full vector?
        if (TENSOR_SIZE == RAJA::internal::expt::TENSOR_FULL)
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_packed++;
#endif
          self.load_packed(ptr);
        }
        // partial
        else
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_packed_n++;
#endif
          self.load_packed_n(ptr, ref.m_tile.m_size[0]);
        }
      }
      // strided data
      else
      {
        // full vector?
        if (TENSOR_SIZE == RAJA::internal::expt::TENSOR_FULL)
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_strided++;
#endif
          self.load_strided(ptr, ref.m_stride[0]);
        }
        // partial
        else
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_strided_n++;
#endif
          self.load_strided_n(ptr, ref.m_stride[0], ref.m_tile.m_size[0]);
        }
      }
    }

    /*!
     * @brief Performs load specified by TensorRef object.
     */
    RAJA_HOST_DEVICE

    RAJA_INLINE
    static void store_ref(self_type const& self, RefType& ref)
    {

      auto ptr = ref.m_pointer + ref.m_tile.m_begin[0] * ref.m_stride[0];

      // check for packed data
      if (STRIDE_ONE_DIM == 0)
      {
        // full vector?
        if (TENSOR_SIZE == RAJA::internal::expt::TENSOR_FULL)
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_packed++;
#endif
          self.store_packed(ptr);
        }
        // partial
        else
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_packed_n++;
#endif
          self.store_packed_n(ptr, ref.m_tile.m_size[0]);
        }
      }
      // strided data
      else
      {
        // full vector?
        if (TENSOR_SIZE == RAJA::internal::expt::TENSOR_FULL)
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_strided++;
#endif
          self.store_strided(ptr, ref.m_stride[0]);
        }
        // partial
        else
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_strided_n++;
#endif
          self.store_strided_n(ptr, ref.m_stride[0], ref.m_tile.m_size[0]);
        }
      }
    }
  };

  template<typename POINTER_TYPE,
           typename INDEX_TYPE,
           RAJA::internal::expt::TensorTileSize TENSOR_SIZE,
           INDEX_TYPE STRIDE_VALUE,
           INDEX_TYPE BEGIN_VALUE,
           INDEX_TYPE SIZE_VALUE,
           camp::idx_t STRIDE_ONE_DIM>
  struct RefBridge<RAJA::internal::expt::StaticTensorRef<
      POINTER_TYPE,
      INDEX_TYPE,
      TENSOR_SIZE,
      camp::int_seq<INDEX_TYPE, STRIDE_VALUE>,
      camp::int_seq<INDEX_TYPE, BEGIN_VALUE>,
      camp::int_seq<INDEX_TYPE, SIZE_VALUE>,
      STRIDE_ONE_DIM>>
  {

    using RefType = RAJA::internal::expt::StaticTensorRef<
        POINTER_TYPE,
        INDEX_TYPE,
        TENSOR_SIZE,
        camp::int_seq<INDEX_TYPE, STRIDE_VALUE>,
        camp::int_seq<INDEX_TYPE, BEGIN_VALUE>,
        camp::int_seq<INDEX_TYPE, SIZE_VALUE>,
        STRIDE_ONE_DIM>;

    /*!
     * @brief Performs load specified by StaticTensorRef object.
     */
    RAJA_HOST_DEVICE

    RAJA_INLINE
    static void load_ref(self_type& self, RefType const& ref)
    {

      auto ptr = ref.m_pointer + ref.m_tile.m_begin[0] * ref.m_stride[0];

      // check for packed data
      if (STRIDE_ONE_DIM == 0)
      {
        // full vector?
        if (TENSOR_SIZE == RAJA::internal::expt::TENSOR_FULL)
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_packed++;
#endif
          self.load_packed(ptr);
        }
        // partial
        else
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_packed_n++;
#endif
          self.load_packed_n(ptr, ref.m_tile.m_size[0]);
        }
      }
      // strided data
      else
      {
        // full vector?
        if (TENSOR_SIZE == RAJA::internal::expt::TENSOR_FULL)
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_strided++;
#endif
          self.load_strided(ptr, ref.m_stride[0]);
        }
        // partial
        else
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_strided_n++;
#endif
          self.load_strided_n(ptr, ref.m_stride[0], ref.m_tile.m_size[0]);
        }
      }
    }

    /*!
     * @brief Performs load specified by StaticTensorRef object.
     */
    RAJA_HOST_DEVICE

    RAJA_INLINE
    static void store_ref(self_type const& self, RefType& ref)
    {

      auto ptr = ref.m_pointer + ref.m_tile.m_begin[0] * ref.m_stride[0];

      // check for packed data
      if (STRIDE_ONE_DIM == 0)
      {
        // full vector?
        if (TENSOR_SIZE == RAJA::internal::expt::TENSOR_FULL)
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_packed++;
#endif
          self.store_packed(ptr);
        }
        // partial
        else
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_packed_n++;
#endif
          self.store_packed_n(ptr, ref.m_tile.m_size[0]);
        }
      }
      // strided data
      else
      {
        // full vector?
        if (TENSOR_SIZE == RAJA::internal::expt::TENSOR_FULL)
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_strided++;
#endif
          self.store_strided(ptr, ref.m_stride[0]);
        }
        // partial
        else
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_strided_n++;
#endif
          self.store_strided_n(ptr, ref.m_stride[0], ref.m_tile.m_size[0]);
        }
      }
    }
  };

  /*!
   * Loads a dense full vector from memory
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& load_packed(element_type const* ptr)
  {
    for (camp::idx_t reg = 0; reg < s_num_full_registers; ++reg)
    {
      m_registers[reg].load_packed(ptr + reg * s_register_num_elem);
    }
    if (s_num_partial_lanes)
    {
      m_registers[s_final_register].load_packed_n(
          ptr + s_final_register * s_register_num_elem, s_num_partial_lanes);
    }
    return *this;
  }

  /*!
   * Loads a strided full vector from memory
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& load_strided(element_type const* ptr, int stride)
  {
    for (camp::idx_t reg = 0; reg < s_num_full_registers; ++reg)
    {
      m_registers[reg].load_strided(ptr + reg * s_register_num_elem * stride,
                                    stride);
    }
    if (s_num_partial_lanes)
    {
      m_registers[s_final_register].load_strided_n(
          ptr + s_final_register * s_register_num_elem * stride, stride,
          s_num_partial_lanes);
    }
    return *this;
  }

  /*!
   * Loads a dense partial vector from memory
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& load_packed_n(element_type const* ptr, int N)
  {
    for (camp::idx_t reg = 0; reg < s_num_full_registers; ++reg)
    {
      if (N >= reg * s_register_num_elem + s_register_num_elem)
      {
        m_registers[reg].load_packed(ptr + reg * s_register_num_elem);
      }
      else
      {
        m_registers[reg].load_packed_n(ptr + reg * s_register_num_elem,
                                       N - reg * s_register_num_elem);

        for (camp::idx_t r = reg + 1; r < s_num_full_registers; ++r)
        {
          m_registers[r].broadcast(0);
        }
        return *this;
      }
    }
    if (s_num_partial_lanes)
    {
      m_registers[s_final_register].load_packed_n(
          ptr + s_final_register * s_register_num_elem,
          N - s_final_register * s_register_num_elem);
    }
    return *this;
  }

  /*!
   * Loads a strided partial vector from memory
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& load_strided_n(element_type const* ptr, int stride, int N)
  {
    for (camp::idx_t reg = 0; reg < s_num_full_registers; ++reg)
    {
      if (N >= reg * s_register_num_elem + s_register_num_elem)
      {
        m_registers[reg].load_strided(ptr + reg * s_register_num_elem * stride,
                                      stride);
      }
      else
      {
        m_registers[reg].load_strided_n(ptr +
                                            reg * s_register_num_elem * stride,
                                        stride, N - reg * s_register_num_elem);
        for (camp::idx_t r = reg + 1; r < s_num_full_registers; ++r)
        {
          m_registers[r].broadcast(0);
        }
        return *this;
      }
    }
    if (s_num_partial_lanes)
    {
      m_registers[s_final_register].load_strided_n(
          ptr + s_final_register * s_register_num_elem * stride, stride,
          N - s_final_register * s_register_num_elem);
    }
    return *this;
  }

  /*!
   * @brief Generic gather operation for full vector.
   *
   * Must provide another register containing offsets of all values
   * to be loaded relative to supplied pointer.
   *
   * Offsets are element-wise, not byte-wise.
   *
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type& gather(element_type const* ptr, int_vector_type offsets)
  {
    for (camp::idx_t reg = 0; reg < s_num_full_registers; ++reg)
    {
      m_registers[reg].gather(ptr, offsets.vec(reg));
    }
    if (s_num_partial_lanes)
    {
      m_registers[s_final_register].gather_n(ptr, offsets.vec(s_final_register),
                                             s_num_partial_lanes);
    }
    return *this;
  }

  /*!
   * @brief Generic gather operation for n-length subvector.
   *
   * Must provide another register containing offsets of all values
   * to be loaded relative to supplied pointer.
   *
   * Offsets are element-wise, not byte-wise.
   *
   */
  RAJA_INLINE
  self_type& gather_n(element_type const* ptr,
                      int_vector_type offsets,
                      camp::idx_t N)
  {
    for (camp::idx_t reg = 0; reg < s_num_full_registers; ++reg)
    {
      if (N >= reg * s_register_num_elem + s_register_num_elem)
      {
        m_registers[reg].gather(ptr, offsets.vec(reg));
      }
      else
      {
        m_registers[reg].gather_n(ptr, offsets.vec(reg),
                                  N - reg * s_register_num_elem);
        for (camp::idx_t r = reg + 1; r < s_num_full_registers; ++r)
        {
          m_registers[r].broadcast(0);
        }
        return *this;
      }
    }
    if (s_num_partial_lanes)
    {
      m_registers[s_final_register].gather_n(ptr, offsets.vec(s_final_register),
                                             N - s_final_register *
                                                     s_register_num_elem);
    }
    return *this;
  }

  /*!
   * Loads a dense full vector from memory
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& store_packed(element_type* ptr) const
  {
    for (camp::idx_t reg = 0; reg < s_num_full_registers; ++reg)
    {
      m_registers[reg].store_packed(ptr + reg * s_register_num_elem);
    }
    if (s_num_partial_lanes)
    {
      m_registers[s_final_register].store_packed_n(
          ptr + s_final_register * s_register_num_elem, s_num_partial_lanes);
    }
    return *this;
  }

  /*!
   * Loads a strided full vector from memory
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& store_strided(element_type* ptr, int stride) const
  {
    for (camp::idx_t reg = 0; reg < s_num_full_registers; ++reg)
    {
      m_registers[reg].store_strided(ptr + reg * s_register_num_elem * stride,
                                     stride);
    }
    if (s_num_partial_lanes)
    {
      m_registers[s_final_register].store_strided_n(
          ptr + s_final_register * s_register_num_elem * stride, stride,
          s_num_partial_lanes);
    }
    return *this;
  }

  /*!
   * Loads a dense partial vector from memory
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& store_packed_n(element_type* ptr, int N) const
  {
    for (camp::idx_t reg = 0; reg < s_num_full_registers; ++reg)
    {
      if (N >= reg * s_register_num_elem + s_register_num_elem)
      {
        m_registers[reg].store_packed(ptr + reg * s_register_num_elem);
      }
      else
      {
        m_registers[reg].store_packed_n(ptr + reg * s_register_num_elem,
                                        N - reg * s_register_num_elem);
        return *this;
      }
    }
    if (s_num_partial_lanes)
    {
      m_registers[s_final_register].store_packed_n(
          ptr + s_final_register * s_register_num_elem,
          N - s_final_register * s_register_num_elem);
    }
    return *this;
  }

  /*!
   * Loads a strided partial vector from memory
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& store_strided_n(element_type* ptr, int stride, int N) const
  {
    for (camp::idx_t reg = 0; reg < s_num_full_registers; ++reg)
    {
      if (N >= reg * s_register_num_elem + s_register_num_elem)
      {
        m_registers[reg].store_strided(ptr + reg * s_register_num_elem * stride,
                                       stride);
      }
      else
      {
        m_registers[reg].store_strided_n(ptr +
                                             reg * s_register_num_elem * stride,
                                         stride, N - reg * s_register_num_elem);
        return *this;
      }
    }
    if (s_num_partial_lanes)
    {
      m_registers[s_final_register].store_strided_n(
          ptr + s_final_register * s_register_num_elem * stride, stride,
          N - s_final_register * s_register_num_elem);
    }
    return *this;
  }

  /*!
   * @brief Generic scatter operation for full vector.
   *
   * Must provide another register containing offsets of all values
   * to be stored relative to supplied pointer.
   *
   * Offsets are element-wise, not byte-wise.
   *
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& scatter(element_type* ptr,
                           int_vector_type const& offsets) const
  {
    for (camp::idx_t reg = 0; reg < s_num_full_registers; ++reg)
    {
      m_registers[reg].scatter(ptr, offsets.vec(reg));
    }
    if (s_num_partial_lanes)
    {
      m_registers[s_final_register].scatter_n(
          ptr, offsets.vec(s_final_register), s_num_partial_lanes);
    }
    return *this;
  }

  /*!
   * @brief Generic scatter operation for n-length subvector.
   *
   * Must provide another register containing offsets of all values
   * to be stored relative to supplied pointer.
   *
   * Offsets are element-wise, not byte-wise.
   *
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& scatter_n(element_type* ptr,
                             int_vector_type const& offsets,
                             camp::idx_t N) const
  {
    for (camp::idx_t reg = 0; reg < s_num_full_registers; ++reg)
    {
      if (N >= reg * s_register_num_elem + s_register_num_elem)
      {
        m_registers[reg].scatter(ptr, offsets.vec(reg));
      }
      else
      {
        m_registers[reg].scatter_n(ptr, offsets.vec(reg),
                                   N - reg * s_register_num_elem);

        return *this;
      }
    }
    if (s_num_partial_lanes)
    {
      m_registers[s_final_register].scatter_n(
          ptr, offsets.vec(s_final_register),
          N - s_num_full_registers * s_register_num_elem);
    }
    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide(self_type const& den) const
  {
    self_type result;
    for (camp::idx_t reg = 0; reg < s_num_full_registers; ++reg)
    {
      result.vec(reg) = m_registers[reg].divide(den.vec(reg));
    }
    if (s_num_partial_lanes)
    {
      result.vec(s_final_register) = m_registers[s_final_register].divide_n(
          den.vec(s_final_register), s_num_partial_lanes);
    }
    return result;
  }

  /*!
   * @brief Divide n elements of this vector by another vector
   * @param x Vector to divide by
   * @param n Number of elements to divide
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide_n(self_type const& b, camp::idx_t n) const
  {
    self_type q(*this);
    for (camp::idx_t i = 0; i < n; ++i)
    {
      q.set(this->get(i) / b.get(i), i);
    }
    return q;
  }

  /*!
   * @brief Divide n elements of this vector by a scalar
   * @param x Scalar to divide by
   * @param n Number of elements to divide
   * @return Value of (*this)+x
   */
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide_n(element_type const& b, camp::idx_t n) const
  {
    self_type q(*this);
    for (camp::idx_t i = 0; i < n; ++i)
    {
      q.set(this->get(i) / b, i);
    }
    return q;
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  element_type min() const
  {
    // special case where there's just one parital register
    if (s_num_full_registers == 0)
    {
      return m_registers[0].min_n(s_num_partial_lanes);
    }

    element_type result = m_registers[0].min();
    for (camp::idx_t i = 1; i < s_num_full_registers; ++i)
    {
      result = RAJA::min<element_type>(result, m_registers[i].min());
    }
    if (s_num_partial_lanes)
    {
      result = RAJA::min<element_type>(
          result, m_registers[s_final_register].min_n(s_num_partial_lanes));
    }
    return result;
  }

  /*!
   * @brief Returns the smallest element over the first N lanes
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  element_type min_n(int N) const
  {
    // special case where there's just one parital register
    if (N < s_register_num_elem)
    {
      return m_registers[0].min_n(N);
    }

    element_type result = m_registers[0].min();
    for (camp::idx_t reg = 1; reg < s_num_full_registers; ++reg)
    {
      if (N >= reg * s_register_num_elem + s_register_num_elem)
      {
        result = RAJA::min<element_type>(result, m_registers[reg].min());
      }
      else
      {
        return RAJA::min<element_type>(
            result, m_registers[reg].min_n(N - reg * s_register_num_elem));
      }
    }
    if (N - s_num_full_registers * s_register_num_elem > 0)
    {
      result = RAJA::min<element_type>(
          result, m_registers[s_final_register].min_n(
                      N - s_final_register * s_register_num_elem));
    }
    return result;
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  element_type max() const
  {
    // special case where there's just one parital register
    if (s_num_full_registers == 0)
    {
      return m_registers[0].max_n(s_num_partial_lanes);
    }

    element_type result = m_registers[0].max();
    for (camp::idx_t i = 1; i < s_num_full_registers; ++i)
    {
      result = RAJA::max<element_type>(result, m_registers[i].max());
    }
    if (s_num_partial_lanes)
    {
      result = RAJA::max<element_type>(
          result, m_registers[s_final_register].max_n(s_num_partial_lanes));
    }
    return result;
  }

  /*!
   * @brief Returns the largest element over the first N lanes
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  element_type max_n(int N) const
  {
    // special case where there's just one parital register
    if (N < s_register_num_elem)
    {
      return m_registers[0].max_n(N);
    }

    element_type result = m_registers[0].max();
    for (camp::idx_t reg = 1; reg < s_num_full_registers; ++reg)
    {
      if (N >= reg * s_register_num_elem + s_register_num_elem)
      {
        result = RAJA::max<element_type>(result, m_registers[reg].max());
      }
      else
      {
        return RAJA::max<element_type>(
            result, m_registers[reg].max_n(N - reg * s_register_num_elem));
      }
    }
    if (N - s_num_full_registers * s_register_num_elem > 0)
    {
      result = RAJA::max<element_type>(
          result, m_registers[s_final_register].max_n(
                      N - s_final_register * s_register_num_elem));
    }
    return result;
  }

  /*!
   * @brief Returns the sum of all elements
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  element_type sum() const
  {
    // first do a vector sum of all registers
    register_type s = m_registers[0];
    for (camp::idx_t i = 1; i < s_num_registers; ++i)
    {
      s += m_registers[i];
    }
    // then a horizontal sum of result
    return s.sum();
  }

  /*!
   * @brief The * operator of two vectors is a element-wise multiply
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type operator*(self_type const& x) const { return this->multiply(x); }

  /*!
   * @brief The dot product of two vectors
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  element_type dot(self_type const& x) const
  {
    element_type dp(0);
    for (camp::idx_t i = 0; i < s_num_registers; ++i)
    {
      dp += m_registers[i].dot(x.vec(i));
    }
    return dp;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& set(element_type val, int idx)
  {
    m_registers[to_register(idx)].set(val, to_lane(idx));
    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  element_type get(int idx) const
  {
    return m_registers[to_register(idx)].get(to_lane(idx));
  }

  /*!
   * @brief Converts to vector to a string
   *
   *
   */
  RAJA_INLINE
  std::string to_string() const
  {
    std::string s = "Vector(" + std::to_string(s_num_elem) + ")[ ";

    //
    for (camp::idx_t i = 0; i < s_num_elem; ++i)
    {
      s += std::to_string(this->get(i)) + " ";
    }

    camp::idx_t physical_size = s_num_registers * s_register_num_elem;
    if (s_num_elem < physical_size)
    {
      s += "{";
      for (camp::idx_t i = s_num_elem; i < physical_size; ++i)
      {
        s += std::to_string(this->get(i)) + " ";
      }
      s += "}";
    }


    s += " ]\n";

    return s;
  }
};


}  // namespace expt
}  // namespace RAJA

// Bring in the register policy file so we get the default register type
// and all of the register traits setup
#include "RAJA/policy/tensor/arch.hpp"


#endif
